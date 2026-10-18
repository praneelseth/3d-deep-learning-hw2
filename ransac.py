"""Sequential multi-type RANSAC for planes, spheres and cones.

Each round, on the points not yet explained:
  1. Hypotheses: for every type, draw minimal samples from local
     neighborhoods (3 points for a plane, 4 for a sphere, 3 points + normals
     for a cone) and solve for the parameters in closed form.
  2. Scoring: a point is an inlier if it lies within `eps` of the surface and
     its normal is within `max_angle_deg` of the surface normal there.
  3. Refinement: the best hypothesis per type is refit by weighted least
     squares on the largest connected component of its inliers, repeatedly.
  4. Selection: among types whose support is within `support_tol` of the
     largest, the first in `preference` wins unless a later one fits its
     support with under `residual_ratio` times the mean residual (on clean
     CAD data the true type fits far tighter). The preference order puts
     planes first and cones before spheres: a thin cone band and the sphere
     tangent to it along a circle fit equally well, and cones are ~7x more
     common in the data.
  5. Type re-check: every other type is least-squares fitted to the winning
     support and replaces the winner under the same residual rule, which
     catches segments found correctly whose own type had only poor minimal
     samples (thin cone bands).
  6. Cylinder rejection: if a cylinder fits the winner's support better than
     the winner does, the cylinder's connected inliers are removed as
     explained background instead of being reported. Otherwise the winner's
     points are labeled and removed.
Terminates after `patience` consecutive rounds that produce no candidate
with `min_inliers` connected inliers.
"""
from dataclasses import astuple, dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from primitives import (CONE, PLANE, PRIMITIVE_CLASSES, SPHERE, Cylinder, cone_geometry, fit_primitive,
                        plane_geometry, sphere_geometry)


@dataclass
class RansacConfig:
    types: tuple = (PLANE, SPHERE, CONE)
    # Defaults tuned on 300 val shapes (outputs/ransac/sweep*.log); the data is
    # clean CAD samples, so tight thresholds win.
    eps: float = 0.002              # inlier distance, unit-sphere units
    max_angle_deg: float = 8.0      # inlier normal deviation
    hypotheses: int = 100           # per type per round
    neighborhood: int = 64          # minimal samples come from a point's k nearest neighbors
    min_inliers: int = 50
    conn_scale: float = 5.0         # connectivity radius = conn_scale * median NN spacing
    refine_iters: int = 3
    patience: int = 3
    preference: tuple = (PLANE, CONE, SPHERE)
    support_tol: float = 0.1
    residual_ratio: float = 0.5
    max_sphere_radius: float = 1.5
    max_sphere_center: float = 3.0
    min_cone_angle_deg: float = 3.0   # below this a "cone" is really a cylinder
    max_cone_angle_deg: float = 85.0  # above this it is really a plane
    max_cone_apex: float = 20.0       # apex distance from the origin
    reject_cylinders: bool = True
    max_cylinder_radius: float = 1.5
    max_primitives: int = 60
    seed: int = 0


SAMPLE_SIZE = {PLANE: 3, SPHERE: 4, CONE: 3}
GEOMETRY = {PLANE: plane_geometry, SPHERE: sphere_geometry, CONE: cone_geometry}


# Minimal solvers: P, N are (K, m, 3) samples; return (valid mask, batched params).

def _plane_hypotheses(P, N):
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    length = np.linalg.norm(n, axis=1)
    n = n / np.maximum(length, 1e-12)[:, None]
    return length > 1e-12, (n, (n * P[:, 0]).sum(1))


def _sphere_hypotheses(P, N):
    # |p_i - c|^2 = |p_0 - c|^2  ->  2 (p_i - p_0) . c = |p_i|^2 - |p_0|^2
    A = 2 * (P[:, 1:] - P[:, :1])
    b = (P[:, 1:] ** 2).sum(-1) - (P[:, :1] ** 2).sum(-1)
    ok = np.abs(np.linalg.det(A)) > 1e-12
    A[~ok] = np.eye(3)
    c = np.linalg.solve(A, b[..., None])[..., 0]
    return ok, (c, np.linalg.norm(P[:, 0] - c, axis=1))


def _cone_hypotheses(P, N):
    # The three tangent planes meet at the apex; the unit directions from the
    # apex to the samples lie on a circle whose normal is the axis.
    b = (N * P).sum(-1)
    ok = np.abs(np.linalg.det(N)) > 1e-9
    A = N.copy()
    A[~ok] = np.eye(3)
    apex = np.linalg.solve(A, b[..., None])[..., 0]
    u = P - apex[:, None]
    u /= np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-12)
    axis = np.cross(u[:, 1] - u[:, 0], u[:, 2] - u[:, 0])
    length = np.linalg.norm(axis, axis=1)
    ok &= length > 1e-12
    axis /= np.maximum(length, 1e-12)[:, None]
    axis[(axis * u.sum(1)).sum(-1) < 0] *= -1
    angle = np.arccos(np.clip((u * axis[:, None]).sum(-1), -1, 1)).mean(1)
    return ok, (apex, axis, angle)


HYPOTHESES = {PLANE: _plane_hypotheses, SPHERE: _sphere_hypotheses, CONE: _cone_hypotheses}


def _plausible(type_id, params, cfg):
    """Reject degenerate parameters: near-flat spheres, cylinder- or plane-like cones."""
    if type_id == SPHERE:
        center, radius = params
        return (radius < cfg.max_sphere_radius) & (np.linalg.norm(center, axis=-1) < cfg.max_sphere_center)
    if type_id == CONE:
        apex, _, angle = params
        return ((angle > np.deg2rad(cfg.min_cone_angle_deg)) & (angle < np.deg2rad(cfg.max_cone_angle_deg))
                & (np.linalg.norm(apex, axis=-1) < cfg.max_cone_apex))
    return np.ones(len(params[0]), dtype=bool)


def _as_batch(prim):
    return tuple(np.asarray(v)[None] for v in astuple(prim))


def _inlier_mask(dist, surface_normals, normals, cfg):
    cos = np.abs(np.einsum("...j,...j->...", surface_normals, normals))
    return (dist < cfg.eps) & (cos > np.cos(np.deg2rad(cfg.max_angle_deg)))


def _largest_component(points, idx, radius):
    if len(idx) < 2:
        return idx
    pairs = cKDTree(points[idx]).query_pairs(radius, output_type="ndarray")
    n = len(idx)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    return idx[comp == np.bincount(comp).argmax()]


def _sample(rng, remaining, rem_idx, nbrs, k, m):
    """k minimal samples of size m: a random seed point plus m-1 remaining neighbors."""
    seeds = rng.choice(rem_idx, k)
    cand = nbrs[seeds, 1:]
    valid = remaining[cand]
    keys = np.where(valid, rng.random(cand.shape), 2.0)
    order = np.argsort(keys, axis=1)[:, : m - 1]
    ok = np.take_along_axis(valid, order, 1).all(1)
    return np.c_[seeds, np.take_along_axis(cand, order, 1)][ok]


def _best_hypothesis(t, points, normals, remaining, rem_idx, nbrs, rng, cfg):
    sample = _sample(rng, remaining, rem_idx, nbrs, cfg.hypotheses, SAMPLE_SIZE[t])
    if len(sample) == 0:
        return None
    ok, params = HYPOTHESES[t](points[sample], normals[sample])
    ok &= _plausible(t, params, cfg)
    if not ok.any():
        return None
    params = tuple(p[ok] for p in params)
    dist, surface_normals = GEOMETRY[t](points[rem_idx], *params)
    counts = _inlier_mask(dist, surface_normals, normals[rem_idx][None], cfg).sum(1)
    k = counts.argmax()
    if counts[k] < cfg.min_inliers:
        return None
    return PRIMITIVE_CLASSES[t](*(p[k] for p in params))


def _refine(t, prim, rem_idx, points, normals, cfg, radius):
    """Alternate support = largest connected inlier set, prim = weighted LS fit to it."""
    best_prim, best_support = prim, np.zeros(0, dtype=np.int64)
    for it in range(cfg.refine_iters + 1):
        dist, surface_normals = prim.geometry(points[rem_idx])
        inl = _inlier_mask(dist, surface_normals, normals[rem_idx], cfg)
        support = _largest_component(points, rem_idx[inl], radius)
        if len(support) <= len(best_support):
            break
        best_prim, best_support = prim, support
        if it == cfg.refine_iters or len(support) < cfg.min_inliers:
            break
        # Tukey weights: points near the surface count most.
        w = (1 - (prim.distance(points[support]) / cfg.eps) ** 2) ** 2 + 1e-6
        prim = fit_primitive(t, points[support], normals[support], w)
        if not _plausible(t, _as_batch(prim), cfg)[0]:
            break
    return best_prim, best_support


def detect(points, normals, cfg=RansacConfig()):
    """Returns (labels (N,) with -1 for unassigned, {label: primitive})."""
    rng = np.random.default_rng(cfg.seed)
    nn_dist, nbrs = cKDTree(points).query(points, k=min(cfg.neighborhood + 1, len(points)))
    radius = cfg.conn_scale * np.median(nn_dist[:, 1])
    labels = np.full(len(points), -1)
    prims = {}
    remaining = np.ones(len(points), dtype=bool)
    failures = 0
    while failures < cfg.patience and remaining.sum() >= cfg.min_inliers and len(prims) < cfg.max_primitives:
        rem_idx = np.flatnonzero(remaining)
        candidates = []
        for t in sorted(cfg.types, key=cfg.preference.index):
            prim = _best_hypothesis(t, points, normals, remaining, rem_idx, nbrs, rng, cfg)
            if prim is None:
                continue
            prim, support = _refine(t, prim, rem_idx, points, normals, cfg, radius)
            if len(support) >= cfg.min_inliers:
                candidates.append((prim, support, prim.distance(points[support]).mean()))
        if not candidates:
            failures += 1
            continue
        failures = 0
        prim, support = _select(candidates, cfg)
        prim, support = _recheck_type(prim, support, rem_idx, points, normals, cfg, radius)
        if cfg.reject_cylinders:
            explained = _cylinder_support(prim, support, rem_idx, points, normals, cfg, radius)
            if explained is not None:
                remaining[explained] = False
                continue
        prims[len(prims)] = prim
        labels[support] = len(prims) - 1
        remaining[support] = False
    return labels, prims


def _select(candidates, cfg):
    """candidates: (prim, support, mean residual) in preference order."""
    top = max(len(c[1]) for c in candidates)
    competitive = [c for c in candidates if len(c[1]) >= (1 - cfg.support_tol) * top]
    best = competitive[0]
    for c in competitive[1:]:
        if c[2] < cfg.residual_ratio * best[2]:
            best = c
    return best[0], best[1]


def _cylinder_support(prim, support, rem_idx, points, normals, cfg, radius):
    """Points to discard as a cylinder if one explains the support better, else None."""
    cyl = Cylinder.fit(points[support], normals[support])
    if cyl.radius >= cfg.max_cylinder_radius:
        return None
    if cyl.distance(points[support]).mean() >= prim.distance(points[support]).mean():
        return None
    dist, surface_normals = cyl.geometry(points[rem_idx])
    inl = _inlier_mask(dist, surface_normals, normals[rem_idx], cfg)
    return np.union1d(support, _largest_component(points, rem_idx[inl], radius))


def _recheck_type(prim, support, rem_idx, points, normals, cfg, radius):
    residual = prim.distance(points[support]).mean()
    for t in cfg.preference:
        if t == prim.type_id or t not in cfg.types:
            continue
        alt = fit_primitive(t, points[support], normals[support])
        if not _plausible(t, _as_batch(alt), cfg)[0]:
            continue
        if alt.distance(points[support]).mean() >= cfg.residual_ratio * residual:
            continue
        alt, alt_support = _refine(t, alt, rem_idx, points, normals, cfg, radius)
        if len(alt_support) >= (1 - cfg.support_tol) * len(support):
            prim, support = alt, alt_support
            residual = prim.distance(points[support]).mean()
    return prim, support
