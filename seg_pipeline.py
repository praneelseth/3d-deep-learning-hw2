"""Second stage of the learned pipeline: group points by predicted type into
instances and fit each instance by weighted least squares.

Methods, applied separately to each predicted type's points:
  cc      connected components of a radius graph; for planes an edge also
          needs near-parallel normals, which separates faces meeting at an
          edge (normals are consistently oriented in this data, so the
          signed angle also separates the two faces of a thin plate).
          Each component is fitted with iteratively reweighted least
          squares, and points far from the fit are left unassigned.
  ransac  the RANSAC detector restricted to that type.
  hybrid  cc, then components whose fit is poor (several surfaces merged)
          are re-split with type-restricted RANSAC.
With recheck, every instance is also least-squares fitted with the other
two types and switches to one that fits with under residual_ratio times the
mean residual, correcting per-point type errors at the instance level.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from primitives import CONE, PLANE, SPHERE, fit_primitive
from ransac import RansacConfig, _as_batch, _plausible, detect


@dataclass
class GroupConfig:
    method: str = "hybrid"
    min_points: int = 50
    conn_scale: float = 5.0           # graph radius = conn_scale * median NN spacing
    plane_angle_deg: float = 10.0     # max normal angle across a plane edge
    curved_angle_deg: float = 45.0    # same for sphere/cone neighbors
    fit_iters: int = 3
    trim_eps: float = 0.01            # farther from the fit -> unassigned
    split_residual: float = 0.002     # hybrid: re-split components with a larger mean residual
    recheck: bool = True
    residual_ratio: float = 0.5


def smooth_probs(points, normals, probs, iters=5, k=16, max_angle_deg=30.0):
    """Average class probabilities over each point's neighbors (and itself),
    `iters` times, ignoring neighbors across a normal discontinuity, so points
    on one surface converge to a common type instead of a speckle."""
    if iters <= 0:
        return probs
    nbrs = cKDTree(points).query(points, k=k + 1)[1]
    ok = (normals[nbrs] * normals[:, None]).sum(-1) > np.cos(np.deg2rad(max_angle_deg))
    w = ok / ok.sum(1, keepdims=True)
    probs = probs.astype(np.float32)
    for _ in range(iters):
        probs = np.einsum("nk,nkc->nc", w, probs[nbrs])
    return probs


def _components(points, normals, idx, radius, max_angle_deg):
    pairs = cKDTree(points[idx]).query_pairs(radius, output_type="ndarray")
    cos = (normals[idx[pairs[:, 0]]] * normals[idx[pairs[:, 1]]]).sum(1)
    pairs = pairs[cos > np.cos(np.deg2rad(max_angle_deg))]
    n = len(idx)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    return [idx[comp == c] for c in np.unique(comp)]


def _robust_fit(t, points, normals, idx, cfg):
    """IRLS with Tukey weights at scale trim_eps; returns the primitive and its inlier indices."""
    prim = fit_primitive(t, points[idx], normals[idx])
    for _ in range(cfg.fit_iters):
        d = prim.distance(points[idx])
        w = np.clip(1 - (d / cfg.trim_eps) ** 2, 0, None) ** 2
        if (w > 0).sum() < 5:
            break
        prim = fit_primitive(t, points[idx], normals[idx], w + 1e-9)
    return prim, idx[prim.distance(points[idx]) < cfg.trim_eps]


def _recheck(prim, pts, nrm, cfg, ransac_cfg):
    residual = prim.distance(pts).mean()
    for t in (PLANE, CONE, SPHERE):
        if t == prim.type_id:
            continue
        alt = fit_primitive(t, pts, nrm)
        if _plausible(t, _as_batch(alt), ransac_cfg)[0] and alt.distance(pts).mean() < cfg.residual_ratio * residual:
            prim, residual = alt, alt.distance(pts).mean()
    return prim


def group_and_fit(points, normals, types, cfg=GroupConfig(), ransac_cfg=RansacConfig()):
    """types: (N,) predicted type per point. Returns (labels with -1 unassigned, {label: primitive})."""
    labels = np.full(len(points), -1)
    prims = {}
    spacing = np.median(cKDTree(points).query(points, k=2)[0][:, 1])

    def add(prim, members):
        if cfg.recheck:
            prim = _recheck(prim, points[members], normals[members], cfg, ransac_cfg)
        prims[len(prims)] = prim
        labels[members] = len(prims) - 1

    def run_ransac(t, idx):
        rcfg = replace(ransac_cfg, types=(t,), reject_cylinders=False, min_inliers=cfg.min_points)
        sub_labels, sub_prims = detect(points[idx], normals[idx], rcfg)
        for l, prim in sub_prims.items():
            add(prim, idx[sub_labels == l])

    for t in (PLANE, SPHERE, CONE):
        idx = np.flatnonzero(types == t)
        if len(idx) < cfg.min_points:
            continue
        if cfg.method == "ransac":
            run_ransac(t, idx)
            continue
        angle = cfg.plane_angle_deg if t == PLANE else cfg.curved_angle_deg
        for comp in _components(points, normals, idx, cfg.conn_scale * spacing, angle):
            if len(comp) < cfg.min_points:
                continue
            prim, inliers = _robust_fit(t, points, normals, comp, cfg)
            poor = prim.distance(points[comp]).mean() > cfg.split_residual
            if cfg.method == "hybrid" and poor:
                run_ransac(t, comp)
            elif len(inliers) >= cfg.min_points:
                add(prim, inliers)
    return labels, prims
