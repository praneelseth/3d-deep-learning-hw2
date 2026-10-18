"""Weighted least-squares fitting and point-to-surface distances for planes,
spheres and cones.

Every fitter takes points (N, 3), optional normals (N, 3) and optional
per-point weights (N,). Given the minimum number of points (3 for a plane,
4 for a sphere, 3 + normals for a cone) the same code returns the exact
minimal-sample solution, so RANSAC can reuse it for hypothesis generation.
"""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.optimize import least_squares

BACKGROUND, PLANE, SPHERE, CONE = 0, 1, 2, 3
TYPE_NAMES = ("background", "plane", "sphere", "cone")


# Batched geometry: K primitives against M shared points -> distances (K, M)
# and unit surface normals (K, M, 3) at the closest-direction surface point.
# Normal signs are arbitrary; compare them with |dot|.

def plane_geometry(pts, normal, offset):
    dist = np.abs(normal @ pts.T - offset[:, None])
    return dist, np.broadcast_to(normal[:, None, :], (len(normal), len(pts), 3))


def sphere_geometry(pts, center, radius):
    d = pts[None] - center[:, None]
    norm = np.linalg.norm(d, axis=-1)
    return np.abs(norm - radius[:, None]), d / np.maximum(norm, 1e-12)[..., None]


def cone_geometry(pts, apex, axis, angle, with_normals=True):
    d = pts[None] - apex[:, None]
    h = np.einsum("kmj,kj->km", d, axis)
    radial = d - h[..., None] * axis[:, None]
    r = np.linalg.norm(radial, axis=-1)
    ca, sa = np.cos(angle)[:, None], np.sin(angle)[:, None]
    # In the (h, r) half-plane the surface is the ray from the apex along (cos, sin).
    along = h * ca + r * sa
    dist = np.where(along >= 0, np.abs(h * sa - r * ca), np.sqrt(h * h + r * r))
    if not with_normals:
        return dist, None
    rhat = radial / np.maximum(r, 1e-12)[..., None]
    normals = ca[..., None] * rhat - sa[..., None] * axis[:, None]
    return dist, normals


def _normalized_weights(n, weights):
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=np.float64)
    return w / w.sum()


def _smallest_eigvec(x, w):
    """Direction of least weighted variance of the rows of x."""
    m = w @ x
    d = x - m
    cov = (d * w[:, None]).T @ d
    return np.linalg.eigh(cov)[1][:, 0], m


@dataclass
class Plane:
    """Points x with normal . x = offset."""
    normal: np.ndarray
    offset: float
    type_id: ClassVar[int] = PLANE

    def geometry(self, pts):
        dist, normals = plane_geometry(pts, self.normal[None], np.array([self.offset]))
        return dist[0], normals[0]

    def distance(self, pts):
        return np.abs(pts @ self.normal - self.offset)

    @classmethod
    def fit(cls, pts, normals=None, weights=None):
        w = _normalized_weights(len(pts), weights)
        n, c = _smallest_eigvec(pts, w)
        return cls(n, float(n @ c))


@dataclass
class Sphere:
    center: np.ndarray
    radius: float
    type_id: ClassVar[int] = SPHERE

    def geometry(self, pts):
        dist, normals = sphere_geometry(pts, self.center[None], np.array([self.radius]))
        return dist[0], normals[0]

    def distance(self, pts):
        return np.abs(np.linalg.norm(pts - self.center, axis=1) - self.radius)

    @classmethod
    def fit(cls, pts, normals=None, weights=None, refine=True):
        w = _normalized_weights(len(pts), weights)
        sw = np.sqrt(w)
        # Algebraic fit |p|^2 = 2 c.p + k with k = r^2 - |c|^2, centered for conditioning.
        m = w @ pts
        p = pts - m
        A = np.c_[2 * p, np.ones(len(p))]
        b = (p ** 2).sum(1)
        x = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)[0]
        c = x[:3]
        sphere = cls(c + m, float(np.sqrt(max(x[3] + c @ c, 1e-12))))
        if refine and len(pts) > 4:
            sphere = sphere._refine(pts, sw)
        return sphere

    def _refine(self, pts, sw, max_nfev=50):
        def residuals(x):
            return sw * (np.linalg.norm(pts - x[:3], axis=1) - x[3])

        sol = least_squares(residuals, np.r_[self.center, self.radius], max_nfev=max_nfev)
        return Sphere(sol.x[:3], float(abs(sol.x[3])))


@dataclass
class Cone:
    """Single-nappe cone: apex, unit axis pointing into the opening, half-angle in radians."""
    apex: np.ndarray
    axis: np.ndarray
    angle: float
    type_id: ClassVar[int] = CONE

    def geometry(self, pts):
        dist, normals = cone_geometry(pts, self.apex[None], self.axis[None], np.array([self.angle]))
        return dist[0], normals[0]

    def distance(self, pts):
        return cone_geometry(pts, self.apex[None], self.axis[None], np.array([self.angle]), False)[0][0]

    @classmethod
    def fit(cls, pts, normals, weights=None, refine=True):
        if normals is None:
            raise ValueError("Cone fitting needs normals")
        w = _normalized_weights(len(pts), weights)
        sw = np.sqrt(w)
        # Every tangent plane passes through the apex: n_i . a = n_i . p_i.
        # This and everything below is invariant to the sign of the normals.
        b = (normals * pts).sum(1)
        apex = np.linalg.lstsq(normals * sw[:, None], b * sw, rcond=None)[0]
        # Unit directions from the apex lie on a circle around the axis.
        u = pts - apex
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        axis, mean_u = _smallest_eigvec(u, w)
        if axis @ mean_u < 0:
            axis = -axis
        angle = float(w @ np.arccos(np.clip(u @ axis, -1.0, 1.0)))
        cone = cls(apex, axis, float(np.clip(angle, 1e-3, np.pi / 2 - 1e-3)))
        if refine and len(pts) > 3:
            cone = cone._refine(pts, sw)
        return cone

    def _refine(self, pts, sw, max_nfev=100):
        def residuals(x):
            return sw * Cone(x[:3], x[3:6] / np.linalg.norm(x[3:6]), x[6]).distance(pts)

        x0 = np.r_[self.apex, self.axis, self.angle]
        lo = np.r_[np.full(6, -np.inf), 1e-3]
        hi = np.r_[np.full(6, np.inf), np.pi / 2 - 1e-3]
        sol = least_squares(residuals, x0, bounds=(lo, hi), max_nfev=max_nfev)
        if sol.cost >= 0.5 * np.sum(residuals(x0) ** 2):
            return self
        return Cone(sol.x[:3], sol.x[3:6] / np.linalg.norm(sol.x[3:6]), float(sol.x[6]))


@dataclass
class Cylinder:
    """Not a target type: RANSAC uses it to explain away cylinders, which make
    up ~23% of the points and otherwise get covered by thin, tilted cones."""
    center: np.ndarray
    axis: np.ndarray
    radius: float

    def geometry(self, pts):
        d = pts - self.center
        radial = d - (d @ self.axis)[:, None] * self.axis
        r = np.linalg.norm(radial, axis=1)
        return np.abs(r - self.radius), radial / np.maximum(r, 1e-12)[:, None]

    def distance(self, pts):
        return self.geometry(pts)[0]

    @classmethod
    def fit(cls, pts, normals, weights=None):
        w = _normalized_weights(len(pts), weights)
        # Normals are perpendicular to the axis; this second moment ignores their sign.
        axis = np.linalg.eigh((normals * w[:, None]).T @ normals)[1][:, 0]
        u = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        # Algebraic circle fit in the plane perpendicular to the axis.
        m = w @ pts
        q = np.c_[(pts - m) @ u, (pts - m) @ v]
        sw = np.sqrt(w)
        A = np.c_[2 * q, np.ones(len(q))]
        x = np.linalg.lstsq(A * sw[:, None], (q ** 2).sum(1) * sw, rcond=None)[0]
        c = x[:2]
        return cls(m + c[0] * u + c[1] * v, axis, float(np.sqrt(max(x[2] + c @ c, 1e-12))))


PRIMITIVE_CLASSES = {PLANE: Plane, SPHERE: Sphere, CONE: Cone}


def fit_primitive(type_id, pts, normals=None, weights=None, **kwargs):
    return PRIMITIVE_CLASSES[type_id].fit(pts, normals, weights, **kwargs)
