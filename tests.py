"""Sanity tests for the fitters and metrics. Run with `python tests.py` or pytest."""
import numpy as np

from metrics import MetricAccumulator, evaluate_shape
from primitives import CONE, PLANE, SPHERE, Cone, Plane, Sphere

rng = np.random.default_rng(0)


def _frame(axis):
    axis = axis / np.linalg.norm(axis)
    u = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    return axis, u, np.cross(axis, u)


def sample_plane(n=500):
    normal, u, v = _frame(np.array([1.0, 2.0, 3.0]))
    pts = 0.3 * normal + rng.uniform(-1, 1, (n, 1)) * u + rng.uniform(-1, 1, (n, 1)) * v
    return pts, np.tile(normal, (n, 1)), Plane(normal, 0.3)


def sample_sphere(n=500, center=np.array([0.1, -0.2, 0.3]), radius=0.7):
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return center + radius * d, d, Sphere(center, radius)


def sample_cone(n=800):
    apex, angle = np.array([0.2, 0.1, -0.4]), np.deg2rad(25)
    axis, u, v = _frame(np.array([0.3, -0.5, 0.8]))
    t = rng.uniform(0.2, 1.0, n)
    phi = rng.uniform(0, 1.5 * np.pi, n)
    radial = np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v
    pts = apex + t[:, None] * (np.cos(angle) * axis + np.sin(angle) * radial)
    normals = np.cos(angle) * radial - np.sin(angle) * axis
    normals *= rng.choice([-1, 1], (n, 1))  # fitting must not depend on normal orientation
    return pts, normals, Cone(apex, axis, angle)


def test_fitters_exact_and_minimal():
    for sample, cls, k in [(sample_plane, Plane, 3), (sample_sphere, Sphere, 4), (sample_cone, Cone, 3)]:
        pts, normals, true = sample()
        assert true.distance(pts).max() < 1e-12
        assert cls.fit(pts, normals).distance(pts).max() < 1e-8
        assert cls.fit(pts[:k], normals[:k]).distance(pts).max() < 1e-8


def test_cone_distance_matches_brute_force():
    _, _, cone = sample_cone()
    axis, u, v = _frame(cone.axis)
    t = np.linspace(0, 4, 2000)[:, None, None]
    phi = np.linspace(0, 2 * np.pi, 400)[None, :, None]
    gen = np.cos(cone.angle) * axis + np.sin(cone.angle) * (np.cos(phi) * u + np.sin(phi) * v)
    surface = (cone.apex + t * gen).reshape(-1, 3)
    q = rng.uniform(-1, 1, (100, 3))
    brute = np.sqrt(((q[:, None] - surface[None]) ** 2).sum(-1)).min(1)
    assert np.abs(brute - cone.distance(q)).max() < 2e-3  # surface sampling resolution


def test_metrics_hand_example():
    # 10 points: GT plane on 0-5, GT sphere on 6-9.
    pts, _, plane = sample_plane(10)
    gt_labels = np.array([0] * 6 + [1] * 4)
    gt_types = np.array([PLANE] * 6 + [SPHERE] * 4)
    # Prediction 7 covers 0-3 (IoU 4/6 with the plane), prediction 8 covers 4-9 labeled cone
    # (IoU 2/6 with the plane, 4/6 with the sphere). Hungarian picks 7->plane, 8->sphere.
    pred_labels = np.array([7] * 4 + [8] * 6)
    recs = evaluate_shape(pts, gt_labels, gt_types, pred_labels,
                          {7: plane, 8: Cone(np.zeros(3), np.array([0, 0, 1.0]), 0.5)}, min_gt_points=1)
    assert np.allclose([r["iou"] for r in recs], [4 / 6, 4 / 6])
    assert [r["pred_type"] for r in recs] == [PLANE, CONE]
    assert recs[0]["residual"] < 1e-12
    acc = MetricAccumulator()
    acc.add(recs)
    s = acc.summary()
    assert np.isclose(s["seg_iou"], 4 / 6) and np.isclose(s["type_acc"], 0.5)

    # Too-small and non-target GT segments are not scored; empty predictions score 0.
    assert evaluate_shape(pts, gt_labels, gt_types, pred_labels, {}, min_gt_points=7) == []
    recs = evaluate_shape(pts, gt_labels, gt_types, np.full(10, -1), {}, min_gt_points=1)
    assert [r["iou"] for r in recs] == [0.0, 0.0] and not any(r["matched"] for r in recs)


def sample_cylinder(n=1500, center=np.array([3.0, -2.0, 0.0]), radius=0.5):
    axis, u, v = _frame(np.array([0.0, 1.0, 1.0]))
    phi = rng.uniform(0, 2 * np.pi, n)
    radial = np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v
    return center + radius * radial + rng.uniform(-1, 1, (n, 1)) * axis, radial


def test_ransac_synthetic_scene():
    from data import normalize
    from ransac import detect

    plane, plane_n, _ = sample_plane(3000)
    sphere, sphere_n, _ = sample_sphere(1500, center=np.array([3.0, 3.0, 0.0]), radius=0.6)
    cone, cone_n, _ = sample_cone(1500)
    cyl, cyl_n = sample_cylinder()
    pts = np.r_[plane, sphere, cone + np.array([-3.0, 0.0, 2.0]), cyl]
    normals = np.r_[plane_n, sphere_n, cone_n, cyl_n]
    gt_labels = np.repeat([0, 1, 2, 3], [3000, 1500, 1500, 1500])
    gt_types = np.repeat([PLANE, SPHERE, CONE, 0], [3000, 1500, 1500, 1500])
    pts, _, _ = normalize(pts)

    labels, prims = detect(pts, normals)
    recs = evaluate_shape(pts, gt_labels, gt_types, labels, prims)
    assert [r["pred_type"] for r in recs] == [PLANE, SPHERE, CONE]
    assert min(r["iou"] for r in recs) > 0.95
    assert (labels[gt_labels == 3] >= 0).mean() < 0.05  # the cylinder is not reported


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
