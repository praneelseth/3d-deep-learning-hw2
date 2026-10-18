"""Evaluation metrics shared by the RANSAC and learned pipelines.

A prediction for one shape is
    pred_labels: (N,) int instance id per point, -1 for unassigned
    pred_prims:  {instance id: Plane | Sphere | Cone}

Ground-truth instances are the plane/sphere/cone segments with at least
`min_gt_points` points; cylinders, splines and slivers are not scored.
Predicted and ground-truth instances are matched one-to-one by maximum IoU
(Hungarian matching). Then
    Seg-IoU:  mean IoU over ground-truth instances (0 if unmatched),
              averaged per shape and then over shapes
    Type acc: fraction of matched instances whose predicted type is correct
    Residual: mean distance from a matched ground-truth instance's points to
              its predicted surface, averaged over matched instances
              (normalized units, i.e. the shape fits in the unit sphere)
The residual mean is dominated by a few spurious low-IoU matches (a ground
truth surface paired with a sliver of prediction on another surface), so the
summary also gives its median and its mean over matches with IoU >= 0.5.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from primitives import CONE, PLANE, SPHERE, TYPE_NAMES

EVAL_TYPES = (PLANE, SPHERE, CONE)
DEFAULT_MIN_GT_POINTS = 50


def _dense_ids(ids, keep):
    """Map the kept values of ids to 0..K-1 and everything else to -1."""
    out = np.full(len(ids), -1, dtype=np.int64)
    out[np.isin(ids, keep)] = np.searchsorted(keep, ids[np.isin(ids, keep)])
    return out


def evaluate_shape(points, gt_labels, gt_types, pred_labels, pred_prims,
                   min_gt_points=DEFAULT_MIN_GT_POINTS):
    """Per ground-truth instance records for one shape, or [] if nothing is scored."""
    ids, first, counts = np.unique(gt_labels, return_index=True, return_counts=True)
    keep = np.isin(gt_types[first], EVAL_TYPES) & (counts >= min_gt_points)
    gt_ids = ids[keep]
    if len(gt_ids) == 0:
        return []
    pred_ids = np.unique(pred_labels[pred_labels >= 0])

    g = _dense_ids(gt_labels, gt_ids)
    p = _dense_ids(pred_labels, pred_ids)
    G, P = len(gt_ids), len(pred_ids)
    both = (g >= 0) & (p >= 0)
    inter = np.bincount(g[both] * P + p[both], minlength=G * P).reshape(G, P)
    gt_size = np.bincount(g[g >= 0], minlength=G)
    pred_size = np.bincount(p[p >= 0], minlength=P)
    iou = inter / np.maximum(gt_size[:, None] + pred_size[None, :] - inter, 1)

    match = np.full(G, -1)
    if P > 0:
        rows, cols = linear_sum_assignment(iou, maximize=True)
        ok = iou[rows, cols] > 0
        match[rows[ok]] = cols[ok]

    records = []
    for k, gid in enumerate(gt_ids):
        mask = g == k
        rec = {"gt_type": int(gt_types[mask][0]), "size": int(gt_size[k]),
               "iou": 0.0, "matched": False, "pred_type": None, "residual": np.nan}
        if match[k] >= 0:
            prim = pred_prims[int(pred_ids[match[k]])]
            rec.update(iou=float(iou[k, match[k]]), matched=True,
                       pred_type=prim.type_id,
                       residual=float(prim.distance(points[mask]).mean()))
        records.append(rec)
    return records


class MetricAccumulator:
    def __init__(self):
        self.shapes = []

    def add(self, records):
        if records:
            self.shapes.append(records)

    def summary(self):
        recs = [r for shape in self.shapes for r in shape]
        matched = [r for r in recs if r["matched"]]
        out = {
            "num_shapes": len(self.shapes),
            "num_instances": len(recs),
            "seg_iou": float(np.mean([np.mean([r["iou"] for r in s]) for s in self.shapes])),
            "type_acc": float(np.mean([r["pred_type"] == r["gt_type"] for r in matched])) if matched else np.nan,
            "residual": float(np.mean([r["residual"] for r in matched])) if matched else np.nan,
            "residual_median": float(np.median([r["residual"] for r in matched])) if matched else np.nan,
            "residual_iou50": float(np.mean([r["residual"] for r in matched if r["iou"] >= 0.5] or [np.nan])),
            "match_rate": len(matched) / max(len(recs), 1),
            "per_type": {},
        }
        for t in EVAL_TYPES:
            rt = [r for r in recs if r["gt_type"] == t]
            mt = [r for r in rt if r["matched"]]
            out["per_type"][TYPE_NAMES[t]] = {
                "count": len(rt),
                "iou": float(np.mean([r["iou"] for r in rt])) if rt else np.nan,
                "type_acc": float(np.mean([r["pred_type"] == t for r in mt])) if mt else np.nan,
                "residual": float(np.mean([r["residual"] for r in mt])) if mt else np.nan,
            }
        return out


def format_summary(s):
    lines = [f"shapes={s['num_shapes']}  instances={s['num_instances']}  matched={s['match_rate']:.1%}",
             f"Seg-IoU={s['seg_iou']:.4f}  TypeAcc={s['type_acc']:.4f}  Residual={s['residual']:.2e}"
             f"  (median {s['residual_median']:.2e}, IoU>=0.5 {s['residual_iou50']:.2e})",
             f"{'type':<8}{'count':>8}{'IoU':>10}{'TypeAcc':>10}{'Residual':>12}"]
    for name, v in s["per_type"].items():
        lines.append(f"{name:<8}{v['count']:>8}{v['iou']:>10.4f}{v['type_acc']:>10.4f}{v['residual']:>12.2e}")
    return "\n".join(lines)
