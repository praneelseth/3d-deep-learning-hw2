"""Step-1 check: score the ground truth itself through the full pipeline.

Each ground-truth plane/sphere/cone segment becomes a "prediction" with its
true type and parameters fitted by least squares. Seg-IoU and type accuracy
must come out at 1 and the residual near 0; anything else is a bug in the
loader, the fitters or the metrics. Also prints how well each fitter explains
each true type, which is the separation RANSAC has to rely on.

    python eval_oracle.py --data test_data.h5 --n 500
"""
import argparse
import os
import warnings

import numpy as np

from data import ABCParts
from metrics import DEFAULT_MIN_GT_POINTS, EVAL_TYPES, MetricAccumulator, evaluate_shape, format_summary
from primitives import TYPE_NAMES, fit_primitive
from viz import instance_colors, plot_panels, type_colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="test_data.h5")
    ap.add_argument("--n", type=int, default=500, help="number of shapes (random subset)")
    ap.add_argument("--min-gt-points", type=int, default=DEFAULT_MIN_GT_POINTS)
    ap.add_argument("--viz", type=int, default=4, help="number of shapes to plot")
    ap.add_argument("--out", default="outputs/oracle")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.makedirs(args.out, exist_ok=True)

    ds = ABCParts(args.data)
    indices = np.random.default_rng(0).permutation(len(ds))[: args.n]
    acc = MetricAccumulator()
    cross = {(t, f): [] for t in EVAL_TYPES for f in EVAL_TYPES}  # (true type, fitted type) -> residuals

    for k, i in enumerate(indices):
        s = ds[i]
        pts, normals, labels, types = s["points"], s["normals"], s["labels"], s["types"]
        pred_labels = np.full(len(pts), -1)
        prims = {}
        for l in np.unique(labels):
            mask = labels == l
            t = int(types[mask][0])
            if t not in EVAL_TYPES or mask.sum() < 4:
                continue
            prims[int(l)] = fit_primitive(t, pts[mask], normals[mask])
            pred_labels[mask] = l
            if mask.sum() >= args.min_gt_points:
                for f in EVAL_TYPES:
                    fit = fit_primitive(f, pts[mask], normals[mask])
                    cross[t, f].append(fit.distance(pts[mask]).mean())
        acc.add(evaluate_shape(pts, labels, types, pred_labels, prims, args.min_gt_points))

        if k < args.viz:
            dist = np.zeros(len(pts))
            for l, prim in prims.items():
                dist[pred_labels == l] = prim.distance(pts[pred_labels == l])
            heat = np.clip(dist / 0.01, 0, 1)
            heat_colors = np.c_[heat, 0.2 * np.ones_like(heat), 1 - heat]
            heat_colors[pred_labels < 0] = 0.85
            plot_panels(pts, [(instance_colors(labels), "GT instances"),
                              (type_colors(types), "GT types (plane/sphere/cone)"),
                              (heat_colors, "fit residual (red >= 0.01)")],
                        os.path.join(args.out, f"shape_{i:05d}.png"))

    print(format_summary(acc.summary()))
    print("\nMedian mean-residual of each fitter on each true type (rows true, cols fitted):")
    print(f"{'':<8}" + "".join(f"{TYPE_NAMES[f]:>10}" for f in EVAL_TYPES))
    for t in EVAL_TYPES:
        print(f"{TYPE_NAMES[t]:<8}" + "".join(f"{np.median(cross[t, f]) if cross[t, f] else np.nan:>10.1e}"
                                             for f in EVAL_TYPES))
    print(f"\nFigures in {args.out}/")


if __name__ == "__main__":
    main()
