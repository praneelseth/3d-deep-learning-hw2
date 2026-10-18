"""Recompute metrics from saved predictions without rerunning a detector.

    python rescore.py outputs/ransac/val_full --data val_data.h5
    python rescore.py outputs/seg/test --data test_data.h5 --min-gt-points 20
"""
import argparse
import os
import pickle
import warnings

from data import ABCParts
from metrics import DEFAULT_MIN_GT_POINTS, MetricAccumulator, evaluate_shape, format_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="directory containing preds.pkl")
    ap.add_argument("--data", required=True)
    ap.add_argument("--min-gt-points", type=int, default=DEFAULT_MIN_GT_POINTS)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    with open(os.path.join(args.run_dir, "preds.pkl"), "rb") as f:
        preds = pickle.load(f)
    ds = ABCParts(args.data)
    acc = MetricAccumulator()
    for i, pred in preds.items():
        s = ds[i]
        labels, prims = pred[0], pred[1]
        acc.add(evaluate_shape(s["points"], s["labels"], s["types"], labels, prims, args.min_gt_points))
    print(format_summary(acc.summary()))


if __name__ == "__main__":
    main()
