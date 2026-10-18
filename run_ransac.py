"""Run RANSAC on a split, report metrics, save predictions and figures.

    python run_ransac.py --data val_data.h5 --n 300 --tag base
    python run_ransac.py --data val_data.h5 --n 300 --tag eps01 --eps 0.01
    python run_ransac.py --data test_data.h5 --n 0 --tag final   # whole split

Every RansacConfig field is a flag (--types takes e.g. "plane,sphere").
Besides the assignment metrics it prints how many primitives are predicted
per shape and the fraction of predicted points that lie on ground-truth
background (cylinders, splines), i.e. false detections the metrics ignore.
"""
import argparse
import json
import os
import pickle
import time
import warnings
from dataclasses import asdict, fields
from multiprocessing import Pool

import numpy as np

from data import ABCParts
from metrics import DEFAULT_MIN_GT_POINTS, MetricAccumulator, evaluate_shape, format_summary
from primitives import BACKGROUND, TYPE_NAMES
from ransac import RansacConfig, detect
from viz import instance_colors, plot_panels, type_colors

_ds = None


def _work(job):
    global _ds
    path, i, cfg, min_gt_points = job
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if _ds is None:
        _ds = ABCParts(path)
    s = _ds[i]
    t0 = time.time()
    labels, prims = detect(s["points"], s["normals"], cfg)
    seconds = time.time() - t0
    records = evaluate_shape(s["points"], s["labels"], s["types"], labels, prims, min_gt_points)
    on_background = int(((labels >= 0) & (s["types"] == BACKGROUND)).sum())
    return i, labels, prims, records, seconds, on_background, int((labels >= 0).sum())


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="val_data.h5")
    ap.add_argument("--n", type=int, default=300, help="random subset size, 0 for the whole split")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-gt-points", type=int, default=DEFAULT_MIN_GT_POINTS)
    ap.add_argument("--viz", type=int, default=6)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--types", default="plane,sphere,cone")
    scalar_fields = [f for f in fields(RansacConfig) if f.name not in ("types", "preference")]
    for f in scalar_fields:
        if isinstance(f.default, bool):
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    args = ap.parse_args()
    cfg_kwargs = {f.name: getattr(args, f.name) for f in scalar_fields}
    cfg = RansacConfig(types=tuple(TYPE_NAMES.index(t) for t in args.types.split(",")), **cfg_kwargs)
    return args, cfg


def main():
    args, cfg = parse_args()
    out = os.path.join("outputs", "ransac", args.tag)
    os.makedirs(out, exist_ok=True)
    ds = ABCParts(args.data)
    indices = np.arange(len(ds)) if args.n <= 0 else np.random.default_rng(0).permutation(len(ds))[: args.n]

    acc = MetricAccumulator()
    preds, seconds, num_prims, on_bg, assigned = {}, [], [], 0, 0
    jobs = [(args.data, int(i), cfg, args.min_gt_points) for i in indices]
    with Pool(args.workers) as pool:
        for i, labels, prims, records, sec, bg, n_assigned in pool.imap(_work, jobs, chunksize=4):
            acc.add(records)
            preds[i] = (labels, prims)
            seconds.append(sec)
            num_prims.append(len(prims))
            on_bg += bg
            assigned += n_assigned

    summary = acc.summary()
    summary.update(config=asdict(cfg), data=args.data, n=len(indices), sec_per_shape=float(np.mean(seconds)),
                   prims_per_shape=float(np.mean(num_prims)), pred_points_on_background=on_bg / max(assigned, 1))
    print(f"[{args.tag}] {args.data} n={len(indices)}  {np.mean(seconds):.2f}s/shape  "
          f"{np.mean(num_prims):.1f} prims/shape  pred pts on GT background {summary['pred_points_on_background']:.1%}")
    print(format_summary(summary))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out, "preds.pkl"), "wb") as f:
        pickle.dump(preds, f)

    for i in indices[: args.viz]:
        s = ds[int(i)]
        labels, prims = preds[int(i)]
        pred_types = np.zeros(len(labels), dtype=np.int64)
        for l, prim in prims.items():
            pred_types[labels == l] = prim.type_id
        plot_panels(s["points"], [(instance_colors(s["labels"]), "GT instances"),
                                  (instance_colors(labels), "RANSAC instances"),
                                  (type_colors(s["types"]), "GT types"),
                                  (type_colors(pred_types), "RANSAC types")],
                    os.path.join(out, f"shape_{int(i):05d}.png"))
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()
