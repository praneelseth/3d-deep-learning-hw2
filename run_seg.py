"""Evaluate the learned pipeline (type segmentation network + grouping and fitting).

    python run_seg.py --data val_data.h5 --n 300 --ckpt checkpoints/typeseg/best.pt --tag dev
    python run_seg.py --data val_data.h5 --n 300 --gt-types --tag oracle_types   # perfect classifier
    python run_seg.py --data test_data.h5 --n 0 --ckpt checkpoints/typeseg/best.pt --tag test

Network inference runs in the main process (MPS/CUDA if available); grouping
and fitting run in --workers processes. Every GroupConfig field is a flag.
Besides the primitive metrics it reports per-point type accuracy and IoU of
the network on the full 10k-point clouds.
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
import torch

from data import ABCParts
from metrics import DEFAULT_MIN_GT_POINTS, MetricAccumulator, evaluate_shape, format_summary
from primitives import BACKGROUND, TYPE_NAMES
from seg_model import TypeSegNet, predict_type_probs
from seg_pipeline import GroupConfig, group_and_fit, smooth_probs
from train_seg import CLASS_FREQ, confusion, seg_scores
from viz import instance_colors, plot_panels, type_colors

_ds = None


def _work(job):
    global _ds
    path, i, types, cfg, min_gt_points = job
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if _ds is None:
        _ds = ABCParts(path)
    s = _ds[i]
    t0 = time.time()
    labels, prims = group_and_fit(s["points"], s["normals"], types, cfg)
    seconds = time.time() - t0
    records = evaluate_shape(s["points"], s["labels"], s["types"], labels, prims, min_gt_points)
    on_background = int(((labels >= 0) & (s["types"] == BACKGROUND)).sum())
    return i, labels, prims, records, seconds, on_background, int((labels >= 0).sum())


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="val_data.h5")
    ap.add_argument("--n", type=int, default=300, help="random subset size, 0 for the whole split")
    ap.add_argument("--ckpt", default="checkpoints/typeseg/best.pt")
    ap.add_argument("--gt-types", action="store_true", help="use ground-truth types instead of the network")
    ap.add_argument("--passes", type=int, default=2, help="inference passes over random splits, averaged")
    ap.add_argument("--prior-power", type=float, default=0.0,
                    help="predict argmax(p / freq^a): a > 0 favors rare classes")
    ap.add_argument("--probs-cache", default=None, help="pickle of network probabilities to reuse or create")
    ap.add_argument("--smooth-iters", type=int, default=0, help="edge-aware probability smoothing iterations")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-gt-points", type=int, default=DEFAULT_MIN_GT_POINTS)
    ap.add_argument("--viz", type=int, default=6)
    ap.add_argument("--tag", default="default")
    for f in fields(GroupConfig):
        if isinstance(f.default, bool):
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    args = ap.parse_args()
    return args, GroupConfig(**{f.name: getattr(args, f.name) for f in fields(GroupConfig)})


def network_probs(args, ds, indices):
    if args.probs_cache and os.path.exists(args.probs_cache):
        with open(args.probs_cache, "rb") as f:
            cached = pickle.load(f)
        if all(int(i) in cached for i in indices):
            return cached
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    model = TypeSegNet(features=ck["args"].get("features", "raw")).to(device)
    model.load_state_dict(ck["model"])
    num_points, k = ck["args"]["num_points"], ck["args"]["k"]
    probs = {}
    for i in indices:
        s = ds[int(i)]
        probs[int(i)] = predict_type_probs(model, s["points"], s["normals"], device, num_points, k, args.passes,
                                           seed=int(i)).astype(np.float16)
    if args.probs_cache:
        with open(args.probs_cache, "wb") as f:
            pickle.dump(probs, f)
    return probs


def predict_all(args, ds, indices):
    if args.gt_types:
        return {int(i): ds[int(i)]["types"] for i in indices}, None
    probs = network_probs(args, ds, indices)
    scale = CLASS_FREQ ** -args.prior_power
    types, conf = {}, torch.zeros(4, 4, dtype=torch.long)
    for i in indices:
        s = ds[int(i)]
        p = smooth_probs(s["points"], s["normals"], probs[int(i)], args.smooth_iters)
        types[int(i)] = (p * scale).argmax(1)
        conf += confusion(torch.from_numpy(types[int(i)]), torch.from_numpy(s["types"]))
    return types, seg_scores(conf)


def main():
    args, cfg = parse_args()
    out = os.path.join("outputs", "seg", args.tag)
    os.makedirs(out, exist_ok=True)
    ds = ABCParts(args.data)
    indices = np.arange(len(ds)) if args.n <= 0 else np.random.default_rng(0).permutation(len(ds))[: args.n]

    t0 = time.time()
    types, point_scores = predict_all(args, ds, indices)
    infer_sec = (time.time() - t0) / len(indices)

    acc = MetricAccumulator()
    preds, seconds, num_prims, on_bg, assigned = {}, [], [], 0, 0
    jobs = [(args.data, int(i), types[int(i)], cfg, args.min_gt_points) for i in indices]
    with Pool(args.workers) as pool:
        for i, labels, prims, records, sec, bg, n_assigned in pool.imap(_work, jobs, chunksize=4):
            acc.add(records)
            preds[i] = (labels, prims, types[i])
            seconds.append(sec)
            num_prims.append(len(prims))
            on_bg += bg
            assigned += n_assigned

    summary = acc.summary()
    summary.update(config=asdict(cfg), data=args.data, n=len(indices), gt_types=args.gt_types,
                   prior_power=args.prior_power, smooth_iters=args.smooth_iters,
                   ckpt=None if args.gt_types else args.ckpt, point_scores=point_scores,
                   infer_sec_per_shape=infer_sec, group_sec_per_shape=float(np.mean(seconds)),
                   prims_per_shape=float(np.mean(num_prims)), pred_points_on_background=on_bg / max(assigned, 1))
    print(f"[{args.tag}] {args.data} n={len(indices)}  types={'GT' if args.gt_types else args.ckpt}  "
          f"infer {infer_sec:.2f}s + group {np.mean(seconds):.2f}s per shape  "
          f"{np.mean(num_prims):.1f} prims/shape  pred pts on GT background {summary['pred_points_on_background']:.1%}")
    if point_scores:
        print("per-point types: " + "  ".join(f"{k}={v:.4f}" for k, v in point_scores.items()))
    print(format_summary(summary))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out, "preds.pkl"), "wb") as f:
        pickle.dump(preds, f)

    for i in indices[: args.viz]:
        s = ds[int(i)]
        labels, prims, point_types = preds[int(i)]
        plot_panels(s["points"], [(instance_colors(s["labels"]), "GT instances"),
                                  (instance_colors(labels), "predicted instances"),
                                  (type_colors(s["types"]), "GT types"),
                                  (type_colors(point_types), "network types")],
                    os.path.join(out, f"shape_{int(i):05d}.png"))
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()
