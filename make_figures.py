"""Report figures: training curves and RANSAC vs. learned-pipeline comparisons.

    python make_figures.py --data test_data.h5 --ransac outputs/ransac/test --seg outputs/seg/test \
        --history checkpoints/typeseg_geo2/history.json --out outputs/figures

--history also accepts a last.pt checkpoint, which stores the history so far.

Writes training_curves.png, and for shapes picked by per-shape Seg-IoU (best
and worst of each method, and where the methods disagree most) a figure with
GT instances / RANSAC instances / learned instances and the type maps.
"""
import argparse
import json
import os
import pickle
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import ABCParts
from metrics import evaluate_shape
from viz import instance_colors, plot_panels, type_colors


def load_preds(run_dir):
    with open(os.path.join(run_dir, "preds.pkl"), "rb") as f:
        return pickle.load(f)


def pred_types(labels, prims):
    types = np.zeros(len(labels), dtype=np.int64)
    for l, prim in prims.items():
        types[labels == l] = prim.type_id
    return types


def shape_iou(s, labels, prims):
    recs = evaluate_shape(s["points"], s["labels"], s["types"], labels, prims)
    return np.mean([r["iou"] for r in recs]) if recs else None


def training_curves(history_path, path):
    """history_path: history.json, or a last.pt checkpoint (which stores the history so far)."""
    if history_path.endswith(".pt"):
        import torch

        hist = torch.load(history_path, map_location="cpu", weights_only=False)["history"]
    else:
        with open(history_path) as f:
            hist = json.load(f)
    epochs = [h["epoch"] + 1 for h in hist]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["loss"] for h in hist], label="train loss (weighted CE)")
    axes[0].set_xlabel("epoch (1500 steps)")
    axes[0].legend()
    for key, name in [("miou", "mean"), ("iou_background", "background"), ("iou_plane", "plane"),
                      ("iou_sphere", "sphere"), ("iou_cone", "cone")]:
        axes[1].plot(epochs, [h[key] for h in hist], label=name, lw=2.5 if key == "miou" else 1.2)
    axes[1].set_xlabel("epoch (1500 steps)")
    axes[1].set_ylabel("val per-point IoU")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="test_data.h5")
    ap.add_argument("--ransac", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--history", default=None)
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--per-group", type=int, default=3)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.makedirs(args.out, exist_ok=True)
    if args.history:
        training_curves(args.history, os.path.join(args.out, "training_curves.png"))

    ds = ABCParts(args.data)
    ransac, seg = load_preds(args.ransac), load_preds(args.seg)
    common = sorted(set(ransac) & set(seg))
    scores = {}
    for i in common:
        s = ds[i]
        r, l = shape_iou(s, *ransac[i][:2]), shape_iou(s, *seg[i][:2])
        if r is not None:
            scores[i] = (r, l)
    ids = np.array(list(scores))
    r = np.array([scores[i][0] for i in ids])
    l = np.array([scores[i][1] for i in ids])
    k = args.per_group
    groups = {
        "learned_better": ids[np.argsort(r - l)[:k]],
        "ransac_better": ids[np.argsort(l - r)[:k]],
        "both_good": ids[np.argsort(-(r + l))[:k]],
        "both_bad": ids[np.argsort(r + l)[:k]],
    }
    for group, members in groups.items():
        for i in members:
            s = ds[int(i)]
            rl, rp = ransac[int(i)][:2]
            sl, sp = seg[int(i)][:2]
            ri, li = scores[int(i)]
            plot_panels(s["points"], [
                (instance_colors(s["labels"]), "GT instances"),
                (instance_colors(rl), f"RANSAC (IoU {ri:.2f})"),
                (instance_colors(sl), f"learned (IoU {li:.2f})"),
                (type_colors(s["types"]), "GT types"),
                (type_colors(pred_types(rl, rp)), "RANSAC types"),
                (type_colors(pred_types(sl, sp)), "learned types"),
            ], os.path.join(args.out, f"{group}_{int(i):05d}.png"))
    print(f"{len(ids)} shapes compared; figures in {args.out}/")


if __name__ == "__main__":
    main()
