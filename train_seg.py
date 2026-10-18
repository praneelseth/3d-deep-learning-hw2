"""Train the per-point type segmentation network.

    python train_seg.py --epochs 12 --steps-per-epoch 1500 --out checkpoints/typeseg

An "epoch" is --steps-per-epoch batches of random training shapes (with
replacement across epochs), followed by validation on a fixed val subset.
The checkpoint with the best val mean IoU over the four classes is kept as
best.pt; last.pt is written every epoch so a run can be resumed.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from primitives import TYPE_NAMES
from seg_model import NUM_CLASSES, TypeSegDataset, TypeSegNet

# Point fractions of background/plane/sphere/cone in the training split.
CLASS_FREQ = np.array([0.561, 0.386, 0.0136, 0.0391])


def confusion(pred, target):
    return torch.bincount(target * NUM_CLASSES + pred, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def seg_scores(conf):
    conf = conf.double()
    tp = conf.diag()
    iou = tp / (conf.sum(0) + conf.sum(1) - tp).clamp(min=1)
    return {"acc": float(tp.sum() / conf.sum()), "miou": float(iou.mean()),
            **{f"iou_{TYPE_NAMES[c]}": float(iou[c]) for c in range(NUM_CLASSES)}}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for feats, idx, idx_wide, y in loader:
        pred = model(feats.to(device), idx.to(device), idx_wide.to(device)).argmax(1).cpu()
        conf += confusion(pred.flatten(), y.flatten())
    return seg_scores(conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_data.h5")
    ap.add_argument("--val", default="val_data.h5")
    ap.add_argument("--out", default="checkpoints/typeseg")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps-per-epoch", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-points", type=int, default=5000)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--val-shapes", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--features", default="geo2", choices=["raw", "geo", "geo2"])
    ap.add_argument("--weight-power", type=float, default=0.75, help="class weight = freq ** -power")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TypeSegDataset(args.train, args.num_points, args.k, augment=True, features=args.features)
    val_idx = np.random.default_rng(0).permutation(len(TypeSegDataset(args.val).shapes))[: args.val_shapes]
    val_ds = TypeSegDataset(args.val, args.num_points, args.k, augment=False, indices=val_idx, features=args.features)
    sampler = torch.utils.data.RandomSampler(train_ds, replacement=True,
                                             num_samples=args.steps_per_epoch * args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_ds, args.batch_size, sampler=sampler, drop_last=True,
                                               num_workers=args.workers, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_ds, args.batch_size, num_workers=args.workers)

    model = TypeSegNet(features=args.features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * args.steps_per_epoch,
                                                pct_start=0.05)
    weights = CLASS_FREQ ** -args.weight_power
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    start, best, history = 0, -1.0, []
    if args.resume and os.path.exists(os.path.join(args.out, "last.pt")):
        ck = torch.load(os.path.join(args.out, "last.pt"), map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best, history = ck["epoch"] + 1, ck["best"], ck["history"]

    for epoch in range(start, args.epochs):
        model.train()
        t0, losses = time.time(), []
        conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
        for feats, idx, idx_wide, y in train_loader:
            feats, idx, idx_wide, y = feats.to(device), idx.to(device), idx_wide.to(device), y.to(device)
            logits = model(feats, idx, idx_wide)
            loss = F.cross_entropy(logits, y, weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            conf += confusion(logits.argmax(1).flatten().cpu(), y.flatten().cpu())
        train = seg_scores(conf)
        val = validate(model, val_loader, device)
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "train_miou": train["miou"], **val,
                  "minutes": (time.time() - t0) / 60}
        history.append(record)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in record.items()}), flush=True)
        state = {"model": model.state_dict(), "args": vars(args)}
        if val["miou"] > best:
            best = val["miou"]
            torch.save(state, os.path.join(args.out, "best.pt"))
        torch.save({**state, "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
                    "best": best, "history": history}, os.path.join(args.out, "last.pt"))
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
