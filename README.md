# 3d-deep-learning-hw2

Primitive detection (planes, spheres, cones) on the ABCParts point clouds from
ParSeNet, using RANSAC and a learned per-point segmentation pipeline.

## Setup

Requires Python 3 with `numpy`, `scipy`, `h5py`, `matplotlib` and `torch`.
Put the data splits in the repo root (they are git-ignored):

```
train_data.h5  val_data.h5  test_data.h5
```

Each file stores 10k points per shape with normals, a per-point segment id
(`labels`) and a ParSeNet primitive code (`prim`: 1 plane, 3 cone, 4 cylinder,
5 sphere, the rest splines). Plane/sphere/cone are the target types;
everything else counts as background. Shapes are centered and scaled to the
unit sphere on load, so all distances below are in those units.

## Code

| File | Contents |
|---|---|
| `data.py` | HDF5 loader, normalization, prim code to type mapping |
| `primitives.py` | weighted least-squares fitters and distances for plane / sphere / cone (+ cylinder, used only for rejection) |
| `metrics.py` | Seg-IoU (Hungarian matching on IoU), type accuracy, residual |
| `ransac.py` | multi-type RANSAC detector |
| `run_ransac.py` | runs RANSAC on a split, evaluates, saves predictions and figures |
| `eval_oracle.py` | scores the fitted ground truth as a pipeline sanity check |
| `viz.py` | matplotlib point-cloud figures |
| `tests.py` | fitter, metric and RANSAC tests |

## Running

Tests:

```
python tests.py
```

Sanity check: ground-truth segments with least-squares fits should score
Seg-IoU 1, type accuracy 1 and a residual near 0.

```
python eval_oracle.py --data test_data.h5 --n 500
```

RANSAC evaluation (8 worker processes, about 1.7 s per shape per worker):

```
python run_ransac.py --data val_data.h5 --n 300 --tag dev      # random 300-shape subset
python run_ransac.py --data val_data.h5 --n 0 --tag val_full   # whole split, ~14 min
python run_ransac.py --data test_data.h5 --n 0 --tag test      # final test numbers
```

Every `RansacConfig` field in `ransac.py` is also a flag, e.g.
`--eps 0.005 --max_angle_deg 15 --reject_cylinders false`.
`--types plane` runs a single primitive type. Other options: `--workers`,
`--viz` (number of shapes to plot), `--min-gt-points` (default 50).

Each run prints the metrics and writes to `outputs/ransac/<tag>/`:
`summary.json` (metrics and config), `preds.pkl` (per-shape labels and fitted
primitives) and `shape_*.png` figures (GT vs. predicted instances and types).

## Evaluation protocol

Ground-truth instances are the plane/sphere/cone segments with at least 50
points (smaller ones are mostly slivers holding about 1% of target points).
Predicted and ground-truth instances are matched one-to-one by IoU.

- **Seg-IoU**: mean IoU over ground-truth instances (0 if unmatched),
  averaged per shape, then over shapes.
- **Type accuracy**: fraction of matched instances whose predicted type is
  correct.
- **Residual**: mean distance from a matched ground-truth instance's points
  to its predicted surface.

`run_ransac.py` also reports the fraction of predicted points that lie on
ground-truth background, a false-detection measure the metrics above do not
capture.

## Results

RANSAC with default settings on the full validation split (4000 shapes,
23476 scored instances):

| Type | Count | IoU | Type acc | Residual |
|---|---|---|---|---|
| plane | 21158 | 0.900 | 0.985 | 2.7e-4 |
| sphere | 310 | 0.686 | 0.795 | 5.7e-3 |
| cone | 2008 | 0.797 | 0.887 | 2.6e-3 |
| **all** | 23476 | **0.888** | **0.975** | **5.3e-4** |

13.7 predicted primitives per shape; 35.8% of predicted points lie on
ground-truth background.
