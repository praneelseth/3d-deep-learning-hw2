# 3d-deep-learning-hw2

Primitive detection (planes, spheres, cones) on the ABCParts point clouds from
ParSeNet, using RANSAC (Task 1) and a learned per-point type segmentation
network followed by grouping and least-squares fitting (Task 2, Option 1).

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
| `seg_model.py` | per-point type network (static-graph DGCNN with curvature inputs), dataset, inference |
| `train_seg.py` | trains the network |
| `seg_pipeline.py` | groups predicted types into instances and fits them (Option 1 second stage) |
| `run_seg.py` | runs the learned pipeline on a split, evaluates, saves predictions and figures |
| `eval_oracle.py` | scores the fitted ground truth as a pipeline sanity check |
| `rescore.py` | recomputes metrics from saved predictions |
| `make_figures.py` | training curves and RANSAC vs. learned comparison figures |
| `viz.py` | matplotlib point-cloud figures |
| `tests.py` | fitter, metric, RANSAC, grouping and network tests |

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

### Learned pipeline (Task 2, Option 1)

Training (Apple MPS or CUDA; ~9.5 min per 1500-step epoch on an M-series Mac,
12 epochs). The best checkpoint by validation mean IoU is saved as `best.pt`:

```
python train_seg.py --epochs 12 --steps-per-epoch 1500 --features geo2 \
    --weight-power 0.75 --rotation cube --out checkpoints/typeseg_geo2
```

Evaluation (network inference ~0.3 s per shape, grouping in 8 workers):

```
python run_seg.py --data val_data.h5 --n 500 --tag dev
python run_seg.py --data test_data.h5 --n 0 --tag test \
    --probs-cache outputs/seg/cache/test_probs.pkl
python run_seg.py --gt-types --data val_data.h5 --n 300 --tag oracle   # perfect classifier
```

Defaults: `--ckpt checkpoints/typeseg_geo2/best.pt`, `--prior-power 0.5`
(predict argmax of p / freq^0.5 to favor rare classes), `--smooth-iters 10`
(edge-aware probability smoothing), `--method hybrid` (connected components
per predicted type, re-split with type-restricted RANSAC when a component fits
poorly, then a least-squares type re-check per instance). `--probs-cache`
stores the network outputs so post-processing settings can be swept without
rerunning the network. Outputs go to `outputs/seg/<tag>/`.

Figures for the report (after both test runs):

```
python make_figures.py --data test_data.h5 --ransac outputs/ransac/test \
    --seg outputs/seg/test --history checkpoints/typeseg_geo2/history.json
```

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

Both runners also report the fraction of predicted points that lie on
ground-truth background, a false-detection measure the metrics above do not
capture, and the residual's median and its mean over matches with IoU >= 0.5
(the mean is dominated by a few spurious low-IoU matches).

## Results

Test split (4163 shapes, 3695 with scored instances, 23899 instances).
Residuals are in unit-sphere units; mean / median.

| Method | Seg-IoU | Type acc | Residual | Plane IoU | Sphere IoU | Cone IoU | Pred. pts on GT background |
|---|---|---|---|---|---|---|---|
| RANSAC | **0.897** | **0.975** | 6.2e-4 / 9.9e-6 | 0.899 | 0.702 | 0.777 | 36.4% |
| Learned (Option 1) | 0.807 | 0.936 | 1.7e-3 / 3.1e-5 | 0.800 | 0.638 | 0.639 | 37.3% |
| Learned, plain argmax, no smoothing | 0.762 | 0.977 | 8.3e-4 / 2.9e-5 | 0.773 | 0.335 | 0.353 | 19.6% |

The network alone reaches per-point accuracy 0.822 and mean IoU 0.458 on test
(background 0.722, plane 0.774, sphere 0.153, cone 0.184). With ground-truth
types the same grouping and fitting reach Seg-IoU 0.953 on 300 val shapes,
so per-point classification is the bottleneck of the learned pipeline.

Full validation split (4000 shapes): RANSAC Seg-IoU 0.888 / type acc 0.975;
learned 0.797 / 0.937. RANSAC settings were tuned on 300 val shapes and the
learned post-processing on 500; both are consistent across val and test.
