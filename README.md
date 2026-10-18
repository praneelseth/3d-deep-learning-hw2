# 3d-deep-learning-hw2

Plane, sphere and cone detection on the ABCParts point clouds (ParSeNet):
RANSAC, and a per-point type segmentation network followed by grouping and
least-squares fitting.

## Setup

Python 3 with `numpy`, `scipy`, `h5py`, `matplotlib`, `scikit-learn` and
`torch`. Place the data splits in the repository root:

```
train_data.h5  val_data.h5  test_data.h5
```

## Code

| File | Contents |
|---|---|
| `data.py` | HDF5 loader, normalization, primitive code to type mapping |
| `primitives.py` | least-squares fitters and distances for plane / sphere / cone (and cylinder) |
| `metrics.py` | Seg-IoU, type accuracy, residual |
| `ransac.py` | RANSAC detector |
| `run_ransac.py` | runs RANSAC on a split and evaluates it |
| `seg_model.py` | type segmentation network, dataset, inference |
| `train_seg.py` | trains the network |
| `seg_pipeline.py` | groups predicted types into instances and fits them |
| `run_seg.py` | runs the learned pipeline on a split and evaluates it |
| `eval_oracle.py` | evaluates least-squares fits of the ground-truth segments |
| `rescore.py` | recomputes metrics from saved predictions |
| `make_figures.py` | training curves and comparison figures |
| `viz.py` | point-cloud plotting |
| `tests.py` | tests |

## Running

Tests:

```
python tests.py
```

Ground-truth check (should give Seg-IoU 1, type accuracy 1, residual near 0):

```
python eval_oracle.py --data test_data.h5 --n 500
```

### RANSAC

```
python run_ransac.py --data val_data.h5 --n 300 --tag dev      # random 300-shape subset
python run_ransac.py --data test_data.h5 --n 0 --tag test      # whole split
```

Every field of `RansacConfig` in `ransac.py` is also a flag, e.g.
`--eps 0.005 --max_angle_deg 15 --reject_cylinders false`; `--types plane`
runs a single primitive type. Other flags: `--workers` (default 8), `--viz`
(number of shapes to plot), `--min-gt-points` (default 50).

### Learned pipeline

Training (12 epochs of 1500 steps, about 2 hours on an Apple M-series GPU):

```
python train_seg.py --epochs 12 --steps-per-epoch 1500 --features geo2 \
    --weight-power 0.75 --rotation cube --out checkpoints/typeseg_geo2
```

The checkpoint with the best validation mean IoU is saved as
`checkpoints/typeseg_geo2/best.pt`; `--resume` continues from `last.pt`.

Evaluation:

```
python run_seg.py --data val_data.h5 --n 500 --tag dev
python run_seg.py --data test_data.h5 --n 0 --tag test \
    --probs-cache outputs/seg/cache/test_probs.pkl
python run_seg.py --gt-types --data val_data.h5 --n 300 --tag oracle   # ground-truth types
```

Defaults: `--ckpt checkpoints/typeseg_geo2/best.pt --prior-power 0.5
--smooth-iters 10 --method hybrid`. Every field of `GroupConfig` in
`seg_pipeline.py` is also a flag. `--probs-cache` saves the network outputs
so post-processing settings can be changed without rerunning the network.

### Outputs

Each run writes to `outputs/ransac/<tag>/` or `outputs/seg/<tag>/`:
`summary.json` (metrics and settings), `preds.pkl` (per-shape labels and
fitted primitives) and `shape_*.png` figures.

Comparison figures and training curves, after both test runs:

```
python make_figures.py --data test_data.h5 --ransac outputs/ransac/test \
    --seg outputs/seg/test --history checkpoints/typeseg_geo2/history.json
```

## Results

Test split: 4163 shapes, 23899 scored instances (plane, sphere and cone
segments with at least 50 points). Distances are in units of the shape
normalized to the unit sphere; residual is given as mean / median over
matched instances.

| Method | Seg-IoU | Type acc | Residual | Plane IoU | Sphere IoU | Cone IoU |
|---|---|---|---|---|---|---|
| RANSAC | 0.897 | 0.975 | 6.2e-4 / 9.9e-6 | 0.899 | 0.702 | 0.777 |
| Learned | 0.807 | 0.936 | 1.7e-3 / 3.1e-5 | 0.800 | 0.638 | 0.639 |

Validation split (4000 shapes): RANSAC Seg-IoU 0.888, type accuracy 0.975;
learned 0.797, 0.937.

Per-point type segmentation on test (network argmax): accuracy 0.822, mean
IoU 0.458 (background 0.722, plane 0.774, sphere 0.153, cone 0.184).
