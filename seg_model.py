"""Per-point primitive type segmentation network (background/plane/sphere/cone).

A DGCNN-style network with a static graph: the k nearest neighbors are
computed once per cloud in xyz (on the CPU, in the data loader) and shared by
the EdgeConv layers, which is much cheaper on MPS than DGCNN's per-layer
feature-space kNN. Input per point is xyz + normal.

`features` selects the input variant (stored in checkpoints):
  raw   first EdgeConv sees the standard [x_i, x_j - x_i].
  geo   first EdgeConv sees scale-normalized edge geometry, because raw
        neighbor offsets (~0.02) and normal differences are tiny next to
        absolute coordinates: [x_i, (p_j - p_i) / r_i, curvature_ij], with r_i
        the mean neighbor distance and curvature_ij = (n_j - n_i) / |p_j - p_i|.
  geo2  geo, plus per-point principal curvatures estimated from the normal
        field (plane (0, 0), cylinder/cone (k, 0), sphere (k, k)), and the
        second and third EdgeConv use a dilated graph (every 4th of the 4k
        nearest neighbors) for a 4x wider receptive field at the same cost.
Curvatures are multiplied by CURVATURE_SCALE (~ a typical point spacing) and
clipped, so they are dimensionless and O(1).
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from data import ABCParts

NUM_CLASSES = 4
CURVATURE_SCALE = 0.05
DILATION = 4


def _gather(x, idx):
    B, C, N = x.shape
    k = idx.shape[-1]
    nbr = torch.gather(x, 2, idx.reshape(B, 1, N * k).expand(-1, C, -1)).reshape(B, C, N, k)
    return nbr, x.unsqueeze(-1).expand(-1, -1, -1, k)


class EdgeConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(2 * cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))

    def forward(self, x, idx):
        """x (B, C, N), idx (B, N, k) -> (B, cout, N): max over edge features [x_i, x_j - x_i]."""
        nbr, center = _gather(x, idx)
        return self.mlp(torch.cat([center, nbr - center], 1)).max(-1)[0]


class GeoEdgeConv(nn.Module):
    """First layer for the geo variants; channels 0-2 of x are xyz and 3-5 the normal."""

    def __init__(self, cin, cout):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(cin + 6, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))

    def forward(self, x, idx):
        nbr, center = _gather(x, idx)
        dp = nbr[:, :3] - center[:, :3]
        dist = dp.norm(dim=1, keepdim=True)
        rel = dp / dist.mean(-1, keepdim=True).clamp(min=1e-6)
        curv = (nbr[:, 3:6] - center[:, 3:6]) / dist.clamp(min=1e-4) * CURVATURE_SCALE
        return self.mlp(torch.cat([center, rel, curv.clamp(-5, 5)], 1)).max(-1)[0]


def _conv1d(cin, cout):
    return nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout), nn.LeakyReLU(0.2))


class TypeSegNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3, features="raw"):
        super().__init__()
        self.features = features
        if features == "raw":
            first = EdgeConv(6, 64)
        else:
            first = GeoEdgeConv(8 if features == "geo2" else 6, 64)
        self.edge = nn.ModuleList([first, EdgeConv(64, 64), EdgeConv(64, 128)])
        self.glob = _conv1d(256, 512)
        self.head = nn.Sequential(_conv1d(256 + 512, 256), nn.Dropout(dropout), _conv1d(256, 128),
                                  nn.Conv1d(128, num_classes, 1))

    def forward(self, feats, idx, idx_wide=None):
        """feats (B, C, N), idx / idx_wide (B, N, k) -> logits (B, num_classes, N)."""
        if idx_wide is None:
            idx_wide = idx
        xs = []
        x = feats
        for layer, graph in zip(self.edge, (idx, idx_wide, idx_wide)):
            x = layer(x, graph)
            xs.append(x)
        local = torch.cat(xs, 1)
        g = self.glob(local).max(-1, keepdim=True)[0].expand(-1, -1, local.shape[-1])
        return self.head(torch.cat([local, g], 1))


def principal_curvatures(points, normals, nbr_idx):
    """(N, 2) signed principal curvatures, largest magnitude first.

    Least-squares fit of the normal's differential, dn ~ dp A, over each
    point's neighbors; the eigenvalues of A symmetrized and projected onto the
    tangent plane are the principal curvatures (1/R for a sphere of radius R).
    """
    dp = points[nbr_idx] - points[:, None]
    dn = normals[nbr_idx] - normals[:, None]
    M = np.einsum("nki,nkj->nij", dp, dp)
    M += 1e-3 * np.trace(M, axis1=1, axis2=2)[:, None, None] * np.eye(3)
    A = np.linalg.solve(M, np.einsum("nki,nkj->nij", dp, dn))
    P = np.eye(3) - normals[:, :, None] * normals[:, None, :]
    S = P @ ((A + A.transpose(0, 2, 1)) / 2) @ P
    ev = np.linalg.eigvalsh(S)
    order = np.argsort(-np.abs(ev), axis=1)[:, :2]
    return np.take_along_axis(ev, order, 1)


def make_input(points, normals, k, features="raw"):
    """Network inputs for one cloud: feats (C, N), idx (N, k), idx_wide (N, k)."""
    wide = features == "geo2"
    nbrs = cKDTree(points).query(points, k=(k * DILATION if wide else k) + 1)[1][:, 1:]
    idx = nbrs[:, :k]
    cols = [points, normals]
    if wide:
        curv = principal_curvatures(points, normals, idx)
        cols.append(np.clip(curv * CURVATURE_SCALE, -5, 5))
    feats = np.concatenate(cols, 1).T.astype(np.float32)
    idx_wide = nbrs[:, ::DILATION] if wide else idx
    return feats, idx.astype(np.int64), idx_wide.astype(np.int64)


def random_rotation(rng, mode):
    """so3: uniform random rotation. cube: one of the 48 axis permutations with
    sign flips, which keeps the canonical axis alignment of the CAD parts (84%
    of plane points, but 6% of background, have axis-aligned normals)."""
    if mode == "so3":
        return Rotation.random(random_state=rng.integers(1 << 31)).as_matrix()
    return np.eye(3)[rng.permutation(3)] * rng.choice([-1.0, 1.0], 3)[:, None]


class TypeSegDataset(torch.utils.data.Dataset):
    """Random num_points-subsets of shapes; with augment, random rotation and scale."""

    def __init__(self, path, num_points=5000, k=16, augment=True, indices=None, features="raw", rotation="cube"):
        self.shapes = ABCParts(path)
        self.indices = np.arange(len(self.shapes)) if indices is None else np.asarray(indices)
        self.num_points, self.k, self.augment, self.features = num_points, k, augment, features
        self.rotation = rotation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = int(self.indices[j])
        s = self.shapes[i]
        # Validation uses a fixed subset per shape so its numbers are comparable across epochs.
        rng = np.random.default_rng() if self.augment else np.random.default_rng(i)
        sel = rng.choice(len(s["points"]), self.num_points, replace=False)
        pts, nrm = s["points"][sel], s["normals"][sel]
        if self.augment:
            R = random_rotation(rng, self.rotation)
            pts = pts @ R.T * rng.uniform(0.85, 1.15)
            nrm = nrm @ R.T
        feats, idx, idx_wide = make_input(pts, nrm, self.k, self.features)
        return (torch.from_numpy(feats), torch.from_numpy(idx), torch.from_numpy(idx_wide),
                torch.from_numpy(s["types"][sel]))


@torch.no_grad()
def predict_type_probs(model, points, normals, device, num_points=5000, k=16, passes=2, seed=0):
    """Per-point class probabilities (N, 4) for a full cloud.

    The cloud is split into random chunks of ~num_points (the training density)
    so every point is predicted once per pass; passes average over splits.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    n = len(points)
    probs = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    chunks = max(1, round(n / num_points))
    for _ in range(passes):
        for part in np.array_split(rng.permutation(n), chunks):
            inputs = make_input(points[part], normals[part], k, model.features)
            logits = model(*(torch.from_numpy(a)[None].to(device) for a in inputs))
            probs[part] += torch.softmax(logits[0], 0).T.cpu().numpy()
    return probs / passes
