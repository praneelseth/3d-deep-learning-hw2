"""Per-point primitive type segmentation network (background/plane/sphere/cone).

A DGCNN-style network with a static graph: the k nearest neighbors are
computed once per cloud in xyz (on the CPU, in the data loader) and shared by
all EdgeConv layers, which is much cheaper on MPS than DGCNN's per-layer
feature-space kNN. Input per point is xyz + normal.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from data import ABCParts

NUM_CLASSES = 4


class EdgeConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(2 * cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))

    def forward(self, x, idx):
        """x (B, C, N), idx (B, N, k) -> (B, cout, N): max over edge features [x_i, x_j - x_i]."""
        B, C, N = x.shape
        k = idx.shape[-1]
        nbr = torch.gather(x, 2, idx.reshape(B, 1, N * k).expand(-1, C, -1)).reshape(B, C, N, k)
        center = x.unsqueeze(-1).expand(-1, -1, -1, k)
        return self.mlp(torch.cat([center, nbr - center], 1)).max(-1)[0]


def _conv1d(cin, cout):
    return nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout), nn.LeakyReLU(0.2))


class TypeSegNet(nn.Module):
    def __init__(self, in_ch=6, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.edge = nn.ModuleList([EdgeConv(in_ch, 64), EdgeConv(64, 64), EdgeConv(64, 128)])
        self.glob = _conv1d(256, 512)
        self.head = nn.Sequential(_conv1d(256 + 512, 256), nn.Dropout(dropout), _conv1d(256, 128),
                                  nn.Conv1d(128, num_classes, 1))

    def forward(self, feats, idx):
        """feats (B, 6, N), idx (B, N, k) -> logits (B, num_classes, N)."""
        xs = []
        x = feats
        for layer in self.edge:
            x = layer(x, idx)
            xs.append(x)
        local = torch.cat(xs, 1)
        g = self.glob(local).max(-1, keepdim=True)[0].expand(-1, -1, local.shape[-1])
        return self.head(torch.cat([local, g], 1))


def knn_indices(points, k):
    return cKDTree(points).query(points, k=k + 1)[1][:, 1:]


def make_input(points, normals, k):
    feats = np.concatenate([points, normals], 1).T.astype(np.float32)
    return feats, knn_indices(points, k).astype(np.int64)


class TypeSegDataset(torch.utils.data.Dataset):
    """Random num_points-subsets of shapes; with augment, random rotation and scale."""

    def __init__(self, path, num_points=5000, k=16, augment=True, indices=None):
        self.shapes = ABCParts(path)
        self.indices = np.arange(len(self.shapes)) if indices is None else np.asarray(indices)
        self.num_points, self.k, self.augment = num_points, k, augment

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
            R = Rotation.random(random_state=rng.integers(1 << 31)).as_matrix()
            pts = pts @ R.T * rng.uniform(0.85, 1.15)
            nrm = nrm @ R.T
        feats, idx = make_input(pts, nrm, self.k)
        return torch.from_numpy(feats), torch.from_numpy(idx), torch.from_numpy(s["types"][sel])


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
            feats, idx = make_input(points[part], normals[part], k)
            logits = model(torch.from_numpy(feats)[None].to(device), torch.from_numpy(idx)[None].to(device))
            probs[part] += torch.softmax(logits[0], 0).T.cpu().numpy()
    return probs / passes
