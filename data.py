"""Loading the ABCParts (ParSeNet) HDF5 splits.

Each file holds, per shape, 10k points with normals, a per-point segment id
(`labels`) and a per-point ParSeNet primitive code (`prim`).
"""
import h5py
import numpy as np

from primitives import BACKGROUND, CONE, PLANE, SPHERE

# ParSeNet primitive codes, checked by fitting each code's segments:
# 1 plane, 3 cone, 4 cylinder, 5 sphere; 0, 2 and 6-9 are splines and other
# freeform surfaces. Everything but plane/sphere/cone is background here.
PRIM_CODE_TO_TYPE = np.full(10, BACKGROUND, dtype=np.int64)
PRIM_CODE_TO_TYPE[1] = PLANE
PRIM_CODE_TO_TYPE[5] = SPHERE
PRIM_CODE_TO_TYPE[3] = CONE


def normalize(points):
    """Center on the bounding box and scale to fit the unit sphere.

    Raw shapes range from tens to thousands of units across, so every
    threshold and distance in this project is measured after this step.
    """
    center = (points.max(0) + points.min(0)) / 2
    scale = np.linalg.norm(points - center, axis=1).max()
    return (points - center) / scale, center, scale


class ABCParts:
    """Random access to one split, e.g. ABCParts("val_data.h5")[i]."""

    def __init__(self, path, normalize=True):
        self.path = path
        self.normalize = normalize
        self._file = None
        with h5py.File(path, "r") as f:
            self._len = len(f["points"])

    def _h5(self):
        # Opened lazily so the object can be handed to DataLoader workers.
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        f = self._h5()
        points = f["points"][i].astype(np.float64)
        center, scale = np.zeros(3), 1.0
        if self.normalize:
            points, center, scale = normalize(points)
        prim = f["prim"][i].astype(np.int64)
        return {
            "points": points,
            "normals": f["normals"][i].astype(np.float64),
            "labels": f["labels"][i].astype(np.int64),
            "prim": prim,
            "types": PRIM_CODE_TO_TYPE[prim],
            "center": center,
            "scale": scale,
        }
