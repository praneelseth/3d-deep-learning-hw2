"""Matplotlib point-cloud figures for debugging and the report."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# background, plane, sphere, cone
TYPE_COLORS = np.array([[0.80, 0.80, 0.80], [0.12, 0.47, 0.71], [1.00, 0.50, 0.05], [0.17, 0.63, 0.17]])
UNASSIGNED = np.array([0.85, 0.85, 0.85])


def type_colors(types):
    return TYPE_COLORS[types]


def instance_colors(labels, seed=0):
    """A random color per instance id; -1 (unassigned) is light gray."""
    ids = np.unique(labels)
    palette = np.random.default_rng(seed).uniform(0.1, 0.9, (len(ids), 3))
    colors = palette[np.searchsorted(ids, labels)]
    colors[labels < 0] = UNASSIGNED
    return colors


def plot_panels(points, panels, path, max_points=5000, seed=0):
    """Save side-by-side views of one cloud; panels is a list of (colors (N,3), title)."""
    idx = np.random.default_rng(seed).permutation(len(points))[:max_points]
    pts = points[idx]
    fig = plt.figure(figsize=(5 * len(panels), 5))
    for k, (colors, title) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), k + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors[idx], s=1.5, depthshade=False)
        ax.set_title(title)
        ax.set_box_aspect((1, 1, 1))
        lim = np.abs(pts).max()
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
