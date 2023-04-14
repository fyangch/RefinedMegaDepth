"""Module to visualize the overlap score."""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import megadepth.visualization.view_projections as view_projections


def show_matrix(matrix: np.ndarray, path: Optional[str] = None):
    """Makes a plot from the matrix.

    Args:
        matrix: NxN np.ndarray
        path: filepath to store the plot
    """
    fig = plt.figure()
    plt.tight_layout()
    plt.imshow(matrix, interpolation="nearest")
    if path is None:
        plt.show()
    else:
        fig.savefig(path, dpi=900, bbox_inches="tight")


def vis_overlap(matrix, poses, points=np.zeros((0, 3)), path=None, *args, **kwargs):
    """Shows overlap score as opacity on lines between camera poses.

    For performance all overlaps smaller than 0.1 are thrown away.
    An overlap of 1 will result in an opacity of 0.1.

    Args:
        matrix: Overlap matrix NxN np.ndarray
        poses: Nx3 np.ndarray
        points: (optional)
        path: filepath to store the plot
    """
    fig = plt.figure()
    plt.tight_layout()

    # Calculate coordinates and weights of all pairs of cameras
    i, j = np.triu_indices(len(poses), k=1)

    weights = (matrix + matrix.T)[i, j].astype(float) / 10

    # sorted = np.argsort(-weights)[:10000]
    sorted = weights > 0.01
    i = i[sorted]
    j = j[sorted]
    weights = weights[sorted]

    x = np.stack((poses[i, 0], poses[j, 0]), axis=-1)
    y = np.stack((poses[i, 1], poses[j, 1]), axis=-1)

    # Create LineCollection object with coordinates and weights
    lines = LineCollection(np.stack([x, y], axis=-1), linewidths=1, colors="orange", alpha=weights)

    # Add LineCollection to plot
    ax = plt.gca()
    ax.add_collection(lines)

    view_projections.plot_view_projection(Data=[points, poses], view=0, limit=2.5, *args, **kwargs)

    if path is None:
        plt.show()
    else:
        fig.savefig(path, dpi=900, bbox_inches="tight")
    plt.close(fig)
