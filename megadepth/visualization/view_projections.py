"""This module provides functions to visualize 3D points in an axis aligned plot."""

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pycolmap

from megadepth.utils.projections import get_camera_poses


def pca(data: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Computes pca on the data matrix and returns a coordinate transform as a lambda function.

    Args:
        data: np.ndarray of shape (N, 3)

    Returns:
        lambda x: np.ndarray of shape (N, 3) -> np.ndarray of shape (N, 3)

    """
    mean = data.mean(axis=0)
    standardized_data = data - mean
    scale = data.std()
    standardized_data /= scale
    covariance_matrix = np.cov(standardized_data, ddof=0, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    order_of_importance = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, order_of_importance]
    # Ensure handedness doesnt change
    if np.linalg.det(sorted_eigenvectors) < 0:
        sign = np.sign(np.ones((1, 3)) @ sorted_eigenvectors)
        det_sign = np.linalg.det(sorted_eigenvectors)
        sorted_eigenvectors = sorted_eigenvectors * sign * det_sign
    return lambda x: (x - mean) / scale @ sorted_eigenvectors


def plot_view_projection(
    Data: np.ndarray, view: int, limit: int = 10, s: float = 1, alpha: float = 0.1, *args, **kwargs
) -> None:
    """Creates an axis aligned plot.

    Args:
        Data (np.ndarray): List of np.ndarrays of shape (N, 3) [np.ndarray, ..]
        view (int): Int to select from ["Top View", "Front View", "Right View"]
        limit (int, optional): Limits of the plot limit=3 should contain 99% of the camera
        positions. Defaults to 10.
        s (float, optional): Size for dots. Defaults to 1.
        alpha (float, optional): Set transparency for dots. Defaults to 0.1.
    """
    view_names = ["Top View", "Front View", "Right View"]
    id1, id2 = [(0, 1), (0, 2), (1, 2)][view]
    for data in Data:
        labels = ["X", "Y", "Z"]
        plt.scatter(data[:, id1], data[:, id2], s=s, alpha=alpha, *args, **kwargs)
        plt.title(view_names[view])
    plt.axis("scaled")
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    plt.axis("off")
    plt.xlabel(labels[id1], labelpad=0)
    plt.ylabel(labels[id2], labelpad=0)


def create_view_projection_figure(
    data: np.ndarray,
    view: Optional[int] = None,
    path: Optional[str] = None,
    *args,
    **kwargs,
) -> None:
    """Creates an axis aligned plot.

    Args:
        data (np.ndarray): List of np.ndarrays of shape (N, 3) [np.ndarray, ..]
        view (Optional[int], optional): Int to select from ["Top View", "Front View", "Right View"]
        path (Optional[str], optional): Filepath to store the plot. Defaults to None.
    """
    fig = plt.figure()
    plt.tight_layout()
    if view is None:
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plot_view_projection(data, i, *args, **kwargs)
    else:
        plot_view_projection(data, view, *args, **kwargs)
    if path is None:
        plt.show()
    else:
        fig.savefig(path, dpi=900, bbox_inches="tight")
    plt.close(fig)


def align_models(
    reconstruction_anchor: pycolmap.Reconstruction,
    reconstruction_align: pycolmap.Reconstruction,
    min_common_images: int = 6,
) -> pycolmap.Reconstruction:
    """Aligns two reconstructions by aligning the camera positions.

    Args:
        reconstruction1 (pycolmap.Reconstruction): First reconstruction to align to.
        reconstruction2 (pycolmap.Reconstruction): Second reconstruction to be aligned.
        min_common_images (int, optional): Minimum number of common images between the two
        reconstructions. Defaults to 6.

    Returns:
        pycolmap.Reconstruction: Aligned reconstructions.
    """
    image_names = [img.name for img in reconstruction_anchor.images.values()]
    locations = get_camera_poses(reconstruction_anchor)

    _ = reconstruction_align.align_robust(image_names, locations, min_common_images)

    return reconstruction_align
