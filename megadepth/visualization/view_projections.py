"""This module provides functions to visualize 3D points in an axis aligned plot."""
import matplotlib.pyplot as plt
import numpy as np


def pca(data):
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


def plot_view_projection(Data, view, limit=10, s=1, alpha=0.1, *args, **kwargs):
    """Creates an axis aligned plot.

    Args:
        Data: list of np.ndarrays of shape (N, 3) [np.ndarray, ..]
        view: int to select from ["Top View", "Front View", "Right View"]
        alpha: set transparency for dots
        s: size for dots
        limit: limits of the plot limit=3 should contain 99% of the camera positions
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


def create_view_projection_figure(data, view=None, path=None, *args, **kwargs):
    """Creates an axis aligned plot.

    Args:
        data: np.ndarray of shape (N, 3)
        view: int to select from ["Top View", "Front View", "Right View"]
        path: filepath to store the plot
        alpha: set transparency for dots
        s: size for dots
        limit: limits of the plot limit=3 should contain 99% of the camera positions

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
