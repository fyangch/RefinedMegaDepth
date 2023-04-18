"""Utility functions."""

import numpy as np
import pycolmap


def camera_pixel_grid(
    camera: pycolmap.Camera, downsample: int, reverse_x: bool = False, reverse_y: bool = False
) -> np.ndarray:
    """Generate array of 2d grid points within the bounds of the given camera.

    Args:
        camera (pycolmap.Camera): Camera to use to generate grid points
        downsample (int): number of samples
        reverse_x (bool, optional): whether to reverse x-axis. Defaults to False.
        reverse_y (bool, optional): whether to reverse y-axis. Defaults to False.

    Returns:
        np.ndarray: 2d list of points of shape (n, 2)
    """
    if reverse_x:
        xl = np.linspace(camera.width - 1, 0, camera.width)[::downsample]
    else:
        xl = np.linspace(0, camera.width - 1, camera.width)[::downsample]

    if reverse_y:
        yl = np.linspace(camera.height - 1, 0, camera.height)[::downsample]
    else:
        yl = np.linspace(0, camera.height - 1, camera.height)[::downsample]

    xv, yv = np.meshgrid(xl, yl)
    return np.vstack((np.ravel(xv), np.ravel(yv))).T


def filter_mask(data: np.ndarray, w: int, h: int) -> np.ndarray:
    """Generate a mask that filters points between [0, w] and [0, h].

    Args:
        data (np.ndarray): array of shape (n, 2)
        w (int): upper bound for x
        h (int): upper bound for y

    Returns:
        np.ndarray: a mask of 0 and 1.
    """
    x = data[:, 0].astype(int)
    y = data[:, 1].astype(int)
    return (x >= 0) & (x < w) & (y >= 0) & (y < h)
