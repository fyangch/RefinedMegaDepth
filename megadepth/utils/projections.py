"""Projection functions."""

from typing import Tuple, Union

import numpy as np
import pycolmap

from megadepth.utils.utils import filter_mask


def forward_project(
    points_3d: np.ndarray,
    image: pycolmap.Image,
    camera: pycolmap.Camera,
    return_depth: bool = False,
    return_mask: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Project array of 3D points onto the given image. Invalid points are discarded.

    Args:
        points_3d (np.ndarray): Array of 3D points with shape (n, 3).
        image (pycolmap.Image): Image onto which the points are projected.
        camera (pycolmap.Camera): Camera associated with the image.
        return_depth (bool, optional): Whether to return the depth values of projected 2D points.
        return_mask (bool, optional): Whether to return the mask that filters out invalid 2D points.

    Returns:
        (np.ndarray: Array of valid 2D points on the given image.
         np.ndarray (optional)): Array of depth values associated with each 2D point returned.
    """
    points_2d = np.array(camera.world_to_image(image.project(points_3d)), dtype=int)

    # filter out 2D points that lie outside of the image
    mask = filter_mask(points_2d, camera.width, camera.height)
    points_2d = points_2d[mask]

    if return_depth:
        depths = np.array(image.transform_to_image(points_3d))[:, 2]
        depths = depths[mask]
        return (points_2d, depths, mask) if return_mask else (points_2d, depths)

    return (points_2d, mask) if return_mask else points_2d


def backward_project(
    points_2d: np.ndarray, image: pycolmap.Image, camera: pycolmap.Camera, depth: np.ndarray
) -> np.ndarray:
    """Project array of 2D points into the 3D world space given a depth map.

    Args:
        points2d (np.ndarray): Array of 2D points with shape (n, 2).
        image (pycolmap.Image): Image from which the points are taken.
        camera (pycolmap.Camera): Camera associated with the image.
        depth (np.ndarray): Associated depth map with shape (n,).

    Returns:
        np.ndarray: Array of backward projected 3D points from the given 2D points.
    """
    p_world = np.array(camera.image_to_world(points_2d))
    p_world = np.stack([p_world[:, 0], p_world[:, 1], np.ones_like(p_world[:, 0])]) * depth
    p_world = np.array(image.transform_to_world(p_world.T))

    return p_world
