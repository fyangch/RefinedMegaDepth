"""Projection functions."""

import numpy as np
import pycolmap

from megadepth.utils.utils import filter_mask


def forward_project(
    points_3d: np.ndarray,
    image: pycolmap.Image,
    camera: pycolmap.Camera,
    return_mask: bool = True,
    return_depth: bool = False,
) -> tuple:
    """Project list of 3d points onto the given image.

    Args:
        points_3d (np.ndarray): list of 3d points; shape (n, 3)
        image (pycolmap.Image): image onto which the points are projected
        camera (pycolmap.Camera): camera associated with the image
        return_mask (bool, optional): mask of points inside the camera bounds. (Default: True)
        return_depth (bool, optional): depth values of projected 2d points. (Default: False)

    Returns:
        (np.ndarray: list of 2d points on the given image,
         np.ndarray (optional): a mask which removes the points outside of the image bounds,
         np.ndarray (optional)): an array of depth values associated with each 2d point returned
    """
    uv = np.array(camera.world_to_image(image.project(points_3d)), dtype=int)

    return_array = [uv]

    if return_mask:
        mask = filter_mask(uv, camera.width, camera.height)
        return_array.append(mask)
    if return_depth:
        depth = np.array(image.transform_to_image(points_3d))[:, 2]
        return_array.append(depth)

    return uv if not return_mask and not return_depth else tuple(return_array)


def backward_project(
    points2d: np.ndarray, image: pycolmap.Image, camera: pycolmap.Camera, depth: np.ndarray
) -> np.ndarray:
    """Project list of 2d points into the 3d world space given a depth map.

    Args:
        points2d (np.ndarray): list of 2d points; shape (n, 2)
        image (pycolmap.Image): image from which the points are taken
        camera (pycolmap.Camera): camera associated with the image
        depth (np.ndarray): associated depth map
        colors (np.ndarray, optional): a list of colors for each 2d point. Defaults to None.

    Returns:
        np.ndarray: a list of backward projected 3d points from the given 2d points
    """
    if len(depth.shape) > 1:
        depth_d = depth.ravel()
    else:
        depth_d = depth

    p_world = np.array(camera.image_to_world(points2d))
    p_world = np.stack([p_world[:, 0], p_world[:, 1], np.ones_like(p_world[:, 0])]) * depth_d
    p_world = np.array(image.transform_to_world(p_world.T))

    return p_world
