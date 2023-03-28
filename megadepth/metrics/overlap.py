"""Functions to compute the overlap metrics between a list of images."""

import os
from pathlib import Path
from typing import Union

import numpy as np
import pycolmap

from megadepth.utils.projections import backward_project, forward_project
from megadepth.utils.read_write_dense import read_array
from megadepth.utils.utils import camera_pixel_grid


def sparse_overlap(reconstruction: pycolmap.Reconstruction) -> np.ndarray:
    """Computes the sparse overlap metric between a list of images.

    For each pair of images (i,j), this score indicates the fraction of sparse features in image i
    that are also contained in the image j. If i=j, the score is simply 1.
    The computation of this score is based on:
    https://github.com/mihaidusmanu/d2-net/blob/master/megadepth_utils/preprocess_scene.py#L202

    Args:
        reconstruction (pycolmap.Reconstruction): Reconstruction object.

    Returns:
        np.ndarray: Array of shape (N, N) with the overlap metric between each pair of images.
    """
    images = reconstruction.images
    N = len(images)

    # pre-compute hash sets with 3D point IDs for faster lookups later
    img_to_ids = {}
    for i, k in enumerate(images.keys()):
        img = images[k]
        img_to_ids[i] = {p.point3D_id for p in img.get_valid_points2D()}

    # compute overlap scores for each image pair
    scores = np.ones((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            # compute number of common 3D point IDs
            ids_1 = img_to_ids[i]
            ids_2 = img_to_ids[j]
            n_common_ids = len(ids_1.intersection(ids_2))

            # set overlap scores
            scores[i, j] = n_common_ids / len(ids_1)
            scores[j, i] = n_common_ids / len(ids_2)

    return scores


def dense_overlap(
    reconstruction: pycolmap.Reconstruction,
    depth_path: Union[Path, str],
    downsample: int = 10,
    rel_thresh: float = 0.03,
) -> np.ndarray:
    """Computes the dense overlap metric between a list of images.

    For each pair of images (i,j), this score indicates the fraction of dense features in image i
    that are also contained in the image j. If i=j, the score is simply 1.

    Args:
        reconstruction (pycolmap.Reconstruction): Reconstruction object.
        depth_path (Union[Path, str]): Path to the directory that contains the depth maps.
        downsample (int, optional): By which factor to downsample depth maps.
        rel_thresh (float, optional): Dense points with an absolute relative depth error below this
            threshold are considered as inliers.

    Returns:
        np.ndarray: Array of shape (N, N) with the overlap metric between each pair of images.
    """
    cameras = reconstruction.cameras
    images = reconstruction.images
    N = len(images)

    # compute overlap scores for each image pair
    scores = np.ones((N, N))
    for i, k1 in enumerate(images.keys()):
        for j, k2 in enumerate(images.keys()):
            if i == j:
                continue

            # load images, cameras and depth maps
            image_1 = images[k1]
            image_2 = images[k2]
            camera_1 = cameras[image_1.camera_id]
            camera_2 = cameras[image_2.camera_id]
            depth_map_1 = read_array(os.path.join(depth_path, f"{image_1.name}.geometric.bin"))
            depth_map_2 = read_array(os.path.join(depth_path, f"{image_2.name}.geometric.bin"))

            # gather depth values that we want to check in a vector
            depth_1 = depth_map_1[::downsample, ::downsample].ravel()

            # get the corresponding 2D coordinates in image 1
            points_2d = camera_pixel_grid(camera_1, downsample)

            # filter out invalid depth values
            valid_depth_mask = depth_1 > 0.0
            depth_1 = depth_1[valid_depth_mask]
            points_2d = points_2d[valid_depth_mask]

            # number of dense features we are considering for the score computation
            n_features = depth_1.size

            # backproject all valid 2D points from image 1 to 3D
            points_3d = backward_project(
                points_2d=points_2d,
                image=image_1,
                camera=camera_1,
                depth=depth_1,
            )

            # project all 3D points to image 2 to obtain 2D points and associated depth values
            proj_points_2d, proj_depths = forward_project(
                points_3d=points_3d, image=image_2, camera=camera_2, return_depth=True
            )

            # get corresponding depth values from the second depth map
            # Note: the depth map values are stored column-wise and not row-wise!
            depth_2 = np.array(
                [depth_map_2[coords[1] - 1, coords[0] - 1] for coords in proj_points_2d]
            )

            # compute inliers based on the absolute relative depth errors
            abs_rel_error = np.abs(depth_2 / proj_depths - 1.0)
            n_inliners = np.count_nonzero(abs_rel_error < rel_thresh)

            # final score
            scores[i, j] = n_inliners / n_features

    return scores
