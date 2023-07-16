"""Functions to compute the overlap metrics between a list of images."""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

import numpy as np
import pycolmap
import xarray as xr
from omegaconf import DictConfig
from tqdm import tqdm

from megadepth.utils.io import load_depth_map
from megadepth.utils.projections import backward_project, forward_project
from megadepth.utils.utils import camera_pixel_grid


def sparse_overlap(reconstruction: pycolmap.Reconstruction) -> xr.DataArray:
    """Computes the sparse overlap metric between a list of images.

    For each pair of images (i,j), this score indicates the percentage of sparse features in image i
    that are also contained in the image j. If i=j, the score is simply 100. Each score is an
    integer in the range [0, 100].
    The computation of this score is based on:
    https://github.com/mihaidusmanu/d2-net/blob/master/megadepth_utils/preprocess_scene.py#L202

    Args:
        reconstruction (pycolmap.Reconstruction): Reconstruction object.

    Returns:
        xr.DataArray: DataArray of shape (N, N) with the overlap metric between each pair of images.
    """
    images = reconstruction.images
    image_fns = [images[k].name for k in images.keys()]
    N = len(images)

    # pre-compute hash sets with 3D point IDs for faster lookups later
    img_to_ids = {}
    for i, k in enumerate(images.keys()):
        img = images[k]
        img_to_ids[i] = {p.point3D_id for p in img.get_valid_points2D()}

    # compute overlap scores for each image pair
    scores = np.full((N, N), 100, dtype=np.int8)
    for i in tqdm(range(N), desc="Computing sparse overlap...", ncols=80):
        for j in range(i + 1, N):
            # compute number of common 3D point IDs
            ids_1 = img_to_ids[i]
            ids_2 = img_to_ids[j]
            n_common_ids = len(ids_1.intersection(ids_2))

            # set overlap scores
            scores[i, j] = np.rint(100 * n_common_ids / len(ids_1))
            scores[j, i] = np.rint(100 * n_common_ids / len(ids_2))

    # create and return data array
    coords = {"img1": image_fns, "img2": image_fns}
    return xr.DataArray(scores, coords=coords, dims=["img1", "img2"])


def _compute_dense_row(
    i: int,
    k1: int,
    paths: DictConfig,
    downsample: int,
    rel_thresh: float,
    cosine_weighted: bool,
) -> Tuple[int, np.ndarray]:
    """Compute the i-th row of the dense overlap matrix."""
    reconstruction = pycolmap.Reconstruction(os.path.join(paths.dense, "sparse"))
    depth_path = os.path.join(paths.dense, "stereo", "depth_maps")
    normal_path = os.path.join(paths.dense, "stereo", "normal_maps")

    cameras = reconstruction.cameras
    images = reconstruction.images
    N = len(images)

    row = np.full((N,), 100, dtype=np.int8)

    # load first image, camera and depth map
    image_1 = images[k1]
    camera_1 = cameras[image_1.camera_id]
    depth_map_1 = load_depth_map(os.path.join(depth_path, f"{image_1.name}.geometric.bin"))

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

    for j, k2 in enumerate(images.keys()):
        if i == j and not cosine_weighted:
            continue

        # load second image, camera and depth map
        image_2 = images[k2]
        camera_2 = cameras[image_2.camera_id]
        depth_map_2 = load_depth_map(os.path.join(depth_path, f"{image_2.name}.geometric.bin"))

        # project all 3D points to image 2 to obtain 2D points and associated depth values
        proj_points_2d, proj_depths, proj_mask = forward_project(
            points_3d=points_3d,
            image=image_2,
            camera=camera_2,
        )

        # get corresponding depth values from the second depth map
        # depth map values are stored row-wise => first index with y-coordinate
        depth_2 = depth_map_2[proj_points_2d[:, 1], proj_points_2d[:, 0]]

        # compute inliers based on the absolute relative depth errors
        abs_rel_error = np.abs(depth_2 / proj_depths - 1.0)
        if cosine_weighted:
            normal_map_1 = load_depth_map(
                os.path.join(str(normal_path), f"{image_1.name}.geometric.bin")
            )
            normal_map_2 = load_depth_map(
                os.path.join(str(normal_path), f"{image_2.name}.geometric.bin")
            )
            norm_1 = normal_map_1[
                points_2d[proj_mask][:, 1].astype(int), points_2d[proj_mask][:, 0].astype(int)
            ]
            norm_2 = normal_map_2[proj_points_2d[:, 1], proj_points_2d[:, 0]]
            cos_w_2 = np.minimum(np.abs(norm_1[:, 2]), np.abs(norm_2[:, 2]))
            n_inliners = np.sum(cos_w_2[abs_rel_error < rel_thresh])
        else:
            n_inliners = np.count_nonzero(abs_rel_error < rel_thresh)

        # final score
        row[j] = np.rint(100 * n_inliners / n_features)

    return (i, row)  # add row index because the tasks for each row don't finish in-order


def dense_overlap(
    paths: DictConfig,
    downsample: int = 50,
    rel_thresh: float = 0.03,
    cosine_weighted: bool = False,
) -> xr.DataArray:
    """Computes the dense overlap metric between a list of images.

    For each pair of images (i,j), this score indicates the fraction of dense features in image i
    that are also contained in the image j. If i=j, the score is simply 100. Each score is an
    integer in the range [0, 100].

    Args:
        paths (DictConfig): Data paths.
        downsample (int, optional): By which factor to downsample depth maps.
        rel_thresh (float, optional): Dense points with an absolute relative depth error below this
            threshold are considered as inliers.
        cosine_weighted (bool): Whether to cosine-weight the overlap scores. Defaults to False.

    Returns:
        xr.DataArray: DataArray of shape (N, N) with the overlap metric between each pair of images.
    """
    reconstruction = pycolmap.Reconstruction(os.path.join(paths.dense, "sparse"))
    images = reconstruction.images
    image_fns = [images[k].name for k in images.keys()]
    N = len(images)

    # args for subprocesses
    kwargs = dict(
        paths=paths,
        downsample=downsample,
        rel_thresh=rel_thresh,
        cosine_weighted=cosine_weighted,
    )

    # compute overlap scores for each image pair
    scores = np.full((N, N), 100, dtype=np.int8)
    with ProcessPoolExecutor() as executor:
        logging.info(f"Using {os.cpu_count()} workers to compute the dense overlap.")

        futures = [
            executor.submit(_compute_dense_row, i, k1, **kwargs)
            for i, k1 in enumerate(images.keys())
        ]

        pbar = tqdm(total=len(futures), desc="Computing dense overlap...", ncols=80)
        for future in as_completed(futures):
            index, row = future.result()
            scores[index, :] = row
            pbar.update(1)

    # create and return data array
    coords = {"img1": image_fns, "img2": image_fns}
    return xr.DataArray(scores, coords=coords, dims=["img1", "img2"])
