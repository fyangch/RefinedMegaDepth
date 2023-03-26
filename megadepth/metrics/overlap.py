"""Functions to compute the overlap metrics between a list of images."""


import numpy as np
import pycolmap


def sparse_overlap(reconstruction: pycolmap.Reconstruction) -> np.ndarray:
    """Computes the sparse overlap metric between a list of images.

    Computation is based on:
        https://github.com/mihaidusmanu/d2-net/blob/master/megadepth_utils/preprocess_scene.py#L202

    Args:
        reconstruction (pycolmap.Reconstruction): Reconstruction object.

    Returns:
        np.ndarray: Array of shape (N, N) with the overlap metric between each pair of images.
    """
    images = reconstruction.images
    N = len(images)

    # pre-compute hash sets with 3D point IDs for faster look ups later
    img_to_ids = {}
    for i in range(N):
        img = images[i + 1]  # image indices start at 1
        img_to_ids[i] = set([p.point3D_id for p in img.get_valid_points2D()])

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
    reconstruction: pycolmap.Reconstruction, stride: int = 10, reproj_thresh: int = 2000
) -> np.ndarray:
    """Computes the dense overlap metric between a list of images.

    Args:
        reconstruction (pycolmap.Reconstruction): Reconstruction object.
        stride (int, optional): Which stride to use when accessing the image and depth map pixels.
        reproj_thresh (int, optional): Dense features with a reprojection error below this
            threshold are considered as valid.

    Returns:
        np.ndarray: Array of shape (N, N) with the overlap metric between each pair of images.
    """
    # use image, camera, and 3D point functions from pycolmap
    # images = reconstruction.images
    # cameras = reconstruction.cameras
    # points3D = reconstruction.points3D

    raise NotImplementedError()
