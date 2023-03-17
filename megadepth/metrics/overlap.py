"""Functions to compute the overlap metric between a list of images."""


import numpy as np
import pycolmap


# TODO: Change function signature if necessary
def overlap(reconstruction: pycolmap.Reconstruction) -> np.ndarray:
    """Computes the overlap metric between a list of images.

    Args:
        reconstruction (pycolmap.Reconstruction): Reconstruction object.

    Returns:
        np.ndarray: Array of shape (N, N) with the overlap metric between each pair of images.
    """
    # use image, camera, and 3D point functions from pycolmap
    # images = reconstruction.images
    # cameras = reconstruction.cameras
    # points3D = reconstruction.points3D

    raise NotImplementedError()
