"""Functions to compute the overlap metric between a list of images."""
from typing import List

import numpy as np


# TODO: Change function signature if necessary
def overlap(images: List[np.ndarray], depth_maps: List[np.ndarray]) -> np.ndarray:
    """Computes the overlap metric between a list of images.

    Args:
        images (List[np.ndarray]): List of images.
        depth_maps (List[np.ndarray]): List of depth maps.

    Returns:
        float: Overlap metric.
    """
    raise NotImplementedError()
