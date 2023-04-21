"""Functions that perform image processing steps required for the cleanup."""
import numpy as np


# TODO: Change function signature if necessary
def filter_unstable_depths(
    depth_map: np.ndarray,
    kernel_size: int = 5,
    threshold: float = 1.15,
) -> np.ndarray:
    """Filter unstable depths by removing unstable pixels.

    Args:
        depth_map (np.ndarray): _description_
        kernel_size (int, optional): Size of the median filter kernel. Defaults to 5.
        threshold (float, optional): Threshold for the depth difference. Defaults to 1.15.

    Returns:
        np.ndarray: Filtered depth map.
    """
    # TODO: Supplementary material, Algorithm 1, line 10-11
    # 1. apply 5x5 median filter
    # 2. remove "unstable pixels"

    raise NotImplementedError()


# TODO: Change function signature if necessary
def erode_and_remove(depth_map: np.ndarray) -> np.ndarray:
    """Erode the depth map and remove the eroded pixels.

    Args:
        depth_map (np.ndarray): Depth map.

    Returns:
        np.ndarray: Eroded depth map and filtered depth map.
    """
    # TODO: Supplementary material, Algorithm 1, line 18

    raise NotImplementedError()
