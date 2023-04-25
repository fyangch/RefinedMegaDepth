"""Functions that perform image processing steps required for the cleanup."""
import cv2
import numpy as np


def filter_unstable_depths(
    depth_map: np.ndarray,
    kernel_size: int = 5,
    threshold: float = 1.15,
) -> np.ndarray:
    """Apply median filter and keep depth values that are close enough to the corresponding median.

    Args:
        depth_map (np.ndarray): Depth map.
        kernel_size (int, optional): Size of the median filter kernel. Defaults to 5.
        threshold (float, optional): Threshold for the relative depth difference. Defaults to 1.15.

    Returns:
        np.ndarray: Filtered depth map.
    """
    # apply median filter
    median_filtered = cv2.medianBlur(depth_map, kernel_size)

    # a depth value D is unstable if max(D/M, M/D) > threshold where M is the corresponding median
    unstable_1 = median_filtered > threshold * depth_map  # avoid division by 0....
    unstable_2 = depth_map > threshold * median_filtered
    unstable = unstable_1 | unstable_2

    # only consider removing valid depth values from the original depth map
    valid = depth_map > 0.0
    unstable = unstable & valid

    # remove unstable pixels from the median filtered depth map
    stable = np.logical_not(unstable)
    return median_filtered * stable


def erode_and_remove(depth_map: np.ndarray, segmentation_map: np.ndarray) -> np.ndarray:
    """Erode the segmentation map and remove the small connected components.

    Args:
        depth_map (np.ndarray): Depth map.

    Returns:
        np.ndarray: Eroded depth map and filtered depth map.
    """
    # TODO: Supplementary material, Algorithm 1, line 18

    raise NotImplementedError()
