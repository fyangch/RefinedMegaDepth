"""Functions that perform image processing steps required for the cleanup."""
import cv2
import numpy as np
from skimage.measure import label

# disk-shaped kernel with radius 2
disk_r2 = np.array(
    [
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
    ],
    dtype=np.uint8,
)

# disk-shaped kernel with radius 4
disk_r4 = np.array(
    [
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
    ],
    dtype=np.uint8,
)


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
    median = cv2.medianBlur(depth_map, kernel_size)

    # a depth value D is unstable if max(D/M, M/D) > threshold where M is the corresponding median
    unstable_1 = median > threshold * depth_map  # avoid division by 0....
    unstable_2 = depth_map > threshold * median
    unstable = unstable_1 | unstable_2

    # remove unstable pixels from the depth map
    stable = np.logical_not(unstable)
    return depth_map * stable


def erode_and_remove(depth_map: np.ndarray, n_pixels: int = 200) -> np.ndarray:
    """Erode the depth map and remove small connected components.

    Args:
        depth_map (np.ndarray): Depth map.
        n_pixels (int): Connected components with less pixels will be removed. Defaults to 200.

    Returns:
        np.ndarray: Eroded depth map and filtered depth map.
    """
    # create binary mask of the depth map
    mask = (depth_map > 0.0).astype(np.uint8)

    # apply closing and erosion
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk_r2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, disk_r4)

    # remove small connected components
    labeled_mask, num_components = label(mask, background=0, connectivity=2, return_num=True)
    for i in range(1, num_components + 1):
        component_size = labeled_mask[labeled_mask == i].size
        if component_size < n_pixels:
            depth_map[labeled_mask == i] = 0.0

    return depth_map * (mask == 1)
