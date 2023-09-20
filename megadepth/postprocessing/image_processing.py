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


def remove_small_components(depth_map: np.ndarray, n_pixels: int) -> np.ndarray:
    """Remove small connected components from the depth map.

    Args:
        depth_map (np.ndarray): Depth map.
        n_pixels (int): Connected components with less pixels will be removed.

    Returns:
        np.ndarray: Depth map without small connected components.
    """
    # binary mask of the depth map
    mask = (depth_map > 0.0).astype(np.uint8)

    # check each connected component
    labeled_mask, num_components = label(mask, background=0, connectivity=2, return_num=True)
    for i in range(1, num_components + 1):
        component_mask = labeled_mask == i
        component_size = np.count_nonzero(component_mask)

        if component_size < n_pixels:
            depth_map[component_mask] = 0.0

    return depth_map


def _resize_depth_map(depth_map: np.ndarray, max_size: int):
    """Resize the depth map such that the larger dimension is equal to `max_size`"""
    original_shape = depth_map.shape
    ratio = max_size / max(original_shape)
    new_shape = (int(original_shape[1] * ratio), int(original_shape[0] * ratio))
    depth_map = cv2.resize(depth_map, new_shape, interpolation=cv2.INTER_NEAREST)
    return depth_map


def filter_unstable_depths(
    depth_map: np.ndarray,
    kernel_size: int = 5,
    threshold: float = 1.15,
    max_size: int = 1600,
) -> np.ndarray:
    """Apply median filter and keep depth values that are close enough to the corresponding median.

    Args:
        depth_map (np.ndarray): Depth map.
        kernel_size (int, optional): Size of the median filter kernel. Defaults to 5.
        threshold (float, optional): Threshold for the relative depth difference. Defaults to 1.15.
        max_size (int, optional): The larger dim. will be scaled to this size. Defaults to 1600.

    Returns:
        np.ndarray: Filtered depth map.
    """
    # rescale depth map
    original_shape = depth_map.shape
    depth_map = _resize_depth_map(depth_map, max_size)

    # apply median filter
    median = cv2.medianBlur(depth_map, kernel_size)
    # median = cv2.blur(depth_map, (kernel_size, kernel_size))

    # a depth value D is unstable if max(D/M, M/D) > threshold where M is the corresponding median
    unstable_1 = median > threshold * depth_map  # avoid division by 0....
    unstable_2 = depth_map > threshold * median
    unstable = unstable_1 | unstable_2

    # remove unstable pixels from the depth map
    stable = np.logical_not(unstable)
    depth_map = depth_map * stable

    # scale back to original dimensions
    return cv2.resize(depth_map, original_shape[::-1], interpolation=cv2.INTER_NEAREST)


def erode_and_remove(
    depth_map: np.ndarray, n_pixels: int = 200, max_size: int = 1600
) -> np.ndarray:
    """Erode the depth map and remove small connected components.

    Args:
        depth_map (np.ndarray): Depth map.
        n_pixels (int): Connected components with less pixels will be removed. Defaults to 200.
        max_size (int, optional): The larger dim. will be scaled to this size. Defaults to 1600.

    Returns:
        np.ndarray: Eroded depth map and filtered depth map.
    """
    # rescale depth map
    original_shape = depth_map.shape
    depth_map = _resize_depth_map(depth_map, max_size)

    # create binary mask of the depth map
    mask = (depth_map > 0.0).astype(np.uint8)

    # apply closing and erosion
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk_r2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, disk_r4)

    depth_map = remove_small_components(depth_map, n_pixels)
    depth_map = depth_map * (mask == 1)

    # scale back to original dimensions
    return cv2.resize(depth_map, original_shape[::-1], interpolation=cv2.INTER_NEAREST)
