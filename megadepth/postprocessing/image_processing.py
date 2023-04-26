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


def median_filter(img, kernel_size=5):
    """Median filter that ignores zeros and nans."""
    # Replace zeros with NaNs
    img_nan = np.where(img == 0, np.nan, img)
    # compute kernel radius
    r = (kernel_size - 1) // 2
    # Pad the image with NaNs
    img_padded = np.pad(img_nan, r, mode="constant", constant_values=np.nan)
    # Apply median filter
    IPc = np.lib.stride_tricks.sliding_window_view(img_padded, (kernel_size, kernel_size))
    # compute nanmedian along window dimensions
    filtered_img = np.nanmedian(IPc, axis=(2, 3))
    return filtered_img


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
    median = median_filter(depth_map, kernel_size)

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


if __name__ == "__main__":
    depth = np.array(
        [
            [0, 0, 0, 0, 0],
            [2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    # depth = np.where(depth == 0, np.nan, depth)
    print(median_filter(depth, 3))
    print(cv2.medianBlur(depth, 3))
    print()

    # [[2. 2. 2. 2. 2.]
    # [2. 2. 2. 2. 2.]
    # [2. 2. 2. 2. 2.]
    # [2. 2. 2. 2. 2.]
    # [2. 2. 2. 2. 2.]
    # [2. 2. 2. 2. 2.]]

    # [[0 0 0 0 0]
    # [0 0 0 0 0]
    # [0 0 0 0 0]
    # [0 0 0 0 0]
    # [0 0 0 0 0]
    # [0 0 0 0 0]]