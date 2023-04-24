"""Functions that perform image processing steps required for the cleanup."""
import numpy as np


def median_filter(img, kernel_size=3):
    """Median filter that ignores zeros and nans."""
    # Replace zeros with NaNs
    img_nan = np.where(img == 0, np.nan, img)

    # Pad the image with NaNs
    img_padded = np.pad(img_nan, 2, mode="constant", constant_values=np.nan)

    # Apply median filter
    filtered_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Extract the subarray centered at (i, j)
            subarray = img_padded[:5, :5]  # ,img_padded[i : i + kernel_size, j : j + kernel_size]
            # Ignore NaNs and zeros in the subarray
            subarray = subarray[np.logical_and(~np.isnan(subarray), subarray != 0)]
            # Calculate the median of the subarray
            if subarray.size > 0:
                filtered_img[i, j] = np.nanmedian(subarray)
            else:
                filtered_img[i, j] = 0

    return filtered_img


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
    median_filtered = median_filter(depth_map, kernel_size)
    # avoid division by zero, max(D/Df, Df/D)> thresh
    unstable1 = median_filtered > threshold * depth_map
    unstable2 = depth_map > threshold * median_filtered
    # a pixel is stable unless one of the criterions makes it unstable
    stable = 1 - unstable1 * unstable2
    # 2. remove "unstable pixels"
    # filtered = cv2.bitwise_and(depth_map, depth_map, mask=stable)
    filtered = depth_map * stable
    return filtered


# TODO: Change function signature if necessary
def erode_and_remove(depth_map: np.ndarray, segmentation_map: np.ndarray) -> np.ndarray:
    """Erode the segmentation map and remove the small connected components.

    Args:
        depth_map (np.ndarray): Depth map.

    Returns:
        np.ndarray: Eroded depth map and filtered depth map.
    """
    # TODO: Supplementary material, Algorithm 1, line 18

    # 1. compute connected components from foreground segmentation

    # 2. for each connected component
    # if C is small: continue

    # a. count percentage of valid depths

    # if percentage is < 50% mask out the component

    raise NotImplementedError()
