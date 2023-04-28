"""Functions that implement the semantic filtering steps for the cleanup.."""
import cv2
import numpy as np
from skimage.measure import label

from megadepth.postprocessing.image_processing import disk_r4, remove_small_components

background_labels = [0, 1, 16, 25, 42, 48, 61, 84]
foreground_labels = [8, 19, 30, 36, 43, 87, 93, 100, 115, 134, 136, 144, 149]
transportation_labels = [20, 76, 80, 83, 102, 103, 127]
water_labels = [21, 26, 60, 128]

mask_type_to_labels = {
    "background": background_labels,
    "foreground": foreground_labels,
    "sky": [2],
    "removal": [2] + foreground_labels + transportation_labels + water_labels,
    "transportation": transportation_labels,
    "water": water_labels,
    "tree": [4],
    "plant": [17],
    "human": [12],
    "animal": [126],
    "fountain": [104],
    "sculpture": [132],
}


def get_mask(segmentation_map: np.ndarray, mask_type: str) -> np.ndarray:
    """Return the segmentation mask given a segmentation map and mask type.

    Args:
        segmentation_map (np.ndarray): Predicted segmentation map.
        mask_type (str]): Mask type.

    Returns:
        np.ndarray: Mask with the same shape as the segmentation map.
    """
    if mask_type not in mask_type_to_labels:
        raise ValueError(f"Invalid mask type: {mask_type}")

    labels = mask_type_to_labels[mask_type]
    if len(labels) == 1:
        return segmentation_map == labels[0]
    else:
        mask = np.in1d(segmentation_map, labels)
        return np.reshape(mask, segmentation_map.shape)


def apply_semantic_filtering(
    depth_map: np.ndarray,
    segmentation_map: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Apply semantic filtering to the depth map.

    Args:
        depth_map (np.ndarray): Depth map.
        segmentation_map (np.ndarray): Predicted segmentation map.
        threshold (float, optional): Threshold for the semantic filtering. Defaults to 0.5.

    Returns:
        np.ndarray: Filtered depth map.
    """
    # check connected components of different types of foreground masks
    for mask_type in ["tree", "plant", "human", "animal", "fountain", "sculpture"]:
        # get connected components from the current mask
        mask = get_mask(segmentation_map, mask_type)
        labeled_mask, num_components = label(mask, background=0, connectivity=2, return_num=True)

        # iterate over connected components
        for i in range(1, num_components + 1):
            # compute fraction of valid depths in the current component
            depth_vector = depth_map[labeled_mask == i]
            fraction = np.count_nonzero(depth_vector) / depth_vector.size

            # set depth values to 0 if the fraction is smaller than the threshold
            if fraction < threshold:
                depth_map[labeled_mask == i] = 0.0

    # get removal mask, dilate it and remove all corresponding depth values
    removal_mask = get_mask(segmentation_map, "removal").astype(np.uint8)
    removal_mask = cv2.morphologyEx(removal_mask, cv2.MORPH_DILATE, disk_r4)
    return depth_map * (removal_mask == 0)


def is_selfie_image(
    depth_map: np.ndarray, segmentation_map: np.ndarray, threshold: float = 0.3
) -> bool:
    """Check if the image is a "selfie image".

    Args:
        depth_map (np.ndarray): Depth map.
        segmentation_map (np.ndarray): Predicted segmentation map.
        threshold (float, optional): Threshold for the selfie criterion. Defaults to 0.3.

    Returns:
        bool: True if the image is a selfie image and False otherwise.
    """
    # get all depth values that don't belong to the sky region
    sky_mask = get_mask(segmentation_map, "sky")
    depth_values = depth_map[np.logical_not(sky_mask)]

    # check if the fraction of valid depths among the non-sky depths is smaller than the threshold
    num_valid = np.count_nonzero(depth_values)
    return num_valid < threshold * depth_values.size  # avoid division by 0....


def get_ordinal_map(
    depth_map: np.ndarray,
    segmentation_map: np.ndarray,
    fg_label: int = 1,
    bg_label: int = 2,
    min_fg_fraction: float = 0.05,
    min_bg_fraction: float = 0.05,
    depth_quantile: float = 0.75,
    n_pixels: int = 1000,
) -> np.ndarray:
    """Return an ordinal map given the depth map and the segmentation map.

    Args:
        depth_map (np.ndarray): Depth map.
        segmentation_map (np.ndarray): Predicted segmentation map.
        fg_label (int, optional): Foreground label in the ordinal map. Defaults to 1.
        bg_label (int, optional): Background label in the ordinal map. Defaults to 2.
        min_fg_fraction (float, optional): Foreground components need to occupy a larger fraction of
            the image to be included in the ordinal map. Defaults to 0.05.
        min_bg_fraction (float, optional): Background components need to occupy a larger fraction of
            the image to be included in the ordinal map. Defaults to 0.05.
        depth_quantile (float, optional): A background pixel needs to have a depth value larger than
            this quantile over all valid depths to be included in the ordinal map. Defaults to 0.75.
        n_pixels (int): Connected components with less pixels will be removed. Defaults to 1000.

    Returns:
        np.ndarray: Ordinal map.
    """
    ordinal_map = np.zeros_like(depth_map)

    # check each component of each foreground mask
    for mask_type in ["human", "animal", "sculpture", "transportation", "fountain", "foreground"]:
        # get connected components from the current mask
        foreground_mask = get_mask(segmentation_map, mask_type)
        labeled_mask, num_components = label(
            foreground_mask, background=0, connectivity=2, return_num=True
        )
        for i in range(1, num_components + 1):
            # check size of current component
            component_mask = labeled_mask == i
            component_size = np.count_nonzero(component_mask)
            if float(component_size) / depth_map.size < min_fg_fraction:
                continue

            # add component pixels to the ordinal map
            ordinal_map[component_mask] = fg_label

    # create mask for depth values that are large enough
    valid_depths = depth_map[depth_map > 0.0]
    if valid_depths.size == 0:
        return ordinal_map
    depth_mask = depth_map >= np.quantile(valid_depths, depth_quantile)

    # check each background component
    background_mask = get_mask(segmentation_map, "background")
    labeled_mask, num_components = label(
        background_mask, background=0, connectivity=2, return_num=True
    )
    for i in range(1, num_components + 1):
        # check size of current component
        component_mask = labeled_mask == i
        component_size = np.count_nonzero(component_mask)
        if float(component_size) / depth_map.size < min_bg_fraction:
            continue

        # add component pixels to the ordinal map if the corresponding depths are large enough
        combined_mask = component_mask & depth_mask
        ordinal_map[combined_mask] = bg_label

    # clean up final ordinal map
    ordinal_map = remove_small_components(ordinal_map, n_pixels)
    return ordinal_map
