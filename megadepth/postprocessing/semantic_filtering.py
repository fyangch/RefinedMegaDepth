"""Functions that implement the semantic filtering steps for the cleanup.."""
import cv2
import numpy as np
from skimage.measure import label

from megadepth.postprocessing.image_processing import disk_r4

background_labels = np.array([1, 25, 48, 68, 84, 113, 16])
removal_labels = np.array(
    [2, 20, 80, 102, 83, 76, 103, 36, 134, 136, 87, 100, 144, 149, 43, 93, 8, 115, 21, 26, 60, 128]
)


def get_mask(segmentation_map: np.ndarray, mask_type: str) -> np.ndarray:
    """Return the segmentation mask given a segmentation map and mask type.

    Args:
        segmentation_map (np.ndarray): Predicted segmentation map.
        mask_type (str]): Mask type.

    Returns:
        np.ndarray: Mask with the same shape as the segmentation map.
    """
    if mask_type == "background":
        mask = np.in1d(segmentation_map, background_labels)
        return np.reshape(mask, segmentation_map.shape)
    elif mask_type == "removal":
        mask = np.in1d(segmentation_map, removal_labels)
        return np.reshape(mask, segmentation_map.shape)
    elif mask_type == "sky":
        return segmentation_map == 2
    elif mask_type == "tree":
        return segmentation_map == 4
    elif mask_type == "plant":
        return segmentation_map == 17
    elif mask_type == "creature":
        mask = np.in1d(segmentation_map, np.array([12, 126]))
        return np.reshape(mask, segmentation_map.shape)
    elif mask_type == "fountain":
        return segmentation_map == 104
    elif mask_type == "sculpture":
        return segmentation_map == 132
    else:
        raise ValueError(f"Invalid mask type: {mask_type}")


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
    for mask_type in ["tree", "plant", "creature", "fountain", "sculpture"]:
        # get connected components from the current mask
        mask = get_mask(segmentation_map, mask_type)
        labeled_mask, num_components = label(mask, background=0, connectivity=2, return_num=True)

        # iterate over connected components
        for i in range(1, num_components + 1):
            # compute fraction of valid depths in the current component
            depth_vector = depth_map[labeled_mask == i]
            fraction = np.count_nonzero(depth_vector) / depth_vector.size

            # set depth values to 0 if the fraction is smaller than 0.5
            if fraction < threshold:
                depth_map[labeled_mask == i] = 0.0

    # get removal mask, dilate it and remove all corresponding depth values
    removal_mask = get_mask(segmentation_map, "removal").astype(np.uint8)
    removal_mask = cv2.morphologyEx(removal_mask, cv2.MORPH_DILATE, disk_r4)
    return depth_map * (removal_mask == 0)


def is_selfie_image(depth_map: np.ndarray, segmentation_map: np.ndarray, threshold=0.35) -> bool:
    """Check if the image is a selfie image.

    Returns True if depth coverage is bigger than threshold.
    for scenes [168, 229, 212, 768] they use 0.2 as threshold.

    Args:
        depth_map (np.ndarray): The depth map.
        segmentation_map (np.ndarray): The predicted segmentation map.

    Returns:
        bool: True if the image is a selfie image, False otherwise.
    """
    # ignore sky
    sky_mask = get_mask(segmentation_map, "sky")
    num_valid = np.count_nonzero(sky_mask != 0)
    num_valid_depth = np.count_nonzero(depth_map)
    return num_valid > threshold * num_valid_depth


# not sure yet about this step
def get_ordinal_labels():
    """Get the ordinal labels."""
    # TODO
    return
