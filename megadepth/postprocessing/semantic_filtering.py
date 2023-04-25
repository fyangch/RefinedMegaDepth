"""Functions that implement the semantic filtering steps for the cleanup.."""
from typing import Literal

import numpy as np
from skimage.measure import label

foreground_labels = np.array(
    [
        12,
        15,
        19,
        31,
        43,
        66,
        67,
        69,
        76,
        80,
        83,
        87,
        88,
        100,
        102,
        103,
        104,
        115,
        116,
        119,
        126,
        127,
        132,
        136,
        144,
    ]
)
background_labels = np.array([1, 25, 48, 68, 84, 113, 16])


def get_mask(
    segmentation_map: np.ndarray, mask_type: Literal["foreground", "background", "sky"]
) -> np.ndarray:
    """Return the foreground/background/sky mask given a segmentation mask.

    Args:
        segmentation_map (np.ndarray): Predicted segmentation map.
        mask_type (Literal["foreground", "background", "sky"]): Mask type.

    Returns:
        np.ndarray: Mask with the same shape as the segmentation map.
    """
    if mask_type == "foreground":
        mask = np.in1d(segmentation_map, foreground_labels)
        return np.reshape(mask, segmentation_map.shape)
    elif mask_type == "background":
        mask = np.in1d(segmentation_map, background_labels)
        return np.reshape(mask, segmentation_map.shape)
    elif mask_type == "sky":
        return segmentation_map == 2
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
    foreground_mask = get_mask(segmentation_map, "foreground")
    sky_mask = get_mask(segmentation_map, "sky")

    # compute connected components from foreground segmentation
    labeled_mask, num_components = label(
        foreground_mask, background=0, connectivity=2, return_num=True
    )

    # iterate over connected components
    for i in range(1, num_components + 1):
        # compute fraction of valid depths in the current component
        depth_vector = depth_map[labeled_mask == i]
        fraction = np.count_nonzero(depth_vector) / depth_vector.size

        # set depth values to 0 if the fraction is smaller than 0.5
        if fraction < threshold:
            depth_map[labeled_mask == i] = 0.0

    # filter out all depths from the sky region
    depth_map[sky_mask] = 0.0

    return depth_map


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
    raise NotImplementedError()
