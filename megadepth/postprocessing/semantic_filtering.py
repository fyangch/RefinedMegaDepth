"""Functions that implement the semantic filtering steps for the cleanup.."""
from typing import Literal

import numpy as np

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


# TODO: Change function signature if necessary
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
    # TODO: Supplementary material, Algorithm 1, line 12-17

    raise NotImplementedError()


# TODO: Change function signature if necessary
def is_selfie_image(depth_map: np.ndarray, segmentation_map: np.ndarray) -> bool:
    """Check if the image is a selfie image.

    Args:
        depth_map (np.ndarray): The depth map.
        segmentation_map (np.ndarray): The predicted segmentation map.

    Returns:
        bool: True if the image is a selfie image, False otherwise.
    """
    # I guess this is the very last step? (not shown in Algorithm 1 of the supplements...)

    raise NotImplementedError()


# not sure yet about this step
def get_ordinal_labels():
    """Get the ordinal labels."""
    raise NotImplementedError()
