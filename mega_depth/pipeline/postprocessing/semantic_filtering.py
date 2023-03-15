"""Functions that implement the semantic filtering steps of the MegaDepth pipeline."""
import numpy as np
from torch import nn


# TODO: Change function signature if necessary
def get_segmentation_map(img_path: str, model: nn.Module) -> np.ndarray:
    """Extract the segmentation map from the image.

    Args:
        img_path (str): Path to the image.
        model (nn.Module): Segmentation model to predict the segmentation map.

    Returns:
        np.ndarray: Predicted segmentation map.
    """
    # TODO: implemtent this function

    raise NotImplementedError()


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
