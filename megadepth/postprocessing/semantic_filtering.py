"""Functions that implement the semantic filtering steps for the cleanup.."""
import numpy as np


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
