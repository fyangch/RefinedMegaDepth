import numpy as np
from torch import nn


# TODO: Change function signature if necessary
def get_segmentation_map(img_path: str, model: nn.Module) -> np.ndarray:
    """ 
    Read the original RGB image from the given path and use the 
    pretrained segmentation model to obtain its segmentation map. 
    """
    
    # TODO

    raise NotImplementedError()


# TODO: Change function signature if necessary
def apply_semantic_filtering(
    depth_map: np.ndarray, 
    segmentation_map: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """ TODO """

    # TODO: Supplementary material, Algorithm 1, line 12-17

    raise NotImplementedError()


# TODO: Change function signature if necessary
def is_selfie_image(depth_map: np.ndarray, segmentation_map: np.ndarray) -> bool:
    """ TODO """

    # I guess this is the very last step? (not shown in Algorithm 1 of the supplements...)

    raise NotImplementedError()


# not sure yet about this step
def get_ordinal_labels():
    raise NotImplementedError()
