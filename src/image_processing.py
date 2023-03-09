import numpy as np


# TODO: Change function signature if necessary
def filter_unstable_depths(
    depth_map: np.ndarray, 
    kernel_size: int = 5, 
    threshold: float = 1.15,
) -> np.ndarray:
    """ 
    Apply median filter to the input depth map and filter
    all unstable pixels from the resulting depth map.
    """
    
    # TODO: Supplementary material, Algorithm 1, line 10-11
    # 1. apply 5x5 median filter 
    # 2. remove "unstable pixels"

    raise NotImplementedError()


# TODO: Change function signature if necessary
def erode_and_remove(depth_map: np.ndarray) -> np.ndarray:
    """
    Apply morphological erosion followed by a removal of 
    small connected components to obtain the final depth map.
    """
    
    # TODO: Supplementary material, Algorithm 1, line 18

    raise NotImplementedError()
