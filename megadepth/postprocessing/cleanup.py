"""This is a re-implementation of the MegaDepth postprocessing steps."""

import os

from megadepth.postprocessing.image_processing import erode_and_remove, filter_unstable_depths
from megadepth.postprocessing.semantic_filtering import (
    apply_semantic_filtering,
    get_ordinal_labels,
    get_segmentation_map,
    is_selfie_image,
)
from megadepth.postprocessing.semantic_segmentation import get_segmentation_model
from megadepth.utils.io import load_depth_map


def refine_depth_maps(
    image_dir: str,
    depth_map_dir: str,
    output_dir: str,
    # TODO: add optional params, e.g. segmentation model, threshold values, etc.
) -> None:
    """Refine the depth maps.

    Args:
        image_dir (str): Path to the directory that contains the undistorted RGB images.
        depth_map_dir (str): Path to the directory that contains the raw depth maps.
        output_dir (str): Path to the directory where the refined depth maps should be saved.
    """
    # TODO: implement this function

    feature_extractor, model = get_segmentation_model()

    for image_path in os.listdir(image_dir):
        depth_map_path = os.path.join(depth_map_dir, "TODO")  # TODO: set correct path
        depth_map = load_depth_map(depth_map_path)
        segmentation_map = get_segmentation_map(image_path, model)

        depth_map = filter_unstable_depths(depth_map)
        depth_map = apply_semantic_filtering(depth_map, segmentation_map)
        depth_map = erode_and_remove(depth_map)

        # not sure about this part...
        if is_selfie_image(depth_map, segmentation_map):
            # ordinal_labels = get_ordinal_labels() # commented out because it's not used
            _ = get_ordinal_labels()

        # TODO: save depth map (+ ordinal labels) ==> check format of the original MegaDepth dataset
