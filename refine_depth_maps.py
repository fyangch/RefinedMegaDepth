import os
import argparse
import numpy as np

from src.semantic_filtering import *
from src.image_processing import *
from src.utils import *


def refine_depth_maps(
    image_dir: str,
    depth_map_dir: str,
    output_dir: str,
    # TODO: add optional params, e.g. segmentation model, threshold values, etc.
):
    """ TODO """

    model = get_segmentation_model() # TODO

    for image_path in os.listdir(image_dir):
        depth_map_path = os.path(depth_map_dir, "TODO") # TODO: set correct path
        depth_map = read_depth_map(depth_map_path)
        segmentation_map = get_segmentation_map(image_path, model)

        depth_map = filter_unstable_depths(depth_map)
        depth_map = apply_semantic_filtering(depth_map, segmentation_map)
        depth_map = erode_and_remove(depth_map)

        # not sure about this part...
        if is_selfie_image(depth_map, segmentation_map):
            ordinal_labels = get_ordinal_labels() 

        # TODO: save depth map (+ ordinal labels) ==> check format of the original MegaDepth dataset!!!


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        help="Path to the directory that contains the original RGB images.", 
        type=str, 
        required=True,
    )
    parser.add_argument(
        "--depth_map_dir",
        help="Path to the directory that contains the raw depth maps.", 
        type=str, 
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Path to the output directory that contains the refined depth maps.", 
        type=str, 
        required=True,
    )
    args = parser.parse_args()
    
    refine_depth_maps(
        image_dir=args.image_dir,
        depth_map_dir=args.depth_map_dir,
        output_dir=args.output_dir,
    )
