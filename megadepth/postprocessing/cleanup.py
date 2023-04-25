"""This is a re-implementation of the MegaDepth postprocessing steps."""

import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from megadepth.postprocessing.image_processing import erode_and_remove, filter_unstable_depths
from megadepth.postprocessing.semantic_filtering import (
    apply_semantic_filtering,
    get_ordinal_labels,
    is_selfie_image,
)
from megadepth.postprocessing.semantic_segmentation import (
    get_segmentation_map,
    get_segmentation_model,
)
from megadepth.utils.io import load_depth_map


def refine_depth_maps(
    image_dir: Path,
    depth_map_dir: Path,
    output_dir: Path,
) -> None:
    """Refine the depth maps.

    Args:
        image_dir (Path): Path to the directory that contains the undistorted RGB images.
        depth_map_dir (Path): Path to the directory that contains the raw depth maps.
        output_dir (Path): Path to the directory where the refined depth maps should be saved.
    """
    model = get_segmentation_model()

    # TODO: consider creating batches for the prediction of the segmentation maps
    for image_fn in tqdm(os.listdir(image_dir)):
        image = Image.open(image_dir / image_fn).convert("RGB")
        depth_map = load_depth_map(os.path.join(depth_map_dir, f"{image_fn}.geometric.bin"))

        segmentation_map = get_segmentation_map(image, model)

        depth_map = filter_unstable_depths(depth_map)
        depth_map = apply_semantic_filtering(depth_map, segmentation_map)
        depth_map = erode_and_remove(depth_map, segmentation_map)

        # TODO: finish ordinal labeling
        if is_selfie_image(depth_map, segmentation_map):
            _ = get_ordinal_labels()

        # TODO: save depth map (+ ordinal labels) ==> check format of the original MegaDepth dataset
