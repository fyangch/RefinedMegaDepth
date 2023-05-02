"""This is a re-implementation of the MegaDepth postprocessing steps."""

import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from megadepth.postprocessing.image_processing import erode_and_remove, filter_unstable_depths
from megadepth.postprocessing.semantic_filtering import apply_semantic_filtering, get_ordinal_map
from megadepth.postprocessing.semantic_segmentation import get_segmentation_model
from megadepth.utils.io import load_depth_map


def refine_depth_maps(
    image_dir: Path,
    depth_map_dir: Path,
    output_dir: Path,
    segmentation_model: str = "segformer",
) -> None:
    """Refine the depth maps and save the final depth maps, ordinal maps and segmentation maps.

    Args:
        image_dir (Path): Path to the directory that contains the undistorted RGB images.
        depth_map_dir (Path): Path to the directory that contains the raw depth maps.
        output_dir (Path): Path to the output directory.
        segmentation_model (str): Which segmentation model to use. Defaults to "segformer".
    """
    # create subdirectories for the refined depth maps, ordinal maps and segmentation maps
    os.makedirs(output_dir / "depth_maps", exist_ok=True)
    os.makedirs(output_dir / "ordinal_maps", exist_ok=True)
    os.makedirs(output_dir / "segmentation_maps", exist_ok=True)

    model = get_segmentation_model(segmentation_model)

    for image_fn in tqdm(os.listdir(image_dir)):
        image = Image.open(image_dir / image_fn).convert("RGB")
        depth_map = load_depth_map(os.path.join(depth_map_dir, f"{image_fn}.geometric.bin"))

        segmentation_map = model.get_segmentation_map(image)

        depth_map = filter_unstable_depths(depth_map)
        depth_map = apply_semantic_filtering(depth_map, segmentation_map)
        depth_map = erode_and_remove(depth_map)

        ordinal_map = get_ordinal_map(depth_map, segmentation_map)

        with open(output_dir / "ordinal_maps" / f"{image_fn}.npy", "wb") as f:
            np.save(f, ordinal_map)
        with open(output_dir / "depth_maps" / f"{image_fn}.npy", "wb") as f:
            np.save(f, depth_map)
        with open(output_dir / "segmentation_maps" / f"{image_fn}.npy", "wb") as f:
            np.save(f, segmentation_map)
