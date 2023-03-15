"""Several helper functions for IO."""
import argparse
import os
from typing import List

import numpy as np
from PIL import Image


def load_image(image_path: str) -> np.ndarray:
    """Load the image from the given path and return it as a numpy array.

    Args:
        image_path (str): Path to the image.

    Returns:
        np.ndarray: Image as a numpy array.
    """
    img = Image.open(image_path)
    return np.array(img)


def save_image(image: np.ndarray, image_path: str) -> None:
    """Save the given image to the given path.

    Args:
        image (np.ndarray): Image to be saved.
        image_path (str): Path to the image.
    """
    img = Image.fromarray(image)
    img.save(image_path)


def get_scene_image_paths(scene: str, args: argparse.Namespace) -> List[str]:
    """Return a list of image paths for the given scene.

    Args:
        scene (str): The name of the scene to be processed.
        args (argparse.Namespace): The parsed command line arguments.

    Returns:
        List[str]: A list of image paths for the given scene.
    """
    scene_path = os.path.join(args.image_path, scene)
    if not os.path.exists(scene_path):
        raise ValueError(f"Scene does not exist: {scene_path}")

    image_paths = [fname for fname in os.listdir(scene_path)]

    # TODO: filter

    return image_paths


def read_depth_map(path: str) -> np.ndarray:
    """Read depth map from the given path and return it as a numpy array.

    This function was copied from:
    https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_dense.py

    Args:
        path (str): Path to the depth map.

    Returns:
        np.ndarray: Depth map as a numpy array.
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()
