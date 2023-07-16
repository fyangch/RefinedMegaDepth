"""Several helper functions for IO."""
import argparse

# import glob
import os
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image

# from typing import List


# from megadepth.utils.constants import ModelType


def load_image(image_path: str) -> np.ndarray:
    """Load the image from the given path and return it as a numpy array.

    Args:
        image_path (str): Path to the image.

    Returns:
        np.ndarray: Image as a numpy array.
    """
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def save_image(image: np.ndarray, image_path: str) -> None:
    """Save the given image to the given path.

    Args:
        image (np.ndarray): Image to be saved.
        image_path (str): Path to the image.
    """
    img = Image.fromarray(image)
    img.save(image_path)


def load_depth_map(path: str) -> np.array:
    """Load the depth map from the given path and return it as a numpy array.

    The code for this function was copied from:
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


def get_image_dir(scene: str, args: argparse.Namespace) -> Path:
    """Return the path to the given scene.

    Args:
        scene (str): The name of the scene to be processed.
        args (argparse.Namespace): The parsed command line arguments.

    Returns:
        str: The path to the given scene.
    """
    image_dir = os.path.join(args.data_path, args.image_path, scene)
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory does not exist at: {image_dir}")

    return Path(image_dir)


def model_exists(path: Path) -> bool:
    """Check if the model exists.

    Args:
        path (Path): Path to the model.

    Returns:
        True if the model exists, False otherwise.
    """
    try:
        pycolmap.Reconstruction(path)
        return True
    except ValueError:
        return False
