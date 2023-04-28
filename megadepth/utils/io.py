"""Several helper functions for IO."""
import argparse
import glob
import os
from pathlib import Path
from typing import List

import h5py
import numpy as np
from PIL import Image

from megadepth.utils.constants import ModelType


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


def make_gif(store_path: str, movie_path: str):
    """Crashes when used with many more frames than 50."""
    frames = [Image.open(image) for image in sorted(glob.glob(f"{store_path}/*.png"))]
    frame_one = frames[0]
    frame_one.save(
        movie_path, format="GIF", append_images=frames, save_all=True, duration=10, loop=0
    )


def load(file_path):
    """Load image file and depth maps."""
    _, ext = os.path.splitext(file_path)
    # print(ext)
    if ext == ".npy":
        return np.load(file_path)
    if ext == ".h5":
        with h5py.File(file_path, "r") as f:
            # List all the keys in the file
            # print("Keys: %s" % f.keys())
            # Get the dataset
            dataset = f["depth"]
            # Get the data from the dataset
            data = dataset[:]
            return data
    if ext == ".bin":
        return load_depth_map(file_path)
    return Image.open(file_path).convert("RGB")


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


def get_model_dir(scene: str, model: ModelType, args: argparse.Namespace) -> Path:
    """Return the path to the given scene.

    Args:
        scene (str): The name of the scene to be processed.
        args (argparse.Namespace): The parsed command line arguments.

    Returns:
        str: The path to the given scene.
    """
    model_dir = None

    if model == ModelType.SPARSE:
        model_dir = os.path.join(args.data_path, args.sparse_path, scene)
    elif model == ModelType.DENSE:
        model_dir = os.path.join(args.data_path, args.dense_path, scene)
    else:
        raise ValueError(f"Unknown model type: {model}")

    if not os.path.exists(model_dir):
        raise ValueError(f"Model does not exist at: {model_dir}")

    return Path(model_dir)


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
