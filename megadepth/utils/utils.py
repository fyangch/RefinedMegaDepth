"""Utility functions."""

import argparse
import datetime
import logging
import os

import numpy as np
import pycolmap


def camera_pixel_grid(
    camera: pycolmap.Camera, downsample: int, reverse_x: bool = False, reverse_y: bool = False
) -> np.ndarray:
    """Generate array of 2d grid points within the bounds of the given camera.

    Args:
        camera (pycolmap.Camera)
        : Camera to use to generate grid points
        downsample (int): number of samples
        reverse_x (bool, optional): whether to reverse x-axis. Defaults to False.
        reverse_y (bool, optional): whether to reverse y-axis. Defaults to False.

    Returns:
        np.ndarray: 2d list of points of shape (n, 2)
    """
    if reverse_x:
        xl = np.linspace(camera.width - 1, 0, camera.width)[::downsample]
    else:
        xl = np.linspace(0, camera.width - 1, camera.width)[::downsample]

    if reverse_y:
        yl = np.linspace(camera.height - 1, 0, camera.height)[::downsample]
    else:
        yl = np.linspace(0, camera.height - 1, camera.height)[::downsample]

    xv, yv = np.meshgrid(xl, yl)
    return np.vstack((np.ravel(xv), np.ravel(yv))).T


def filter_mask(data: np.ndarray, w: int, h: int) -> np.ndarray:
    """Generate a mask that filters points between [0, w] and [0, h].

    Args:
        data (np.ndarray): array of shape (n, 2)
        w (int): upper bound for x
        h (int): upper bound for y

    Returns:
        np.ndarray: a mask of 0 and 1.
    """
    x = data[:, 0]
    y = data[:, 1]
    mask = (x >= 0) & (x <= w) & (y >= 0) & (y <= h)
    return mask


def setup() -> argparse.Namespace:
    """Setup the logging and command line arguments.

    Returns:
        The parsed command line arguments.
    """
    # Parse command line arguments.
    parser = argparse.ArgumentParser()

    # DATA RELATED ARGUMENTS
    parser.add_argument(
        "--image_path",
        type=str,
        default="data/00_raw",
        help="Path to the image to be processed.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="The scene to be processed.",
    )

    # MODEL RELATED ARGUMENTS

    # LOGGING RELATED ARGUMENTS
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Path to the log file.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO"],
        default="INFO",
        help="The logging level.",
    )

    args = parser.parse_args()

    # Setup logging.
    if args.log_dir:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(args.log_dir, f"{date}_log.txt")
    else:  # log to stdout
        log_file = None

    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        level=args.log_level,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
    )

    logging.debug("Command line arguments:")
    for arg, val in vars(args).items():
        logging.debug(f"\t{arg}: {val}")

    return args
