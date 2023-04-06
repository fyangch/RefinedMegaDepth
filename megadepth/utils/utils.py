"""Utility functions."""

import argparse
import datetime
import logging
import os
from pathlib import Path

import numpy as np
import pycolmap
from hloc import extract_features, match_features

import megadepth.utils.projections as projections
from megadepth.utils.constants import Features, Matcher, Retrieval


def camera_pixel_grid(
    camera: pycolmap.Camera, downsample: int, reverse_x: bool = False, reverse_y: bool = False
) -> np.ndarray:
    """Generate array of 2d grid points within the bounds of the given camera.

    Args:
        camera (pycolmap.Camera): Camera to use to generate grid points
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
    x = data[:, 0].astype(int)
    y = data[:, 1].astype(int)
    mask = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return mask


def setup() -> argparse.Namespace:
    """Setup the logging and command line arguments.

    Returns:
        The parsed command line arguments.
    """
    args = setup_args()
    setup_logger(args)

    logging.debug("Command line arguments:")
    for arg, val in vars(args).items():
        logging.debug(f"\t{arg}: {val}")

    return args


def setup_args() -> argparse.Namespace:
    """Setup the command line arguments.

    Returns:
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser()

    # DATA RELATED ARGUMENTS
    parser.add_argument(
        "--data_path",
        type=Path,
        default="data",
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        required=True,
        help="The scene to be processed.",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default="images",
        help="Path to the images.",
    )
    parser.add_argument(
        "--features_dir",
        type=Path,
        default="features",
        help="Path to the metadata.",
    )
    parser.add_argument(
        "--matches_dir",
        type=Path,
        default="matches",
        help="Path to the sparse model.",
    )
    parser.add_argument(
        "--sparse_dir",
        type=Path,
        default="sparse",
        help="Path to the sparse model.",
    )
    parser.add_argument(
        "--dense_dir",
        type=Path,
        default="dense",
        help="Path to the dense model.",
    )
    parser.add_argument(
        "--metrics_dir",
        type=Path,
        default="metrics",
        help="Path to the metrics.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default="results",
        help="Path to the results.",
    )

    # MODEL RELATED ARGUMENTS
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model. if not specified, it will be set to <feature>-<matcher>",
    )

    parser.add_argument(
        "--retrieval",
        type=str,
        choices=[r.value for r in Retrieval],
        default=Retrieval.NETVLAD.value,
        help="The retrieval method used in the model.",
    )

    parser.add_argument(
        "--n_retrieval_matches",
        type=int,
        default=50,
        help="The number of retrieval matches.",
    )

    parser.add_argument(
        "--features",
        type=str,
        choices=[f.value for f in Features],
        default=Features.SIFT.value,
        help="The features used in the model.",
    )

    parser.add_argument(
        "--matcher",
        type=str,
        choices=[m.value for m in Matcher],
        default=Matcher.NN_RATIO.value,
        help="The matcher used in the model.",
    )

    # PIPELINE RELATED ARGUMENTS
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the pipeline.",
    )

    parser.add_argument(
        "--colmap",
        action="store_true",
        help="Use COLMAP for the pipeline.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing results.",
    )

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
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Specify verbosity.",
    )

    return parser.parse_args()


def setup_logger(args: argparse.Namespace) -> None:
    """Setup the logger.

    Args:
        args (argparse.Namespace): The parsed command line arguments.
    """
    if args.log_dir:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(args.log_dir, f"{date}_log.txt")
    else:  # log to stdout
        log_file = None

    logging.basicConfig(
        format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=args.log_level,
        filename=log_file,
    )


def get_configs(args: argparse.Namespace) -> dict:
    """Return a dictionary of configuration parameters.

    Args:
        args (argparse.Namespace): The parsed command line arguments.

    Returns:
        dict: A dictionary of configuration parameters.
    """
    confs = {
        "retrieval": extract_features.confs[args.retrieval]
        if args.retrieval
        not in [Retrieval.POSES.value, Retrieval.COVISIBILITY.value, Retrieval.EXHAUSTIVE.value]
        else None,
        "feature": extract_features.confs[args.features],
        "matcher": match_features.confs[args.matcher],
    }

    if args.features == Features.SIFT.value:
        confs["feature"]["preprocessing"]["resize_max"] = 3200
        # confs["feature"]["preprocessing"]["grayscale"] = False # Raises error

    return confs


class DataPaths:
    """Class for handling the data paths."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the data paths.

        Args:
            args (argparse.Namespace): The parsed command line arguments.
        """
        self.model_name = self.get_model_name(args)

        retrieval_name = f"{args.retrieval}"
        retrieval_name += (
            ".txt"
            if args.retrieval != Retrieval.EXHAUSTIVE.value
            else f"-{args.n_retrieval_matches}.txt"
        )

        # paths
        self.data = Path(os.path.join(args.data_path, args.scene))
        self.images = Path(os.path.join(self.data, args.image_dir))

        # retrieval
        self.features_retrieval = Path(
            os.path.join(self.data, args.features_dir, f"{args.retrieval}.h5")
        )
        self.matches_retrieval = Path(
            os.path.join(self.data, args.matches_dir, "retrieval", retrieval_name)
        )

        # features
        self.features = Path(os.path.join(self.data, args.features_dir, f"{args.features}.h5"))

        # matches
        self.matches = Path(os.path.join(self.data, args.matches_dir, f"{self.model_name}.h5"))

        # models
        self.sparse = Path(os.path.join(self.data, args.sparse_dir, self.model_name))
        self.db = Path(os.path.join(self.sparse, "database.db"))
        self.dense = Path(os.path.join(self.data, args.dense_dir, self.model_name))
        self.baseline_model = Path(os.path.join(self.data, args.sparse_dir, "baseline"))

        # output
        self.metrics = Path(os.path.join(self.data, args.metrics_dir, self.model_name))
        self.results = Path(os.path.join(self.data, args.results_dir, self.model_name))

        logging.debug("Data paths:")
        for path, val in vars(self).items():
            logging.debug(f"\t{path}: {val}")

    def get_model_name(self, args: argparse.Namespace) -> str:
        """Return the model name.

        Args:
            args (argparse.Namespace): The parsed command line arguments.

        Returns:
            str: The model name.
        """
        if args.model_name:
            return args.model_name
        elif args.colmap:
            return "colmap"
        else:
            return f"{args.features}-{args.matcher}-{args.retrieval}-{args.n_retrieval_matches}"


def get_camera_poses(reconstruction) -> np.ndarray:
    """Extracts camera positions from reconstruction.

    Args:
        reconstruction: pycolmap.Reconstruction(/path)

    Returns:
        np.ndarray: of shape (N, 3)
    """
    cameras = reconstruction.cameras
    images = reconstruction.images

    N = len(images)
    camera_poses = np.zeros((N, 3))
    for i, k1 in enumerate(images.keys()):
        image_1 = images[k1]
        camera_1 = cameras[image_1.camera_id]
        camera_poses[i] = projections.backward_project(
            points_2d=np.array([[0, 0]]),
            image=image_1,
            camera=camera_1,
            depth=0,
        )
    return camera_poses
