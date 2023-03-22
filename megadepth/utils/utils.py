"""Utility functions."""

import argparse
import datetime
import logging
import os
from pathlib import Path

from hloc import extract_features, match_features

from megadepth.utils.enums import Features, Matcher, Retrieval


def setup() -> argparse.Namespace:
    """Setup the logging and command line arguments.

    Returns:
        The parsed command line arguments.
    """
    # Parse command line arguments.
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
        "--image_path",
        type=Path,
        default="00_images",
        help="Path to the images.",
    )
    parser.add_argument(
        "--features_path",
        type=Path,
        default="01_features",
        help="Path to the metadata.",
    )
    parser.add_argument(
        "--matches_path",
        type=Path,
        default="02_matches",
        help="Path to the sparse model.",
    )
    parser.add_argument(
        "--sparse_path",
        type=Path,
        default="03_sparse",
        help="Path to the sparse model.",
    )
    parser.add_argument(
        "--dense_path",
        type=Path,
        default="04_dense",
        help="Path to the dense model.",
    )
    parser.add_argument(
        "--metrics_path",
        type=Path,
        default="05_metrics",
        help="Path to the metrics.",
    )
    parser.add_argument(
        "--results_path",
        type=Path,
        default="04_results",
        help="Path to the results.",
    )

    # MODEL RELATED ARGUMENTS
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
        default=5,
        help="The number of retrieval matches. 0 for exhaustive matching.",
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
        format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=args.log_level,
        filename=log_file,
    )

    logging.debug("Command line arguments:")
    for arg, val in vars(args).items():
        logging.debug(f"\t{arg}: {val}")

    return args


def get_configs(args: argparse.Namespace) -> dict:
    """Return a dictionary of configuration parameters.

    Args:
        args (argparse.Namespace): The parsed command line arguments.

    Returns:
        dict: A dictionary of configuration parameters.
    """
    return {
        "retrieval": extract_features.confs[args.retrieval],
        "feature": extract_features.confs[args.features],
        "matcher": match_features.confs[args.matcher],
    }


class DataPaths:
    """Class for handling the data paths."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the data paths.

        Args:
            args (argparse.Namespace): The parsed command line arguments.
        """
        # data
        self.data = Path(args.data_path)
        self.model_name = f"{args.features}-{args.matcher}"

        self.images = Path(os.path.join(self.data, args.image_path, args.scene))

        # features
        self.features = Path(
            os.path.join(self.data, args.features_path, args.scene, f"{args.features}.h5")
        )

        retrieval_name = f"{args.retrieval}"
        retrieval_name += (
            f"-{args.n_retrieval_matches}.txt" if args.n_retrieval_matches > 0 else ".txt"
        )
        self.features_retrieval = Path(
            os.path.join(self.data, args.features_path, args.scene, f"{args.retrieval}.h5")
        )

        os.makedirs(self.features.parent, exist_ok=True)
        os.makedirs(self.features_retrieval.parent, exist_ok=True)

        # matches
        self.matches = Path(
            os.path.join(self.data, args.matches_path, args.scene, f"{self.model_name}.h5")
        )

        self.matches_retrieval = Path(
            os.path.join(self.data, args.matches_path, args.scene, "retrieval", retrieval_name)
        )

        os.makedirs(self.matches.parent, exist_ok=True)
        os.makedirs(self.matches_retrieval.parent, exist_ok=True)

        # models
        self.sparse = Path(os.path.join(self.data, args.sparse_path, args.scene, self.model_name))
        self.dense = Path(os.path.join(self.data, args.dense_path, args.scene, self.model_name))
        self.metrics = Path(os.path.join(self.data, args.metrics_path, args.scene, self.model_name))
        self.results = Path(os.path.join(self.data, args.results_path, args.scene, self.model_name))

        os.makedirs(self.sparse, exist_ok=True)
        os.makedirs(self.dense, exist_ok=True)
        os.makedirs(self.metrics, exist_ok=True)
        os.makedirs(self.results, exist_ok=True)

        # logging.debug("Data paths:")
        # for path, val in vars(self).items():
        #     logging.debug(f"\t{path}: {val}")
