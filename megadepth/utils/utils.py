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
        "--stereo_dir",
        type=Path,
        default="stereo",
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

    # PIPELINE RELATED ARGUMENTS
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the pipeline.",
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
        self.data = Path(os.path.join(args.data_path, args.scene))

        # file names
        self.model_name = f"{args.features}-{args.matcher}"

        retrieval_name = f"{args.retrieval}"
        retrieval_name += (
            f"-{args.n_retrieval_matches}.txt" if args.n_retrieval_matches > 0 else ".txt"
        )

        self.images = Path(os.path.join(self.data, args.image_dir))

        # features
        self.features = Path(os.path.join(self.data, args.features_dir, f"{args.features}.h5"))

        self.features_retrieval = Path(
            os.path.join(self.data, args.features_dir, f"{args.retrieval}.h5")
        )
        os.makedirs(self.features.parent, exist_ok=True)

        # matches
        self.matches = Path(os.path.join(self.data, args.matches_dir, f"{self.model_name}.h5"))

        self.matches_retrieval = Path(
            os.path.join(self.data, args.matches_dir, "retrieval", retrieval_name)
        )

        os.makedirs(self.matches.parent, exist_ok=True)
        os.makedirs(self.matches_retrieval.parent, exist_ok=True)

        # models
        self.sparse = Path(os.path.join(self.data, args.sparse_dir, self.model_name))
        self.dense = Path(os.path.join(self.data, args.stereo_dir, self.model_name))
        self.metrics = Path(os.path.join(self.data, args.metrics_dir, self.model_name))
        self.results = Path(os.path.join(self.data, args.results_dir, self.model_name))

        os.makedirs(self.sparse, exist_ok=True)
        os.makedirs(self.dense, exist_ok=True)
        os.makedirs(self.metrics, exist_ok=True)
        os.makedirs(self.results, exist_ok=True)

        logging.debug("Data paths:")
        for path, val in vars(self).items():
            logging.debug(f"\t{path}: {val}")
