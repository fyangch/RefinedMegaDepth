"""Setup the everything for the pipeline."""

import argparse
import datetime
import logging
import os
from pathlib import Path

from hloc import extract_features, match_dense, match_features
from pixsfm.base import interpolation_default_conf
from pixsfm.bundle_adjustment import BundleAdjuster
from pixsfm.features.extractor import FeatureExtractor
from pixsfm.keypoint_adjustment import KeypointAdjuster

from megadepth.utils.args import setup_args
from megadepth.utils.constants import Features, Matcher, Retrieval


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

    logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.ERROR)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.ERROR)


def get_configs(args: argparse.Namespace) -> dict:
    """Return a dictionary of configuration parameters.

    Args:
        args (argparse.Namespace): The parsed command line arguments.

    Returns:
        dict: A dictionary of configuration parameters.
    """
    # retrieval config
    retrieval_conf = (
        extract_features.confs[args.retrieval]
        if args.retrieval
        not in [Retrieval.POSES.value, Retrieval.COVISIBILITY.value, Retrieval.EXHAUSTIVE.value]
        else None
    )

    # feature config
    feature_config = extract_features.confs[args.features]
    if args.features == Features.SIFT.value:
        feature_config["preprocessing"]["resize_max"] = 3200

    # matcher config
    if args.matcher == Matcher.LOFTR.value:
        matcher_conf = match_dense.confs[args.matcher]
        matcher_conf["preprocessing"]["resize_max"] = 840
    else:
        matcher_conf = match_features.confs[args.matcher]

    # refinement config
    refinement_conf = {
        "dense_features": {**FeatureExtractor.default_conf},
        "interpolation": interpolation_default_conf,
        "KA": {
            **KeypointAdjuster.default_conf,
            "interpolation": "${..interpolation}",
        },
        "BA": {
            **BundleAdjuster.default_conf,
            "interpolation": "${..interpolation}",
        },
    }

    return {
        "retrieval": retrieval_conf,
        "feature": feature_config,
        "matcher": matcher_conf,
        "refinement": refinement_conf,
    }


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
            if args.retrieval == Retrieval.EXHAUSTIVE.value
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
        if args.matcher == Matcher.LOFTR.value:
            self.features = Path(
                os.path.join(self.data, args.features_dir, f"{self.model_name}.h5")
            )
        else:
            self.features = Path(os.path.join(self.data, args.features_dir, f"{args.features}.h5"))

        # matches
        self.matches = Path(os.path.join(self.data, args.matches_dir, f"{self.model_name}.h5"))

        # models
        self.sparse = Path(os.path.join(self.data, args.sparse_dir, self.model_name))
        self.sparse_baseline = Path(os.path.join(self.data, args.sparse_dir, "baseline"))
        self.refined_sparse = Path(
            os.path.join(self.data, args.sparse_dir, self.model_name, "refined")
        )
        self.db = Path(os.path.join(self.sparse, "database.db"))
        self.dense = Path(os.path.join(self.data, args.dense_dir, self.model_name))
        self.baseline_model = Path(os.path.join(self.data, args.sparse_dir, "baseline"))

        # output
        self.metrics = Path(os.path.join(self.data, args.metrics_dir, self.model_name))
        self.results = Path(os.path.join(self.data, args.results_dir, self.model_name))
        self.visualizations = Path(
            os.path.join(self.data, args.visualizations_dir, self.model_name)
        )

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
        elif args.matcher == Matcher.LOFTR.value:
            return f"{args.matcher}-{args.retrieval}-{args.n_retrieval_matches}-{args.refinements}"
        elif args.retrieval == Retrieval.EXHAUSTIVE.value:
            return f"{args.features}-{args.matcher}-{args.retrieval}-{args.refinements}"
        else:
            return (
                f"{args.features}"
                + f"-{args.matcher}"
                + f"-{args.retrieval}"
                + f"-{args.n_retrieval_matches}"
                + f"-{args.refinements}"
            )
