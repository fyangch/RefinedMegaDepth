"""Provides the command line arguments for the pipeline."""

import argparse
from pathlib import Path

from megadepth.utils.constants import Features, Matcher, Retrieval


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
    parser.add_argument(
        "--visualizations_dir",
        type=Path,
        default="visualizations",
        help="Path to the visualizations.",
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
        default=Features.SUPERPOINT_MAX.value,
        help="The features used in the model.",
    )

    parser.add_argument(
        "--matcher",
        type=str,
        choices=[m.value for m in Matcher],
        default=Matcher.SUPERGLUE.value,
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
