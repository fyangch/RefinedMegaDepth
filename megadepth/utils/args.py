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
        help="Name of the image directory inside the scene directory.",
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
        "--refinements",
        type=str,
        choices=["", "KA", "BA", "KA+BA"],
        default="BA",
        help="The refinement steps to be performed.",
    )

    parser.add_argument(
        "--low_memory",
        action="store_true",
        help="Use low memory mode in PixSfM. Will write data to path specified by $TMPDIR.",
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
