"""Utility functions."""

import argparse
import datetime
import logging
import os


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
