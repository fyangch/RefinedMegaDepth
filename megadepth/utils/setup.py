"""Setup the everything for the pipeline."""

import datetime
import logging
import os
from pathlib import Path

import pixsfm
from hloc import extract_features, match_dense, match_features
from omegaconf import DictConfig, OmegaConf

from megadepth.utils.constants import Matcher, Retrieval


def set_up_logger(config: DictConfig) -> None:
    """Set up the logger.

    Args:
        config (DictConfig): Config with values from the yaml file and CLI.
    """
    if config.logging.log_dir:
        if not os.path.exists(config.logging.log_dir):
            os.makedirs(config.logging.log_dir)
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(config.logging.log_dir, f"{date}_log.txt")
    else:  # log to stdout
        log_file = None

    logging.basicConfig(
        format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=config.logging.log_level,
        filename=log_file,
    )

    logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.ERROR)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.ERROR)


def get_configs(config: DictConfig) -> dict:
    """Return a dictionary of configuration parameters.

    Args:
        config (DictConfig): Config with values from the yaml file and CLI.

    Returns:
        dict: A dictionary of configuration parameters.
    """
    # retrieval config
    retrieval_conf = (
        extract_features.confs[config.retrieval.name]
        if config.retrieval.name
        not in [Retrieval.POSES.value, Retrieval.COVISIBILITY.value, Retrieval.EXHAUSTIVE.value]
        else None
    )

    # feature config
    feature_config = extract_features.confs[config.features]
    # if config.features == Features.SIFT.value:
    # feature_config["preprocessing"]["resize_max"] = 3200

    # matcher config
    if config.matcher == Matcher.LOFTR.value:
        matcher_conf = match_dense.confs[config.matcher]
        matcher_conf["preprocessing"]["resize_max"] = 840
    else:
        matcher_conf = match_features.confs[config.matcher]

    # refinement config

    # set cache path
    if config.refinement.low_memory:
        refinement_conf = OmegaConf.load(pixsfm.configs.parse_config_path("low_memory"))
    else:
        refinement_conf = OmegaConf.load(pixsfm.configs.parse_config_path("default"))

    return {
        "retrieval": retrieval_conf,
        "feature": feature_config,
        "matcher": matcher_conf,
        "refinement": refinement_conf,
    }


def get_model_name(config: DictConfig) -> str:
    """Define the model name based on the config values.

    Args:
        config (DictConfig): Config with values from the yaml file and CLI.

    Returns:
        str: The model name.
    """
    refinement_suffix = f"-{config.refinement.steps}" if config.refinement.steps else ""

    if config.colmap:
        return "colmap"
    elif config.matcher == Matcher.LOFTR.value:
        return (
            f"{config.matcher}"
            + f"-{config.retrieval.name}"
            + f"-{config.retrieval.n_matches}"
            + refinement_suffix
        )
    elif config.retrieval.name == Retrieval.EXHAUSTIVE.value:
        return (
            f"{config.features}"
            + f"-{config.matcher}"
            + f"-{config.retrieval.name}"
            + refinement_suffix
        )
    else:
        return (
            f"{config.features}"
            + f"-{config.matcher}"
            + f"-{config.retrieval.name}"
            + f"-{config.retrieval.n_matches}"
            + refinement_suffix
        )


def set_up_paths(config: DictConfig) -> DictConfig:
    """Convert types to pathlib.Path and make some path adjustments if necessary.

    Args:
        config (DictConfig): Config with values from the yaml file and CLI.

    Returns:
        DictConfig: Config with final paths.
    """
    # this also makes sure that the results of the variable interpolations are fixed
    # such that we can actually replace some parts of the paths below
    for path in config.paths:
        config.paths[path] = Path(config.paths[path])

    # remove the n_matches part of the filename
    if config.retrieval.name == Retrieval.EXHAUSTIVE.value:
        config.paths.matches_retrieval = (
            config.paths.matches_retrieval.parent / f"{config.retrieval.name}.txt"
        )

    # replace the features name by the model name
    if config.matcher == Matcher.LOFTR.value:
        config.paths.features = config.paths.features.parent / f"{config.model_name}.h5"

    return config
