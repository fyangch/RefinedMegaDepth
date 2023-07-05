"""Functions to check the config values."""

import logging

from omegaconf import DictConfig, OmegaConf

from megadepth.utils.constants import Features, Matcher, Retrieval


def check_config(config: DictConfig) -> None:
    """Check if the config values are valid.

    Args:
        config (DictConfig): Config with values from the yaml file and CLI.
    """
    config.scene  # throws exception if it was not passed as a command line arg

    if config.retrieval.name not in [r.value for r in Retrieval]:
        raise ValueError(f"Invalid retrieval name: {config.retrieval.name}")
    if config.features not in [f.value for f in Features]:
        raise ValueError(f"Invalid feature name: {config.features}")
    if config.matcher not in [m.value for m in Matcher]:
        raise ValueError(f"Invalid matcher name: {config.matcher}")

    logging.debug("Config values:")
    logging.debug(OmegaConf.to_yaml(config))