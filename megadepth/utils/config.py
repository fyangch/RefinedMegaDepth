"""Functions to check the config values."""

from omegaconf import DictConfig

from megadepth.pipelines.pipeline import Pipeline
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
        raise ValueError(f"Invalid features name: {config.features}")
    if config.matcher not in [m.value for m in Matcher]:
        raise ValueError(f"Invalid matcher name: {config.matcher}")

    # check if the pipeline steps in the config are valid methods
    methods = [name for name in dir(Pipeline) if callable(getattr(Pipeline, name))]
    for step in config.steps:
        if step not in methods:
            raise ValueError(f"Invalid pipeline step: {step}")
