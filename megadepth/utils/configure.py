"""Setup the everything for the pipeline."""
import os
from pathlib import Path

from omegaconf import DictConfig

from megadepth.pipelines.pipeline import Pipeline


def get_model_name(config: DictConfig) -> str:
    """Define the model name based on the config values.

    Args:
        config (DictConfig): Config with values from the yaml file and CLI.

    Returns:
        str: The model name.
    """
    return "+".join(sorted(config.ensembles.keys()))


def set_up_paths(config: DictConfig) -> DictConfig:
    """Convert types to pathlib.Path and make some path adjustments if necessary.

    Args:
        config (DictConfig): Config with values from the yaml file and CLI.

    Returns:
        DictConfig: Config with final paths.
    """
    # set up cache dir for euler
    if config.refinement.dense_features.use_cache:
        cache_dir = os.environ.get("TMPDIR")
        if cache_dir is None:
            raise ValueError(
                "TMPDIR environment variable not set. "
                + "Set it using export TMPDIR=/path/to/tmpdir"
            )

        config.paths.cache = cache_dir

    # convert to pathlib.Path
    for path in config.paths:
        config.paths[path] = Path(config.paths[path])

    return config


def check_config(config: DictConfig, pipeline: Pipeline) -> None:
    """Check if the config values are valid.

    Args:
        config (DictConfig): Config with values from the yaml file and CLI.
        pipeline (Pipeline): The pipeline to check the config for.
    """
    config.scene  # throws exception if it was not passed as a command line argument

    # check if the pipeline steps in the config are valid methods
    # TODO: might not hold for all pipelines
    methods = [name for name in dir(pipeline) if callable(getattr(pipeline, name))]
    for step in config.steps:
        if step not in methods:
            raise ValueError(f"Invalid pipeline step: {step}")
