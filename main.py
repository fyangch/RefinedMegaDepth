"""This project is a re-implementation of the MegaDepth pipeline."""

import logging

import hydra
from omegaconf import DictConfig

from megadepth.pipelines.pipeline import Pipeline
from megadepth.utils.config import check_config
from megadepth.utils.constants import Matcher
from megadepth.utils.setup import get_model_name, set_up_logger, set_up_paths


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    """Run the mega depth pipeline."""
    set_up_logger(config)
    check_config(config)

    if not config.model_name:
        config.model_name = get_model_name(config)
    config = set_up_paths(config)

    # create and run pipeline
    pipeline = get_pipeline(config)
    logging.info(f"Pipeline: {type(pipeline).__name__}")
    pipeline.run()


def get_pipeline(config: DictConfig) -> Pipeline:
    """Get pipeline based on arguments.

    Args:
        config (DictConfig): Config with values from the yaml file and CLI.

    Returns:
        Pipeline: Pipeline to run.
    """
    if config.colmap:
        from megadepth.pipelines.colmap import ColmapPipeline

        return ColmapPipeline(config)
    elif config.matcher == Matcher.LOFTR.value:
        from megadepth.pipelines.loftr import LoftrPipeline

        return LoftrPipeline(config)
    elif "KA" in config.refinement.steps:
        from megadepth.pipelines.pixsfm import PixSfMPipeline

        return PixSfMPipeline(config)
    else:
        from megadepth.pipelines.hloc import HlocPipeline

        return HlocPipeline(config)


if __name__ == "__main__":
    main()
