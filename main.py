"""This project is a re-implementation of the MegaDepth pipeline."""

import logging

import hydra
from omegaconf import DictConfig

from megadepth.metrics.metadata import collect_metrics
from megadepth.pipelines.pipeline import Pipeline
from megadepth.utils.config import check_config
from megadepth.utils.constants import Matcher, ModelType
from megadepth.utils.setup import DataPaths, setup_logger


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    """Run the mega depth pipeline."""
    setup_logger(config)
    check_config(config)

    paths = DataPaths(config)

    # create pipeline
    pipeline = get_pipeline(config)
    logging.info(f"Pipeline: {type(pipeline).__name__}")

    if config.evaluate:
        pipeline.align_with_baseline()
        collect_metrics(paths, config, ModelType.SPARSE)
        collect_metrics(paths, config, ModelType.REFINED)
        # collect_metrics(paths, config, ModelType.DENSE)
        return

    # run pipeline
    pipeline.run()

    collect_metrics(paths, config, model_type=ModelType.SPARSE)
    collect_metrics(paths, config, model_type=ModelType.REFINED)
    # collect_metrics(paths, config, model_type=ModelType.DENSE)


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
