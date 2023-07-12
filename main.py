"""This project is a re-implementation of the MegaDepth pipeline."""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from megadepth.pipelines.sparse_pipeline import SparsePipeline
from megadepth.utils.configure import check_config, get_model_name, set_up_paths


@hydra.main(version_base=None, config_path="megadepth/configs", config_name="default")
def main(config: DictConfig):
    """Run the mega depth pipeline."""
    print(OmegaConf.to_yaml(config))

    if not config.model_name:
        config.model_name = get_model_name(config)
    config = set_up_paths(config)

    # create and run pipeline
    pipeline = SparsePipeline(config)
    check_config(config, pipeline)

    logging.info(f"Pipeline: {type(pipeline).__name__}")
    pipeline.run()


if __name__ == "__main__":
    main()
