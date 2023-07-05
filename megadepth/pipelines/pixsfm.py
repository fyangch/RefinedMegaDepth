"""Pipeline using PixSfM."""

import logging
import os

import pycolmap
from omegaconf import DictConfig
from pixsfm.refine_hloc import PixSfM

from megadepth.pipelines.hloc import HlocPipeline
from megadepth.utils.constants import ModelType


class PixSfMPipeline(HlocPipeline):
    """Pipeline using PixSfM."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize pipeline.

        Args:
            config (DictConfig): Config with values from the yaml file and CLI.
        """
        super().__init__(config)

    def sfm(self) -> None:
        """Run Structure from Motion."""
        self.log_step("Running Structure from Motion...")

        os.makedirs(self.paths.sparse, exist_ok=True)

        if self.model_exists(ModelType.SPARSE) and not self.config.overwrite:
            logging.info("Sparse model already exists. Skipping.")
            return

        refiner = PixSfM(conf=self.configs["refinement"])
        model, outputs = refiner.run(
            output_dir=self.paths.sparse,
            image_dir=self.paths.images,
            pairs_path=self.paths.matches_retrieval,
            features_path=self.paths.features,
            matches_path=self.paths.matches,
            cache_path=self.paths.cache,
            verbose=self.config.logging.verbose,
        )

        self.sparse_model = model

        logging.debug(f"Outputs:\n{outputs}")

        logging.debug("Aligning reconstruction with baseline...")
        self.align_with_baseline()

    def refinement(self) -> None:
        """Skip refinement as it is done in SFM."""
        if not self.model_exists(ModelType.SPARSE):
            raise ValueError("Sparse model does not exist. Cannot continue.")

        os.makedirs(self.paths.refined_sparse, exist_ok=True)
        self.refined_model: pycolmap.Reconstruction = self.sparse_model
        self.refined_model.write(str(self.paths.refined_sparse))
        logging.info(f"Refined model written to {self.paths.refined_sparse}")
