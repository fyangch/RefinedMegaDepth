"""Pipeline using LoFTR."""
import logging
import os

from hloc import match_dense
from omegaconf import DictConfig

from megadepth.pipelines.pixsfm import PixSfMPipeline


class LoftrPipeline(PixSfMPipeline):
    """HLoc-based pipeline for LoFTR."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize the pipeline."""
        super().__init__(config)

    def extract_features(self) -> None:
        """Extract features from images."""
        # features are extracted during the dense matching step
        return

    def match_features(self) -> None:
        """Match features between images."""
        self.log_step("Matching and extracting features...")

        logging.debug("Matching and extracting features with LoFTR")
        logging.debug(f"Matcher config: {self.configs['matcher']}")
        logging.debug(f"Loading pairs from {self.paths.matches_retrieval}")
        logging.debug(f"Storing matches to {self.paths.matches}")
        logging.debug(f"Storing features to {self.paths.features}")

        os.makedirs(self.paths.matches.parent, exist_ok=True)
        os.makedirs(self.paths.features.parent, exist_ok=True)

        match_dense.main(
            conf=self.configs["matcher"],
            image_dir=self.paths.images,
            pairs=self.paths.matches_retrieval,
            features=self.paths.features,
            matches=self.paths.matches,
        )
