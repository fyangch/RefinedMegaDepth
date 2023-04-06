"""Pipeline using LoFTR."""
import argparse
import datetime
import logging
import os
import time

from hloc import match_dense

from megadepth.pipelines.hloc import HlocPipeline


class LoftrPipeline(HlocPipeline):
    """HLoc-based pipeline for LoFTR."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the pipeline."""
        super().__init__(args)

    def extract_features(self) -> None:
        """Extract features from images."""
        # features are extracted during the dense matching step
        return

    def _preprocess_images(self) -> None:
        """Preprocess images for dense matching."""
        # TODO
        raise NotImplementedError()

    def match_features(self) -> None:
        """Match features between images."""
        self.log_step("Matching and extracting features...")
        start = time.time()

        logging.debug("Matching and extracting features with LoFTR")
        logging.debug(f"Matcher config: {self.configs['matcher']}")
        logging.debug(f"Loading pairs from {self.paths.matches_retrieval}")
        logging.debug(f"Storing matches to {self.paths.matches}")
        logging.debug(f"Storing features to {self.paths.features}")

        os.makedirs(self.paths.matches.parent, exist_ok=True)
        os.makedirs(self.paths.features.parent, exist_ok=True)

        # preprocess images if not done yet
        # TODO: check if directory with preprocessed images already exists
        self._preprocess_images()

        match_dense.main(
            conf=self.configs["matcher"],
            image_dir=self.paths.images,  # TODO: replace with path to preprocessed directory
            pairs=self.paths.matches_retrieval,
            features=self.paths.features,
            matches=self.paths.matches,
        )

        end = time.time()
        logging.info(
            f"Time to match and extract features: {datetime.timedelta(seconds=end - start)}"
        )
