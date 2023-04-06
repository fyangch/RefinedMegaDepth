"""Pipeline using LoFTR."""
import argparse
import datetime
import logging
import os
import time

import cv2
import numpy as np
from hloc import match_dense

from megadepth.pipelines.hloc import HlocPipeline


class LoftrPipeline(HlocPipeline):
    """HLoc-based pipeline for LoFTR."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the pipeline."""
        super().__init__(args)
        self.preproc_dir = os.path.join(self.paths.data, "images_loftr")

    def extract_features(self) -> None:
        """Extract features from images."""
        # features are extracted during the dense matching step
        return

    def _already_preprocessed(self) -> bool:
        """Check if all images are already preprocessed."""
        if not os.path.exists(self.preproc_dir):
            return False

        return len(os.listdir(self.paths.images)) == len(os.listdir(self.preproc_dir))

    def _preprocess_images(self, size: int = 840) -> None:
        """Preprocess images for dense matching."""
        logging.debug("Preprocessing images for LoFTR")

        os.makedirs(self.preproc_dir, exist_ok=True)
        for fn in os.listdir(self.paths.images):
            img = cv2.imread(os.path.join(self.paths.images, fn))
            height, width = img.shape[:2]
            aspect_ratio = float(width) / float(height)

            if aspect_ratio < 1:
                new_width = int(size * aspect_ratio)
                new_height = size
            else:
                new_width = size
                new_height = int(size / aspect_ratio)

            new_img = np.zeros((size, size, 3), dtype=np.uint8)
            new_img[:new_height, :new_width, :] = cv2.resize(img, (new_width, new_height))
            cv2.imwrite(os.path.join(self.preproc_dir, fn), new_img)

        logging.debug("Preprocessing done")

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
        if not self._already_preprocessed():
            self._preprocess_images()

        match_dense.main(
            conf=self.configs["matcher"],
            image_dir=self.preproc_dir,
            pairs=self.paths.matches_retrieval,
            features=self.paths.features,
            matches=self.paths.matches,
        )

        end = time.time()
        logging.info(
            f"Time to match and extract features: {datetime.timedelta(seconds=end - start)}"
        )
