"""Abstract pipeline class."""
import argparse
import datetime
import logging
import os
import time
from abc import abstractmethod

import pycolmap

from megadepth.utils.constants import ModelType
from megadepth.utils.utils import DataPaths, get_configs


class Pipeline:
    """Abstract pipeline class."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the pipeline.

        Args:
            args: Arguments from the command line.
        """
        self.args = args
        self.configs = get_configs(args)
        self.paths = DataPaths(args)
        self.n_images = len(os.listdir(self.paths.images))
        self.sparse_model = None
        self.dense_model = None

    def log_step(self, title: str) -> None:
        """Log a title.

        Args:
            title: The title to log.
        """
        logging.info(f"{'=' * 80}")
        logging.info(title)
        logging.info(f"{'=' * 80}")

    def model_exists(self, model: ModelType) -> bool:
        """Check if the model exists.

        Args:
            model: The model to check.

        Returns:
            True if the model exists, False otherwise.
        """
        if model == ModelType.SPARSE:
            try:
                self.sparse_model = pycolmap.Reconstruction(self.paths.sparse)
                return True
            except Exception:
                return False
        elif model == ModelType.DENSE:
            try:
                self.dense_model = pycolmap.Reconstruction(self.paths.dense)
                return True
            except Exception:
                return False
        else:
            raise ValueError(f"Invalid model type: {model}")

    @abstractmethod
    def get_pairs(self) -> None:
        """Get pairs of images to match."""
        pass

    @abstractmethod
    def extract_features(self) -> None:
        """Extract features from the images."""
        pass

    @abstractmethod
    def match_features(self) -> None:
        """Match features between images."""
        pass

    @abstractmethod
    def sfm(self) -> None:
        """Run Structure from Motion."""
        pass

    def refinement(self) -> None:
        """Refine the reconstruction using PixSFM."""
        self.log_step("Refining the reconstruction...")
        start = time.time()

        os.makedirs(self.paths.sparse, exist_ok=True)

        # TODO: decide if this can be done in the abstract class

        # TODO: implement pixSFM

        end = time.time()
        logging.info(
            f"Time to refine the reconstruction: {datetime.timedelta(seconds=end - start)}"
        )

    def mvs(self) -> None:
        """Run Multi-View Stereo."""
        self.log_step("Running Multi-View Stereo...")
        start = time.time()

        os.makedirs(self.paths.dense, exist_ok=True)

        # TODO: decide if this can be done in the abstract class

        # TODO: implement MVS
        # pycolmap.undistort_images(mvs_path, output_path, image_dir)
        # pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
        # pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)

        end = time.time()
        logging.info(f"Time to run MVS: {datetime.timedelta(seconds=end - start)}")

    def cleanup(self) -> None:
        """Clean up the pipeline."""
        self.log_step("Cleaning up...")
        start = time.time()

        # TODO: decide if this can be done in the abstract class

        # TODO: implement cleanup

        os.makedirs(self.paths.results, exist_ok=True)

        end = time.time()
        logging.info(f"Time to clean up: {datetime.timedelta(seconds=end - start)}")

    def run(self) -> None:
        """Run the pipeline."""
        self.get_pairs()
        self.extract_features()
        self.match_features()
        self.sfm()
        self.refinement()
        self.mvs()
        self.cleanup()
