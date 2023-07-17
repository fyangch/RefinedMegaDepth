"""Abstract pipeline class."""
import datetime
import json
import logging
import os
import time
from abc import abstractmethod
from typing import Optional

import pycolmap
from omegaconf import DictConfig

from megadepth.metrics.metadata import collect_metrics
from megadepth.postprocessing.cleanup import refine_depth_maps
from megadepth.utils.preprocessing import remove_problematic_images, rotate_images
from megadepth.utils.utils import time_function


class Pipeline:
    """Abstract pipeline class."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize the pipeline.

        Args:
            config (DictConfig): Config with values from the yaml file and CLI.
        """
        self.config = config
        self.paths = config.paths
        self.overwrite = config.overwrite
        self.n_images = len(os.listdir(self.paths.images))
        self.sparse_model: Optional[pycolmap.Reconstruction] = None
        self.dense_model: Optional[pycolmap.Reconstruction] = None

    def log_step(self, title: str) -> None:
        """Log a title.

        Args:
            title: The title to log.
        """
        logging.info(f"{'=' * 80}")
        logging.info(title)
        logging.info(f"{'=' * 80}")

    def preprocess(self) -> None:
        """Preprocess all images."""
        self.log_step("Preprocessing images...")

        if self.config.preprocessing.remove_problematic_images:
            remove_problematic_images(self.paths)

        if self.config.preprocessing.rotate_images:
            rotate_images(self.paths)

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

    def mvs(self) -> None:
        """Run Multi-View Stereo."""
        self.log_step("Running Multi-View Stereo...")

        os.makedirs(self.paths.dense, exist_ok=True)

        logging.info("Running undistort_images...")
        pycolmap.undistort_images(
            output_path=self.paths.dense,
            input_path=self.paths.sparse,
            image_path=self.paths.images,
            verbose=self.config.logging.verbose,
        )

        logging.info("Running patch_match_stereo...")
        pycolmap.patch_match_stereo(
            workspace_path=self.paths.dense,
            verbose=self.config.logging.verbose,
        )

        logging.info("Running stereo_fusion...")
        pycolmap.stereo_fusion(
            output_path=self.paths.dense / "dense.ply",
            workspace_path=self.paths.dense,
            verbose=self.config.logging.verbose,
        )

    def postprocess(self) -> None:
        """Postprocess the raw depth maps."""
        self.log_step("Postprocessing depth maps...")

        os.makedirs(self.paths.results, exist_ok=True)

        refine_depth_maps(
            image_dir=self.paths.dense / "images",
            depth_map_dir=self.paths.dense / "stereo" / "depth_maps",
            output_dir=self.paths.results,
        )

    def compute_metrics(self) -> None:
        """Compute various metrics for the results."""
        self.log_step("Computing metrics...")
        self.paths.metrics.mkdir(exist_ok=True, parents=True)

        collect_metrics(self.paths, self.config, model_type="sparse")
        collect_metrics(self.paths, self.config, model_type="dense")

    def run(self) -> None:
        """Run the pipeline."""
        start_time = time.time()
        # run each step of the pipeline
        timings = {step: time_function(getattr(self, step))() for step in self.config.steps}
        total_time = time.time() - start_time

        # log timings
        logging.info("Timings:")
        for k, v in timings.items():
            logging.info(f"  {k}: {datetime.timedelta(seconds=v)} ({v / total_time:.2%})")
        logging.info(f"  Total: {datetime.timedelta(seconds=total_time)}")

        # save timings to file
        timings_path = self.paths.metrics / "timings.json"
        with open(timings_path, "w") as f:
            json.dump(timings, f, indent=4)
