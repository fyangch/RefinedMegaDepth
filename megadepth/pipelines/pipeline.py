"""Abstract pipeline class."""
import datetime
import json
import logging
import os
import time
from abc import abstractmethod
from typing import Optional

import pycolmap
from hloc.reconstruction import create_empty_db, get_image_ids, import_images
from omegaconf import DictConfig

from megadepth.postprocessing.cleanup import refine_depth_maps
from megadepth.utils.constants import ModelType
from megadepth.utils.setup import DataPaths, get_configs
from megadepth.utils.utils import time_function
from megadepth.visualization.view_projections import align_models


class Pipeline:
    """Abstract pipeline class."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize the pipeline.

        Args:
            config (DictConfig): Config with values from the yaml file and CLI.
        """
        self.config = config
        self.configs = get_configs(config)
        self.paths = DataPaths(config)
        self.n_images = len(os.listdir(self.paths.images))
        self.sparse_model: Optional[pycolmap.Reconstruction] = None
        self.refined_model: Optional[pycolmap.Reconstruction] = None
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
        elif model == ModelType.REFINED:
            try:
                self.refined_model = pycolmap.Reconstruction(self.paths.refined_sparse)
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

    def align_with_baseline(self, overwrite: bool = True) -> pycolmap.Reconstruction:
        """Align the sparse model to the baseline reconstruction.

        Args:
            overwrite (bool, optional): Whether to overwrite the existing model. Defaults to True.

        Returns:
            pycolmap.Reconstruction: The aligned reconstruction.
        """
        logging.info("Trying to align model to baseline reconstruction...")

        if not self.model_exists(ModelType.SPARSE):
            raise ValueError("No sparse model found.")

        try:
            baseline_model = pycolmap.Reconstruction(self.paths.sparse_baseline)
            logging.info(f"Loaded baseline model from {self.paths.sparse_baseline}")
        except Exception:
            logging.warning("No baseline model found. Skipping alignment.")
            return self.sparse_model

        # align sparse model to baseline model
        self.sparse_model = align_models(
            reconstruction_anchor=baseline_model, reconstruction_align=self.sparse_model
        )

        self.sparse_model.write_binary(str(self.paths.sparse))
        logging.info("Aligned sparse model to baseline model.")

        return self.sparse_model

    def preprocess(self) -> None:
        """Remove corrupted and other problematic images as a preprocessing step."""
        self.log_step("Preprocessing images...")

        # create dummy database and try to import all images
        database = self.paths.data / "tmp_database.db"
        create_empty_db(database)
        import_images(self.paths.images, database, pycolmap.CameraMode.AUTO)
        image_ids = get_image_ids(database)

        logging.debug(f"Successfully imported {len(image_ids)} valid images.")

        # delete image files that were not successfully imported
        image_fns = [fn for fn in os.listdir(self.paths.images) if fn not in image_ids]
        for fn in image_fns:
            path = self.paths.images / fn
            logging.info(f"Deleting invalid image at {path}")
            path.unlink()

        # delete dummy database
        database.unlink()

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

    @abstractmethod
    def refinement(self) -> None:
        """Refine the reconstruction using PixSFM."""
        pass

    def mvs(self) -> None:
        """Run Multi-View Stereo."""
        self.log_step("Running Multi-View Stereo...")

        os.makedirs(self.paths.dense, exist_ok=True)

        # TODO: decide if this can be done in the abstract class

        logging.info("Running undistort_images...")
        pycolmap.undistort_images(
            output_path=self.paths.dense,
            input_path=self.paths.refined_sparse,
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

    def run(self) -> None:
        """Run the pipeline."""
        start_time = time.time()
        timings = {
            "preprocessing": time_function(self.preprocess)(),
            "pairs-extraction": time_function(self.get_pairs)(),
            "feature-extraction": time_function(self.extract_features)(),
            "feature-matching": time_function(self.match_features)(),
            "sfm": time_function(self.sfm)(),
            "refinement": time_function(self.refinement)(),
            "mvs": time_function(self.mvs)(),
            "postprocessing": time_function(self.postprocess)(),
        }
        total_time = time.time() - start_time

        logging.info("Timings:")
        for k, v in timings.items():
            logging.info(f"  {k}: {datetime.timedelta(seconds=v)} ({v / total_time:.2%})")
        logging.info(f"  Total: {datetime.timedelta(seconds=total_time)}")

        timings_path = self.paths.metrics / "timings.json"
        with open(timings_path, "w") as f:
            json.dump(timings, f, indent=4)
