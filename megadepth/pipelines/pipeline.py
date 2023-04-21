"""Abstract pipeline class."""
import argparse
import datetime
import logging
import os
import time
from abc import abstractmethod
from typing import Optional

import pycolmap
from hloc.reconstruction import create_empty_db, get_image_ids, import_images

from megadepth.utils.constants import ModelType
from megadepth.utils.setup import DataPaths, get_configs
from megadepth.visualization.view_projections import align_models


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

        if overwrite:
            self.sparse_model.write_binary(str(self.paths.sparse))

        return self.sparse_model

    def preprocess(self) -> None:
        """Remove corrupted and other problematic images as a preprocessing step."""
        self.log_step("Preprocessing images...")
        start = time.time()

        # create dummy database and try to import all images
        database = self.paths.data / "tmp_database.db"
        create_empty_db(database)
        import_images(self.paths.images, database, pycolmap.CameraMode.AUTO)
        image_ids = get_image_ids(database)

        # delete image files that were not successfully imported
        image_fns = [fn for fn in os.listdir(self.paths.images) if fn not in image_ids]
        for fn in image_fns:
            path = self.paths.images / fn
            logging.info(f"Deleting invalid image at {path}")
            path.unlink()

        # delete dummy database
        database.unlink()

        end = time.time()
        logging.info(f"Time to preprocess images: {datetime.timedelta(seconds=end - start)}")

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
        start = time.time()

        os.makedirs(self.paths.dense, exist_ok=True)

        # TODO: decide if this can be done in the abstract class

        logging.info("Running undistort_images...")
        pycolmap.undistort_images(
            output_path=self.paths.dense,
            input_path=self.paths.refined_sparse,
            image_path=self.paths.images,
            verbose=self.args.verbose,
        )

        logging.info("Running patch_match_stereo...")
        pycolmap.patch_match_stereo(
            workspace_path=self.paths.dense,
            verbose=self.args.verbose,
        )

        logging.info("Running stereo_fusion...")
        pycolmap.stereo_fusion(
            output_path=self.paths.dense / "dense.ply",
            workspace_path=self.paths.dense,
            verbose=self.args.verbose,
        )

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
        self.preprocess()
        self.get_pairs()
        self.extract_features()
        self.match_features()
        self.sfm()
        self.refinement()
        self.mvs()
        self.cleanup()
