"""Pipeline using COLMAP."""
import logging
import os
import shutil

import pycolmap
from omegaconf import DictConfig

from megadepth.pipelines.pipeline import Pipeline
from megadepth.utils.constants import ModelType


class ColmapPipeline(Pipeline):
    """Pipeline for COLMAP."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize the pipeline."""
        super().__init__(config)

    def get_pairs(self) -> None:
        """Get pairs of images to match."""
        self.log_step("Getting pairs...")
        logging.info("No retrieval, using colmap")

    def extract_features(self) -> None:
        """Extract features from images."""
        self.log_step("Extracting features...")

        os.makedirs(self.paths.db.parent, exist_ok=True)
        if os.path.exists(self.paths.db):
            if self.config.overwrite:
                logging.info("Database already exists, deleting it...")
                # delete file
                fname = str(self.paths.db)
                os.remove(fname)
            else:
                logging.info("Database already exists, skipping...")
                return

        pycolmap.extract_features(
            database_path=self.paths.db,
            image_path=self.paths.images,
            verbose=self.config.logging.verbose,
        )

    def match_features(self) -> None:
        """Match features between images."""
        self.log_step("Matching features...")

        logging.debug("Exhaustive matching features with colmap")
        pycolmap.match_exhaustive(self.paths.db, verbose=self.config.logging.verbose)

    def sfm(self) -> None:
        """Run Structure from Motion."""
        self.log_step("Running Structure from Motion...")

        if self.model_exists(ModelType.SPARSE) and not self.config.overwrite:
            logging.info(f"Reconstruction exists at {self.paths.sparse}. Skipping SFM...")
            return

        logging.debug("Running SFM with colmap")
        pycolmap.incremental_mapping(self.paths.db, self.paths.images, self.paths.sparse)
        # copy latest model to sfm dir
        model_id = sorted(
            [dir for dir in os.listdir(self.paths.sparse) if os.path.isdir(self.paths.sparse / dir)]
        )[-1]
        for filename in ["images.bin", "cameras.bin", "points3D.bin"]:
            shutil.copy(
                str(self.paths.sparse / model_id / filename), str(self.paths.sparse / filename)
            )

        self.sparse_model = pycolmap.Reconstruction(self.paths.sparse)

    def refinement(self) -> None:
        """Run refinement."""
        if not self.model_exists(ModelType.SPARSE):
            raise ValueError("Sparse model does not exist. Cannot continue.")

        os.makedirs(self.paths.refined_sparse, exist_ok=True)
        self.refined_model: pycolmap.Reconstruction = self.sparse_model
        self.refined_model.write(str(self.paths.refined_sparse))
        logging.info(f"Refined model written to {self.paths.refined_sparse}")

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
