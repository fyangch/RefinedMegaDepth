"""Implementation of the MegaDepth pipeline using colmap, hloc and pixsfm."""
import argparse
import datetime
import logging
import os
import shutil
import time

import pycolmap
from hloc import (
    extract_features,
    match_features,
    pairs_from_covisibility,
    pairs_from_exhaustive,
    pairs_from_poses,
    pairs_from_retrieval,
    reconstruction,
)

from megadepth.utils.constants import Retrieval
from megadepth.utils.utils import DataPaths, get_configs


def log_step(title: str) -> None:
    """Log a title.

    Args:
        title: The title to log.
    """
    logging.info(f"{'=' * 80}")
    logging.info(title)
    logging.info(f"{'=' * 80}")


class Pipeline:
    """The MegaDepth pipeline."""

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

    def _extract_pairs_from_retrieval(self):
        """Extract pairs from retrieval."""
        logging.debug(f"Retrieval config: {self.configs['retrieval']}")

        # extract global features
        extract_features.main(
            conf=self.configs["retrieval"],
            image_dir=self.paths.images,
            feature_path=self.paths.features_retrieval,
        )

        logging.debug(f"Using {self.args.retrieval}")
        logging.debug(f"Storing pairs to {self.paths.matches_retrieval}")

        pairs_from_retrieval.main(
            descriptors=self.paths.features_retrieval,
            num_matched=self.args.n_retrieval_matches,
            output=self.paths.matches_retrieval,
        )

    def get_pairs(self) -> None:
        """Get pairs of images to be matched."""
        log_step("Getting pairs...")
        logging.debug(f"Storing retrieval features at {self.paths.features_retrieval}")
        start = time.time()

        if self.args.colmap:
            logging.debug("No retrieval, using colmap")
            return

        # match global features to get pairs
        os.makedirs(self.paths.matches_retrieval.parent, exist_ok=True)

        if self.args.retrieval == Retrieval.EXHAUSTIVE.value:
            logging.debug("Using exhaustive retrieval")
            logging.debug(f"Storing pairs to {self.paths.matches_retrieval}")

            pairs_from_exhaustive.main(
                features=self.paths.features_retrieval, output=self.paths.matches_retrieval
            )

        elif self.args.retrieval == Retrieval.POSES.value:
            logging.debug(f"Using {self.args.retrieval}")

            pairs_from_poses.main(
                model=self.paths.baseline_model,
                output=self.paths.matches_retrieval,
                num_matched=self.args.n_retrieval_matches,
            )

        elif self.args.retrieval == Retrieval.COVISIBILITY.value:
            logging.debug(f"Using {self.args.retrieval}")

            pairs_from_covisibility.main(
                model=self.paths.baseline_model,
                output=self.paths.matches_retrieval,
                num_matched=self.args.n_retrieval_matches,
            )

        elif self.args.retrieval in [Retrieval.NETVLAD.value, Retrieval.COSPLACE.value]:
            self._extract_pairs_from_retrieval()

        else:
            raise NotImplementedError(f"Retrieval method {self.args.retrieval} not implemented")

        end = time.time()
        logging.info(f"Time to get pairs: {datetime.timedelta(seconds=end - start)}")

    def extract_features(self) -> None:
        """Extract features from the images."""
        log_step("Extracting features...")
        start = time.time()

        if self.args.colmap:
            logging.debug("Extracting features with colmap")

            os.makedirs(self.paths.db.parent, exist_ok=True)
            if os.path.exists(self.paths.db):
                logging.warning("Database already exists, deleting it...")
                # delete file
                fname = str(self.paths.db)
                os.remove(fname)

            pycolmap.extract_features(
                database_path=self.paths.db, image_path=self.paths.images, verbose=self.args.verbose
            )

            end = time.time()
            logging.info(f"Time to extract features: {datetime.timedelta(seconds=end - start)}")
            return

        logging.debug("Extracting features with hloc")
        logging.debug(f"Feature config: {self.configs['feature']}")
        logging.debug(f"Storing features to {self.paths.features}")

        extract_features.main(
            conf=self.configs["feature"],
            image_dir=self.paths.images,
            feature_path=self.paths.features,
        )

        end = time.time()
        logging.info(f"Time to extract features: {datetime.timedelta(seconds=end - start)}")

    def match_features(self) -> None:
        """Match features between images."""
        log_step("Matching features...")
        start = time.time()

        if self.args.colmap:
            logging.debug("Exhaustive matching features with colmap")
            pycolmap.match_exhaustive(self.paths.db, verbose=self.args.verbose)

            end = time.time()
            logging.info(f"Time to match features: {datetime.timedelta(seconds=end - start)}")
            return

        logging.debug("Matching features with hloc")
        logging.debug(f"Matcher config: {self.configs['matcher']}")
        logging.debug(f"Loading pairs form {self.paths.matches_retrieval}")
        logging.debug(f"Loading features from {self.paths.features}")
        logging.debug(f"Storing matches to {self.paths.matches}")

        match_features.main(
            conf=self.configs["matcher"],
            pairs=self.paths.matches_retrieval,
            features=self.paths.features,
            matches=self.paths.matches,
        )

        end = time.time()
        logging.info(f"Time to match features: {datetime.timedelta(seconds=end - start)}")

    def sfm(self) -> None:
        """Run Structure from Motion."""
        log_step("Running Structure from Motion...")
        start = time.time()

        os.makedirs(self.paths.sparse, exist_ok=True)

        if not self.args.overwrite:
            try:
                self.sparse_model = pycolmap.Reconstruction(self.paths.sparse)
                logging.info(f"Loaded reconstruction from {self.paths.sparse}. Skipping SFM...")
                return
            except Exception:
                logging.warning("No reconstruction found. Running SFM...")

        if self.args.colmap:
            logging.debug("Running SFM with colmap")
            pycolmap.incremental_mapping(self.paths.db, self.paths.images, self.paths.sparse)
            self.sparse_model = pycolmap.Reconstruction(self.paths.sparse)
            # copy latest model to sfm dir
            model_id = sorted(
                [
                    dir
                    for dir in os.listdir(self.paths.sparse)
                    if os.path.isdir(self.paths.sparse / dir)
                ]
            )[-1]
            for filename in ["images.bin", "cameras.bin", "points3D.bin"]:
                shutil.copy(
                    str(self.paths.sparse / model_id / filename), str(self.paths.sparse / filename)
                )

            end = time.time()
            logging.info(f"Time to run SFM: {datetime.timedelta(seconds=end - start)}")
            return

        logging.debug("Running SFM with hloc")
        mapper_options = pycolmap.IncrementalMapperOptions().todict()
        self.sparse_model = reconstruction.main(
            sfm_dir=self.paths.sparse,
            image_dir=self.paths.images,
            pairs=self.paths.matches_retrieval,
            features=self.paths.features,
            matches=self.paths.matches,
            verbose=self.args.verbose,
            mapper_options=mapper_options,
        )

        # delete empty "models" folder
        shutil.rmtree(str(self.paths.sparse / "models"))

        end = time.time()
        logging.info(f"Time to run SFM: {datetime.timedelta(seconds=end - start)}")

    def refinement(self) -> None:
        """Refine the reconstruction using PixSFM."""
        log_step("Refining the reconstruction...")
        start = time.time()

        os.makedirs(self.paths.sparse, exist_ok=True)

        # TODO: implement refinement

        end = time.time()
        logging.info(
            f"Time to refine the reconstruction: {datetime.timedelta(seconds=end - start)}"
        )

    def mvs(self) -> None:
        """Run Multi-View Stereo."""
        log_step("Running Multi-View Stereo...")
        start = time.time()

        os.makedirs(self.paths.dense, exist_ok=True)

        # TODO: implement MVS
        # pycolmap.undistort_images(mvs_path, output_path, image_dir)
        # pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
        # pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)

        end = time.time()
        logging.info(f"Time to run MVS: {datetime.timedelta(seconds=end - start)}")

    def cleanup(self) -> None:
        """Clean up the pipeline."""
        # TODO: implement cleanup
        log_step("Cleaning up...")
        start = time.time()

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


class ColmapPipeline(Pipeline):
    """Pipeline for COLMAP."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the pipeline."""
        super().__init__(args)

    def get_pairs(self) -> None:
        """Get pairs of images to match."""
        log_step("Getting pairs...")
        logging.info("No retrieval, using colmap")

    def extract_features(self) -> None:
        """Extract features from images."""
        log_step("Extracting features...")
        start = time.time()
        logging.debug("Extracting features with colmap")

        os.makedirs(self.paths.db.parent, exist_ok=True)
        if os.path.exists(self.paths.db):
            logging.warning("Database already exists, deleting it...")
            # delete file
            fname = str(self.paths.db)
            os.remove(fname)

        pycolmap.extract_features(
            database_path=self.paths.db, image_path=self.paths.images, verbose=self.args.verbose
        )

        end = time.time()
        logging.info(f"Time to extract features: {datetime.timedelta(seconds=end - start)}")

    def match_features(self) -> None:
        """Match features between images."""
        log_step("Matching features...")
        start = time.time()

        logging.debug("Matching features with colmap")
        pycolmap.match_features(self.paths.db, verbose=self.args.verbose)

        end = time.time()
        logging.info(f"Time to match features: {datetime.timedelta(seconds=end - start)}")

    def sfm(self) -> None:
        """Run Structure from Motion."""
        log_step("Running Structure from Motion...")
        start = time.time()

        os.makedirs(self.paths.sparse, exist_ok=True)

        if not self.args.overwrite:
            try:
                self.sparse_model = pycolmap.Reconstruction(self.paths.sparse)
                logging.info(f"Loaded reconstruction from {self.paths.sparse}. Skipping SFM...")
                return
            except Exception:
                logging.warning("No reconstruction found. Running SFM...")

        logging.debug("Running SFM with colmap")
        pycolmap.incremental_mapping(self.paths.db, self.paths.images, self.paths.sparse)
        self.sparse_model = pycolmap.Reconstruction(self.paths.sparse)
        # copy latest model to sfm dir
        model_id = sorted(os.listdir(self.paths.sparse))[-1]
        for filename in ["images.bin", "cameras.bin", "points3D.bin"]:
            shutil.copy(
                str(self.paths.sparse / model_id / filename), str(self.paths.sparse / filename)
            )

        end = time.time()
        logging.info(f"Time to run SFM: {datetime.timedelta(seconds=end - start)}")
