"""Implementation of the MegaDepth pipeline using colmap, hloc and pixsfm."""
import argparse
import logging
import os
import shutil

import pycolmap
from hloc import (
    extract_features,
    match_features,
    pairs_from_exhaustive,
    pairs_from_retrieval,
    reconstruction,
)

from megadepth.utils.utils import DataPaths, get_configs

N_EXHAUSTIVE = 300


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
        self.model = None

    def get_pairs(self) -> None:
        """Get pairs of images to be matched."""
        log_step("Getting pairs...")
        logging.debug(f"Retrieval config: {self.configs['retrieval']}")
        logging.debug(f"Storing retrieval features at {self.paths.features_retrieval}")

        # TODO: use pairs from poses top 20, 30
        # or netvlad, cosplace

        if self.args.colmap:
            logging.debug("No retrieval, using colmap")
            return

        # extract global features
        # os.makedirs(self.paths.features_retrieval.parent, exist_ok=True)
        extract_features.main(
            conf=self.configs["retrieval"],
            image_dir=self.paths.images,
            feature_path=self.paths.features_retrieval,
        )

        # match global features to get pairs
        os.makedirs(self.paths.matches_retrieval.parent, exist_ok=True)

        if self.n_images <= N_EXHAUSTIVE:
            logging.debug("Using exhaustive retrieval")
            logging.debug(f"Storing pairs to {self.paths.matches_retrieval}")
            pairs_from_exhaustive.main(
                features=self.paths.features_retrieval, output=self.paths.matches_retrieval
            )
        else:
            logging.debug(f"Using {self.args.retrieval}")
            logging.debug(f"Storing pairs to {self.paths.matches_retrieval}")
            pairs_from_retrieval.main(
                descriptors=self.paths.features_retrieval,
                num_matched=self.args.n_retrieval_matches,
                output=self.paths.matches_retrieval,
            )

    def extract_features(self) -> None:
        """Extract features from the images."""
        log_step("Extracting features...")

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
            return

        logging.debug("Extracting features with hloc")
        logging.debug(f"Feature config: {self.configs['feature']}")
        logging.debug(f"Storing features to {self.paths.features}")

        # os.makedirs(self.paths.features.parent, exist_ok=True)
        extract_features.main(
            conf=self.configs["feature"],
            image_dir=self.paths.images,
            feature_path=self.paths.features,
        )

    def match_features(self) -> None:
        """Match features between images."""
        log_step("Matching features...")

        if self.args.colmap:
            if self.n_images <= N_EXHAUSTIVE:
                logging.debug("Exhaustive matching features with colmap")
                pycolmap.match_exhaustive(self.paths.db, verbose=self.args.verbose)
            else:
                logging.debug("Sequential matching features with colmap")
                pycolmap.match_sequential(self.paths.db, verbose=self.args.verbose)
            return

        logging.debug("Matching features with hloc")
        logging.debug(f"Matcher config: {self.configs['matcher']}")
        logging.debug(f"Loading pairs form {self.paths.matches_retrieval}")
        logging.debug(f"Loading features from {self.paths.features}")
        logging.debug(f"Storing matches to {self.paths.matches}")

        # os.makedirs(self.paths.matches.parent, exist_ok=True)
        match_features.main(
            conf=self.configs["matcher"],
            pairs=self.paths.matches_retrieval,
            features=self.paths.features,
            matches=self.paths.matches,
        )

    def sfm(self) -> None:
        """Run Structure from Motion."""
        log_step("Running Structure from Motion...")
        os.makedirs(self.paths.sparse, exist_ok=True)

        if self.args.colmap:
            logging.debug("Running SFM with colmap")
            pycolmap.incremental_mapping(self.paths.db, self.paths.images, self.paths.sparse)
            # move files to sfm dir
            for filename in ["images.bin", "cameras.bin", "points3D.bin"]:
                shutil.move(
                    str(self.paths.sparse / "0" / filename), str(self.paths.sparse / filename)
                )
            # delete "0" folder
            shutil.rmtree(str(self.paths.sparse / "0"))
            return

        logging.debug("Running SFM with hloc")
        mapper_options = pycolmap.IncrementalMapperOptions().todict()
        self.model = reconstruction.main(
            sfm_dir=self.paths.sparse,
            image_dir=self.paths.images,
            pairs=self.paths.matches_retrieval,
            features=self.paths.features,
            matches=self.paths.matches,
            verbose=self.args.verbose,
            mapper_options=mapper_options,
        )

        # delete "models" folder
        shutil.rmtree(str(self.paths.sparse / "models"))

    def refinement(self) -> None:
        """Refine the reconstruction using PixSFM."""
        log_step("Refining the reconstruction...")
        os.makedirs(self.paths.sparse, exist_ok=True)
        # TODO: implement refinement

    def mvs(self) -> None:
        """Run Multi-View Stereo."""
        log_step("Running Multi-View Stereo...")
        os.makedirs(self.paths.dense, exist_ok=True)

        # TODO: implement MVS
        # pycolmap.undistort_images(mvs_path, output_path, image_dir)
        # pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
        # pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)

    def cleanup(self) -> None:
        """Clean up the pipeline."""
        # TODO: implement cleanup
        log_step("Cleaning up...")

        os.makedirs(self.paths.results, exist_ok=True)

    def run(self) -> None:
        """Run the pipeline."""
        self.get_pairs()
        self.extract_features()
        self.match_features()
        self.sfm()
        self.refinement()
        self.mvs()
        self.cleanup()
