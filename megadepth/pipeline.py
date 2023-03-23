"""Implementation of the MegaDepth pipeline using colmap, hloc and pixsfm."""
import argparse
import logging

from hloc import (
    extract_features,
    match_features,
    pairs_from_exhaustive,
    pairs_from_retrieval,
    reconstruction,
)

from megadepth.utils.utils import DataPaths, get_configs


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
        self.model = None

    def get_pairs(self) -> None:
        """Get pairs of images to be matched."""
        logging.info("Getting pairs...")
        logging.debug(f"Retrieval config: {self.configs['retrieval']}")
        logging.debug(f"Storing retrieval features at {self.paths.features_retrieval}")

        # use: pairs from poses top 20, 30
        # netvlad cosplace

        extract_features.main(
            conf=self.configs["retrieval"],
            image_dir=self.paths.images,
            feature_path=self.paths.features_retrieval,
        )

        if self.args.n_retrieval_matches <= 0:
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
        logging.info("Extracting features...")
        logging.debug(f"Feature config: {self.configs['feature']}")
        logging.debug(f"Storing features to {self.paths.features}")
        extract_features.main(
            conf=self.configs["feature"],
            image_dir=self.paths.images,
            feature_path=self.paths.features,
        )

    def match_features(self) -> None:
        """Match features between images."""
        logging.info("Matching features...")
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

    def sfm(self) -> None:
        """Run Structure from Motion."""
        logging.info("Running Structure from Motion...")
        self.model = reconstruction.main(
            sfm_dir=self.paths.sparse,
            image_dir=self.paths.images,
            pairs=self.paths.matches_retrieval,
            features=self.paths.features,
            matches=self.paths.matches,
            verbose=self.args.verbose,
        )

    def refinement(self) -> None:
        """Refine the reconstruction using PixSFM."""
        pass

    def mvs(self) -> None:
        """Run Multi-View Stereo."""
        pass

    def cleanup(self) -> None:
        """Clean up the pipeline."""
        pass

    def run(self) -> None:
        """Run the pipeline."""
        self.get_pairs()
        self.extract_features()
        self.match_features()
        self.sfm()
        self.refinement()
        self.mvs()
        self.cleanup()
