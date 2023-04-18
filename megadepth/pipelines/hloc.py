"""Pipeline using HLoc."""
import argparse
import datetime
import logging
import os
import time

from hloc import (
    extract_features,
    match_features,
    pairs_from_covisibility,
    pairs_from_exhaustive,
    pairs_from_poses,
    pairs_from_retrieval,
    reconstruction,
)

from megadepth.pipelines.pipeline import Pipeline
from megadepth.utils.constants import ModelType, Retrieval

# from pixsfm.refine_hloc import PixSfM


class HlocPipeline(Pipeline):
    """Pipeline for HLoc."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the pipeline."""
        super().__init__(args)

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
        """Get pairs of images to match."""
        self.log_step("Getting pairs...")
        start = time.time()

        # match global features to get pairs
        os.makedirs(self.paths.features.parent, exist_ok=True)
        os.makedirs(self.paths.matches_retrieval.parent, exist_ok=True)

        if self.args.retrieval == Retrieval.EXHAUSTIVE.value:
            logging.debug("Using exhaustive retrieval")
            logging.debug(f"Storing pairs to {self.paths.matches_retrieval}")

            image_list = os.listdir(self.paths.images)
            image_list = [img for img in image_list if img.endswith(".jpg")]

            pairs_from_exhaustive.main(image_list=image_list, output=self.paths.matches_retrieval)

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
        """Extract features from images."""
        self.log_step("Extracting features...")
        start = time.time()

        logging.debug("Extracting features with hloc")
        logging.debug(f"Feature config: {self.configs['feature']}")
        logging.debug(f"Storing features to {self.paths.features}")

        os.makedirs(self.paths.features.parent, exist_ok=True)

        extract_features.main(
            conf=self.configs["feature"],
            image_dir=self.paths.images,
            feature_path=self.paths.features,
        )

        end = time.time()
        logging.info(f"Time to extract features: {datetime.timedelta(seconds=end - start)}")

    def match_features(self) -> None:
        """Match features between images."""
        self.log_step("Matching features...")
        start = time.time()

        logging.debug("Matching features with hloc")
        logging.debug(f"Matcher config: {self.configs['matcher']}")
        logging.debug(f"Loading pairs from {self.paths.matches_retrieval}")
        logging.debug(f"Loading features from {self.paths.features}")
        logging.debug(f"Storing matches to {self.paths.matches}")

        os.makedirs(self.paths.matches.parent, exist_ok=True)

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
        self.log_step("Running Structure from Motion...")
        start = time.time()

        os.makedirs(self.paths.sparse, exist_ok=True)

        if self.model_exists(ModelType.SPARSE) and not self.args.overwrite:
            logging.info(f"Reconstruction exists at {self.paths.sparse}. Skipping SFM...")
            return

        logging.debug("Running SFM with hloc")
        self.sparse_model = reconstruction.main(
            sfm_dir=self.paths.sparse,
            image_dir=self.paths.images,
            pairs=self.paths.matches_retrieval,
            features=self.paths.features,
            matches=self.paths.matches,
            verbose=self.args.verbose,
        )

        logging.debug("Aligning reconstruction with baseline...")
        self.align_with_baseline()

        end = time.time()
        logging.info(f"Time to run SFM: {datetime.timedelta(seconds=end - start)}")

    # def refinement(self) -> None:
    #     """Refine the reconstruction using PixSFM."""
    #     self.log_step("Refining the reconstruction...")
    #     start = time.time()

    #     os.makedirs(self.paths.sparse, exist_ok=True)
    #     os.makedirs(self.paths.ref_sparse, exist_ok=True)

    #     refiner = PixSfM(conf=self.configs["refinement"])

    #     logging.debug("Refining the reconstruction with PixSfM")
    #     logging.debug(f"Refiner config: {self.configs['refinement']}")
    #     logging.debug(f"Loading pairs from {self.paths.matches_retrieval}")
    #     logging.debug(f"Loading features from {self.paths.features}")
    #     logging.debug(f"Loading matches from {self.paths.matches}")
    #     logging.debug(f"Loading sparse model from {self.paths.sparse}")
    #     logging.debug(f"Storing refined model to {self.paths.ref_sparse}")

    #     model, outputs = refiner.run(
    #         output_dir=self.paths.ref_sparse,
    #         image_dir=self.paths.images,
    #         pairs_path=self.paths.matches_retrieval,
    #         features_path=self.paths.features,
    #         matches_path=self.paths.matches,
    #         reference_model_path=self.paths.sparse,
    #     )

    #     self.refined_model = model

    #     # TODO: explore outputs. Maybe save to metrics?

    #     end = time.time()
    #     logging.info(
    #         f"Time to refine the reconstruction: {datetime.timedelta(seconds=end - start)}"
    #     )
