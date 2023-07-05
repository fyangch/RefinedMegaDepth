"""Pipeline using HLoc."""
import logging
import os

from hloc import (
    extract_features,
    match_features,
    pairs_from_covisibility,
    pairs_from_exhaustive,
    pairs_from_poses,
    pairs_from_retrieval,
    reconstruction,
)
from omegaconf import DictConfig
from pixsfm.refine_hloc import PixSfM

from megadepth.pipelines.pipeline import Pipeline
from megadepth.utils.constants import ModelType, Retrieval


class HlocPipeline(Pipeline):
    """Pipeline for HLoc."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize the pipeline."""
        super().__init__(config)

    def _extract_pairs_from_retrieval(self):
        """Extract pairs from retrieval."""
        logging.debug(f"Retrieval config: {self.configs['retrieval']}")

        # extract global features
        extract_features.main(
            conf=self.configs["retrieval"],
            image_dir=self.paths.images,
            feature_path=self.paths.features_retrieval,
        )

        logging.debug(f"Using {self.config.retrieval.name}")
        logging.debug(f"Storing pairs to {self.paths.matches_retrieval}")

        pairs_from_retrieval.main(
            descriptors=self.paths.features_retrieval,
            num_matched=self.config.retrieval.n_matches,
            output=self.paths.matches_retrieval,
        )

    def get_pairs(self) -> None:
        """Get pairs of images to match."""
        self.log_step("Getting pairs...")

        # match global features to get pairs
        os.makedirs(self.paths.features.parent, exist_ok=True)
        os.makedirs(self.paths.matches_retrieval.parent, exist_ok=True)

        if self.config.retrieval.name == Retrieval.EXHAUSTIVE.value:
            logging.debug("Using exhaustive retrieval")
            logging.debug(f"Storing pairs to {self.paths.matches_retrieval}")

            image_list = os.listdir(self.paths.images)

            pairs_from_exhaustive.main(image_list=image_list, output=self.paths.matches_retrieval)

        elif self.config.retrieval.name == Retrieval.POSES.value:
            logging.debug(f"Using {self.config.retrieval.name}")

            pairs_from_poses.main(
                model=self.paths.baseline_model,
                output=self.paths.matches_retrieval,
                num_matched=self.config.retrieval.n_matches,
            )

        elif self.config.retrieval.name == Retrieval.COVISIBILITY.value:
            logging.debug(f"Using {self.config.retrieval.name}")

            pairs_from_covisibility.main(
                model=self.paths.baseline_model,
                output=self.paths.matches_retrieval,
                num_matched=self.config.retrieval.n_matches,
            )

        elif self.config.retrieval.name in [Retrieval.NETVLAD.value, Retrieval.COSPLACE.value]:
            self._extract_pairs_from_retrieval()

        else:
            raise NotImplementedError(
                f"Retrieval method {self.config.retrieval.name} not implemented"
            )

    def extract_features(self) -> None:
        """Extract features from images."""
        self.log_step("Extracting features...")

        logging.debug("Extracting features with hloc")
        logging.debug(f"Feature config: {self.configs['feature']}")
        logging.debug(f"Storing features to {self.paths.features}")

        os.makedirs(self.paths.features.parent, exist_ok=True)

        extract_features.main(
            conf=self.configs["feature"],
            image_dir=self.paths.images,
            feature_path=self.paths.features,
        )

    def match_features(self) -> None:
        """Match features between images."""
        self.log_step("Matching features...")

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

    def sfm(self) -> None:
        """Run Structure from Motion."""
        self.log_step("Running Structure from Motion...")

        os.makedirs(self.paths.sparse, exist_ok=True)

        if self.model_exists(ModelType.SPARSE) and not self.config.overwrite:
            logging.info(f"Reconstruction exists at {self.paths.sparse}. Skipping SFM...")
            return

        logging.debug("Running SFM with hloc")
        self.sparse_model = reconstruction.main(
            sfm_dir=self.paths.sparse,
            image_dir=self.paths.images,
            pairs=self.paths.matches_retrieval,
            features=self.paths.features,
            matches=self.paths.matches,
            verbose=self.config.logging.verbose,
        )

        logging.debug("Aligning reconstruction with baseline...")
        self.align_with_baseline()

    def refinement(self) -> None:
        """Refine the reconstruction using PixSFM."""
        self.log_step("Refining the reconstruction...")

        if self.model_exists(ModelType.REFINED) and not self.config.overwrite:
            logging.info(
                f"Reconstruction exists at {self.paths.refined_sparse}. Skipping refinement..."
            )
            return

        if "KA" not in self.config.refinement.steps:
            logging.info("Skipping refinement...")
            return

        os.makedirs(self.paths.sparse, exist_ok=True)
        os.makedirs(self.paths.refined_sparse, exist_ok=True)

        refiner = PixSfM(conf=self.configs["refinement"])

        logging.debug("Refining the reconstruction with PixSfM")
        logging.debug(f"Refiner config: {self.configs['refinement']}")
        logging.debug(f"Loading pairs from {self.paths.matches_retrieval}")
        logging.debug(f"Loading features from {self.paths.features}")
        logging.debug(f"Loading matches from {self.paths.matches}")
        logging.debug(f"Loading sparse model from {self.paths.sparse}")
        logging.debug(f"Storing refined model to {self.paths.refined_sparse}")

        logging.info("Running PixSfM BA...")
        reconstruction, ba_data, feature_manager = refiner.run_ba(
            reconstruction=self.sparse_model,
            image_dir=self.paths.images,
            cache_path=self.paths.cache,
        )

        reconstruction.write(str(self.paths.refined_sparse))

        self.refined_model = reconstruction
