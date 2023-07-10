"""Implementation of the sparse pipeline."""

import logging
import os
import shutil

import pycolmap
from hloc import extract_features, match_features, pairs_from_exhaustive, pairs_from_retrieval
from hloc.utils.io import list_h5_names
from pixsfm.refine_hloc import PixSfM

from megadepth.pipelines.pipeline import Pipeline
from megadepth.utils.concatenate import concat_features, concat_matches
from megadepth.utils.io import model_exists


class SparsePipeline(Pipeline):
    """Implementation of the sparse pipeline."""

    def get_pairs(self) -> None:
        """Get pairs of images to match."""
        self.log_step("Getting pairs...")

        # match global features to get pairs
        os.makedirs(self.paths.features.parent, exist_ok=True)
        os.makedirs(self.paths.pairs.parent, exist_ok=True)

        if self.config.retrieval.name == "exhaustive":
            logging.debug("Using exhaustive retrieval")
            logging.debug(f"Storing pairs to {self.paths.pairs}")

            image_list = os.listdir(self.paths.images)

            pairs_from_exhaustive.main(image_list=image_list, output=self.paths.pairs)
            return

        """Extract pairs from retrieval."""
        logging.debug(f"Retrieval config: {self.config.retrieval}")

        # extract global features
        extract_features.main(
            conf=self.config.retrieval,
            image_dir=self.paths.images,
            feature_path=self.paths.features_retrieval,
        )

        logging.debug(f"Using {self.config.retrieval.name}")
        logging.debug(f"Storing pairs to {self.paths.pairs}")

        pairs_from_retrieval.main(
            descriptors=self.paths.features_retrieval,
            num_matched=self.config.retrieval.n_matches,
            output=self.paths.pairs,
        )

    def concat_features_and_matches(self) -> None:
        """Concatenate features and matches."""
        self.log_step("Creating ensemble")

        for i, ens in enumerate(self.config.ensembles.values()):
            features_path = self.paths.features.parent / f"{ens.features.output}.h5"
            matches_path = self.paths.matches.parent / f"{ens.matchers.output}.h5"

            if i == 0:
                # copy first feature and matches to final output
                shutil.copyfile(features_path, self.paths.features)
                shutil.copyfile(matches_path, self.paths.matches)
                continue

            concat_features(
                features1=self.paths.features,
                features2=features_path,
                out_path=self.paths.features,
            )

            concat_matches(
                matches1_path=self.paths.matches,
                matches2_path=matches_path,
                ensemble_features_path=self.paths.features,
                out_path=self.paths.matches,
            )

        # write pairs file
        # TODO: check if this is necessary
        pairs = sorted(list(list_h5_names(self.paths.matches)))
        with open(self.paths.pairs, "w") as f:
            for pair in pairs:
                p = pair.split("/")
                f.write(f"{p[0]} {p[1]}\n")

    def extract_features(self) -> None:
        """Extract features from the images."""
        self.log_step("Extract features")

        for ens in self.config.ensembles.values():
            ens_feature_path = self.paths.features.parent / f"{ens.features.output}.h5"

            if ens_feature_path.exists() and self.overwrite:
                logging.info(f"Removing {ens_feature_path}")
                ens_feature_path.unlink()

            extract_features.main(
                conf=ens.features,
                image_dir=self.paths.images,
                feature_path=ens_feature_path,
            )

    def match_features(self) -> None:
        """Match features between images."""
        self.log_step("Match features")

        if self.paths.matches.exists() and self.overwrite:
            logging.info(f"Removing {self.paths.matches}")
            self.paths.matches.unlink()

        for ens in self.config.ensembles.values():
            ens_match_path = self.paths.matches.parent / f"{ens.matchers.output}.h5"

            if ens_match_path.exists() and self.overwrite:
                logging.info(f"Removing {ens_match_path}")
                ens_match_path.unlink()

            match_features.main(
                conf=ens.matchers,
                pairs=self.paths.pairs,
                features=self.paths.features.parent / f"{ens.features.output}.h5",
                matches=ens_match_path,
            )

        # create final ensemble by concatenating features and matches
        self.concat_features_and_matches()

    def sfm(self) -> None:
        """Run SfM."""
        self.log_step("Run SfM")

        if model_exists(self.paths.sparse) and not self.overwrite:
            logging.info("Model already exists. Skipping...")
            self.sparse_model = pycolmap.Reconstruction(self.paths.sparse)
            return

        os.makedirs(self.paths.sparse.parent, exist_ok=True)

        refiner = PixSfM(conf=self.config.refinement)
        model, outputs = refiner.run(
            output_dir=self.paths.sparse,
            image_dir=self.paths.images,
            pairs_path=self.paths.pairs,
            features_path=self.paths.features,
            matches_path=self.paths.matches,
            cache_path=self.paths.cache,
            verbose=self.config.logging.verbose,
        )

        self.sparse_model = model

        logging.debug(f"Outputs:\n{outputs}")
