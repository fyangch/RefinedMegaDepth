"""Setup the everything for the pipeline."""

import datetime
import logging
import os
from pathlib import Path

import pixsfm
from hloc import extract_features, match_dense, match_features
from omegaconf import DictConfig, OmegaConf

from megadepth.utils.constants import Matcher, Retrieval


def setup_logger(config: DictConfig) -> None:
    """Setup the logger.

    Args:
        config (DictConfig): Config with values from the yaml file and CLI.
    """
    if config.logging.log_dir:
        if not os.path.exists(config.logging.log_dir):
            os.makedirs(config.logging.log_dir)
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(config.logging.log_dir, f"{date}_log.txt")
    else:  # log to stdout
        log_file = None

    logging.basicConfig(
        format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=config.logging.log_level,
        filename=log_file,
    )

    logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.ERROR)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.ERROR)


def get_configs(config: DictConfig) -> dict:
    """Return a dictionary of configuration parameters.

    Args:
        config (DictConfig): Config with values from the yaml file and CLI.

    Returns:
        dict: A dictionary of configuration parameters.
    """
    # retrieval config
    retrieval_conf = (
        extract_features.confs[config.retrieval.name]
        if config.retrieval.name
        not in [Retrieval.POSES.value, Retrieval.COVISIBILITY.value, Retrieval.EXHAUSTIVE.value]
        else None
    )

    # feature config
    feature_config = extract_features.confs[config.features]
    # if config.features == Features.SIFT.value:
    # feature_config["preprocessing"]["resize_max"] = 3200

    # matcher config
    if config.matcher == Matcher.LOFTR.value:
        matcher_conf = match_dense.confs[config.matcher]
        matcher_conf["preprocessing"]["resize_max"] = 840
    else:
        matcher_conf = match_features.confs[config.matcher]

    # refinement config

    # set cache path
    if config.refinement.low_memory:
        refinement_conf = OmegaConf.load(pixsfm.configs.parse_config_path("low_memory"))
    else:
        refinement_conf = OmegaConf.load(pixsfm.configs.parse_config_path("default"))

    return {
        "retrieval": retrieval_conf,
        "feature": feature_config,
        "matcher": matcher_conf,
        "refinement": refinement_conf,
    }


class DataPaths:
    """Class for handling the data paths."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize the data paths.

        Args:
            config (DictConfig): Config with values from the yaml file and CLI.
        """
        self.model_name = self.get_model_name(config)

        retrieval_name = f"{config.retrieval.name}"
        retrieval_name += (
            ".txt"
            if config.retrieval.name == Retrieval.EXHAUSTIVE.value
            else f"-{config.retrieval.n_matches}.txt"
        )

        # paths
        self.data = Path(os.path.join(config.data_path, config.scene))
        self.images = Path(os.path.join(self.data, "images"))

        # retrieval
        self.features_retrieval = Path(
            os.path.join(self.data, "features", f"{config.retrieval.name}.h5")
        )
        self.matches_retrieval = Path(
            os.path.join(self.data, "matches", "retrieval", retrieval_name)
        )

        # features
        if config.matcher == Matcher.LOFTR.value:
            self.features = Path(os.path.join(self.data, "features", f"{self.model_name}.h5"))
        else:
            self.features = Path(os.path.join(self.data, "features", f"{config.features}.h5"))

        # matches
        self.matches = Path(os.path.join(self.data, "matches", f"{self.model_name}.h5"))

        # models
        self.sparse = Path(os.path.join(self.data, "sparse", self.model_name))
        self.sparse_baseline = Path(os.path.join(self.data, "sparse", "baseline"))
        self.refined_sparse = Path(os.path.join(self.data, "sparse", self.model_name, "refined"))
        self.db = Path(os.path.join(self.sparse, "database.db"))
        self.dense = Path(os.path.join(self.data, "dense", self.model_name))
        self.baseline_model = Path(os.path.join(self.data, "sparse", "baseline"))

        # output
        self.metrics = Path(os.path.join(self.data, "metrics", self.model_name))
        self.results = Path(os.path.join(self.data, "results", self.model_name))
        self.visualizations = Path(os.path.join(self.data, "visualizations", self.model_name))

        # cache
        self.cache = None
        if config.refinement.low_memory:
            cache_dir = os.environ.get("TMPDIR")
            if cache_dir is None:
                raise ValueError(
                    "TMPDIR environment variable not set. "
                    + "Set it using export TMPDIR=/path/to/tmpdir"
                )

            self.cache = Path(cache_dir)

        logging.debug("Data paths:")
        for path, val in vars(self).items():
            logging.debug(f"\t{path}: {val}")

    def get_model_name(self, config: DictConfig) -> str:
        """Return the model name.

        Args:
            config (DictConfig): Config with values from the yaml file and CLI.

        Returns:
            str: The model name.
        """
        if "model_name" in config:
            return config.model_name
        elif config.colmap:
            return "colmap"
        elif config.matcher == Matcher.LOFTR.value:
            return (
                f"{config.matcher}"
                + f"-{config.retrieval.name}"
                + f"-{config.retrieval.n_matches}"
                + f"-{config.refinement.steps}"
            )
        elif config.retrieval.name == Retrieval.EXHAUSTIVE.value:
            return (
                f"{config.features}"
                + f"-{config.matcher}"
                + f"-{config.retrieval.name}"
                + f"-{config.refinement.steps}"
            )
        elif config.refinement.steps not in ["KA", "BA", "KA+BA"]:
            return (
                f"{config.features}"
                + f"-{config.matcher}"
                + f"-{config.retrieval.name}"
                + f"-{config.retrieval.n_matches}"
            )
        else:
            return (
                f"{config.features}"
                + f"-{config.matcher}"
                + f"-{config.retrieval.name}"
                + f"-{config.retrieval.n_matches}"
                + f"-{config.refinement.steps}"
            )
