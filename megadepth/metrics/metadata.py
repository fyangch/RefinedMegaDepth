"""Collect metadata from a COLMAP reconstruction."""

import datetime
import json
import logging
import os
from typing import Any, Dict, Tuple

import numpy as np
import pycolmap
from omegaconf import DictConfig
from tqdm import tqdm

from megadepth.metrics.nighttime import run_day_night_classification
from megadepth.metrics.overlap import dense_overlap, sparse_overlap
from megadepth.utils.io import load_depth_map


def collect_metrics(paths: DictConfig, config: DictConfig, model_type: str) -> Dict[str, Any]:
    """Collect metrics for a COLMAP reconstruction.

    Args:
        paths: The data paths for the reconstruction.
        config (DictConfig): Config with values from the yaml file and CLI.
        model_type: The type of model to collect metrics for.

    Returns:
        A list of dictionaries containing the metrics.
    """
    if model_type == "dense":
        reconstruction = pycolmap.Reconstruction(os.path.join(paths.dense, "sparse"))
        metrics, overlap = collect_dense(reconstruction, paths)
    elif model_type == "sparse":
        reconstruction = pycolmap.Reconstruction(paths.sparse)
        metrics, overlap = collect_sparse(reconstruction)

        # determine nighttime images
        night_df = run_day_night_classification(reconstruction, paths.images)
        metrics["n_night_images"] = len(
            [img for img in reconstruction.images.values() if night_df.loc[img.name, "is_night"]]
        )
        night_df.to_csv(os.path.join(paths.metrics, "night_images.csv"))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # number of images
    n_images = len(
        [img for img in os.listdir(paths.images) if not os.path.isdir(paths.images / img)]
    )
    metrics["n_images"] = n_images
    metrics["perc_reg_images"] = metrics["n_reg_images"] / n_images * 100

    # mean overlap
    metrics["mean_overlap"] = np.mean(overlap.data)

    # add pipeline information
    metrics["scene"] = str(config.scene)
    metrics["model_name"] = config.model_name
    metrics["retrieval"] = config.retrieval.name
    metrics["n_retrieval_matches"] = config.retrieval.n_matches
    metrics["features"] = [ens.features.name for ens in config.ensembles.values()]
    metrics["matchers"] = [ens.matchers.name for ens in config.ensembles.values()]

    logging.info(f"Metrics for {model_type} model:")
    for k, v in metrics.items():
        logging.info(f"\t{k}: {v}")

    os.makedirs(paths.metrics, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    overlap_fn = f"{model_type}-overlap-{timestamp}.nc"
    overlap.to_netcdf(os.path.join(paths.metrics, overlap_fn))
    metrics["overlap_fn"] = overlap_fn

    with open(os.path.join(paths.metrics, f"{model_type}-{timestamp}.json"), "w") as f_metric:
        json.dump(metrics, f_metric, indent=4)

    return metrics


def collect_sparse(reconstruction: pycolmap.Reconstruction) -> Tuple[Dict[str, Any], np.ndarray]:
    """Collect metrics for a sparse COLMAP reconstruction.

    Args:
        reconstruction: The sparse reconstruction.

    Returns:
        A dictionary containing the metrics and a numpy array containing the overlap scores.
    """
    metrics = {
        "n_reg_images": reconstruction.num_reg_images(),
        "mean_reprojection_error": reconstruction.compute_mean_reprojection_error(),
        "n_observations": reconstruction.compute_num_observations(),
        "mean_obs_per_reg_image": reconstruction.compute_mean_observations_per_reg_image(),
        "mean_track_length": reconstruction.compute_mean_track_length(),
    }
    overlap = sparse_overlap(reconstruction)

    return metrics, overlap


def calculate_completeness(reconstruction: pycolmap.Reconstruction, paths: DictConfig) -> float:
    """Calculate the completeness of the dense maps.

    Args:
        reconstruction (pycolmap.Reconstruction): The sparse reconstruction.
        paths (DictConfig): The data paths for the reconstruction.

    Returns:
        float: The completeness of the dense maps.
    """
    depth_path = os.path.join(paths.dense, "stereo", "depth_maps")
    fnames = [img.name for img in reconstruction.images.values()]
    fnames = [f"{fname}.geometric.bin" for fname in fnames]
    fpaths = [os.path.join(depth_path, fname) for fname in fnames]

    vals = []
    for fpath in tqdm(fpaths, desc="Calculating completeness", ncols=80):
        if not os.path.exists(fpath):
            vals.append(0)
            continue

        depth = load_depth_map(fpath)
        vals.append(np.sum(depth > 0) / depth.size)

    return np.mean(vals)


def collect_dense(
    reconstruction: pycolmap.Reconstruction, paths: DictConfig
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Collect metrics for a dense COLMAP reconstruction.

    Args:
        reconstruction: The dense reconstruction.
        paths: The data paths for the reconstruction.

    Returns:
        A dictionary containing the metrics and a numpy array containing the overlap scores.
    """
    metrics: Dict[str, Any] = {
        "n_reg_images": reconstruction.num_reg_images(),
        "mean_reprojection_error": reconstruction.compute_mean_reprojection_error(),
        "n_observations": reconstruction.compute_num_observations(),
        "mean_obs_per_reg_image": reconstruction.compute_mean_observations_per_reg_image(),
        "mean_track_length": reconstruction.compute_mean_track_length(),
        "completeness": calculate_completeness(reconstruction, paths),
    }
    # TODO: add metrics if we loose some images during dense reconstruction

    overlap = dense_overlap(paths)

    return metrics, overlap
