"""Collect metadata from a COLMAP reconstruction."""

import argparse
import datetime
import json
import logging
import os
from typing import Any, Dict

import pycolmap

from megadepth.utils.enums import ModelType
from megadepth.utils.utils import DataPaths


def collect_metrics(
    paths: DataPaths, args: argparse.Namespace, model_type: ModelType
) -> Dict[str, Any]:
    """Collect metrics for a COLMAP reconstruction.

    Args:
        paths: The data paths for the reconstruction.
        args: The parsed command line arguments.
        model_type: The type of model to collect metrics for.

    Returns:
        A list of dictionaries containing the metrics.
    """
    reconstruction = pycolmap.Reconstruction(paths.sparse)

    if model_type == ModelType.SPARSE:
        metrics = collect_sparse(reconstruction)
    elif model_type == ModelType.DENSE:
        # metrics = collect_dense(reconstruction)
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # add pipeline information
    metrics["scene"] = str(args.scene)
    metrics["model_name"] = paths.model_name
    metrics["retrieval"] = args.retrieval
    metrics["n_retrieval_matches"] = args.n_retrieval_matches
    metrics["features"] = args.features
    metrics["matcher"] = args.matcher

    logging.debug(metrics)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(os.path.join(paths.metrics, f"{model_type.value}-{timestamp}.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def collect_sparse(reconstruction: pycolmap.Reconstruction) -> Dict[str, Any]:
    """Collect metrics for a sparse COLMAP reconstruction.

    Args:
        reconstruction: The sparse reconstruction.

    Returns:
        A dictionary containing the metrics.
    """
    n_images = len(reconstruction.images)
    reg_images = reconstruction.num_reg_images()

    return {
        "n_images": n_images,
        "n_reg_images": reg_images,
        "perc_reg_images": reg_images / n_images * 100.0,
        "mean_reprojection_error": reconstruction.compute_mean_reprojection_error(),
        "n_observations": reconstruction.compute_num_observations(),
        "mean_obs_per_reg_image": reconstruction.compute_mean_observations_per_reg_image(),
        "mean_track_length": reconstruction.compute_mean_track_length(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--model_type",
        type=str,
        choices=[m.value for m in ModelType],
        default=ModelType.SPARSE.value,
    )
    args = parser.parse_args()

    model_path = os.path.join(args.data_path, args.scene, args.model_type, args.model_name)
    model = pycolmap.Reconstruction(model_path)

    if args.model_type == ModelType.SPARSE.value:
        metrics = collect_sparse(model)
    elif args.model_type == ModelType.DENSE.value:
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    metrics["scene"] = args.scene
    metrics["model_name"] = args.model_name

    print("Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    metrics_path = os.path.join(args.data_path, args.scene, "metrics", args.model_name)
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(os.path.join(metrics_path, f"{args.model_type}-{timestamp}.json"), "w") as f:
        json.dump(metrics, f, indent=4)
