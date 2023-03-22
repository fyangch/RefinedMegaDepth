"""Collect metadata from a COLMAP reconstruction."""

import argparse
from pathlib import Path
from typing import Any

import pycolmap


def collect_metrics(path: Path) -> dict[str, Any]:
    """Collect metrics for a COLMAP reconstruction.

    Args:
        path: Path to the COLMAP reconstruction.

    Returns:
        A list of dictionaries containing the metrics.
    """
    reconstruction = pycolmap.Reconstruction(path)

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
    parser.add_argument(
        "--model",
        type=Path,
        default="data",
        help="Path to the model dir containing images.bin, cameras.bin, and points3D.bin.",
    )
    args = parser.parse_args()

    metrics = collect_metrics(args.model)

    for key, value in metrics.items():
        print(f"{key}: {value}")
