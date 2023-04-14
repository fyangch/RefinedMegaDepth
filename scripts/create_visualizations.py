"""Script to create visualizations for all reconstructions."""

import argparse
import logging
import os

from megadepth.visualization import create_all_figures


def main(args: argparse.Namespace):
    """Go through all sparse reconstructions and create visualizations."""
    scenes = [os.path.join(args.data_path, f, "sparse") for f in os.listdir(args.data_path)]
    scenes = [f for f in scenes if os.path.isdir(f)]

    for scene in scenes:
        models = os.listdir(scene)
        models = [f for f in models if not f.startswith(".")]
        for model in models:
            file_path = os.path.join(scene, model)
            output_path = os.path.join(
                scene,
                "..",
                "visualizations",
                model,
            )
            print(output_path)
            os.makedirs(output_path, exist_ok=True)
            create_all_figures(file_path, output_path)

    scenes = [os.path.join(args.data_path, f, "dense") for f in os.listdir(args.data_path)]
    scenes = [f for f in scenes if os.path.isdir(f)]

    for scene in scenes:
        models = os.listdir(scene)
        models = [f for f in models if not f.startswith(".")]
        for model in models:
            file_path = os.path.join(scene, model, "sparse")
            depth_path = os.path.join(scene, model, "stereo", "depth_maps")
            output_path = os.path.join(
                scene,
                "..",
                "visualizations",
                model,
            )
            print(output_path)
            os.makedirs(output_path, exist_ok=True)
            create_all_figures(file_path, output_path, depth_path=depth_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default=r"data", help="Path to the directory containing the scenes"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    main(args)
