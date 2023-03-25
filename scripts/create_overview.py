"""Script to create an overview csv file from all metrics files."""

import argparse
import json
import os
from typing import Any, Dict


def main(args: argparse.Namespace):
    """Collect all metrics from the scenes and create an overview csv file."""
    scenes = [os.path.join(args.data_path, f, "metrics") for f in os.listdir(args.data_path)]
    scenes = [f for f in scenes if os.path.isdir(f)]

    overview: Dict[str, Any] = {}
    for scene in scenes:
        models = os.listdir(scene)
        models = [f for f in models if not f.startswith(".")]
        for model in models:
            fnames = os.listdir(os.path.join(scene, model))
            fnames = [f for f in fnames if f.endswith(".json") and not f.startswith(".")]
            for fname in fnames:
                with open(os.path.join(scene, model, fname), "r") as f:
                    metrics = json.load(f)

                if metrics["scene"] not in overview:
                    overview[metrics["scene"]] = []

                overview[metrics["scene"]].append(metrics)

    with open(args.output, "w") as f:
        f.write(json.dumps(overview, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the directory containing the scenes")
    parser.add_argument(
        "--output", default="data/metrics_overview.json", help="Path to the output file"
    )
    args = parser.parse_args()

    main(args)
