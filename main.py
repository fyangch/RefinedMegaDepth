"""This project is a re-implementation of the MegaDepth pipeline."""
import logging

from megadepth.metrics.metadata import collect_metrics
from megadepth.pipeline import Pipeline
from megadepth.utils.utils import DataPaths, setup


def main():
    """Run the mega depth pipeline."""
    args = setup()
    paths = DataPaths(args)

    # create pipeline
    pipeline = Pipeline(args)

    # run pipeline
    pipeline.get_pairs()
    pipeline.extract_features()
    pipeline.match_features()
    pipeline.sfm()
    pipeline.refinement()  # not implemented yet
    pipeline.mvs()  # not implemented yet
    pipeline.cleanup()  # not implemented yet

    # alterative
    # pipeline.run()

    metrics = collect_metrics(paths.sparse)
    logging.info(f"Metrics:\n{metrics}")


if __name__ == "__main__":
    main()
