"""This project is a re-implementation of the MegaDepth pipeline."""

import datetime
import logging
import time

from megadepth.metrics.metadata import collect_metrics
from megadepth.pipeline import Pipeline
from megadepth.utils.constants import ModelType
from megadepth.utils.utils import DataPaths, setup


def main():
    """Run the mega depth pipeline."""
    start = time.time()

    args = setup()
    paths = DataPaths(args)

    if args.evaluate:
        collect_metrics(paths, args, ModelType.SPARSE)
        return

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
    # pipeline.run() # -> run all steps

    collect_metrics(paths, args, model_type=ModelType.SPARSE)

    end = time.time()
    logging.info(f"Total time: {datetime.timedelta(seconds=end - start)}")


if __name__ == "__main__":
    main()
