"""This project is a re-implementation of the MegaDepth pipeline."""

import argparse
import datetime
import logging
import time

from megadepth.metrics.metadata import collect_metrics
from megadepth.pipelines.pipeline import Pipeline
from megadepth.utils.constants import Matcher, ModelType
from megadepth.utils.setup import DataPaths, setup


def main():
    """Run the mega depth pipeline."""
    start = time.time()

    args = setup()
    paths = DataPaths(args)

    # create pipeline
    pipeline = get_pipeline(args)
    logging.info(f"Pipeline: {type(pipeline).__name__}")

    if args.evaluate:
        pipeline.align_with_baseline()
        collect_metrics(paths, args, ModelType.SPARSE)
        collect_metrics(paths, args, ModelType.REFINED)
        # collect_metrics(paths, args, ModelType.DENSE)
        return

    # run pipeline
    pipeline.run()

    collect_metrics(paths, args, model_type=ModelType.SPARSE)
    collect_metrics(paths, args, model_type=ModelType.REFINED)
    # collect_metrics(paths, args, model_type=ModelType.DENSE)

    end = time.time()
    logging.info(f"Total time: {datetime.timedelta(seconds=end - start)}")


def get_pipeline(args: argparse.Namespace) -> Pipeline:
    """Get pipeline based on arguments.

    Args:
        args: Arguments from command line.

    Returns:
        Pipeline: Pipeline to run.
    """
    if args.colmap:
        from megadepth.pipelines.colmap import ColmapPipeline

        return ColmapPipeline(args)
    elif args.matcher == Matcher.LOFTR.value:
        from megadepth.pipelines.loftr import LoftrPipeline

        return LoftrPipeline(args)
    elif "KA" in args.refinements:
        from megadepth.pipelines.pixsfm import PixSfMPipeline

        return PixSfMPipeline(args)
    else:
        from megadepth.pipelines.hloc import HlocPipeline

        return HlocPipeline(args)


if __name__ == "__main__":
    main()
