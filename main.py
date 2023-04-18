"""This project is a re-implementation of the MegaDepth pipeline."""

import datetime
import logging
import time

from megadepth.metrics.metadata import collect_metrics
from megadepth.utils.constants import Matcher, ModelType
from megadepth.utils.setup import DataPaths, setup
from megadepth.visualization.view_sparse_model import create_movie


def main():
    """Run the mega depth pipeline."""
    start = time.time()

    args = setup()
    paths = DataPaths(args)

    # create pipeline
    if args.colmap:
        from megadepth.pipelines.colmap import ColmapPipeline

        pipeline = ColmapPipeline(args)
    elif args.matcher == Matcher.LOFTR.value:
        from megadepth.pipelines.loftr import LoftrPipeline

        pipeline = LoftrPipeline(args)
    else:
        from megadepth.pipelines.hloc import HlocPipeline

        pipeline = HlocPipeline(args)

    if args.evaluate:
        pipeline.align_with_baseline()
        collect_metrics(paths, args, ModelType.SPARSE)
        create_movie(paths)
        return

    # run pipeline
    pipeline.preprocess()
    pipeline.get_pairs()
    pipeline.extract_features()
    pipeline.match_features()
    pipeline.sfm()
    pipeline.refinement()
    pipeline.mvs()  # not implemented yet
    pipeline.cleanup()  # not implemented yet

    # alterative
    # pipeline.run() # -> run all steps

    collect_metrics(paths, args, model_type=ModelType.SPARSE)
    create_movie(paths)

    end = time.time()
    logging.info(f"Total time: {datetime.timedelta(seconds=end - start)}")


if __name__ == "__main__":
    main()
