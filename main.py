"""This project is a re-implementation of the MegaDepth pipeline."""
import logging

import pycolmap
from mega_depth.utils.utils import setup


def main():
    """Run the mega depth pipeline."""
    args = setup()

    # TODO: add code here.
    logging.info("Hello world!")

    img_path = f"data/00_raw/{args.scene}"
    db_path = f"data/01_processed/{args.scene}.db"
    sparse_path = f"data/02_sparse_model/{args.scene}"

    pycolmap.extract_features(db_path, img_path)
    pycolmap.match_exhaustive(db_path)
    maps = pycolmap.incremental_mapping(db_path, img_path, sparse_path)
    maps[0].write(sparse_path)


if __name__ == "__main__":
    main()
