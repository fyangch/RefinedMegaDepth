"""This project is a re-implementation of the MegaDepth pipeline."""
import os
from pathlib import Path

import pycolmap
from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction

from megadepth.utils.utils import setup


def main():
    """Run the mega depth pipeline."""
    args = setup()

    # paths
    # TODO: potentially create path class / dict
    image_dir = Path(os.path.join(args.image_path, args.scene))
    # db_path = Path(os.path.join("data/01_processes", f"{args.scene}.db"))
    sparse_dir = Path(os.path.join("data/02_sparse_model", args.scene))
    sfm_pairs = Path(os.path.join(sparse_dir, "pairs-netvlad.txt"))
    sfm_dir = Path(os.path.join(sparse_dir, "sfm_superpoint+superglue"))
    mvs_dir = Path(os.path.join("data/03_dense_model", args.scene))

    # # configs
    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"]

    # find image pairs via image retrieval
    retrieval_path = extract_features.main(retrieval_conf, image_dir, sparse_dir)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)

    # extract features and match them
    feature_path = extract_features.main(feature_conf, image_dir, sparse_dir)
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf["output"], sparse_dir)

    # run colmap sfm
    # model = reconstruction.main(sfm_dir, image_dir, sfm_pairs, feature_path, match_path)
    _ = reconstruction.main(sfm_dir, image_dir, sfm_pairs, feature_path, match_path)

    # refinement using pixsfm

    # run mvs
    pycolmap.undistort_images(mvs_dir, sfm_dir, image_dir)
    pycolmap.patch_match_stereo(mvs_dir)  # requires compilation with CUDA
    pycolmap.stereo_fusion(mvs_dir / "dense.ply", mvs_dir)

    # cleanup


if __name__ == "__main__":
    main()
