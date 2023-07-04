"""Functions to compute the nighttime metric."""

import os
from pathlib import Path
from typing import Union

import dnc
import pandas as pd
import pycolmap
import torch
from dnc.inference import infer_from_file
from tqdm import tqdm


def run_day_night_classification(
    reconstruction: pycolmap.Reconstruction, image_dir: Union[Path, str]
) -> pd.DataFrame:
    """Run day-night classification on all registered images.

    Args:
        reconstruction (pycolmap.Reconstruction): Sparse reconstruction.
        image_dir (Union[Path, str]): Path to the image directory of the scene.

    Returns:
        pd.DataFrame: Specifies for each registered image whether it is a nighttime image.
    """
    # set up data frame
    image_fns = [img.name for img in reconstruction.images.values()]
    df = pd.DataFrame(index=image_fns)
    df["is_night"] = False

    # run classification model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module_path = Path(dnc.__path__[0])
    weights = os.path.join(module_path.parent, "models", "resnet18.pt")
    for fn in tqdm(image_fns, desc="Running day-night classification...", ncols=80):
        pred = infer_from_file(img_path=os.path.join(image_dir, fn), weights=weights, device=device)
        df.loc[fn, "is_night"] = pred["night"] >= 0.5

    return df
