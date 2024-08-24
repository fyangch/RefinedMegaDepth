"""Preprocessing functions for the pipeline."""

import logging
import os

import albumentations  # TODO: Remove dependency
import cv2
import numpy as np
import pycolmap
import torch
from check_orientation.pre_trained_models import create_model  # TODO: Remove dependency
from hloc.reconstruction import create_empty_db, get_image_ids, import_images
from omegaconf import DictConfig
from tqdm import tqdm

from megadepth.utils.io import load_image


def remove_problematic_images(paths: DictConfig) -> None:
    """Remove corrupted and other problematic images.

    Args:
        paths (DictConfig): Data paths.
    """
    logging.info("Deleting problematic images...")
    # create dummy database and try to import all images
    database = paths.data / "tmp_database.db"
    create_empty_db(database)
    import_images(paths.images, database, pycolmap.CameraMode.AUTO)
    image_ids = get_image_ids(database)

    logging.debug(f"Successfully imported {len(image_ids)} valid images.")

    # delete image files that were not successfully imported
    image_fns = [fn for fn in os.listdir(paths.images) if fn not in image_ids]
    for fn in image_fns:
        path = paths.images / fn
        logging.info(f"Deleting invalid image at {path}")
        path.unlink()

    # delete dummy database
    database.unlink()


def rotate_images(paths: DictConfig) -> None:
    """Rotate images if they are incorrectly oriented.

    Args:
        paths (DictConfig): Data paths.
    """
    logging.info("Rotating images...")

    model = create_model("swsl_resnext50_32x4d")
    model.eval()

    transform = albumentations.Compose(
        [albumentations.Resize(height=224, width=224), albumentations.Normalize()]
    )
    image_paths = [paths.images / fn for fn in os.listdir(paths.images)]

    n_rotated = 0
    for path in tqdm(image_paths, desc="Rotating images...", ncols=80):
        image = transform(image=load_image(path))["image"]
        tensor = torch.Tensor(image).permute(2, 0, 1).float() / 255
        with torch.no_grad():
            pred = model(tensor.unsqueeze(dim=0)).numpy().flatten()

        orientation = np.argmax(pred)
        if orientation == 0:
            continue

        n_rotated += 1

        # rotate image to fix orientation
        original_img = cv2.imread(str(path))
        if orientation == 1:
            rotated_img = cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)
            logging.debug(f"Rotating image at {path} by 90°")
        elif orientation == 2:
            rotated_img = cv2.rotate(original_img, cv2.ROTATE_180)
            logging.debug(f"Rotating image at {path} by 180°")
        else:
            rotated_img = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            logging.debug(f"Rotating image at {path} by 270°")

        cv2.imwrite(str(path), rotated_img)

    logging.info(f"Rotated {n_rotated} ({n_rotated / len(image_paths) * 100:.2f}%) images.")
