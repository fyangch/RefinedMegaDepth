"""Preprocessing functions for the pipeline."""

import logging
import os

import albumentations as albu
import cv2
import numpy as np
import pycolmap
import torch
from check_orientation.pre_trained_models import create_model
from hloc.reconstruction import create_empty_db, get_image_ids, import_images
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import load_rgb
from omegaconf import DictConfig
from tqdm import tqdm


def delete_problematic_images(paths: DictConfig) -> None:
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

    transform = albu.Compose([albu.Resize(height=224, width=224), albu.Normalize()])
    image_paths = [paths.images / fn for fn in os.listdir(paths.images)]

    n_rotated = 0
    for path in tqdm(image_paths, desc="Rotating images...", ncols=80):
        image = transform(image=load_rgb(path))["image"]
        tensor = tensor_from_rgb_image(image)
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
            logging.info(f"Rotating image at {path} by 90°")
        elif orientation == 2:
            rotated_img = cv2.rotate(original_img, cv2.ROTATE_180)
            logging.info(f"Rotating image at {path} by 180°")
        else:
            rotated_img = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            logging.info(f"Rotating image at {path} by 270°")

        cv2.imwrite(str(path), rotated_img)

    logging.info(f"Rotated {n_rotated} ({n_rotated / len(image_paths) * 100:.2f}%) images.")
