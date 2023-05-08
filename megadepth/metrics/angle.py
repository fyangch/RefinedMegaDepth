"""Coangle Matrix."""
import numpy as np
import pycolmap
from tqdm import tqdm


def angle(reconstruction: pycolmap.Reconstruction):
    """Computes the co-angle matrix.

    i,j element contains the angle between the pincipal axis of the images i,j.
    """
    images = reconstruction.images
    N = len(images)

    # compute overlap scores for each image pair
    scores = np.ones((N, N))
    for i, k1 in enumerate(tqdm(images.keys())):
        # load first image, camera and depth map
        image_1 = images[k1]
        principal_1 = image_1.transform_to_world(np.array([[0, 0, 1]]))[0]
        principal_1 = principal_1 / np.linalg.norm(principal_1)
        for j, k2 in enumerate(images.keys()):
            if i <= j:
                continue

            # load second image, camera and depth map
            image_2 = images[k2]
            principal_2 = image_2.transform_to_world(np.array([[0, 0, 1]]))[0]
            principal_2 = principal_2 / np.linalg.norm(principal_2)
            scores[i, j] = np.sum(principal_1 * principal_2)
            scores[j, i] = scores[i, j]
    return scores
