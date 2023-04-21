"""Functions that implement the semantic segmentation steps for the cleanup.."""
from typing import Tuple

import torch
from PIL import Image
from torch import nn
from transformers import BeitFeatureExtractor, BeitForSemanticSegmentation


def get_segmentation_model() -> Tuple[BeitFeatureExtractor, BeitForSemanticSegmentation]:
    """Return the BEiT segmentation model (pre-trained on MIT ADE20K).

    More information about the model:
        - https://huggingface.co/microsoft/beit-base-finetuned-ade-640-640
        - https://github.com/microsoft/unilm/tree/master/beit

    Returns:
        The feature extractor and segmentation model.
    """
    feature_extractor = BeitFeatureExtractor.from_pretrained(
        "microsoft/beit-base-finetuned-ade-640-640"
    )
    model = BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")

    return feature_extractor, model


def get_segmentation_map(
    image: Image, feature_extractor: nn.Module, model: nn.Module
) -> torch.Tensor:
    """Extract the segmentation map from the image.

    Args:
        image (Image): Undistorted image.
        feature_extractor (nn.Module): BEiT feature extractor.
        model (nn.Module): BEiT segmentation model.

    Returns:
        torch.Tensor: Predicted segmentation map.
    """
    # TODO: implemtent this function

    raise NotImplementedError()
