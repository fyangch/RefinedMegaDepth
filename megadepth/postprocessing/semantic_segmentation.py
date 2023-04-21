"""Functions that implement the semantic segmentation steps for the cleanup.."""
from typing import Tuple

from torch import nn
from transformers import BeitFeatureExtractor, BeitForSemanticSegmentation


def get_segmentation_model() -> Tuple[nn.Module, nn.Module]:
    """Return the BEiT segmentation model (pre-trained on MIT ADE20K).

    More information about the model:
        - https://huggingface.co/microsoft/beit-base-finetuned-ade-640-640
        - https://github.com/microsoft/unilm/tree/master/beit

    Returns:
        Tuple[nn.Module, nn.Module]: The feature extractor and segmentation model.
    """
    feature_extractor = BeitFeatureExtractor.from_pretrained(
        "microsoft/beit-base-finetuned-ade-640-640"
    )
    model = BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")

    return feature_extractor, model
