"""Implementation of the segmentation models.

If we use the pre-trained segmentation models from http://sceneparsing.csail.mit.edu/model/pytorch/
we also need to include the corresponding code, see
https://github.com/CSAILVision/semantic-segmentation-pytorch/tree/master/mit_semseg/models
"""
from torch import nn


def get_segmentation_model(model: str) -> nn.Module:
    """Return the chosen segmentation model (pre-trained on MIT ADE20K).

    Args:
        model: The name of the segmentation model to use.

    Returns:
        The segmentation model.
    """
    if model.lower() == "abcdef":
        raise NotImplementedError()
    elif model.lower() == "ghijk":
        raise NotImplementedError()
    else:
        raise ValueError(f"Invalid segmentation model: {model}")
