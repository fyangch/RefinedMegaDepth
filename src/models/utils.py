from torch import nn


def get_segmentation_model(model: str) -> nn.Module:
    """ Return the chosen segmentation model (pre-trained on MIT ADE20K). """
    if model.lower() == "abcdef":
        raise NotImplementedError()
    elif model.lower() == "ghijk":
        raise NotImplementedError()
    else:
        raise ValueError(f"Invalid segmentation model: {model}")
