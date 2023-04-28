"""Functions that implement the semantic segmentation steps for the cleanup.."""
import os
from urllib.request import urlretrieve

import cv2
import numpy as np
import PIL
import torch
from mit_semseg.models import ModelBuilder, SegmentationModule
from torchvision import transforms

encoder_url = (
    "http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-hrnetv2-c1/encoder_epoch_30.pth"
)
decoder_url = (
    "http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-hrnetv2-c1/decoder_epoch_30.pth"
)

pil_to_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def get_segmentation_model() -> SegmentationModule:
    """Return the HRNetV2 segmentation model (pre-trained on MIT ADE20K).

    More information: https://github.com/CSAILVision/semantic-segmentation-pytorch

    Returns:
        SegmentationModule: The HRNetV2 segmentation model.
    """
    # download pre-trained weights if necessary
    encoder_path = os.path.join("pretrained", "hrnetv2", "encoder_epoch_30.pth")
    decoder_path = os.path.join("pretrained", "hrnetv2", "decoder_epoch_30.pth")
    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        os.makedirs(os.path.join("pretrained", "hrnetv2"), exist_ok=True)
        urlretrieve(encoder_url, encoder_path)
        urlretrieve(decoder_url, decoder_path)

    encoder = ModelBuilder.build_encoder("hrnetv2", fc_dim=720, weights=encoder_path)
    decoder = ModelBuilder.build_decoder("c1", fc_dim=720, use_softmax=True, weights=decoder_path)

    crit = torch.nn.NLLLoss(ignore_index=-1)
    model = SegmentationModule(encoder, decoder, crit)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    return model


def get_segmentation_map(
    image: PIL.Image, model: SegmentationModule, max_size: int = 1024
) -> np.ndarray:
    """Extract the segmentation map from the image.

    Args:
        image (PIL.Image): Undistorted image.
        model (SegmentationModule): Segmentation model.
        max_size (int, optional): Maximum width/height for the segmentation. Defaults to 1024.

    Returns:
        np.ndarray: Predicted segmentation map with the original image size.
    """
    original_size = image.size

    # downscale image to such that the largest dimension is max_size
    if max(image.size) > max_size:
        scale = max_size / max(image.size)
        new_size = tuple(int(x * scale) for x in image.size)
        image = image.resize(new_size, PIL.Image.LANCZOS)

    # get segmentation map
    img_data = pil_to_tensor(image)
    output_size = img_data.shape[1:]

    if torch.cuda.is_available():
        batch = {"img_data": img_data[None].cuda()}
    else:
        batch = {"img_data": img_data[None]}

    with torch.no_grad():
        scores = model(batch, segSize=output_size)

    _, pred = torch.max(scores, dim=1)
    segmentation_map = pred.cpu()[0].numpy().astype(np.uint8)

    # upscale segmentation map to original image size
    return cv2.resize(segmentation_map, dsize=original_size, interpolation=cv2.INTER_NEAREST)
