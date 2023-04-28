"""Functions that implement the semantic segmentation steps for the cleanup.."""
import os
from abc import abstractmethod
from typing import Literal
from urllib.request import urlretrieve

import cv2
import numpy as np
import PIL
import torch
from mit_semseg.models import ModelBuilder, SegmentationModule
from torchvision import transforms
from transformers import BeitForSemanticSegmentation, BeitImageProcessor


class SegmentationModel:
    """Abstract class for segmentation models."""

    @abstractmethod
    def get_segmentation_map(self, image: PIL.Image) -> np.ndarray:
        """Extract the segmentation map from the image.

        Args:
            image (PIL.Image): Undistorted image.

        Returns:
            np.ndarray: Predicted segmentation map with the original image size.
        """
        pass


class HRNet(SegmentationModel):
    """HRNetV2 segmentation model.

    More information: https://github.com/CSAILVision/semantic-segmentation-pytorch
    """

    # pre-trained model weights
    encoder_url = (
        "http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-hrnetv2-c1/encoder_epoch_30.pth"
    )
    decoder_url = (
        "http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-hrnetv2-c1/decoder_epoch_30.pth"
    )

    # image preprocessing steps
    pil_to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    def __init__(self, max_size: int = 1024) -> None:
        """Initialize the model.

        Args:
            max_size (int, optional): Maximum width/height for the segmentation. Defaults to 1024.
        """
        self.max_size = max_size

        # download pre-trained weights if necessary
        encoder_path = os.path.join("pretrained", "hrnetv2", "encoder_epoch_30.pth")
        decoder_path = os.path.join("pretrained", "hrnetv2", "decoder_epoch_30.pth")
        if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
            os.makedirs(os.path.join("pretrained", "hrnetv2"), exist_ok=True)
            urlretrieve(HRNet.encoder_url, encoder_path)
            urlretrieve(HRNet.decoder_url, decoder_path)

        encoder = ModelBuilder.build_encoder("hrnetv2", fc_dim=720, weights=encoder_path)
        decoder = ModelBuilder.build_decoder(
            "c1", fc_dim=720, use_softmax=True, weights=decoder_path
        )

        crit = torch.nn.NLLLoss(ignore_index=-1)
        self.model = SegmentationModule(encoder, decoder, crit)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def get_segmentation_map(self, image: PIL.Image) -> np.ndarray:
        """Extract the segmentation map from the image.

        Args:
            image (PIL.Image): Undistorted image.

        Returns:
            np.ndarray: Predicted segmentation map with the original image size.
        """
        original_size = image.size

        # downscale image to such that the largest dimension is max_size
        if max(image.size) > self.max_size:
            scale = self.max_size / max(image.size)
            new_size = tuple(int(x * scale) for x in image.size)
            image = image.resize(new_size, PIL.Image.LANCZOS)

        # get segmentation map
        img_data = HRNet.pil_to_tensor(image)
        output_size = img_data.shape[1:]

        if torch.cuda.is_available():
            batch = {"img_data": img_data[None].cuda()}
        else:
            batch = {"img_data": img_data[None]}

        with torch.no_grad():
            scores = self.model(batch, segSize=output_size)

        _, pred = torch.max(scores, dim=1)
        segmentation_map = pred.cpu()[0].numpy().astype(np.uint8)

        # upscale segmentation map to original image size
        return cv2.resize(segmentation_map, dsize=original_size, interpolation=cv2.INTER_NEAREST)


class BEiT(SegmentationModel):
    """BEiT segmentation model.

    More information: https://huggingface.co/microsoft/beit-base-finetuned-ade-640-640
    """

    def __init__(self) -> None:
        """Initialize the model."""
        self.image_processor = BeitImageProcessor.from_pretrained(
            "microsoft/beit-base-finetuned-ade-640-640"
        )
        self.model = BeitForSemanticSegmentation.from_pretrained(
            "microsoft/beit-base-finetuned-ade-640-640"
        )

        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def get_segmentation_map(self, image: PIL.Image) -> np.ndarray:
        """Extract the segmentation map from the image.

        Args:
            image (PIL.Image): Undistorted image.

        Returns:
            np.ndarray: Predicted segmentation map with the original image size.
        """
        input = self.image_processor(image, return_tensors="pt").pixel_values
        if torch.cuda.is_available():
            input.cuda()

        with torch.no_grad():
            logits = self.model(input).logits
        logits = torch.nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear")

        _, pred = torch.max(logits, dim=1)
        return pred.cpu()[0].numpy().astype(np.uint8)


def get_segmentation_model(
    model: Literal["hrnet", "beit"], max_size: int = 1024
) -> SegmentationModel:
    """Return the HRNetV2 or BEiT segmentation model (pre-trained on MIT ADE20K).

    Args:
        model (Literal["hrnet", "beit"]): Which model to return.
        max_size (int, optional): Maximum width/height for the segmentation. Defaults to 1024.

    Returns:
        SegmentationModel: Segmentation model.
    """
    if model == "hrnet":
        return HRNet(max_size)
    elif model == "beit":
        return BEiT()
    else:
        raise ValueError(f"Invalid segmentation model: {model}")
