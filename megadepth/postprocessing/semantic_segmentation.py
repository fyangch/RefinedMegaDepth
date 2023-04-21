"""Functions that implement the semantic segmentation steps for the cleanup.."""
import numpy as np
import torch
from mit_semseg.models import ModelBuilder, SegmentationModule
from PIL import Image
from torchvision import transforms

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
    encoder = ModelBuilder.build_encoder("hrnetv2", fc_dim=720)
    decoder = ModelBuilder.build_decoder("c1", fc_dim=720, use_softmax=True)

    crit = torch.nn.NLLLoss(ignore_index=-1)
    model = SegmentationModule(encoder, decoder, crit)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    return model


def get_segmentation_map(image: Image, model: SegmentationModule) -> np.ndarray:
    """Extract the segmentation map from the image.

    Args:
        image (Image): Undistorted image.
        model (SegmentationModule): Segmentation model.

    Returns:
        np.ndarray: Predicted segmentation map with shape (height, width).
    """
    img_data = pil_to_tensor(image)
    output_size = img_data.shape[1:]

    if torch.cuda.is_available():
        batch = {"img_data": img_data[None].cuda()}
    else:
        batch = {"img_data": img_data[None]}

    with torch.no_grad():
        scores = model(batch, segSize=output_size)

    _, pred = torch.max(scores, dim=1)
    return pred.cpu()[0].numpy()
