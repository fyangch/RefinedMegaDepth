"""Model Pretrained on segmentation.

https://github.com/msminhas93/DeepLabv3FineTuning/blob/master/model.py

DeepLabv3 Model download and change the head for your prediction.
"""
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


def createDeepLabv3(outputchannels=1, input_size=512):
    """DeepLabv3 class with custom head.

    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.

    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = models.segmentation.deeplabv3_resnet50(weights=weights, progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()

    preprocess = weights.transforms(antialias=True)
    preprocess.resize_size = [input_size]
    return model, preprocess


def load_model(model_state_file):
    """Load file from path."""
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = models.segmentation.deeplabv3_resnet50(weights=weights, progress=True)
    model.classifier = DeepLabHead(2048, 37)
    # Set the model in training mode
    model.eval()
    preprocess = weights.transforms(antialias=True)

    state_dict = torch.load(model_state_file, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict["state_dict"])
    return model, preprocess


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torchvision import transforms as T

    from megadepth.utils.io import load
    from scripts.create_depth_map_comparisons import plot_image_overlay

    img_path = r"/home/andro/RefinedMegaDepth/data/south-building/dense/colmap/images/P1180141.JPG"
    depth_path = (
        r"/home/andro/RefinedMegaDepth/data/south-building/dense/"
        r"colmap/stereo/depth_maps/P1180141.JPG.geometric.bin"
    )
    image = load(img_path)
    depth = load(depth_path)
    plot_image_overlay([image], [depth], ["Depth"])

    image = torch.tensor(np.array(image), dtype=float)
    image = image.permute((2, 0, 1))
    image = image.unsqueeze(0)  # ahape -> [1,3,h,w]

    print(image.shape)
    size = 520
    crop = T.CenterCrop(size)

    depth = torch.tensor(depth, dtype=float)
    depth = depth.unsqueeze(0).unsqueeze(0)  # shape -> [1,1,h,w]

    depth = crop(depth)
    image = crop(image)
    print(image.shape)
    model, preprocess = createDeepLabv3(1, size)
    model.eval()
    batch = preprocess(image)
    print(batch.shape)
    output = model(batch)["out"].detach()

    print(output.shape)
    prediction = output.squeeze(0).squeeze(0)  # shape [1,1,h,w ] -> [h,w]
    plt.imshow(prediction)
    plt.show()
