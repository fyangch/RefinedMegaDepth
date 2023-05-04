"""Training script."""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from megadepth.training.dataset import DepthDataset
from megadepth.training.model import createDeepLabv3


def loss_data(depth_pred, depth_gt):
    """Log loss."""
    log_depth_pred = torch.log(depth_pred)[depth_gt != 0]
    log_depth_gt = torch.log(depth_gt)[depth_gt != 0]
    return F.mse_loss(log_depth_pred, log_depth_gt)


def loss_grad(depth_pred, depth_gt):
    """Gradient loss."""
    return 0


def loss_ord(depth_pred, depth_gt):
    """Ordinal loss."""
    return 0


def loss_si(depth_pred, depth_gt, alpha=0.1, beta=0.1):
    """Scale invariant loss function."""
    return (
        loss_data(depth_pred, depth_gt)
        # + alpha * loss_grad(depth_pred, depth_gt)
        # + beta * loss_ord(depth_pred, depth_gt)
    )


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = DepthDataset(transforms=T.Compose([T.CenterCrop(200)]))
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    model, preprocess = createDeepLabv3(outputchannels=1, input_size=200)
    optimizer = torch.optim.Adam(model.parameters())

    train_epochs = 20
    log_freq = 1

    for epoch in range(train_epochs):
        model.eval()
        train_loss = 0.0
        for i, (input, target) in enumerate(train_loader):
            # Move input and target tensors to the device (CPU or GPU)
            input = input.to(device)
            target = target.to(device)

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(preprocess(input))["out"]

            loss = loss_si(output, target)

            log_depth_pred = torch.log(output)[target != 0]
            log_depth_gt = torch.log(target)[target != 0]
            loss = F.mse_loss(log_depth_pred, log_depth_gt)

            # Backward pass and update weights
            loss.backward()
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()
    if (i + 1) % log_freq == 0:
        print(
            f"Train Epoch: {epoch+1} [{i+1}/{len(train_loader)}]\t"
            f"Loss: {train_loss/log_freq:.4f}"
        )
