"""Dataset definition."""
import glob

import numpy as np
import torch

from megadepth.utils.io import load


def get_mega_image_path(scene, img_name):
    """Compute path to image on cluster for given scene and image_name."""
    return (
        rf"/cluster/project/infk/courses/252-0579-00L/group01/"
        rf"undistorted_md/phoenix/S6/zl548/MegaDepth_v1/{scene}/dense0/imgs/{img_name}.jpg"
    )


def get_mega_depth_map_path(scene, img_name):
    """Compute path to depth map on cluster for given scene and image_name."""
    return (
        rf"/cluster/project/infk/courses/252-0579-00L/group01/"
        rf"undistorted_md/phoenix/S6/zl548/MegaDepth_v1/{scene}/dense0/depths/{img_name}.h5"
    )


def get_our_raw_depth_map_path(scene, img_name):
    """Compute path to depth map on cluster for given scene and image_name."""
    return (
        rf"/cluster/project/infk/courses/252-0579-00L/group01/scenes/{scene}"
        r"/dense/superpoint_max-superglue-netvlad-50-KA+BA/stereo/"
        rf"depth_maps/{img_name}.jpg.geometric.bin"
    )


def get_our_filtered_depth_map_path(scene, img_name):
    """Compute path to depth map on cluster for given scene and image_name."""
    return (
        rf"/cluster/project/infk/courses/252-0579-00L/group01/scenes/{scene}"
        rf"/results/superpoint_max-superglue-netvlad-50-KA+BA/depth_maps/{img_name}.jpg.npy"
    )


def get_our_image_path(scene, img_name):
    """Compute path on cluster for given scene and image_name."""
    return (
        rf"/cluster/project/infk/courses/252-0579-00L/group01/scenes/{scene}/"
        rf"dense/superpoint_max-superglue-netvlad-50-KA+BA/images/{img_name}.jpg"
    )


class DepthDataset:
    """Depth dataset."""

    def __init__(self, scenes=["0229", "0015"]):
        """Select which scenes to use."""
        self.img_list = []
        self.depth_list = []

        for scene in scenes:
            self.img_list += glob.glob(get_our_image_path(scene, "*"), recursive=True)
            self.depth_list += glob.glob(
                get_our_filtered_depth_map_path(scene, "*"), recursive=True
            )

    def __len__(self):
        """Number of images."""
        return len(self.img_list)

    def __getitem__(self, idx):
        """Get Image and Label."""
        image = load(self.img_list[idx])
        image = torch.tensor(np.array(image), dtype=float)
        image = image.permute((0, 3, 1, 2))
        image = image.unsqueeze(0)  # ahape -> [1,3,h,w]

        depth = load(self.depth_list[idx])
        depth = torch.tensor(depth, dtype=float)
        depth = depth.unsqueeze(0).unsqueeze(0)  # shape -> [1,1,h,w]

        # stack = torch.stack((image, depth),axis=0)
        # apply geometric transform
        # image, depth = stack[0:1], stack[1:2]

        return image, depth


if __name__ == "__main__":
    dataset = DepthDataset()
    print(len(dataset))
    img, depth = dataset[0]
    print(img.shape)
    print(depth.shape)
