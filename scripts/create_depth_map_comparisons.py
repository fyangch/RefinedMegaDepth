"""Script to create comparisions between megadepth and our depth maps."""
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image
from tqdm import tqdm

from megadepth.utils.io import load_depth_map


def plot_images(images: list, titles: list, path=None) -> None:
    """Plot image and depth maps with colormap from megadepth."""
    fig = plt.figure(figsize=(15, 15))
    for i in range(len(images)):
        img = np.array(images[i], float) / 255
        fig.add_subplot(1, len(images), i + 1)
        plt.axis("off")
        plt.title(titles[i])
        if len(img.shape) == 3:
            plt.imshow(images[i])
            continue
        amin, amax = np.quantile(img[img > 0], [0.01, 0.985])
        if amin == amax:  # if ordinal labels
            amin, amax = 0, np.max(img)
        img[img == 0] = np.nan
        nan_mask = np.zeros_like(img)
        nan_mask[~np.isnan(img)] = np.nan
        plt.imshow(img, cmap="jet", norm=Normalize(amin, amax), interpolation="nearest")
        plt.imshow(nan_mask, cmap="gray", interpolation="nearest")
    if path is None:
        plt.show()
    else:
        fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def get_all_dense_scenes():
    """Find all scenes that have a dense reconstruction."""
    return glob.glob(
        r"/cluster/project/infk/courses/252-0579-00L/group01/scenes/*/\
            dense/superpoint_max-superglue-netvlad-50-KA+BA/stereo/depth_maps/",
        recursive=True,
    )


def get_image_path(scene, img_name):
    """Compute path on cluster for given scene and image_name."""
    return (
        rf"/cluster/project/infk/courses/252-0579-00L/group01/scenes/{scene}/images/{img_name}.jpg"
    )


def get_mega_depth_map_path(scene, img_name):
    """Compute path to depth map on cluster for given scene and image_name."""
    return rf"/cluster/project/infk/courses/252-0579-00L/group01/\
undistorted_md/phoenix/S6/zl548/MegaDepth_v1/{scene}/dense0/depths/{img_name}.h5"


def get_our_raw_depth_map_path(scene, img_name):
    """Compute path to depth map on cluster for given scene and image_name."""
    return rf"/cluster/project/infk/courses/252-0579-00L/group01/scenes/{scene}\
        /dense/superpoint_max-superglue-netvlad-50-KA+BA/stereo/\
            depth_maps/{img_name}.jpg.geometric.bin"


def get_our_filtered_depth_map_path(scene, img_name):
    """Compute path to depth map on cluster for given scene and image_name."""
    return rf"/cluster/project/infk/courses/252-0579-00L/group01/scenes/{scene}\
        /results/superpoint_max-superglue-netvlad-50-KA+BA/depth_maps/{img_name}.jpg.npy"


def get_img_name(path):
    """Extract image name from file_path."""
    file_name = os.path.basename(path)
    img_name = str.split(file_name, ".")[0]
    return img_name


def load(file_path):
    """Load image file and depth maps."""
    _, ext = os.path.splitext(file_path)
    # print(ext)
    if ext == ".npy":
        return np.load(file_path)
    if ext == ".h5":
        with h5py.File(file_path, "r") as f:
            # List all the keys in the file
            # print("Keys: %s" % f.keys())
            # Get the dataset
            dataset = f["depth"]
            # Get the data from the dataset
            data = dataset[:]
            return data
    if ext == ".bin":
        return load_depth_map(file_path)
    if ext == ".jpg":
        return Image.open(file_path).convert("RGB")


paths = glob.glob(get_our_raw_depth_map_path("*", "*"), recursive=True)
img_names = list(map(get_img_name, paths))

# enumerate all scenes that have a dense reconstruction
scene = "0229"

paths_raw = glob.glob(get_our_raw_depth_map_path(scene, "*"), recursive=True)
names_raw = set(map(get_img_name, paths_raw))
paths_filt = glob.glob(get_our_filtered_depth_map_path(scene, "*"), recursive=True)
names_filt = set(map(get_img_name, paths_filt))
paths_mega = glob.glob(get_mega_depth_map_path(scene, "*"), recursive=True)
names_mega = set(map(get_img_name, paths_mega))

print("raw, filtered, mega: ", len(names_raw), len(names_filt), len(names_mega))


difference_filt = names_raw.difference(names_filt)
difference_mega = names_filt.difference(names_mega)
intersecion = names_raw.intersection(names_filt).intersection(names_mega)
print(len(difference_filt), len(difference_mega), len(intersecion))

intersection_names = list(intersecion)
pixel_count = np.zeros(len(intersecion))
coverage_raw = np.zeros(len(intersecion))
coverage_filt = np.zeros(len(intersecion))
coverage_mega = np.zeros(len(intersecion))

for i, name in enumerate(tqdm(intersection_names)):
    depth_raw = load(get_our_raw_depth_map_path(scene, name))
    depth_filt = load(get_our_filtered_depth_map_path(scene, name))
    depth_mega = load(get_mega_depth_map_path(scene, name))
    coverage_raw[i] = np.count_nonzero(depth_raw)
    coverage_filt[i] = np.count_nonzero(depth_filt)
    coverage_mega[i] = np.count_nonzero(depth_mega)
    pixel_count[i] = depth_raw.size
    pixel_count_mega = depth_mega.size

coverage_raw /= pixel_count
coverage_filt /= pixel_count
coverage_mega /= pixel_count_mega


# cherry picking
cherry_boxes = [
    np.argsort(coverage_filt)[:10],
    np.argsort(coverage_mega)[:10],
    np.argsort(coverage_filt - coverage_mega)[:10],
    np.argsort(coverage_mega - coverage_filt)[:10],
    np.random.permutation(len(intersecion))[:10],
]
boxes = [
    "best_filtered",
    "best_mega",
    "best_improvement",
    "worst_improvement",
    "random_selection",
]

for cherries, box in zip(cherry_boxes, boxes):
    for cherry in cherries:
        img_name = intersection_names[cherry]
        img = load(get_image_path(scene, img_name))
        depth_raw = load(get_our_raw_depth_map_path(scene, img_name))
        depth_filt = load(get_our_filtered_depth_map_path(scene, img_name))
        depth_mega = load(get_mega_depth_map_path(scene, img_name))
        plot_images(
            [img, depth_raw, depth_filt, depth_mega],
            ["Image", "Raw", "Filtered", "Megadepth"],
            path=f"./plots/{scene}_{img_name}_comp_{box}.jpg",
        )


fig = plt.figure(figsize=(15, 15))
plt.xlabel("raw")
plt.ylabel("mega")
plt.scatter(coverage_raw, coverage_mega)
fig.savefig(f"./plots/{scene}_raw_vs_mega.jpg", dpi=600, bbox_inches="tight")
plt.close(fig)

fig = plt.figure(figsize=(15, 15))
plt.xlabel("raw")
plt.ylabel("filtered")
plt.scatter(coverage_raw, coverage_filt)
fig.savefig(f"./plots/{scene}_raw_vs_filtered.jpg", dpi=600, bbox_inches="tight")
plt.close(fig)

fig = plt.figure(figsize=(15, 15))
plt.xlabel("filtered")
plt.ylabel("mega")
plt.scatter(coverage_filt, coverage_mega)
fig.savefig(f"./plots/{scene}_filtered_vs_mega.jpg", dpi=600, bbox_inches="tight")
plt.close(fig)

fig = plt.figure(figsize=(15, 15))
plt.xlabel("raw")
plt.ylabel("mega")
plt.scatter(coverage_raw, coverage_mega)
fig.savefig(f"./plots/{scene}_raw_vs_mega.jpg", dpi=600, bbox_inches="tight")
plt.close(fig)

fig = plt.figure(figsize=(15, 15))
plt.xlabel("raw")
plt.ylabel("filtered")
plt.scatter(coverage_raw, coverage_filt)
fig.savefig(f"./plots/{scene}_raw_vs_filtered.jpg", dpi=600, bbox_inches="tight")
plt.close(fig)

fig = plt.figure(figsize=(15, 15))
plt.title("Depth map coverage for same images")
plt.hist(coverage_raw, alpha=0.5)
plt.hist(coverage_filt, alpha=0.5)
plt.hist(coverage_mega, alpha=0.5)
plt.legend(["Raw", "Filtered", "MegaDepth"])
fig.savefig(f"./plots/{scene}_histogram.jpg", dpi=600, bbox_inches="tight")
plt.close(fig)
