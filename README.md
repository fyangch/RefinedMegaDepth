<p align="center">
  <h1 align="center"><ins>Improving the Ground-Truth:</ins><br>MegaDepth in 2023</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/alexander-veicht/">Alexander Veicht</a>
    ·
    <a href="https://www.linkedin.com/in/felix-yang-ch/">Felix Yang</a>
    ·
    <a href="https://www.linkedin.com/in/andri-horat/de">Andri Horat</a>
    ·
    <a href="https://www.linkedin.com/in/deep-desai-35943b196">Deep Desai</a>
    ·
    <a href="https://www.linkedin.com/in/philipplindenberger/">Philipp Lindenberger</a>
  </p>
  <h2 align="center">
    <a href="report.pdf" align="center">Report</a> | <a href="poster.pdf" align="center">Poster</a>
  </h2>
  
</p>
<p align="center">
    <a><img src="assets/teaser.png" alt="example" width=80%></a>
    <br>
    <em>
      Our reimplemented and improved pipeline for MegaDepth generates high-quality depth maps 
      <br>
      and camera poses for unstructured images of popular tourist landmarks.
    </em>
</p>

MegaDepth, a dataset of unstructured images featuring popular tourist landmarks, was introduced in 2018. By leveraging structure from motion (SfM) and multi-view stereo (MVS) techniques along with data cleaning methods, MegaDepth generates camera poses and depth maps for each image. Nonetheless, the outcomes suffer from limitations like degenerate camera poses, incomplete depth maps, and inaccuracies caused by unregistered images or noise in the pipeline. Despite these flaws, MegaDepth has become an industry-standard dataset for training various computer vision models, including but not limited to single-view depth estimation, local features, feature matching, and multi-view refinement. This is primarily due to the diversity of scenes, occlusions, and appearance changes captured in the dataset, enabling the models to generalize well.
Our project aims to systematically address these problems to establish a refined MegaDepth ground-truth (GT) pipeline using recent methods such as the [hloc](https://github.com/cvg/Hierarchical-Localization) and [Pixel-Perfect Structure-from-Motion](https://github.com/cvg/pixel-perfect-sfm).


## Setup

Clone and install the repository by running the following commands:

```bash
git clone https://github.com/fyangch/RefinedMegaDepth.git
cd RefinedMegaDepth
pip install -e .
```

## First Reconstruction

Download the [south building](https://demuc.de/colmap/datasets) dataset and extract it to the `data` folder.

```bash
mkdir data
wget https://demuc.de/colmap/datasets/south-building.zip -O data/south-building.zip
unzip data/south-building.zip -d data
rm -rf data/south-building.zip data/south-building/sparse data/south-building/database.db
```

Run the following command to start the pipeline:
```bash
python -m megadepth.reconstruction scene=south-building
```


## Reconstructing Custom Scenes

The images are expected to be split by scenes and stored in the following format:
```
data
├── scene_1
│   ├── images
│   │   ├── 00000000.jpg
│   │   ├── 00000001.jpg
│   │   ├── ...
├── scene_2
│   ├── images
│   │   ├── 00000000.jpg
│   │   ├── 00000001.jpg
│   │   ├── ...
├── ...
```

You can simply run the the reconstruction pipeline by specifying the scene name:

```bash
python -m megadepth.reconstruction scene=scene_1
```

The pipeline will read the images from folder and create the following folders for the outputs:

```
data
├── scene_1
│   ├── images
│   ├── features
│   ├── matches
│   ├── sparse
│   ├── dense
│   ├── metrics
│   ├── results
├── scene_2
│   ├── ...
├── ...
```

## Next Steps

- [ ] Fix rotations
- [ ] Test mvs
- [ ] Remove dependencies (xarray, mit_semseg, ...)
- [ ] Check for licenses (segmentation models etc.)
