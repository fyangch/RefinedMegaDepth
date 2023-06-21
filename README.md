# RefinedMegaDepth

MegaDepth, a dataset of unstructured images featuring popular tourist landmarks, was introduced in 2018. By leveraging structure from motion (SfM) and multi-view stereo (MVS) techniques along with data cleaning methods, MegaDepth generates camera poses and depth maps for each image. Nonetheless, the outcomes suffer from limitations like degenerate camera poses, incomplete depth maps, and inaccuracies caused by unregistered images or noise in the pipeline. Despite these flaws, MegaDepth has become an industry-standard dataset for training various computer vision models, including but not limited to single-view depth estimation, local features, feature matching, and multi-view refinement. This is primarily due to the diversity of scenes, occlusions, and appearance changes captured in the dataset, enabling the models to generalize well.
Our project aims to systematically address these problems to establish a refined MegaDepth ground-truth (GT) pipeline using recent deep learning-based methods such as Hierarchical Localization (hloc) and Pixel-Perfect Structure-from-Motion (pixSfM).

# Team
| Name                 | Email                   |
| -------------------- | ----------------------- |
| **Andri Horat**      | horatan@student.ethz.ch |
| **Alexander Veicht** | veichta@student.ethz.ch |
| **Deep Desai**       | ddesai@student.ethz.ch  |
| **Felix Yang**       | fyang@student.ethz.ch   |

# Setup



## Linux
Make sure that you have [conda](https://docs.conda.io/en/latest/miniconda.html) or [pip](https://pip.pypa.io/en/stable/installing/) installed as well as the follwoing dependencies:
- [pycolmap](https://github.com/colmap/pycolmap)
- [hloc](https://github.com/cvg/Hierarchical-Localization)
- [PixSfM](https://github.com/cvg/pixel-perfect-sfm)

If you have access to euler, follow the instruction from this repository to install (py-)colmap, hloc, and pixSfM:
https://github.com/Phil26AT/3DV_euler

### Pip
Make sure that you have python=3.9 installed. Create a new virtual environment using 
```bash
python3 -m venv venv
```

Activate the environment:
```bash
source venv/bin/activate
```

Install the dependencies:
```bash
make install_pip
```

### Conda
Crate a new conda environment using python=3.9:
```bash
conda create -n megadepth python=3.9
```

Activate the environment:
```bash
conda activate megadepth
```

Install the dependencies:
```bash
make install_conda
```

## Apple M1
Make sure you have [homebrew](https://brew.sh/) as well as [miniforge](https://github.com/conda-forge/miniforge) installed. Create a new conda environment using python=3.9:
```bash
conda create -n megadepth python=3.9
```

Activate the environment:
```bash
conda activate megadepth
```

Install the dependencies:
```bash
make install_m1
```

Note that this will not work with pixsfm.

## Install repository

```pip install -e .```

Maybe we also need to do this:
1. Install `build` and `twine` packages from pip:
   ```shell
   pip install build twine 
   ``` 
2. Build the package:
   ```shell
   python -m build
   ``` 

## Data
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
The pipeline will read the scene images from folder and create the following folders for the outputs:
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


# Running the pipeline
Download a small sample dataset from [here](https://polybox.ethz.ch/index.php/s/gQnwxa68aJQTaAW) or using the following command:
```bash
wget https://polybox.ethz.ch/index.php/s/gQnwxa68aJQTaAW/download -O data/south-building.zip
unzip data/south-building.zip -d data
rm data/south-building.zip
```
The pipeline can be run using the following command:
```bash
python main.py --data_path <path_to_data> --scene <scene_name> --low_memory
```
To run on the provided sample data, use the following command:
```bash
python main.py --data_path data --scene south-building --low_memory
```
The `--low_memory` flag will reduce the memory consumption by using the [low_memory.yaml](https://github.com/cvg/pixel-perfect-sfm/blob/55b10155587ca8eed8324c11f04b2ec9decc31d2/pixsfm/configs/low_memory.yaml) config file for pixSfM. Omit this flag if you have enough memory available.