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
Make sure that you have [conda](https://docs.conda.io/en/latest/miniconda.html) or [pip](https://pip.pypa.io/en/stable/installing/) installed.

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