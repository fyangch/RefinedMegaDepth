import numpy as np
from torch import nn


def get_segmentation_model(model: str) -> nn.Module:
    """ Return the chosen segmentation model (pre-trained on MIT ADE20K). """
    if model.lower() == "abcdef":
        raise NotImplementedError()
    elif model.lower() == "ghijk":
        raise NotImplementedError()
    else:
        raise ValueError(f"Invalid segmentation model: {model}")
    

def read_depth_map(path: str) -> np.ndarray:
    """
    Read depth map from the given path and return it as a numpy array.
    This function was copied from: https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_dense.py
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()
