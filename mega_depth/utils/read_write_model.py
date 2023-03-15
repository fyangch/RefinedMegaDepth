"""TODO: Add docstring.

Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
      its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)
"""

# TODO: move everything to utils/io.py
import argparse
import collections
import os
import struct
from io import BufferedReader, BufferedWriter
from typing import Any, Tuple, Union

import numpy as np

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "BaseImage", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    """Image class with additional methods."""

    def qvec2rotmat(self) -> np.ndarray:
        """Transform the quaternion vector to a rotation matrix. TODO: check if this is correct.

        Returns:
            np.ndarray: Rotation matrix.
        """
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def read_next_bytes(
    fid: BufferedReader, num_bytes: int, format_char_sequence: str, endian_character: str = "<"
) -> tuple:
    """Read and unpack the next bytes from a binary file.

    Args:
        fid: File descriptor.
        num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
        format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
        endian_character: Any of {@, =, <, >, !}

    Returns:
        Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def write_next_bytes(
    fid: BufferedWriter,
    data: Union[tuple, list, Any],
    format_char_sequence: str,
    endian_character: str = "<",
) -> None:
    """Pack and write to a binary file.

    Args:
        fid (BufferedReader): File descriptor.
        data (tuple): Data to send, if multiple elements are sent at the same time,
        they should be encapsuled either in a list or a tuple.
        format_char_sequence (str): List of {c, e, f, d, h, H, i, I, l, L, q, Q}. Should be the same
        length as the data list or tuple
        endian_character (str, optional): Any of {@, =, <, >, !}. Defaults to "<".

    Returns:
        None: TODO: add description.
    """
    if isinstance(data, (list, tuple)):
        bytes_ = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes_ = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes_)


def read_cameras_text(path: str) -> dict:
    """Read cameras from a text file.

    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)

    Args:
        path (str): Path to the file.

    Returns:
        dict: Dictionary of cameras.
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id, model=model, width=width, height=height, params=params
                )
    return cameras


def read_cameras_binary(path_to_model_file: str) -> dict:
    """Read cameras from a binary file.

    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)

    Args:
        path_to_model_file (str): Path to the file.

    Returns:
        dict: Dictionary of cameras.
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
            )
            cameras[camera_id] = Camera(
                id=camera_id, model=model_name, width=width, height=height, params=np.array(params)
            )
        assert len(cameras) == num_cameras
    return cameras


def write_cameras_text(cameras: dict, path: str) -> None:
    """Write cameras to a text file.

    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)

    Args:
        cameras (dict): Dictionary of cameras.
        path (str): Path to the file.

    Returns:
        None: TODO: add description.
    """
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + f"# Number of cameras: {len(cameras)}\n"
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")


def write_cameras_binary(cameras: dict, path_to_model_file: str) -> None:
    """Write cameras to a binary file.

    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)

    Args:
        cameras (dict): Dictionary of cameras.
        path_to_model_file (str): Path to the file.
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")


def read_images_text(path: str) -> dict:
    """Read images from a text file.

    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)

    Args:
        path (str): Path to the file.

    Returns:
        dict: Dictionary of images.
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [tuple(map(float, elems[::3])), tuple(map(float, elems[1::3]))]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file: str) -> dict:
    """Read images from a binary file.

    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)

    Args:
        path_to_model_file (str): Path to the file.

    Returns:
        dict: Dictionary of images.
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(
                fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D
            )
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[::3])), tuple(map(float, x_y_id_s[1::3]))]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def write_images_text(images: dict, path: str) -> None:
    """Write images to a text file.

    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)

    Args:
        images (dict): Dictionary of images.
        path (str): Path to the file.
    """
    mean_observations = (
        sum((len(img.point3D_ids) for _, img in images.items())) / len(images) if images else 0
    )
    HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + f"# Number of images: {len(images)}, mean observations per image: {mean_observations}\n"
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for img in images.values():
            image_header = [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = [
                " ".join(map(str, [*xy, point3D_id]))
                for xy, point3D_id in zip(img.xys, img.point3D_ids)
            ]
            fid.write(" ".join(points_strings) + "\n")


def write_images_binary(images: dict, path_to_model_file: str) -> None:
    """Write images to a binary file.

    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)

    Args:
        images (dict): Dictionary of images.
        path_to_model_file (str): Path to the file.
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


def read_points3D_text(path: str) -> dict:
    """Read points3D from a text file.

    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)

    Args:
        path (str): Path to the file.

    Returns:
        dict: Dictionary of points3D.
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs,
                )
    return points3D


def read_points3D_binary(path_to_model_file: str) -> dict:
    """Read points3D from a binary file.

    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)

    Args:
        path_to_model_file (str): Path to the file.

    Returns:
        dict: Dictionary of points3D.
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length
            )
            image_ids = np.array(tuple(map(int, track_elems[::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def write_points3D_text(points3D: dict, path: str) -> None:
    """Write points3D to a text file.

    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)

    Args:
        points3D (dict): Dictionary of points3D.
        path (str): Path to the file.
    """
    mean_track_length = (
        sum((len(pt.image_ids) for _, pt in points3D.items())) / len(points3D) if points3D else 0.0
    )
    HEADER = (
        "# 3D point list with one line of data per point:\n"
        + "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        + f"# Number of points: {len(points3D)}, mean track length: {mean_track_length}\n"
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for pt in points3D.values():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = [
                " ".join(map(str, [image_id, point2D]))
                for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs)
            ]
            fid.write(" ".join(track_strings) + "\n")


def write_points3D_binary(points3D: dict, path_to_model_file: str) -> None:
    """Write points3D to a binary file.

    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)

    Args:
        points3D (dict): Dictionary of points3D.
        path_to_model_file (str): Path to the file.
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")


def detect_model_format(path: str, ext: str) -> bool:
    """Detect model format.

    Args:
        path (str): Path to the model.
        ext (str): Extension of the model files.

    Returns:
        bool: True if the model format is detected.
    """
    if (
        os.path.isfile(os.path.join(path, f"cameras{ext}"))
        and os.path.isfile(os.path.join(path, f"images{ext}"))
        and os.path.isfile(os.path.join(path, f"points3D{ext}"))
    ):
        print(f"Detected model format: '{ext}'")
        return True

    return False


def read_model(path: str, ext: str = "") -> Tuple[dict, dict, dict]:
    """Try to detect the model format and read the model.

    Args:
        path (str): Path to the model.
        ext (str, optional): Extension of the model files. Defaults to "".

    Raises:
        ValueError: If the model format is not in the list of supported formats.

    Returns:
        Tuple[dict, dict, dict]: Tuple of cameras, images, points3D.
    """
    if not ext:
        if detect_model_format(path, ".bin"):
            ext = ".bin"
        elif detect_model_format(path, ".txt"):
            ext = ".txt"
        else:
            raise ValueError("Provide model format: '.bin' or '.txt'")

    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, f"cameras{ext}"))
        images = read_images_text(os.path.join(path, f"images{ext}"))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, f"cameras{ext}"))
        images = read_images_binary(os.path.join(path, f"images{ext}"))
        points3D = read_points3D_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def write_model(cameras: dict, images: dict, points3D: dict, path: str, ext: str = ".bin") -> None:
    """Write the model to a file.

    Args:
        cameras (dict): The camera dictionary.
        images (dict): The image dictionary.
        points3D (dict): The points3D dictionary.
        path (str): Path to the model.
        ext (str, optional): Extension of the model files (can be .bin or .txt). Defaults to ".bin".
    """
    if ext == ".txt":
        write_cameras_text(cameras, os.path.join(path, f"cameras{ext}"))
        write_images_text(images, os.path.join(path, f"images{ext}"))
        write_points3D_text(points3D, os.path.join(path, "points3D") + ext)
    else:
        write_cameras_binary(cameras, os.path.join(path, f"cameras{ext}"))
        write_images_binary(images, os.path.join(path, f"images{ext}"))
        write_points3D_binary(points3D, os.path.join(path, "points3D") + ext)


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix. TODO: check if this is correct.

    Args:
        qvec (np.ndarray): Quaternion vector.

    Returns:
        np.ndarray: Rotation matrix.
    """
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion. TODO: check if this is correct.

    Args:
        R (np.ndarray): Rotation matrix.

    Returns:
        np.ndarray: Quaternion vector.
    """
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


# TODO: check if this is needed
def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Read and write COLMAP binary and text models")
    parser.add_argument("--input_model", help="path to input model folder")
    parser.add_argument(
        "--input_format", choices=[".bin", ".txt"], help="input model format", default=""
    )
    parser.add_argument("--output_model", help="path to output model folder")
    parser.add_argument(
        "--output_format", choices=[".bin", ".txt"], help="outut model format", default=".txt"
    )
    args = parser.parse_args()

    cameras, images, points3D = read_model(path=args.input_model, ext=args.input_format)

    print("num_cameras:", len(cameras))
    print("num_images:", len(images))
    print("num_points3D:", len(points3D))

    if args.output_model is not None:
        write_model(cameras, images, points3D, path=args.output_model, ext=args.output_format)


if __name__ == "__main__":
    main()
