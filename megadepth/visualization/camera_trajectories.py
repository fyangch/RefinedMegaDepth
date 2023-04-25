"""Define camera trajectories for visualization."""

import numpy as np


def dist(t: float, zoom_in: bool = True) -> float:
    """Distance function for camera trajectory.

    Args:
        t (float): Time in [0, 1].
        zoom_in (bool): Whether to zoom in. Defaults to True.

    Returns:
        float: Distance.
    """
    return 9 - np.sin(t * np.pi) * 6 if zoom_in else 10


def rotation_angle(t: float) -> float:
    """Rotation function for camera trajectory.

    Args:
        t (float): Time in [0, 1].

    Returns:
        float: Rotation angle.
    """
    return 2 * np.pi * t


def surround_view(
    transform: np.ndarray, scale: float, num_frames: int = 300, zoom_in: bool = True
) -> np.ndarray:
    """Generate extrinsics for surround view animation.

    Args:
        transform (np.ndarray): PCA transform to align the camera poses.
        scale (float): Scale of the scene.
        num_frames (int): Number of frames in the animation. Defaults to 300.
        zoom_in (bool): Whether to zoom in during the animation. Defaults to True.

    Returns:
        np.ndarray: _description_
    """
    extrinsics = []
    phi = 120

    for i in range(num_frames):
        d = dist(i / num_frames, zoom_in)
        t = np.array([0, 0, d], dtype=np.float64) * scale
        # compute current turntable angle
        angle_theta = rotation_angle(i / num_frames)
        R_z = np.array(
            [
                [np.cos(angle_theta), -np.sin(angle_theta), 0],
                [np.sin(angle_theta), np.cos(angle_theta), 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        # rotate for tilt, 0 from bottom, 90 from side, 180 from top
        angle_phi = np.deg2rad(phi)
        R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_phi), -np.sin(angle_phi)],
                [0, np.sin(angle_phi), np.cos(angle_phi)],
            ],
            dtype=np.float64,
        )

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_x @ R_z
        extrinsic[:3, 3] = t
        # first transform to pca aligned space, and then apply rotation
        extrinsics.append(extrinsic @ transform)

    return extrinsics
