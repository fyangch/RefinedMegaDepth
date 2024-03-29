"""Visualize a sparse reconstruction from Colmap.

Example:
    python megadepth/visualizations/visualize_sparse.py --model_path /path/to/model_dir
"""

import argparse
import logging
import os
import shutil
import typing
from copy import deepcopy

import numpy as np
import open3d as o3d
import pycolmap
from tqdm import tqdm

from megadepth.utils.projections import get_camera_poses
from megadepth.visualization.view_projections import pca


def pcd_from_ply(path: str) -> o3d.geometry.PointCloud:
    """Load a point cloud from a PLY file.

    Args:
        path (str): Path to the PLY file.

    Returns:
        o3d.geometry.PointCloud: Open3D point cloud.
    """
    pcd = o3d.io.read_point_cloud(f"{path}/dense.ply")
    model = pycolmap.Reconstruction(f"{path}/sparse")
    align = pca(get_camera_poses(model))
    pcd.points = o3d.utility.Vector3dVector(align(np.asarray(pcd.points)))

    return pcd.voxel_down_sample(voxel_size=0.01)


def mesh_from_ply(path: str) -> o3d.geometry.TriangleMesh:
    """Load a mesh from a PLY file.

    Args:
        path (str): Path to the PLY file.

    Returns:
        o3d.geometry.TriangleMesh: Open3D mesh.
    """
    # Load the meshed-poisson.ply file
    return o3d.io.read_triangle_mesh(f"{path}/meshed-poisson.ply")


@typing.no_type_check
def render_frames(
    path: str,
    store_path: str,
    draw_cameras: bool = False,
    show_window: bool = True,
) -> None:
    """Rotate the view of the point cloud.

    Args:
        path (str): Path to the reconstruction.
        store_path (str): Path to store the frames of the animation.
        draw_cameras (bool, optional): Whether to draw the cameras. Defaults to True.
        show_window (bool, optional): Whether to open a window to show the animation. Defaults to
        False.
    """
    render_frames.idx = 0
    render_frames.vis = o3d.visualization.Visualizer()
    render_frames.store_path = store_path
    render_frames.pbar = tqdm(total=600)

    render_frames.extrinsics = fly_through()

    if not os.path.exists(store_path):
        os.makedirs(store_path)

    def rotate_view(vis: o3d.visualization.Visualizer) -> bool:
        """Rotate the view of the point cloud.

        Args:
            vis (o3d.visualization.Visualizer): Open3D visualizer.

        Returns:
            bool: Default return value.
        """
        ctr = vis.get_view_control()

        # Stop rotating after 360 degrees
        glb = render_frames
        glb.idx += 1

        if glb.idx < 600:
            camera = ctr.convert_to_pinhole_camera_parameters()
            camera.extrinsic = get_updated_extrinsics(glb.idx)
            # camera.extrinsic = glb.extrinsics[glb.idx]
            ctr.convert_from_pinhole_camera_parameters(camera)

        else:
            vis.close()

        # Store the frame in a plot
        frame_every = 2
        if glb.idx % frame_every == 0:
            filename = os.path.join(glb.store_path, f"{glb.idx//frame_every:04d}.png")
            vis.capture_screen_image(filename, do_render=True)

        # Update the progress bar
        glb.pbar.update(1)

        return False

    vis = render_frames.vis
    vis.create_window(visible=show_window)

    pcd = pcd_from_ply(path)
    vis.add_geometry(pcd)

    # if draw_cameras:
    #     add_cameras(rec, vis)

    opt = vis.get_render_option()
    opt.point_size = 1
    vis.register_animation_callback(rotate_view)
    vis.run()
    vis.destroy_window()

    logging.info("Done rendering frames.")


def add_cameras(rec: pycolmap.Reconstruction, vis: o3d.visualization.Visualizer) -> None:
    """Add cameras to the visualizer.

    Args:
        rec (pycolmap.Reconstruction): Sparse reconstruction from Colmap.
        vis (o3d.visualization.Visualizer): Open3D visualizer.
    """
    camera_lines = {
        camera.camera_id: o3d.geometry.LineSet.create_camera_visualization(
            camera.width,
            camera.height,
            camera.calibration_matrix(),
            np.eye(4),
            scale=0.1,
        )
        for camera in rec.cameras.values()
    }
    # Draw the frustum for each image
    for image in rec.images.values():
        T = np.eye(4)
        T[:3, :4] = image.inverse_projection_matrix()
        cam = deepcopy(camera_lines[image.camera_id]).transform(T)
        cam.paint_uniform_color([1.0, 0.0, 0.0])  # red
        vis.add_geometry(cam)


def fly_through(num_frames: int = 600) -> np.ndarray:
    """Fly through the point cloud.

    Args:
        num_frames (int, optional): Number of frames to render. Defaults to 600.

    Returns:
        np.ndarray: Camera extrinsics for each frame.
    """
    # Define the positions and orientations of the camera for each frame
    positions = np.zeros((num_frames, 3))
    orientations = np.zeros((num_frames, 3))

    extrinsics = np.zeros((num_frames, 4, 4))

    for i in range(num_frames):
        # Define the position of the camera for this frame
        positions[i, 0] = np.sin(i / num_frames * 2 * np.pi) * 0.5
        positions[i, 1] = np.cos(i / num_frames * 2 * np.pi) * 0.5
        positions[i, 2] = np.sin(i / num_frames * 4 * np.pi) * 0.5

        # Define the orientation of the camera for this frame
        orientations[i, 0] = np.sin(i / num_frames * 2 * np.pi)
        orientations[i, 1] = np.cos(i / num_frames * 2 * np.pi)
        orientations[i, 2] = np.sin(i / num_frames * 4 * np.pi)

        # Calculate the extrinsic matrix for this frame
        R = np.array(
            [
                [np.cos(orientations[i, 0]), 0, np.sin(orientations[i, 0])],
                [0, 1, 0],
                [-np.sin(orientations[i, 0]), 0, np.cos(orientations[i, 0])],
            ]
        )
        R = np.dot(
            R,
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(orientations[i, 1]), -np.sin(orientations[i, 1])],
                    [0, np.sin(orientations[i, 1]), np.cos(orientations[i, 1])],
                ]
            ),
        )
        R = np.dot(
            R,
            np.array(
                [
                    [np.cos(orientations[i, 2]), -np.sin(orientations[i, 2]), 0],
                    [np.sin(orientations[i, 2]), np.cos(orientations[i, 2]), 0],
                    [0, 0, 1],
                ]
            ),
        )
        T = np.array(
            [
                [1, 0, 0, positions[i, 0]],
                [0, 1, 0, positions[i, 1]],
                [0, 0, 1, positions[i, 2]],
                [0, 0, 0, 1],
            ]
        )
        R = np.vstack((R, np.array([0, 0, 0])))
        R = np.hstack((R, np.array([[0], [0], [0], [1]])))
        extrinsic = np.dot(R, T)
        extrinsics[i, :, :] = extrinsic

    return extrinsics


def get_updated_extrinsics(idx: int) -> np.ndarray:
    """Get camera extrinsics for a given frame.

    Args:
        idx (int): Frame index.

    Returns:
        np.ndarray: Camera extrinsics.
    """
    extrinsic = np.eye(4)

    t = np.array([0, 0, 10], dtype=np.float64)

    # rotate figure to look at origin
    R = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    angle = (idx / 600) * 2 * np.pi
    R_y = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ],
        dtype=np.float64,
    )

    # rotate around x
    angle = 30
    angle = angle / 180 * np.pi
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )

    R = R_x @ R_y @ R

    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    return extrinsic


def render_movie(store_path: str, movie_path: str, args: argparse.Namespace) -> None:
    """Make a movie from the frames of the animation.

    Args:
        store_path (str): Path to the frames of the animation.
        movie_path (str): Path to store the movie.
        args (argparse.Namespace): Command line arguments.
    """
    cmd = "ffmpeg "
    cmd += f"-framerate 30 -i {store_path}/%04d.png "
    cmd += "-c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "
    size = "1920:1080" if args.quality == "high" else "960:540"
    cmd += f"-vf scale={size} "
    cmd += f"{movie_path} -y"
    os.system(cmd)


def main(args: argparse.Namespace):
    """Main function."""
    super_path = os.path.join(args.data_path, args.scene, "dense", args.model_name)
    movie_dir = os.path.join(args.data_path, args.scene, "visualizations", args.model_name)

    render_frames(super_path, os.path.join(movie_dir, "frames"))

    render_movie(os.path.join(movie_dir, "frames"), os.path.join(movie_dir, "dense.mp4"), args)

    shutil.rmtree(os.path.join(movie_dir, "frames"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data.")
    parser.add_argument("--scene", type=str, required=True, help="Name of the scene.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model.")
    parser.add_argument(
        "--quality", type=str, default="low", choices=["low", "high"], help="Quality of the movie."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
