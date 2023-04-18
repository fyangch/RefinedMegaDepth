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
from megadepth.utils.setup import DataPaths
from megadepth.visualization.view_projections import align_models, pca


def pcd_from_colmap(
    rec: pycolmap.Reconstruction, min_track_length: int = 6, max_reprojection_error: int = 8
) -> o3d.geometry.PointCloud:
    """Create an Open3D point cloud from a Colmap reconstruction.

    Args:
        rec (pycolmap.Reconstruction): Sparse reconstruction from Colmap.
        min_track_length (int, optional): Minimum number of images a point must be visible in to be
        included in the point cloud. Defaults to 6.
        max_reprojection_error (int, optional): Maximum reprojection error for a point to be
        included in the point cloud. Defaults to 8.

    Returns:
        o3d.geometry.PointCloud: Open3D point cloud.
    """
    points = []
    colors = []
    for p3D in rec.points3D.values():
        if p3D.track.length() < min_track_length:
            continue
        if p3D.error > max_reprojection_error:
            continue
        points.append(p3D.xyz)
        colors.append(p3D.color / 255.0)

    # Align the point cloud
    align = pca(get_camera_poses(rec))
    points = align(np.array(points))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack(points))
    pcd.colors = o3d.utility.Vector3dVector(np.stack(colors))
    print(len(points), len(rec.points3D))
    return pcd


@typing.no_type_check
def render_frames(
    rec: pycolmap.Reconstruction,
    store_path: str,
    draw_cameras: bool = False,
    show_window: bool = False,
) -> None:
    """Rotate the view of the point cloud.

    Args:
        rec (pycolmap.Reconstruction): Sparse reconstruction from Colmap.
        store_path (str): Path to store the frames of the animation.
        draw_cameras (bool, optional): Whether to draw the cameras. Defaults to True.
        show_window (bool, optional): Whether to open a window to show the animation. Defaults to
        False.
    """
    render_frames.idx = 0
    render_frames.vis = o3d.visualization.Visualizer()
    render_frames.store_path = store_path
    render_frames.pbar = tqdm(total=600)

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

    pcd = pcd_from_colmap(rec)
    vis.add_geometry(pcd)

    if draw_cameras:
        add_cameras(rec, vis)

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
    camera_lines = {}
    for camera in rec.cameras.values():
        camera_lines[camera.camera_id] = o3d.geometry.LineSet.create_camera_visualization(
            camera.width, camera.height, camera.calibration_matrix(), np.eye(4), scale=0.1
        )

    # Draw the frustum for each image
    for image in rec.images.values():
        T = np.eye(4)
        T[:3, :4] = image.inverse_projection_matrix()
        cam = deepcopy(camera_lines[image.camera_id]).transform(T)
        cam.paint_uniform_color([1.0, 0.0, 0.0])  # red
        vis.add_geometry(cam)


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
    R_z = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    # rotate around x
    angle = 120
    angle = angle / 180 * np.pi
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )

    R = R_x @ R_z

    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    return extrinsic


def render_movie(store_path: str, movie_path: str) -> None:
    """Make a movie from the frames of the animation.

    Args:
        store_path (str): Path to the frames of the animation.
        movie_path (str): Path to store the movie.
    """
    cmd = "ffmpeg "
    cmd += f"-framerate 30 -i {store_path}/%04d.png "
    cmd += "-c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "
    cmd += f"{movie_path} -y"
    os.system(cmd)


def create_movie(paths: DataPaths) -> None:
    """Make a movie from the frames of the animation.

    Args:
        paths (DataPaths): Paths to the data.
    """
    model = pycolmap.Reconstruction(paths.sparse)

    logging.debug("Rendering frames.")
    render_frames(model, os.path.join(paths.visualizations, "frames"))
    logging.debug("Rendering movie.")
    render_movie(
        os.path.join(paths.visualizations, "frames"),
        os.path.join(paths.visualizations, "sparse.mp4"),
    )
    logging.debug("Removing frames.")
    shutil.rmtree(os.path.join(paths.visualizations, "frames"))


def main(args: argparse.Namespace):
    """Main function."""
    super_path = os.path.join(args.data_path, args.scene, "sparse", args.model_name)
    baseline_path = os.path.join(args.data_path, args.scene, "sparse", "baseline")
    movie_dir = os.path.join(args.data_path, args.scene, "visualizations", args.model_name)

    super_model = pycolmap.Reconstruction(super_path)
    try:
        baseline_model = pycolmap.Reconstruction(baseline_path)
        super_model = align_models(
            reconstruction_anchor=baseline_model, reconstruction_align=super_model
        )
    except Exception:
        logging.info("No baseline model found. Skipping alignment.")

    print(super_model.summary())

    render_frames(super_model, os.path.join(movie_dir, "frames"))

    render_movie(os.path.join(movie_dir, "frames"), os.path.join(movie_dir, "sparse.mp4"))

    shutil.rmtree(os.path.join(movie_dir, "frames"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data.")
    parser.add_argument("--scene", type=str, required=True, help="Name of the scene.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
