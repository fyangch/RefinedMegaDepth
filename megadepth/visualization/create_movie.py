"""Create a movie from a sparse or dense reconstruction."""

import argparse
import contextlib
import logging
import os
import typing
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import open3d as o3d
import pycolmap
from tqdm import tqdm

from megadepth.utils.projections import get_camera_poses
from megadepth.visualization.camera_trajectories import surround_view
from megadepth.visualization.view_projections import align_models, pca_matrix
from megadepth.visualization.view_sparse_model import pcd_from_colmap


def compute_pca_on_camera_poses(rec: pycolmap.Reconstruction) -> np.ndarray:
    """Returns pca basis and center.

    Args:
        rec (pycolmap.Reconstruction): Sparse reconstruction from Colmap.

    Returns:
        np.ndarray: pca basis and center
    """
    camera_poses = get_camera_poses(rec)
    return pca_matrix(camera_poses)


def compute_pca_on_points(rec: pycolmap.Reconstruction) -> np.ndarray:
    """Returns pca basis and center.

    Args:
        rec (pycolmap.Reconstruction): Sparse reconstruction from Colmap.

    Returns:
        np.ndarray: pca basis and center
    """
    points = np.array([p.xyz for p in rec.points3D.values()])
    return pca_matrix(points)


def compute_scale_on_camera_poses(rec: pycolmap.Reconstruction) -> np.ndarray:
    """Returns std of camera poses.

    Args:
        rec (pycolmap.Reconstruction): Sparse reconstruction from Colmap.

    Returns:
        np.ndarray: std of camera poses
    """
    camera_poses = get_camera_poses(rec)
    return camera_poses.std()


def compute_scale_on_points(rec: pycolmap.Reconstruction) -> np.ndarray:
    """Returns std of points.

    Args:
        rec (pycolmap.Reconstruction): Sparse reconstruction from Colmap.

    Returns:
        np.ndarray: std of points
    """
    points = np.array([p.xyz for p in rec.points3D.values()])
    std = points.std()
    mean = points.mean(axis=0)

    # keep points within 2 std
    n_std = 3
    points = points[np.all(np.abs(points - mean) < n_std * std, axis=1)]

    return points.std()


@typing.no_type_check
def render_frames(
    point_cloud: o3d.geometry.PointCloud,
    extrinsics: Union[np.ndarray, list],
    cameras: Optional[list] = None,
    store_path: Optional[str] = None,
    show_window: bool = False,
    num_frames: int = 300,
) -> None:
    """Render frames from a point cloud.

    Args:
        point_cloud (o3d.geometry.PointCloud): Point cloud to render.
        extrinsics (Union[np.ndarray, list]): List of extrinsics.
        cameras (Optional[list], optional): List of cameras. Defaults to None.
        store_path (Optional[str], optional): Path to store the frames. Defaults to None.
        show_window (bool, optional): Show the window. Defaults to False.
        num_frames (int, optional): Number of frames to render. Defaults to 300.
    """
    render_frames.idx = 0
    render_frames.vis = o3d.visualization.Visualizer()
    render_frames.store_path = store_path
    render_frames.num_frames = num_frames
    render_frames.pbar = tqdm(total=render_frames.num_frames)

    render_frames.extrinsics = extrinsics

    if store_path is not None and not os.path.exists(store_path):
        os.makedirs(store_path)

    def play_animation(vis: o3d.visualization.Visualizer) -> bool:
        """Play the animation.

        Args:
            vis (o3d.visualization.Visualizer): Open3D visualizer.

        Returns:
            bool: Default return value.
        """
        ctr = vis.get_view_control()

        # Stop rotating after 360 degrees
        glb = render_frames

        # update camera
        if glb.idx < glb.num_frames:
            camera = ctr.convert_to_pinhole_camera_parameters()
            camera.extrinsic = render_frames.extrinsics[glb.idx]
            ctr.convert_from_pinhole_camera_parameters(camera)

        else:
            vis.close()

        glb.idx += 1

        # update view
        vis.poll_events()
        vis.update_renderer()

        # Store the frame in a plot
        if store_path is not None:
            filename = os.path.join(glb.store_path, f"{glb.idx:04d}.png")
            vis.capture_screen_image(filename, do_render=True)

        # Update the progress bar
        glb.pbar.update(1)

        return False

    vis = render_frames.vis
    vis.create_window(visible=show_window)

    # Add the point cloud with support at least 1% of the images
    vis.add_geometry(point_cloud)

    if cameras is not None:
        for cam in cameras:
            vis.add_geometry(cam)

    opt = vis.get_render_option()
    opt.point_size = 1
    vis.register_animation_callback(play_animation)
    vis.run()
    vis.destroy_window()

    logging.info("Done rendering frames.")


def get_cameras(rec: pycolmap.Reconstruction) -> list[o3d.geometry.LineSet]:
    """Get the cameras from the reconstruction.

    Args:
        rec (pycolmap.Reconstruction): Sparse reconstruction from Colmap.

    Returns:
        list[o3d.geometry.LineSet]: List of cameras.
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
    cameras = []
    for image in rec.images.values():
        T = np.eye(4)
        T[:3, :4] = image.inverse_projection_matrix()
        cam = deepcopy(camera_lines[image.camera_id]).transform(T)
        cam.paint_uniform_color([1.0, 0.0, 0.0])  # red
        cameras.append(cam)

    return cameras


def visualize_point_cloud(
    model: pycolmap.Reconstruction,
    name: str,
    extrinsics: list,
    min_track_length: int,
    args: argparse.Namespace,
):
    """Visualize the point cloud.

    Args:
        model (pycolmap.Reconstruction): PyColmap reconstruction.
        name (str): Name of the model.
        extrinsics (list): List of camera extrinsics.
        min_track_length (int): Minimum track length.
        args (argparse.Namespace): Arguments.

    Raises:
        ValueError: If the model type is unknown.
    """
    if model is None:
        return

    # get point cloud
    if args.model_type == "sfm":
        pcd = pcd_from_colmap(model, min_track_length=min_track_length)
    elif args.model_type == "mvs":
        path = os.path.join(args.data_path, args.scene, "dense", name, "dense.ply")
        if not os.path.exists(path):
            logging.warning(f"Could not find dense point cloud at for {name} at {path}")
            return
        pcd = o3d.io.read_point_cloud(path)
    else:
        raise ValueError(f"Unknown model type {args.model_type}")

    cameras = get_cameras(model) if args.cameras else None

    output_path = os.path.join(args.data_path, args.scene, "visualizations", name, "frames")

    render_frames(
        point_cloud=pcd,
        extrinsics=extrinsics,
        cameras=cameras,
        show_window=args.visible,
        store_path=output_path,
    )


def render_movie(name: str, args: argparse.Namespace):
    """Render a movie from the frames.

    Args:
        name (str): Name of the model.
        args (argparse.Namespace): Arguments.
    """
    frames = os.path.join(args.data_path, args.scene, "visualizations", name, "frames")
    output = os.path.join(
        args.data_path, args.scene, "visualizations", name, f"{args.model_type}.mp4"
    )

    if not os.path.exists(frames):
        logging.info(f"Could not find frames at {frames}")
        return

    cmd = "ffmpeg "
    cmd += f"-framerate 30 -i {frames}/%04d.png "
    cmd += "-vcodec libx264 -crf 30 -pix_fmt yuv420p "
    size = "1920:1080" if args.quality == "high" else "960:540"
    cmd += f"-vf scale={size} "
    cmd += f"{output} -y"
    os.system(cmd)

    # remove frames
    for frame in os.listdir(frames):
        with contextlib.suppress(Exception):
            os.remove(os.path.join(frames, frame))
    os.rmdir(frames)


def main(args: argparse.Namespace):
    """Main function."""
    base_dir = os.path.join(args.data_path, args.scene)
    super_path = os.path.join(base_dir, "sparse", args.model_name)
    baseline_path = os.path.join(base_dir, "sparse", "baseline")

    super_model = pycolmap.Reconstruction(super_path)
    try:
        baseline_model = pycolmap.Reconstruction(baseline_path)
        super_model = align_models(
            reconstruction_anchor=baseline_model, reconstruction_align=super_model
        )
        pca_transform = compute_pca_on_camera_poses(super_model)
        scale = compute_scale_on_points(baseline_model)

        logging.info("Found baseline model.")

    except Exception:
        baseline_model = None
        pca_transform = compute_pca_on_camera_poses(super_model)
        scale = compute_scale_on_points(super_model)
        logging.info("No baseline model found. Skipping alignment.")

    logging.info(f"Scale: {scale}")
    logging.info(f"PCA transform:\n{pca_transform}")

    logging.info(f"(points) Baseline scale: {compute_scale_on_points(baseline_model)}")
    logging.info(f"(points) Super scale:    {compute_scale_on_points(super_model)}")

    logging.info(f"(camera) Baseline scale: {compute_scale_on_camera_poses(baseline_model)}")
    logging.info(f"(camera) Super scale:    {compute_scale_on_camera_poses(super_model)}")

    extrinsics = surround_view(transform=pca_transform, scale=scale, zoom_in=args.zoom_in)

    min_track_length = np.ceil(len(super_model.images) * 0.005)  # 0.5% of images
    visualize_point_cloud(baseline_model, "baseline", extrinsics, min_track_length, args)
    visualize_point_cloud(super_model, args.model_name, extrinsics, min_track_length, args)

    render_movie("baseline", args)
    render_movie(args.model_name, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data.")
    parser.add_argument("--scene", type=str, required=True, help="Name of the scene.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model.")
    parser.add_argument(
        "--model_type", type=str, default="sfm", choices=["sfm", "mvs"], help="Type of model."
    )
    parser.add_argument("--zoom_in", action="store_true", help="Zoom in on the point cloud.")
    parser.add_argument("--cameras", action="store_true", help="Draw cameras.")
    parser.add_argument("--visible", action="store_true", help="Show the window.")
    parser.add_argument(
        "--quality", type=str, default="high", choices=["low", "high"], help="Quality of the movie."
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )

    main(args)
