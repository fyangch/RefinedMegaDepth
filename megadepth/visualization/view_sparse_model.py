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
from megadepth.visualization.view_projections import align_models, pca_matrix


def compute_pca_on_camera_poses(rec: pycolmap.Reconstruction):
    """Returns pca basis and center."""
    camera_poses = get_camera_poses(rec)
    return pca_matrix(camera_poses)


def compute_pca_on_points(rec: pycolmap.Reconstruction):
    """Returns pca basis and center."""
    points = np.array([p.xyz for p in rec.points3D.values()])
    return pca_matrix(points)


def compute_scale_on_camera_poses(rec: pycolmap.Reconstruction):
    """Returns std of camera poses."""
    camera_poses = get_camera_poses(rec)
    return camera_poses.std()


def compute_scale_on_points(rec: pycolmap.Reconstruction):
    """Returns std of points."""
    points = np.array([p.xyz for p in rec.points3D.values()])
    return points.std()


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
    logging.info("Creating point cloud from Colmap reconstruction")
    logging.info(f"Min track length: {min_track_length}")
    points = []
    colors = []
    for p3D in rec.points3D.values():
        if p3D.track.length() < min_track_length:
            continue
        if p3D.error > max_reprojection_error:
            continue
        points.append(p3D.xyz)
        colors.append(p3D.color / 255.0)

    pts = np.array(points)
    col = np.array(colors)

    # get idx of points inside 3 std
    n_std = 3
    center = pts.mean(axis=0)
    std = pts.std(axis=0)
    idx = np.where(np.all(np.abs(pts - center) < n_std * std, axis=1))[0]
    pts = pts[idx]
    col = col[idx]

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
    pca_transform=None,
    scale=None,
) -> None:
    """Rotate the view of the point cloud.

    Args:
        rec (pycolmap.Reconstruction): Sparse reconstruction from Colmap.
        store_path (str): Path to store the frames of the animation.
        draw_cameras (bool, optional): Whether to draw the cameras. Defaults to True.
        show_window (bool, optional): Whether to open a window to show the animation. Defaults to
        False.
        pca_transform: pass pca matrix from basemodel to be used for alignment
        scale: pass scale of animation from basemodel
    """
    render_frames.idx = 0
    render_frames.vis = o3d.visualization.Visualizer()
    render_frames.store_path = store_path
    render_frames.num_frames = 150
    render_frames.pbar = tqdm(total=render_frames.num_frames)

    render_frames.phi = lambda x: 120
    render_frames.t = lambda x: 5  # 3.5 + 1.5 * np.cos(x * 2 * np.pi)
    render_frames.theta = lambda x: 2 * np.pi * x
    # render_frames.theta = lambda x: np.deg2rad(
    #     np.cos(2 * np.pi * x) * 45
    # )  # go back and forth 45 deg
    if pca_transform is None:
        render_frames.pca_transform = compute_pca_on_camera_poses(rec)
    else:
        render_frames.pca_transform = pca_transform

    if scale is None:
        render_frames.scale = compute_scale_on_points(rec)
    else:
        render_frames.scale = scale

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

        if glb.idx < glb.num_frames:
            camera = ctr.convert_to_pinhole_camera_parameters()
            camera.extrinsic = get_updated_extrinsics(glb)
            ctr.convert_from_pinhole_camera_parameters(camera)

        else:
            vis.close()
        glb.idx += 1

        # update view
        vis.poll_events()
        vis.update_renderer()

        # Store the frame in a plot
        filename = os.path.join(glb.store_path, f"{glb.idx:04d}.png")
        vis.capture_screen_image(filename, do_render=True)

        # Update the progress bar
        glb.pbar.update(1)

        return False

    vis = render_frames.vis
    vis.create_window(visible=show_window)

    # Add the point cloud with support at least 1% of the images
    pcd = pcd_from_colmap(rec, min_track_length=np.ceil(len(rec.images.keys()) * 0.01))
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


def get_updated_extrinsics(glb) -> np.ndarray:
    """Get camera extrinsics for a given frame.

    Args:
        idx (int): Frame index.

    Returns:
        np.ndarray: Camera extrinsics.
    """
    t0 = 1.0 * glb.idx / glb.num_frames

    t = np.array([0, 0, glb.t(t0)], dtype=np.float64) * glb.scale

    # compute current turntable angle
    theta = glb.theta(t0)
    R_z = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    # rotate for tilt, 0 from bottom, 90 from side, 180 from top
    phi = np.deg2rad(glb.phi(t0))
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)],
        ],
        dtype=np.float64,
    )

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R_x @ R_z
    extrinsic[:3, 3] = t
    # first transform to pca aligned space, and then apply rotation
    return extrinsic @ glb.pca_transform


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
        pca_transform = compute_pca_on_camera_poses(baseline_model)
        scale = compute_scale_on_points(baseline_model)

    except Exception:
        pca_transform = None
        scale = None
        logging.info("No baseline model found. Skipping alignment.")

    print(super_model.summary())

    render_frames(
        super_model, os.path.join(movie_dir, "frames"), pca_transform=pca_transform, scale=scale
    )

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
