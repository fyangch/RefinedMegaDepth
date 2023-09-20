"""Show model from Colmap reconstruction."""
import argparse
import logging

import numpy as np
import open3d as o3d
import pycolmap

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()

# mypy: ignore-errors


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
    print("Creating point cloud from Colmap reconstruction")
    print(f"Min track length: {min_track_length}")
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
    pcd.points = o3d.utility.Vector3dVector(np.stack(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.stack(col))
    # print(len(pts), len(rec.points3D))
    print(f"Number of points: {len(pts)} ({len(rec.points3D)})")
    return pcd


def render_frames(
    rec: pycolmap.Reconstruction,
    show_window: bool = True,
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

    vis = render_frames.vis
    vis.create_window(visible=show_window)

    # Add the point cloud with support at least 1% of the images
    pcd = pcd_from_colmap(rec, min_track_length=np.ceil(len(rec.images.keys()) * 0.01))
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 1
    vis.run()
    vis.destroy_window()

    logging.info("Done rendering frames.")


reconstruction = pycolmap.Reconstruction(args.model_path)

render_frames(reconstruction)
