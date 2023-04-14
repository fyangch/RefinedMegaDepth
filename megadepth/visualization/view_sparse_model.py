"""Visualize a sparse reconstruction from Colmap.

Example:
    python megadepth/visualizations/visualize_sparse.py --model_path /path/to/model_dir
"""

import argparse
import typing
from copy import deepcopy

import numpy as np
import open3d as o3d
import pycolmap


def pcd_from_colmap(
    rec: pycolmap.Reconstruction, min_track_length: int = 4, max_reprojection_error: int = 8
) -> o3d.geometry.PointCloud:
    """Create an Open3D point cloud from a Colmap reconstruction.

    Args:
        rec (pycolmap.Reconstruction): Sparse reconstruction from Colmap.
        min_track_length (int, optional): Minimum number of images a point must be visible in to be
        included in the point cloud. Defaults to 4.
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
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack(points))
    pcd.colors = o3d.utility.Vector3dVector(np.stack(colors))
    print(len(points), len(rec.points3D))
    return pcd


@typing.no_type_check
def custom_draw_geometry_with_rotation(rec: pycolmap.Reconstruction) -> None:
    """Rotate the view of the point cloud.

    Args:
        rec (pycolmap.Reconstruction): Sparse reconstruction from Colmap.
    """
    custom_draw_geometry_with_rotation.idx = 0
    custom_draw_geometry_with_rotation.vis = o3d.visualization.Visualizer()

    def rotate_view(vis: o3d.visualization.Visualizer) -> bool:
        """Rotate the view of the point cloud.

        Args:
            vis (o3d.visualization.Visualizer): Open3D visualizer.

        Returns:
            bool: Default return value.
        """
        ctr = vis.get_view_control()

        # Stop rotating after 360 degrees
        glb = custom_draw_geometry_with_rotation
        glb.idx += 1

        # Rotate the view
        if glb.idx < 450:
            ctr.set_lookat([0, glb.idx / 50, -glb.idx / 50])
            ctr.rotate(0, -1.0)
        elif glb.idx < 1070:
            angle = glb.idx * np.pi / 180
            ctr.rotate(3 * np.sin(0.6 * angle), 0)
        else:
            custom_draw_geometry_with_rotation.vis.register_animation_callback(None)

        # print(glb.idx)
        # print(ctr.convert_to_pinhole_camera_parameters().extrinsic)

        return False

    vis = custom_draw_geometry_with_rotation.vis
    vis.create_window()

    pcd = pcd_from_colmap(rec)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    vis.add_geometry(pcd)
    # Define the camera frustums as lines
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

    opt = vis.get_render_option()
    opt.point_size = 2
    vis.register_animation_callback(rotate_view)
    vis.run()
    vis.destroy_window()


def main(args: argparse.Namespace):
    """Main function."""
    rec = pycolmap.Reconstruction(args.model_path)
    print(rec.summary())

    custom_draw_geometry_with_rotation(rec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to model file.")
    args = parser.parse_args()
    main(args)
