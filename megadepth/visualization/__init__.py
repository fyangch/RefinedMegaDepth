import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pycolmap

import megadepth.visualization.view_overlap as view_overlap
import megadepth.visualization.view_projections as view_projections
from megadepth.metrics import overlap
from megadepth.utils.projections import get_camera_poses


def join(
    output_path: str,
    *parts: str,
) -> str:
    """Join path parts with output_path.

    Args:
        output_path (str): Path to the output folder

    Returns:
        str: Path to the output folder
    """
    return None if output_path is None else os.path.join(output_path, *parts)


def create_all_figures(
    sparse_model_path: str,
    output_path: str,
    depth_path: Optional[str] = None,
) -> None:
    """Loads a reconstrucion and stores all necessary plots.

    Args:
        sparse_model_path (str): Path to the sparse model
        output_path (str): Path to the output folder
        depth_path (Optional[str], optional): Path to the depth maps. Defaults to None.
    """
    try:
        reconstruction = pycolmap.Reconstruction(sparse_model_path)
        camera_poses = get_camera_poses(reconstruction)
        points = np.array([p.xyz for p in reconstruction.points3D.values()])
        align = view_projections.pca(camera_poses)
        # alternative to pca
        # align = lambda x: x @ np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        out_file = join(output_path, "all_views.jpg")
        view_projections.create_view_projection_figure(
            [align(points), align(camera_poses)], limit=3, path=out_file
        )

        # sparse overlap matrix
        sparse_overlap_matrix = overlap.sparse_overlap(reconstruction)
        view_overlap.show_matrix(
            sparse_overlap_matrix, os.path.join(output_path, "sparse_overlap_matrix.jpg")
        )
        view_overlap.vis_overlap(
            sparse_overlap_matrix,
            align(camera_poses),
            align(points),
            os.path.join(output_path, "vis_sparse_overlap.jpg"),
        )

        if depth_path is not None:
            dense_overlap_matrix = overlap.dense_overlap(reconstruction, depth_path=depth_path)
            view_overlap.show_matrix(
                dense_overlap_matrix,
                os.path.join(
                    output_path,
                    "dense_overlap_matrix.jpg",
                ),
            )
            view_overlap.vis_overlap(
                dense_overlap_matrix,
                align(camera_poses),
                align(points),
                os.path.join(output_path, "vis_dense_overlap.jpg"),
            )
    except Exception as e:
        print(e)


def scatter3d(
    data: np.ndarray,
    color: Optional[str] = None,
    s: float = 3,
) -> None:
    """Creates a 3D scatter plot.

    Args:
        data (np.ndarray): Points to plot.
        color (Optional[str], optional): Color of the points. Defaults to None.
        s (float, optional): Size of the points. Defaults to 3.
    """
    # Create a 3D figure
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # Add the scatter plot
    ax.scatter3D(data[0], data[1], data[2], c=color, s=s)

    # Add labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("3D Scatter Plot")
    # ax.invert_xaxis()
    ax.invert_yaxis()
    ax.invert_zaxis()

    # Show the plot
    plt.show()


def camera_pose_and_overlap(camera_poses: np.ndarray, overlap_matrix: np.ndarray) -> None:
    """Creates a 3D plot of the camera poses and the overlap matrix.

    Args:
        camera_poses (np.ndarray): Camera poses in the reconstruction.
        overlap_matrix (np.ndarray): Overlap matrix of the reconstruction.
    """
    points = camera_poses
    weights = overlap_matrix.astype(float)
    weights /= np.max(weights)

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    h, w = overlap_matrix.shape

    # Loop over all pairs of points
    for i in range(h):
        for j in range(w):
            if i != j:
                # Get weight value and set opacity
                weight = weights[i, j]
                alpha = min(1, weight) / 10  # Increase opacity for larger weights

                # Get coordinates of two points
                x = [points[i, 0], points[j, 0]]
                y = [points[i, 1], points[j, 1]]
                z = [points[i, 2], points[j, 2]]

                # Plot line with opacity set by weight
                ax.plot(x, y, color="blue", alpha=alpha)

    # Show plot
    plt.show()
