import os

import matplotlib.pyplot as plt
import numpy as np
import pycolmap

import megadepth.visualization.view_overlap as view_overlap
import megadepth.visualization.view_projections as view_projections
from megadepth.metrics import overlap
from megadepth.utils.projections import get_camera_poses


def join(output_path, *parts):
    if output_path is None:
        return None
    return os.path.join(output_path, *parts)


def create_all_figures(sparse_model_path, output_path):
    """
        Loads a reconstrucion and stores all necessary plots.

    Args:
        sparse_model_path: folder path of the sparse model
        output_path: folder path where the plots should go
    """
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


def scatter3d(data, color=None, s=3):
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


def camera_pose_and_overlap(camera_poses, overlap_matrix, path=None):
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
