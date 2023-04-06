import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread


def pca(data):
    """Computes pca, since cameras tend to lay in a horizontal plane.
    Returns a coordinate transform as a lambda function
    """
    mean = data.mean(axis=0)
    standardized_data = data - mean
    scale = data.std()
    standardized_data /= scale
    covariance_matrix = np.cov(standardized_data, ddof=0, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    order_of_importance = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[order_of_importance]
    sorted_eigenvectors = eigenvectors[:, order_of_importance]
    # Ensure handedness doesnt change
    if np.linalg.det(sorted_eigenvectors) < 0:
        sign = np.sign(np.ones((1, 3)) @ sorted_eigenvectors)
        det_sign = np.linalg.det(sorted_eigenvectors)
        sorted_eigenvectors = sorted_eigenvectors * sign * det_sign
    return lambda x: (x - mean) / scale @ sorted_eigenvectors


def plot_view_projection(Data, view, limit=10, s=1, alpha=0.1, *args, **kwargs):
    """Takes Data, an index that specifies the view direction ["Top View", "Front View", "Right View"]."""
    view_names = ["Top View", "Front View", "Right View"]
    id1, id2 = [(0, 1), (0, 2), (1, 2)][view]
    for data in Data:
        labels = ["X", "Y", "Z"]
        plt.scatter(data[:, id1], data[:, id2], s=s, alpha=alpha, *args, **kwargs)
        plt.title(view_names[view])
    plt.axis("scaled")
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    plt.axis("off")
    plt.xlabel(labels[id1], labelpad=0)
    plt.ylabel(labels[id2], labelpad=0)


def create_view_projection_figure(data, view=None, path=None, *args, **kwargs):
    """
    Creates an axis aligned plot.

    Args:
        data: np.ndarray of shape (N, 3)
        view: int to select from ["Top View", "Front View", "Right View"]
        path: filepath to store the plot
        alpha: set transparency for dots
        s: size for dots
        limit: limits of the plot limit=3 should contain 99% of the camera positions

    """
    fig = plt.figure(frameon=False)
    plt.tight_layout()
    if view is None:
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plot_view_projection(data, i, *args, **kwargs)
    else:
        plot_view_projection(data, view, *args, **kwargs)
    if path is None:
        plt.show()
    else:
        fig.savefig(path, dpi=900)


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


def camera_pose_and_overlap(camera_poses, overlap_matrix):
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
