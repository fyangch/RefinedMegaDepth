import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread


def pca(data):
    """Computes pca, since cameras tend to lay in a horizontal plane.
    Returns a coordinate transform as a lambda function
    """
    mean = data.mean(axis=0)
    standardized_data = data - mean
    scale = standardized_data.std()
    standardized_data /= scale
    covariance_matrix = np.cov(standardized_data, ddof=0, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    order_of_importance = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[order_of_importance]
    sorted_eigenvectors = eigenvectors[:, order_of_importance]  # sort the columns
    k = 3  # select the number of principal components
    reduced_data = standardized_data @ sorted_eigenvectors[:, :k]
    return lambda x: (x - mean) @ sorted_eigenvectors


def orthogonal_projections(data, color=None, limit=10, path=None):
    """Creates 3 orthogonal scatter plots, from top, front and side view."""

    def projection(Data, id1, id2):
        for data in Data:
            labels = ["X", "Y", "Z"]
            if not color is None:
                plt.scatter(data[:, id1], data[:, id2], s=1, color=color, alpha=0.1)
            else:
                plt.scatter(data[:, id1], data[:, id2], s=1, alpha=0.1)
        plt.axis("scaled")
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)
        plt.axis("off")
        plt.xlabel(labels[id1], labelpad=0)
        plt.ylabel(labels[id2], labelpad=0)

    fig = plt.figure()
    plt.tight_layout()
    plt.subplot(1, 3, 1)
    plt.title("Top View")
    projection(data, 0, 1)
    plt.subplot(1, 3, 2)
    plt.title("Front View")
    projection(data, 0, 2)
    plt.subplot(1, 3, 3)
    plt.title("Side View")
    projection(data, 1, 2)
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
