import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread


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
