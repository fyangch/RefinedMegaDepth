"""Tool to visualize epipolar lines in a pair of images.

2D visualization primitives based on Matplotlib.
1) Plot images with `plot_images`.
2) Call `plot_keypoints` or `plot_matches` any number of times.
3) Optionally: save a .png or .pdf plot (nice in papers!) with `save_plot`.
"""

from typing import Optional, Union

import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np


def cm_RdGn(x: np.ndarray) -> np.ndarray:
    """Custom colormap: red (0) -> yellow (0.5) -> green (1).

    Args:
        x (np.ndarray): Array of values in [0, 1].

    Returns:
        np.ndarray: Array of RGB values in [0, 1].
    """
    x = np.clip(x, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0]]) + (2 - x) * np.array([[1.0, 0, 0]])
    return np.clip(c, 0, 1)


def plot_images(
    imgs: list,
    titles: Optional[list] = None,
    cmaps: Union[str, list, tuple] = "gray",
    dpi: int = 100,
    pad: float = 0.5,
    adaptive: bool = True,
) -> None:
    """Plot a set of images horizontally.

    Args:
        imgs (list): A list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles (list, optional): A list of strings, as titles for each image. Defaults to None.
        cmaps (str, optional): Colormaps for monochrome images. Defaults to "gray".
        dpi (int, optional): Dpi of the figure. Defaults to 100.
        pad (float, optional): Padding of the figure. Defaults to 0.5.
        adaptive (bool, optional): Whether the figure size should fit the image aspect ratios.
        Defaults to True.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    ratios = [i.shape[1] / i.shape[0] for i in imgs] if adaptive else [4 / 3] * n

    figsize = [sum(ratios) * 4.5, 4.5]

    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios})
    if n == 1:
        ax = [ax]

    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])

    fig.tight_layout(pad=pad)


def plot_keypoints(
    kpts: list,
    colors: Union[str, list] = "lime",
    ps: float = 4,
    axes: Optional[plt.axes] = None,
    a: float = 1.0,
) -> None:
    """Plot keypoints for existing images.

    Args:
        kpts (list): List of ndarrays of size (N, 2).
        colors (Union[str, list], optional): String or list of list of tuples (one for each kpt).
        ps (float, optional): Size of the keypoints as float. Defaults to 4.
        axes (plt.axes, optional): Axes to plot on. Defaults to None.
        a (float, optional): Alpha opacity of the keypoints. Defaults to 1.0.
    """
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    if axes is None:
        axes = plt.gcf().axes
    assert len(axes) == 2
    for ax, k, c in zip(axes, kpts, colors):
        ax.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0, alpha=a)


def plot_matches(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    color: Optional[list] = None,
    lw: float = 1.5,
    ps: float = 4,
    a: float = 1.0,
    labels: Optional[list] = None,
    axes: Optional[plt.axes] = None,
):
    """Plot matches for a pair of existing images.

    Args:
        kpts0 (np.ndarray): List of ndarrays of size (N, 2).
        kpts1 (np.ndarray): Corresponding keypoints of size (N, 2).
        color (list, optional): Color of each match, string or RGB tuple. Random if not given.
        lw (float, optional): Width of the lines. Defaults to 1.5.
        ps (float, optional): Size of the end points (no endpoint if ps=0). Defaults to 4.
        a (float, optional): Alpha opacity of the match lines. Defaults to 1.0.
        labels (list, optional): Labels for each match. Defaults to None.
        axes (plt.axes, optional): Axes to plot on. Defaults to None.
    """
    fig = plt.gcf()
    if axes is None:
        ax = fig.axes
        ax0, ax1 = ax[0], ax[1]
    else:
        ax0, ax1 = axes
    fig.canvas.draw()

    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        for i in range(len(kpts0)):
            line = matplotlib.patches.ConnectionPatch(
                xyA=(kpts0[i, 0], kpts0[i, 1]),
                xyB=(kpts1[i, 0], kpts1[i, 1]),
                coordsA=ax0.transData,
                coordsB=ax1.transData,
                axesA=ax0,
                axesB=ax1,
                zorder=1,
                color=color[i],
                linewidth=lw,
                clip_on=True,
                alpha=a,
                label=None if labels is None else labels[i],
                picker=5.0,
            )
            line.set_annotation_clip(True)
            fig.add_artist(line)

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def add_text(
    idx: int,
    text: str,
    pos: tuple = (0.01, 0.99),
    fs: float = 15,
    color: str = "w",
    lcolor: str = "k",
    lwidth: float = 2,
    ha: str = "left",
    va: str = "top",
) -> None:
    """Add text to an existing image.

    Args:
        idx (int): Index of the image to add text to.
        text (str): Text to add.
        pos (tuple, optional): Position of the text as a tuple (x, y). Defaults to (0.01, 0.99).
        fs (float, optional): Font size. Defaults to 15.
        color (str, optional): Color of the text. Defaults to "w".
        lcolor (str, optional): Color of the text outline. Defaults to "k".
        lwidth (float, optional): Width of the text outline. Defaults to 2.
        ha (str, optional): Horizontal alignment. Defaults to "left".
        va (str, optional): Vertical alignment. Defaults to "top".
    """
    ax = plt.gcf().axes[idx]
    t = ax.text(*pos, text, fontsize=fs, ha=ha, va=va, color=color, transform=ax.transAxes)
    if lcolor is not None:
        t.set_path_effects(
            [path_effects.Stroke(linewidth=lwidth, foreground=lcolor), path_effects.Normal()]
        )


def draw_epipolar_line(
    line: np.ndarray,
    imshape: tuple,
    axis: plt.axis,
    color: str = "b",
    label: Optional[str] = None,
    alpha: float = 1.0,
    visible: bool = True,
) -> None:
    """Draw an epipolar line on an existing image.

    Args:
        line (np.ndarray): Line to draw.
        imshape (tuple): Shape of the image.
        axis (plt.axis): Axis to draw on.
        color (str, optional): Color of the line. Defaults to "b".
        label (str, optional): Label of the line. Defaults to None.
        alpha (float, optional): Alpha opacity of the line. Defaults to 1.0.
        visible (bool, optional): Whether the line is visible. Defaults to True.
    """
    h, w = imshape[:2]
    # Intersect line with lines representing image borders.
    X1 = np.cross(line, [1, 0, -1])
    X1 = X1[:2] / X1[2]
    X2 = np.cross(line, [1, 0, -w])
    X2 = X2[:2] / X2[2]
    X3 = np.cross(line, [0, 1, -1])
    X3 = X3[:2] / X3[2]
    X4 = np.cross(line, [0, 1, -h])
    X4 = X4[:2] / X4[2]

    # Find intersections which are not outside the image,
    # which will therefore be on the image border.
    Xs = [X1, X2, X3, X4]
    Ps = []
    for p in range(4):
        X = Xs[p]
        if (0 <= X[0] <= (w + 1e-6)) and (0 <= X[1] <= (h + 1e-6)):
            Ps.append(X)
            if len(Ps) == 2:
                break

    # Plot line, if it's visible in the image.
    if len(Ps) == 2:
        axis.plot(
            [Ps[0][0], Ps[1][0]],
            [Ps[0][1], Ps[1][1]],
            color,
            linestyle="dashed",
            label=label,
            alpha=alpha,
            visible=visible,
        )


def get_line(F: np.ndarray, kp: np.ndarray) -> np.ndarray:
    """Get the epipolar line for a given keypoint.

    Args:
        F (np.ndarray): Fundamental matrix.
        kp (np.ndarray): Keypoint.

    Returns:
        np.ndarray: Epipolar line for the given keypoint.
    """
    hom_kp = np.array([list(kp) + [1.0]]).transpose()
    return np.dot(F, hom_kp)


def plot_epipolar_lines(
    pts0: np.ndarray,
    pts1: np.ndarray,
    F: np.ndarray,
    color: str = "b",
    axes: Optional[plt.axes] = None,
    labels: Optional[list] = None,
    a: float = 1.0,
    visible: bool = True,
) -> None:
    """Plot epipolar lines on a pair of images.

    Args:
        pts0 (np.ndarray): Points in the first image.
        pts1 (np.ndarray): Corresponding points in the second image.
        F (np.ndarray): Fundamental matrix.
        color (str, optional): Color of the lines. Defaults to "b".
        axes (plt.axes, optional): Axes to draw on. Defaults to None.
        labels (list, optional): Labels of the lines. Defaults to None.
        a (float, optional): Alpha opacity of the lines. Defaults to 1.0.
        visible (bool, optional): Whether the lines are visible. Defaults to True.
    """
    if axes is None:
        axes = plt.gcf().axes
    assert len(axes) == 2

    for ax, kps in zip(axes, [pts1, pts0]):
        _, w = ax.get_xlim()
        h, _ = ax.get_ylim()

        imshape = (h + 0.5, w + 0.5)
        for i in range(kps.shape[0]):
            if ax == axes[0]:
                line = get_line(F.transpose(0, 1), kps[i])[:, 0]
            else:
                line = get_line(F, kps[i])[:, 0]
            draw_epipolar_line(
                line,
                imshape,
                ax,
                color=color,
                label=None if labels is None else labels[i],
                alpha=a,
                visible=visible,
            )


def save_plot(path: str, **kw) -> None:
    """Save the current figure without any white margin.

    Args:
        path (str): Path to save the figure to.
        **kw: Additional keyword arguments to pass to `plt.savefig`.
    """
    plt.savefig(path, bbox_inches="tight", pad_inches=0, **kw)
