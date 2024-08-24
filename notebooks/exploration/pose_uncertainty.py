"""Calculate pose uncertainty for a reconstruction."""

import sys

sys.path.append("external_dependencies/pyceres/")

import argparse
import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pyceres
import pycolmap
from tqdm import tqdm

from megadepth.utils.projections import get_camera_poses


def get_poses(reconstruction: pycolmap.Reconstruction) -> np.ndarray:
    """Get camera poses from a reconstruction.

    Args:
        reconstruction (pycolmap.Reconstruction): Colmap reconstruction.

    Returns:
        np.ndarray: Array of camera poses.
    """
    cameras = reconstruction.cameras
    images = reconstruction.images

    poses = {}
    for im in images.values():
        camera = cameras[im.camera_id]
        p_world = np.array(camera.image_to_world([[0, 0]]))
        p_world = np.stack([p_world[:, 0], p_world[:, 1], np.ones_like(p_world[:, 0])])
        p_world = np.array(im.transform_to_world(p_world.T))

        poses[im.name] = p_world[0]

    return poses


def plot_model(
    reconstruction: pycolmap.Reconstruction, tilt: float = -45, output: Optional[str] = None
) -> None:
    """Plot a reconstruction.

    Args:
        reconstruction (pycolmap.Reconstruction): Colmap reconstruction.
        tilt (float, optional): Tilt of the camera. Defaults to -45.
        output (str, optional): Output directory. Defaults to None.
    """
    _, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="3d"))

    p3d = np.array(
        [p.xyz for p in reconstruction.points3D.values() if p.track.length() > 2 and p.error < 8]
    )
    cam_tvecs = get_camera_poses(reconstruction)

    p_colors = [(0, 0, 0) for _ in p3d]
    ax.scatter(p3d[:, 0], p3d[:, 1], p3d[:, 2], s=0.1, marker=".", alpha=0.05, c=p_colors)

    cam_colors = [(1, 0, 0) for _ in cam_tvecs]
    ax.scatter(
        cam_tvecs[:, 0], cam_tvecs[:, 1], cam_tvecs[:, 2], s=10, marker=".", alpha=0.5, c=cam_colors
    )

    # bbox = (-5, 5)

    # ax.set_xlim(*bbox)
    # ax.set_ylim(*bbox)
    # ax.set_zlim(*bbox)

    ax.set_xlim(cam_tvecs[:, 0].min(), cam_tvecs[:, 0].max())
    ax.set_ylim(cam_tvecs[:, 1].min(), cam_tvecs[:, 1].max())
    ax.set_zlim(cam_tvecs[:, 2].min(), cam_tvecs[:, 2].max())

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # flip x and z
    ax.view_init(tilt, -90, 0)

    if output is not None:
        plt.savefig(os.path.join(output, "model.png"))
    else:
        plt.show()


def plot_uncs(
    reconstruction: pycolmap.Reconstruction,
    uncertainties: np.ndarray,
    cam_tvecs: np.ndarray,
    tilt: float = -45,
    output: Optional[str] = None,
) -> None:
    """Plot a reconstruction with uncertainties.

    Args:
        reconstruction (pycolmap.Reconstruction): Colmap reconstruction.
        uncertainties (np.ndarray): Uncertainties of the camera poses.
        cam_tvecs (np.ndarray): Camera poses.
        tilt (float, optional): Tilt of the camera. Defaults to -45.
        output (str, optional): Output directory. Defaults to None.
    """
    cmap = dict(cmap="turbo", norm=plt.Normalize(0, uncertainties.max() * 100))
    # cmap = dict(cmap='turbo', norm=plt.Normalize(0, 0.2))

    _, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="3d"))

    p3d = np.array(
        [p.xyz for p in reconstruction.points3D.values() if p.track.length() > 2 and p.error < 8]
    )

    p_colors = [(0, 0, 0) for _ in p3d]
    ax.scatter(p3d[:, 0], p3d[:, 1], p3d[:, 2], s=0.1, marker=".", alpha=0.05, c=p_colors)

    obj = ax.scatter(
        cam_tvecs[:, 0],
        cam_tvecs[:, 1],
        cam_tvecs[:, 2],
        s=100,
        marker=".",
        alpha=0.5,
        c=uncertainties * 100,
        **cmap
    )

    ax.get_figure().colorbar(obj, ax=ax)

    # bbox = (-5, 5)

    # ax.set_xlim(*bbox)
    # ax.set_ylim(*bbox)
    # ax.set_zlim(*bbox)

    ax.set_xlim(cam_tvecs[:, 0].min(), cam_tvecs[:, 0].max())
    ax.set_ylim(cam_tvecs[:, 1].min(), cam_tvecs[:, 1].max())
    ax.set_zlim(cam_tvecs[:, 2].min(), cam_tvecs[:, 2].max())

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # flip x and z
    ax.view_init(tilt, -90, 0)

    if output is not None:
        plt.savefig(os.path.join(output, "uncertainties.png"))
    else:
        plt.show()


def solve(prob: pyceres.Problem) -> None:
    """Solve a ceres problem.

    Args:
        prob (pyceres.Problem): Ceres problem.
    """
    print(
        prob.num_parameter_blocks(),
        prob.num_parameters(),
        prob.num_residual_blocks(),
        prob.num_residuals(),
    )
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    options.minimizer_progress_to_stdout = True
    options.num_threads = -1
    options.max_num_iterations = 0
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    print(summary.BriefReport())


def opt_poses(rec: pycolmap.Reconstruction) -> pyceres.Problem:
    """Run bundle adjustment on the poses of a reconstruction.

    Args:
        rec (pycolmap.Reconstruction): Colmap reconstruction.

    Returns:
        pyceres.Problem: Ceres problem.
    """
    prob = pyceres.Problem()
    loss = pyceres.TrivialLoss()
    for im in tqdm(
        rec.images.values(), total=len(rec.images), desc="Adding image residuals", ncols=80
    ):
        cam = rec.cameras[im.camera_id]
        for p in im.points2D:
            cost = pyceres.factors.BundleAdjustmentCost(cam.model_id, p.xy)
            if p.has_point3D():
                prob.add_residual_block(
                    cost, loss, [im.qvec, im.tvec, rec.points3D[p.point3D_id].xyz, cam.params]
                )
        prob.set_manifold(im.qvec, pyceres.QuaternionManifold())
    for cam in rec.cameras.values():
        prob.set_parameter_block_constant(cam.params)
    for p in rec.points3D.values():
        prob.set_parameter_block_constant(p.xyz)

    return prob


def compute_pose_covariances(problem: pyceres.Problem, qts: dict) -> dict:
    """Compute the covariance of the poses in a reconstruction.

    Args:
        problem (pyceres.Problem): Ceres problem.
        qts (dict): Dictionary of quaternions and translations.

    Returns:
        dict: Dictionary of pose covariances.
    """
    opts = pyceres.CovarianceOptions(dict(num_threads=2))
    cov = pyceres.Covariance(opts)
    blocks = []
    for q, t in qts.values():
        blocks.extend([(t, t), (q, q), (q, t)])
    if not cov.compute(blocks, problem):
        return {}
    covariances = {}
    for k, (q, t) in qts.items():
        c_qq = cov.get_covariance_block_tangent_space(q, q, problem)
        c_tt = cov.get_covariance_block(t, t)
        c_qt = cov.get_covariance_block_tangent_space(q, t, problem)
        covariances[k] = np.block([[c_qq, c_qt], [c_qt.T, c_tt]])
    return covariances


def run(reconstruction: pycolmap.Reconstruction, args: argparse.Namespace) -> None:
    """Run pose uncertainty analysis on a reconstruction.

    Args:
        reconstruction (pycolmap.Reconstruction): Colmap reconstruction.
        args (argparse.Namespace): Arguments.
    """
    plot_model(reconstruction, 0, args.out)

    # run ba
    problem = opt_poses(reconstruction)
    solve(problem)

    # get covars
    qts = {im.name: (im.qvec, im.tvec) for im in reconstruction.images.values()}
    qts_cov = compute_pose_covariances(problem, qts)

    if qts_cov == {}:
        print("No covariances")
        return

    # get uncertainties
    keys = qts.keys()

    poses = get_poses(reconstruction)
    cam_tvecs = np.array([poses[k] for k in keys])
    covars = np.stack([qts_cov[k][3:, 3:] for k in keys])
    uncs = np.sqrt(np.linalg.eig(covars)[0].max(1))

    # plot uncertainties histogram
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.hist(uncs * 100, bins=100)
    ax.set_title("Uncertainty Histogram")

    if args.out is not None:
        plt.savefig(os.path.join(args.out, "hist.png"))
    else:
        plt.show()

    # plot
    plot_uncs(reconstruction, uncs, cam_tvecs, 0, args.out)

    json_file = dict(zip(keys, uncs))
    if args.out is not None:
        with open(os.path.join(args.out, "uncertainties.json"), "w") as f:
            json.dump(json_file, f, indent=4)


def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tilt", type=float, default=-45)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    reconstruction = pycolmap.Reconstruction(args.model_path)

    run(reconstruction, args)


if __name__ == "__main__":
    main()
