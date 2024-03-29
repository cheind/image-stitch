import argparse
import time
from glob import glob

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

import image_stitch as ist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, help="Use first N images", default=-1)
    parser.add_argument(
        "-r",
        type=int,
        help="Choose reference camera to stitch in, -1 for plane",
        default=0,
    )
    parser.add_argument(
        "-px-per-m",
        type=float,
        help="Resolution of virtual camera when r=-1",
        default=1000,
    )
    args = parser.parse_args()

    # Load images, intrinsics and precomputed extrinsics
    files = sorted(glob("data/*.jpg"))[: args.n]
    imgs = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in files]
    data = np.load("data/data.npz")
    K_cam = data["K"]
    t_cam_world = data["t_cam_world"][: args.n]

    # Setup pi wrt. to world. The used chessboard is 4mm thick,
    # so the pi is essentially offsetted 4mm in +z
    t_world_plane = np.eye(4)
    t_world_plane[2, 3] = 0.004

    if args.r >= 0:
        # Stitch in physical camera
        K_ref = K_cam
        t_ref_world = t_cam_world[args.r]
        extent = [-5000, 5000, -5000, 5000]
    else:
        # Stitch in virtual camera corresponding to pi
        px_per_m = args.px_per_m
        K_ref = np.eye(3)
        K_ref[0, 0] = K_ref[1, 1] = px_per_m

        t_plane_ref = np.eye(4)
        t_plane_ref[2, 3] = -1
        t_ref_world = np.linalg.inv(t_world_plane @ t_plane_ref)
        extent = [-3 * px_per_m, 5 * px_per_m, -1 * px_per_m, 2 * px_per_m]

    img_c, K_c, xy_c = ist.stitch(
        imgs,
        K_cam,
        t_cam_world,
        t_world_plane,
        K_ref,
        t_ref_world,
        extent=extent,
    )

    fig, ax = plt.subplots(figsize=plt.figaspect(img_c.shape[0] / img_c.shape[1]))
    ax.imshow(img_c[..., ::-1], origin="upper")
    ax.set_aspect("equal")
    ax.autoscale(False)

    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]

    xys = np.concatenate((xy_c, np.zeros((len(xy_c), 1, 2))), 1)  # (N,5,2)

    paths = [Path(xy, codes) for xy in xys]
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    for idx, path in enumerate(paths):
        patch = patches.PathPatch(
            path,
            lw=1,
            facecolor="none",
            edgecolor=colors[idx % len(colors)],
            label=f"{idx:02d}",
            ls="--",
        )
        ax.add_patch(patch)
    now = time.strftime("%Y%m%d-%H%M%S")
    fig.savefig(f"tmp/stitch-{now}.png")
    plt.show()


if __name__ == "__main__":
    main()
