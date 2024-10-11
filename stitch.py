import argparse
import time
from glob import glob

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as mplPath

from omegaconf import OmegaConf
from pathlib import Path

import image_stitch as ist


def main():
    cfg = OmegaConf.merge(OmegaConf.load("stitch.yml"), OmegaConf.from_cli())
    base_path = Path(cfg.basepath)

    # Load images, intrinsics and precomputed extrinsics
    files = sorted(glob(str(base_path / "*.jpg")))[: cfg.nimages]
    imgs = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in files]
    data = np.load(str(base_path / "data.npz"))
    K_cam = data["K"]
    t_cam_world = data["t_cam_world"][: cfg.nimages]

    # Setup pi wrt. to world. The used chessboard is 4mm thick,
    # so the pi is essentially offsetted 4mm in +z
    t_world_plane = np.eye(4)
    t_world_plane[2, 3] = 0.004

    mode = cfg[cfg.mode]
    if mode.idx >= 0:
        # Stitch in physical camera
        K_ref = K_cam
        t_ref_world = t_cam_world[mode.idx]
        extent = list(mode.extent)
    else:
        # Stitch in virtual camera corresponding to pi
        px_per_m = mode.px_per_m
        K_ref = np.eye(3)
        K_ref[0, 0] = K_ref[1, 1] = px_per_m

        t_plane_ref = np.eye(4)
        t_plane_ref[2, 3] = -1
        t_world_plane[2, 3] += -mode.z  # -z towards ceiling
        t_ref_world = np.linalg.inv(t_world_plane @ t_plane_ref)
        extent = [i * mode.px_per_m for i in mode.extent]

    img_c, K_c, xy_c, w_imgs, w_weights = ist.stitch(
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
        mplPath.MOVETO,
        mplPath.LINETO,
        mplPath.LINETO,
        mplPath.LINETO,
        mplPath.CLOSEPOLY,
    ]

    xys = np.concatenate((xy_c, np.zeros((len(xy_c), 1, 2))), 1)  # (N,5,2)

    paths = [mplPath(xy, codes) for xy in xys]
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
    if cfg.show:
        plt.show()
    if cfg.save_raw:
        np.savez_compressed(
            f"tmp/stitch-{now}.npz",
            img=img_c,
            K=K_c,
            xy=xy_c,
            imgs=w_imgs,
            weights=w_weights,
        )


if __name__ == "__main__":
    main()
