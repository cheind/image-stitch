import argparse
import time
from glob import glob

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as mplPath
from pathlib import Path

import image_stitch as ist

from omegaconf import OmegaConf


def compute_weights_by_color(cfg: OmegaConf, imgs: np.ndarray):

    imgs_lab = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2LAB) for i in imgs], 0)

    yellow = np.array([97.17, -21.29, 94.41])
    d = np.linalg.norm(imgs_lab - yellow.reshape(1, 1, 1, 3), axis=-1, keepdims=True)

    tol = 70
    w = np.exp(-np.maximum((d - tol), 0) * 0.05)
    w[w < 0.5] = 0

    plt.imshow(w.max(0))

    return w


def main():
    cfg = OmegaConf.merge(OmegaConf.load("oof.yml"), OmegaConf.from_cli())

    data = np.load(cfg.rawpath)
    imgs = data["imgs"]
    imgs = imgs.astype(np.float32) / 255

    weights = compute_weights_by_color(cfg, imgs)
    weights = weights / (weights.sum(0, keepdims=True) + 1e-12)

    out = ((weights * imgs).sum(0) * 255).astype(np.uint8)
    fig, ax = plt.subplots(figsize=plt.figaspect(out.shape[0] / out.shape[1]))
    ax.imshow(out[..., ::-1], origin="upper")
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    # python oof.py rawpath=tmp\stitch-20241011-193007.npz
    main()
