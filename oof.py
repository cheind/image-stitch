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

    tol = 60
    w = np.exp(-np.maximum((d - tol), 0) * 0.1)

    return w


def main():
    cfg = OmegaConf.merge(OmegaConf.load("oof.yml"), OmegaConf.from_cli())

    data = np.load(cfg.rawpath)
    imgs = data["imgs"]
    imgs = imgs.astype(np.float32) / 255

    weights = compute_weights_by_color(cfg, imgs)

    # background
    imgs = np.concatenate((imgs, np.ones_like(imgs[:1])), 0)

    weights = np.concatenate(
        (weights, np.clip(1.0 - weights.sum(0, keepdims=True), 0.0, 1.0)), 0
    )

    weights = weights / weights.sum(0, keepdims=True)

    out = ((weights * imgs).sum(0) * 255).astype(np.uint8)
    fig, ax = plt.subplots(figsize=plt.figaspect(out.shape[0] / out.shape[1]))
    ax.imshow(out[..., ::-1], origin="upper")
    ax.set_aspect("equal")
    plt.show()

    # istack = istack.astype(float) / 255.0

    # d = np.sqrt(3) - np.linalg.norm(
    #     istack - np.array([0, 255, 255]).reshape(1, 1, 1, 3) / 255,
    #     2,
    #     axis=-1,
    #     keepdims=True,
    # )
    # d[d < 0.7] = 1e-6
    # d /= d.sum(0, keepdims=True)

    # final = ((d * istack).sum(0) * 255).astype(np.uint8)

    # fig, ax = plt.subplots(figsize=plt.figaspect(img_c.shape[0] / img_c.shape[1]))
    # ax.imshow(img_c[..., ::-1], origin="upper")
    # ax.set_aspect("equal")

    # fig, ax = plt.subplots(figsize=plt.figaspect(img_c.shape[0] / img_c.shape[1]))
    # ax.imshow(final[..., ::-1], origin="upper")
    # ax.set_aspect("equal")

    # plt.show()


if __name__ == "__main__":
    # python oof.py -f data/oof -r -1 -px-per-m 1000 -z 0.03
    main()
