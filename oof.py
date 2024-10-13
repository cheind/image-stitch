import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf


def _softmax(x: np.ndarray, temp: float = 1.0, axis: int = 0):
    x = x / temp
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def plot_weights(weights):
    n = weights.shape[0]
    ncols = 4
    nrows = (n // ncols) + (n % ncols > 0)
    ir, ic = np.unravel_index(np.arange(n), (nrows, ncols))

    W, H = plt.figaspect(nrows / ncols)
    fig = plt.figure(figsize=(W * 2, H * 2), layout="constrained")
    gs = fig.add_gridspec(nrows, ncols, wspace=0, hspace=0)

    for idx, i, j in zip(range(n), ir, ic):
        ax = fig.add_subplot(gs[i, j])
        ax.imshow(weights[idx])


def weights_from_color(cfg: OmegaConf, imgs: np.ndarray, weights: np.ndarray):

    imgs_lab = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2LAB) for i in imgs], 0)
    tgt = np.array(cfg.color.target_lab)

    d = np.linalg.norm(imgs_lab - tgt.reshape(1, 1, 1, 3), axis=-1, keepdims=True)
    w = _softmax(-d, temp=cfg.color.T, axis=0)
    return w


def weights_from_outlier(cfg: OmegaConf, imgs: np.ndarray, weights: np.ndarray):

    imgs_gray = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs], 0)
    imgs_gray = np.expand_dims(imgs_gray, -1)
    mean = imgs_gray.mean(0, keepdims=True)
    var = (imgs_gray - mean) ** 2
    w = _softmax(var, temp=cfg.outlier.T, axis=0)
    return w


def weights_from_blend_masks(cfg: OmegaConf, imgs: np.ndarray, weights: np.ndarray):
    w = weights / (weights.sum(0, keepdims=True) + 1e-12)
    return w


def main():
    cfg = OmegaConf.merge(OmegaConf.load("oof.yml"), OmegaConf.from_cli())

    data = np.load(cfg.rawpath)
    imgs = data["imgs"]
    imgs = imgs.astype(np.float32) / 255

    weights = data["weights"]

    mode_to_fn = {
        "color": weights_from_color,
        "outlier": weights_from_outlier,
        "default": weights_from_blend_masks,
    }
    w = mode_to_fn[cfg.weight_filter](cfg, imgs, weights)
    w = np.where(w < cfg.integrate.min_weight, 0.0, w)

    plot_weights(w)
    out = ((w * imgs).sum(0) * 255).astype(np.uint8)

    fs = plt.figaspect(out.shape[0] / out.shape[1])
    fig, ax = plt.subplots(figsize=(fs[0] * 2, fs[1] * 2))
    ax.imshow(out[..., ::-1], origin="upper")
    ax.set_aspect("equal")
    now = time.strftime("%Y%m%d-%H%M%S")
    fig.savefig(f"tmp/oof-{now}.png", dpi=300)

    fig, ax = plt.subplots(figsize=(fs[0] * 2, fs[1] * 2))
    ax.imshow(w.max(0), origin="upper")
    ax.set_aspect("equal")
    now = time.strftime("%Y%m%d-%H%M%S")
    fig.savefig(f"tmp/oof-weights-{now}.png", dpi=300)

    if cfg.show:
        plt.show()


if __name__ == "__main__":
    # python oof.py rawpath=tmp\stitch-20241011-193007.npz
    main()
