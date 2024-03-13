import numpy as np
import cv2
import warnings

from .blending import compute_blend_weights

U = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]])
D = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])


def stitch(
    imgs: list[np.ndarray],
    K_cam: np.ndarray,
    t_cam_world: list[np.ndarray],
    t_world_plane: np.ndarray,
    K_ref: np.ndarray,
    t_ref_world: np.ndarray,
    weights: np.ndarray | list[np.ndarray] | None = None,
    extent: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Compute the unshifted homographies

    P1 = K_ref @ D @ t_ref_world @ t_world_plane @ U

    Hs = []
    for t_i_world in t_cam_world:
        P2 = K_cam @ D @ t_i_world @ t_world_plane @ U
        H_r_i = P1 @ np.linalg.inv(P2)
        Hs.append(H_r_i)

    # Determine the extent of the resulting image
    H, W = imgs[0].shape[:2]
    bbox = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    bbox = np.concatenate((bbox, np.ones((4, 1))), -1)

    corners = []
    for idx, H_r_i in enumerate(Hs):
        xy = bbox @ H_r_i.T
        xy = xy[:, :2] / xy[:, 2:3]
        corners.append(xy)
        isconvex = cv2.isContourConvex(xy.reshape(-1, 1, 2).astype(np.float32))
        if not isconvex:
            warnings.warn(
                (
                    f"Contour {idx} is not convex, might lead to artefacts "
                    "when computing target image dimensions."
                )
            )
    corners = np.stack(corners, 0)  # (N,4,2)
    minc = corners.min(axis=(0, 1))
    maxc = corners.max(axis=(0, 1))

    if extent is not None:
        minc = np.clip(minc, [extent[0], extent[2]], None)
        maxc = np.clip(maxc, None, [extent[1], extent[3]])

    # Compute shift transform to avoid negative pixels and compute H-hat
    S = np.eye(3)
    S[:2, 2] = -minc
    outsize = np.round(maxc - minc).astype(int)
    HHs = [S @ H for H in Hs]
    HHs = [H / H[-1, -1] for H in HHs]

    # Compute stitched image
    if weights is None:
        weights = compute_blend_weights(imgs[0].shape[:2])

    if not isinstance(weights, list):
        weights = [weights] * len(imgs)

    fimgs = []
    fweights = []
    corners = []
    for img, w, HH_r_i in zip(imgs, weights, HHs):

        fimg = cv2.warpPerspective(img, HH_r_i, outsize)
        fimgs.append(fimg)

        fw = cv2.warpPerspective(w, HH_r_i, outsize)
        fweights.append(fw)

        xy = bbox @ HH_r_i.T
        xy = xy[:, :2] / xy[:, 2:3]
        corners.append(xy)

    fimgs = np.stack(fimgs)
    fweights = np.expand_dims(np.stack(fweights), -1)
    corners = np.stack(corners)

    img_c = (fimgs / 255.0 * fweights).sum(0) / (fweights.sum(0) + 1e-8)
    img_c = (img_c * 255.0).astype(np.uint8)
    K_c = S @ K_ref

    return img_c, K_c, corners
