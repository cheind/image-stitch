import numpy as np
import cv2


def compute_blend_weights(shape: tuple[int, int]):
    """Returns a weight prototype image for the given image shape.

    Puts more emphasis on central pixels. For an explanation see
    Readme.

    Returns:
        w: (H,W) array of weights
    """
    weights = np.ones(shape, dtype=np.uint8) * 255
    weights = np.pad(
        weights,
        [(1, 1), (1, 1)],
        mode="constant",
        constant_values=0,
    )

    weights = cv2.distanceTransform(weights, cv2.DIST_L2, 3)[1:-1, 1:-1]
    weights /= weights.max()
    return weights
    # plt.imshow(weights)
    # plt.show()
