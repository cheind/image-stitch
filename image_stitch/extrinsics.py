import numpy as np
import cv2


def chessboard_points(pattern: tuple[int, int, float]):
    """Returns Nx3 array of chessboard model points."""
    W, H, S = pattern[:3]

    x, y = np.meshgrid(range(W), range(H))

    x = x.reshape(-1)
    y = y.reshape(-1)

    xyz = np.stack([x, y, np.zeros_like(x)], -1).astype(np.float32)
    xyz *= S

    return xyz


def undistort(
    files: list[str],
    K: np.ndarray,
    D: np.ndarray,
    alpha: float = 1.0,
    outsize: tuple[int, int] = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Fisheye undistortion for images taken by the same camera.

    Params:
        files: list of N paths to images
        K: (3,3) intrinsic matrix
        D: OpenCV fisheye distortion coefficients
        alpha: scaling factor for resulting focal length
        outsize: optional output dims (W,H) of undistorted image

    Returns:
        imgs: array of undistorted images
        K: associated intrinsic matrix
    """

    imgs = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in files]
    in_size = imgs[0].shape[:2][::-1]

    if outsize is None:
        outsize = in_size

    Kout = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        D,
        in_size,
        np.eye(3),
        balance=max(min(alpha, 1.0), 0.0),
        new_size=outsize,
        fov_scale=1.0,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), Kout, outsize, cv2.CV_32F
    )

    imgs = [
        cv2.remap(
            img,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        for img in imgs
    ]
    # imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    return imgs, Kout


def find_extrinsics(
    imgs: list[np.ndarray],
    K: np.ndarray,
    pattern: tuple[int, int, float],
    reverse: bool,
) -> list[np.ndarray]:
    """Returns the extrinsics wrt. to a chessboard pattern.

    Params:
        imgs: list of N images (H,W,3)
        K: (3,3) intrinsic matrix
        pattern: Chessboard pattern defined as (#rows,#cols,square size)

    Returns:
        t_cam_world: list of N extrinsic matrices (4,4) representing the pose
            of the chessboard wrt. camera.

    """
    t_cam_world = []
    mpts = chessboard_points(pattern)

    for img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        success = False
        for factor in [1.0, 0.5, 0.25]:

            ret, corners = cv2.findChessboardCornersSB(
                cv2.resize(gray, None, fx=factor, fy=factor),
                pattern[:2],
                flags=cv2.CALIB_CB_EXHAUSTIVE,
                # | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if ret:
                success = True
                corners = (corners + 0.5) / factor - 0.5
                break
        if not success:
            t_cam_world.append(None)
            continue

        if reverse:
            corners = corners[::-1]

        ret, rvec, tvec = cv2.solvePnP(
            mpts.reshape(-1, 1, 3),
            corners.reshape(-1, 1, 2),
            K,
            None,
        )

        if not ret:
            t_cam_world.append(None)
        else:
            m = np.eye(4)
            m[:3, :3] = cv2.Rodrigues(rvec)[0]
            m[:3, 3:4] = tvec
            t_cam_world.append(m)

    return t_cam_world
