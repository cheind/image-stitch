import cv2
import numpy as np
from glob import glob
from pathlib import Path
from image_stitch import extrinsics


def main():
    files = sorted(glob("tmp/imgs/*.JPG"))
    calib = np.load("tmp/hero12.photo.5568x4872.npz")
    pattern = (8, 5, 0.068)
    W, H = (5568 // 2, 4872 // 2)

    imgs, K_cam = extrinsics.undistort_fisheye(
        files, calib["K"], calib["D"], alpha=0.0, outsize=(W, H)
    )
    t_cam_world = extrinsics.find_extrinsics(imgs, K_cam, pattern, reverse=True)
    successes = [t is not None for t in t_cam_world]

    for img, fpath, success in zip(imgs, files, successes):
        if success:
            cv2.imwrite(f"data/{Path(fpath).stem}.jpg", img)

    np.savez(
        "data/data.npz",
        K=K_cam,
        t_cam_world=[t for t, s in zip(t_cam_world, successes) if s],
    )


if __name__ == "__main__":
    main()
