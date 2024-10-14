from omegaconf import OmegaConf
from pathlib import Path
from image_stitch import extrinsics
import numpy as np
import cv2


def parse_intrinsics(path: Path):
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            if line[0] == "#":
                continue
            els = line.split(" ")
            camera = {}
            camera_id = int(els[0])
            camera["id"] = camera_id
            camera["w"] = float(els[2])
            camera["h"] = float(els[3])
            camera["fx"] = float(els[4])
            camera["fy"] = float(els[4])
            camera["k1"] = 0
            camera["k2"] = 0
            camera["k3"] = 0
            camera["k4"] = 0
            camera["p1"] = 0
            camera["p2"] = 0
            camera["cx"] = camera["w"] / 2
            camera["cy"] = camera["h"] / 2
            camera["is_fisheye"] = False
            if els[1] == "SIMPLE_PINHOLE":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
            elif els[1] == "PINHOLE":
                camera["fy"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
            elif els[1] == "RADIAL":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
                camera["k2"] = float(els[8])
            elif els[1] == "OPENCV":
                camera["fy"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
                camera["k1"] = float(els[8])
                camera["k2"] = float(els[9])
                camera["p1"] = float(els[10])
                camera["p2"] = float(els[11])
            elif els[1] == "SIMPLE_RADIAL_FISHEYE":
                camera["is_fisheye"] = True
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
            elif els[1] == "RADIAL_FISHEYE":
                camera["is_fisheye"] = True
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
                camera["k2"] = float(els[8])
            elif els[1] == "OPENCV_FISHEYE":
                camera["is_fisheye"] = True
                camera["fy"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
                camera["k1"] = float(els[8])
                camera["k2"] = float(els[9])
                camera["k3"] = float(els[10])
                camera["k4"] = float(els[11])
            else:
                print("Unknown camera model ", els[1])

        cameras[camera_id] = camera
    return cameras


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def parse_images(path: Path):
    i = 0
    images = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if i % 2 == 0:
                continue  # 3d points
            elems = line.split(
                " "
            )  # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)

            fname = "".join(elems[9:])
            image_id = int(elems[0])
            qvec = np.array(tuple(map(float, elems[1:5])))
            tvec = np.array(tuple(map(float, elems[5:8])))
            R = qvec2rotmat(-qvec)
            t = tvec.reshape([3, 1])
            m = np.concatenate(
                [
                    np.concatenate([R, t], 1),
                    np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4]),
                ],
                0,
            )
            images[image_id] = {"fname": fname, "t_cam_world": m}

    return images


def parse_points(path: Path):
    points = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            elems = line.split(" ")
            xyz = [float(e) for e in elems[1:4]]
            err = float(elems[7])
            if err > 1.0:
                continue
            points.append(xyz)
    return np.array(points).reshape(-1, 3)


def step1_extract_colmap(cfg: OmegaConf):
    colmap_path = Path(cfg.colmapdir)
    assert colmap_path.is_dir()

    image_path = Path(cfg.imagedir)
    assert image_path.is_dir()

    alpha_path = None
    ext = ".jpg"
    if "alphadir" in cfg:
        alpha_path = Path(cfg.alphadir)
        assert alpha_path.is_dir()  # multimodal alpha channel
        ext = ".png"

    outdir = Path(cfg.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    cams = parse_intrinsics(colmap_path / "cameras.txt")
    images = parse_images(colmap_path / "images.txt")
    points = parse_points(colmap_path / "points3d.txt")

    K = np.eye(3)
    K[0, 0] = cams[1]["fx"]
    K[1, 1] = cams[1]["fy"]
    K[0, 2] = cams[1]["cx"]
    K[1, 2] = cams[1]["cy"]

    D = np.array(
        [cams[1]["k1"], cams[1]["k2"], cams[1]["p1"], cams[1]["p2"], cams[1]["k3"]]
    )

    fnames_in = [img["fname"] for img in images.values()]
    fnames_out = [str(Path(f).with_suffix(ext)) for f in fnames_in]

    # Process RGB
    imgs, _ = extrinsics.undistort(
        [str(image_path / f) for f in fnames_in],
        K,
        D,
        outsize=None,
    )

    # Process alpha channel
    if alpha_path is not None:
        h, w = imgs[0].shape[:2]
        imgs_alpha, _ = extrinsics.undistort(
            [str(alpha_path / f) for f in fnames_in],
            K,
            D,
            outsize=None,
            prescalesize=(w, h),
            openmode=cv2.IMREAD_GRAYSCALE,
        )
        imgs_alpha = np.expand_dims(imgs_alpha, -1)
        imgs = np.concatenate((imgs, imgs_alpha), -1)

    for img, fname in zip(imgs, fnames_out):
        cv2.imwrite(outdir / fname, img)

    np.savez(
        outdir / "data.npz",
        K=K,
        t_cam_world=[i["t_cam_world"] for i in images.values()],
        fnames=fnames_out,
    )

    # Process in meshlab to find transform and scale -> step2
    np.savetxt(outdir / "points.csv", points, delimiter=" ")


def step2_correct_transform(cfg: OmegaConf):
    inpath = Path(cfg.npzfile)
    assert inpath.is_file()

    d = np.load(inpath)
    ddict = {k: d[k] for k in d.files}
    tc = np.loadtxt(cfg.transform).reshape(4, 4)
    tc = np.linalg.inv(tc)

    t_cam_world = np.stack([tc @ t for t in ddict["t_cam_world"]], 0)
    t_cam_world[:, :3, 3] /= cfg.get("scale", 1.0)

    ddict["t_cam_world"] = t_cam_world

    np.savez(inpath.with_suffix(".corrected.npz"), **ddict)


def main():
    cfg = OmegaConf.from_cli()

    step = cfg.get("step", 1)
    if step == 1:
        step1_extract_colmap(cfg)
    else:
        step2_correct_transform(cfg)


if __name__ == "__main__":
    main()
