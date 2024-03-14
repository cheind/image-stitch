# Planar Image Stitching

The code in this repository demonstrates stitching of images captured by cameras observing a **planar target**. We analytically derive homographies by assuming the camera pose wrt. to the target are known.

The following image shows views stitched in the green reference camera frame. 

![](etc/stitch-cam3.png)

The following image shows the same scene stitched in a virtual camera whose image plane aligns with the ground plane, having a pixel resolution of 500px/

![](etc/stitch-pi-500.png)

Both images exhibit ghosting artefacts caused by blending moving objects and warping objects that violate the in-target-plane assumption. 

## Theory

See [PlanarImageStitching.pdf](PlanarImageStitching.pdf) for background information on the stitching process.

## Usage

The code provided is for demonstration purposes only. It is limited to a scenario in which a moving fisheye camera observes a ground floor. The extrinsics are computed from knowing the fisheye intrinsics/distortions and the pattern configuration.

```shell
# Stitch in camera 3 view (index starting at zero)
python stitch.py -r 2
```

![](etc/stitch-cam3.png)

```shell
# Stitch in plane pi using px/m of 500
python stitch.py -r -1 -px-per-m 500
```

![](etc/stitch-pi-500.png)

```shell
# Stitch in plane pi using px/m of 10
python stitch.py -r -1 -px-per-m 10
```

![](etc/stitch-pi-10.png)