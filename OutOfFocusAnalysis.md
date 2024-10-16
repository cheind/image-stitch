
# Exploiting Out-of-Focus Properties in Image Stitching
Christoph Heindl, 2024/10, https://github.com/cheind/image-stitch


# Introduction

In [PlanarImageStitching.md](./PlanarImageStitching.md), we highlighted the issue of ghosting artifacts that arise during image stitching when the in-target-plane assumption is not met. Objects that protrude from the target plane will appear misaligned, scattered, and lack opacity in the resulting integrated image.

Surprisingly, exploiting this artifact can create see-through effects that enhance the visibility of in-focus objects, even when they are significantly obscured by out-of-focus elements. This technique is particularly valuable in search-and-rescue operations and ground fire detection, where RGB or thermal signals may be obscured by trees or foliage. For instance, placing the target plane near the ground (in-focus) reduces the impact of trees and foliage (out-of-focus) on the integrated image, enhancing detection rates despite visual obstructions[^1].

[^1]: Kurmi, Indrajit, David C. Schedl, and Oliver Bimber. "Airborne optical sectioning." Journal of Imaging 4.8 (2018): 102.


# Scenario

Here, we consider a simplified search-and-rescue scenario. The RGB images shown below are a subset of 17 captured by a 'drone' flying over a forested area (simulated using plants) that conceals a hidden non-moving object (a rubber duck). 

![](etc/oof_ducky.jpg)

The camera is assumed to be calibrated and viewpoint transformations between images are assumed to be known and have been pre-computed from the visible calibration pattern. The ground plane is parallel to the plane induced by the chessboard pattern, but shifted down by the thickness of the board.

You can download the dataset from [here](https://drive.google.com/file/d/10h1QwlkxLyLN0XluWZdBL7DUDQ0p9CLN/view?usp=sharing).

# Principle

Out-of-focus makes use of the fact that objects not in the focal plane get dispersed during stitching. We illustrate this by comparing two stitching planes. One 4mm above ground (chessboard) and one parallel to it but lifted by 50cm.

Focus plane 4mm above ground     |  Focus plane 50cm above ground
:-------------------------:|:-------------------------:
![](etc/oof_stitch_4.png)  |  ![](etc/oof_stitch_500.png)

As we increase the distance of the target/focal plane to the ground, the chessboard pattern becomes increasingly distorted dispersed across the integrated stiching image. In contrast, portions of the plant become more sharply defined in the second image as the focal plane shifts.

# Properties

The amount of 'dispersion' can be quantified by determining the radius of the [circle of confusion](https://en.wikipedia.org/wiki/Circle_of_confusion). For simplicity, we consider two images taken by camera $i$ and $j$ at height $H$ above ground and distance $D$ apart. Additionally we assume the presence of an occluder of height $h$ above ground and focus on the two eye-rays that pass through the occluder.

![](etc/oof_properties.png)

The circle of confusion with radius $r$ represents the lateral displacement of the occluder as imaged by the target plane $\pi$. The larger $r$ the more dispersed the occluder will appear in the integrated image. We aim to maximize $r$ for occluders as it increases the likelihood that later processing will be able to dimish their effects on the integral image.

From basic trigonometry we have
```math
r=\tan(\alpha/2)h.
```

Hence, $r$ is proportional to both $\alpha$ and $h$. That is, for a fixed occluder height $h$ we can maximize &#8593;$r$ by &#8593;$\alpha$ by &#8595;$H$ or &#8593;$D$.


# Objective

Our goal is to enhance the likelihood of successfully locating and rescuing the rubber duck. By applying the out-of-focus principle to reduce the visibility of the surrounding forest structure, we can improve the reconstruction accuracy of in-focus objects. This increases the probability of detecting the rubber duck hidden beneath the foliage.

# Method

Our method is comprised of the following steps:

1. **Define a focal plane** where we expect the rubber duck to be located.
1. **Warp the images** to a virtual camera, with the image plane corresponding to the focal plane chosen in step 1.
1. **Calculate integration weights** that assign a confidence score to each pixel in each view, based on how well it might contain our rubber duck.
1. **Generate the final image** by integrating the pixel values from the warped images, using the computed weights to determine the most reliable information for each pixel position.


Aside from steps 1 and 3, this approach aligns with the default image stitching pipeline as described in [PlanarImageStitching](./PlanarImageStitching.md).

## Focal plane

Our rubber duck is approximately 4 cm tall, so we position the focal plane parallel to the ground but shifted upward by about 3 cm to align with the expected height of the duck.

```shell
# Perform stitching and save intermediate results. 
# This generates 'tmp/stitch-<DATA>-<TIME>.npz'
python stitch.py basepath=data/oof plane.idx=-1 plane.extent="[-0.5,1,0,1.5]" plane.z=0.03 save_raw=true
```

## Weighting strategies
The integration weights determine the relative importance of pixels of each view during integration. By carefully choosing the weights, we increase the probability of spotting the duck. Here we compare three different approaches.

### Baseline

For the baseline weights, we use the default stitching approach, which generally assigns nearly equal weight to each view for each pixel, except at the image borders where the weights gradually decrease.

### Target Color Assumption

Given that rubber ducks are predominantly yellow, a natural strategy is to assign greater importance to pixels that are closer in color to yellow, while reducing the weight of pixels that deviate from this hue. 

We measure the color distance from yellow in the `L*a*b` color space to more closely mimic perceptual color differences, as this color space is designed to align with human visual perception. 

We then compute the weights $w_{ijv}$ for pixel $i,j$ of view $v$ and target color $c$ as

```math
\begin{aligned}
    g_{ijv} &=d(I^{\textrm{LAB}}_{ijv}, c^{\textrm{LAB}})\\
    w_{ijv} &=\frac{\exp(g_{ijv}/T)}{\sum_k\exp(g_{ijk}/T)},    
\end{aligned}
```

where $I^{\textrm{LAB}}_{ijv}$ is the image of view $v$ in LAB color space and $c^{\textrm{LAB}}$ is the target color in LAB space and $T$ is temperature scaling that allows us to gear the sharpness of the resulting normalization.

### Outlier Assumption

If we assume the rubber duck is more often obscured than visible, we can exploit the fact that the duck is underrepresented in the data. 

We compute the weights $w_{ijv}$ for pixel $i,j$ of view $v$ as
```math
\begin{aligned}
    g_{ijv} &=(I^{\textrm{L}}-\bar{I}^{\textrm{L}}_{ij})^2\\
    w_{ijv} &=\frac{\exp(g_{ijv}/T)}{\sum_k\exp(g_{ijk}/T)},    
\end{aligned}
```

where $I^{\textrm{L}}$ is the luminosity image, $\bar{I}^{\textrm{L}}_{ij}$ is the luminosity grayscale value across views, $T$ is temperature scaling that allows us to gear the sharpness of the resulting normalization.

# Evaluation

We conduct a brief evaluation of the different weighting strategies and provide a subjective assessment of the results.

## Baseline

With the baseline weighting strategy, the rubber duck becomes visible, but it appears blurred and mixed with the surrounding leaves, making it easy for an operator to overlook.

![](etc/oof-baseline-20241013-082545.png)

```shell
# Integrate intermediate results by baseline method
# Generates tmp/oof-weights-baseline-<DATE>-<TIME>.png
python oof.py rawpath=tmp/stitch-<DATA>-<TIME>.npz weight_filter=baseline
```

## Color

For yellow rubber ducks, the color-weighting strategy is highly effective, making the rubber duck stand out clearly against its surroundings.

![](etc/oof-color-20241013-082612.png)

```shell
# Integrate intermediate results by color weighting
# Generates tmp/oof-weights-color-<DATE>-<TIME>.png
python oof.py rawpath=tmp/stitch-<DATA>-<TIME>.npz  weight_filter=color color.T=10
```


## Outlier

The outlier weighting scheme performs just as effectively as the color-based strategy, but it has the advantage of requiring less prior information to achieve similar results.

![](etc/oof-outlier-20241013-082634.png)

```shell
# Integrate intermediate results by outlier weighting
# Generates tmp/oof-weights-outlier-<DATE>-<TIME>.png
python oof.py rawpath=tmp/stitch-<DATA>-<TIME>.npz weight_filter=outlier outlier.T=0.05
```

# Future work

This work is intended as an introductory text on the topic of out-of-focus analysis, also known as "Airborne Optical Sectioning"[¹]. We've briefly covered the basic method and evaluated three different weighting strategies on a challenging rubber duck rescue scenario. Many more advanced weighting methods remain unexplored.

<!-- 
## Ideas
Just ranting some possible improvements
 - optimize local accutance of the image. use a conv net to inter-relate neighboring pixels.
 - use multiple target planes at different heights to capture the height of the rubber duck.
 -->

# References
