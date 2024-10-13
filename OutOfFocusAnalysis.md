
---
title: "Exploiting Out-of-Focus Properties in Image Stitching"
subtitle: "https://github.com/cheind/image-stitch"
author: "Christoph Heindl"
date: "2024-10"
comment: "pandoc -f markdown+tex_math_dollars+yaml_metadata_block+implicit_figures --citeproc OutOfFocusAnalysis.md -o OutOfFocusAnalysis.pdf"
bibliography: biblio.bib
---

# Introduction

In [PlanarImageStitching](PlanarImageStitching.md), we highlighted the issue of ghosting artifacts that arise during image stitching when the target plane assumption is not met. Objects that do not lie on the assumed target plane can appear misaligned or scattered into inconsistent positions across different viewpoints, leading to visual distortions. 

This effect can be exploited to enhance visibility of in-focus objects that are heavily occluded by out-of-focus ones. This technique has valuable applications in areas such as search-and-rescue operations and ground fire detection, where infrared temperature signatures of people or fires may be partially obscured by trees or foliage, allowing for more effective detection despite these visual obstructions [@kurmi2018airborne].

# Scenario

Here, we present a simplified search-and-rescue scenario. The images below are a subset of 17 captured by a 'drone' flying over a forested area (simulated using plants) that conceals a hidden object (a rubber duck). A calibration pattern is included in the scene to facilitate referencing specific viewpoints.

![](etc/oof_ducky.jpg)

# Principle

Out-of-focus makes use of the fact that objects not in the focal plane get dispersed during stitching. We illustrate this by comparing two stitching planes. One 4mm above ground (chessboard) and one parallel to it but lifted by 50cm.

Focus plane 4mm above ground     |  Focus plane 50cm above ground
:-------------------------:|:-------------------------:
![](etc/oof_stitch_4.png)  |  ![](etc/oof_stitch_500.png)

As we raise the target or focal plane, the chessboard pattern becomes increasingly distorted and dispersed across the images. In contrast, portions of the plant become more sharply defined in the second image as the focal plane shifts.

# Method

Our objective is to reconstruct the hidden rubber duck as clearly as possible, enhancing the chances of successful rescue. 

Our method follows these steps:

1. **Define a focal plane** where we expect the rubber duck to be located.
1. **Warp the images** to a virtual camera, with the image plane corresponding to the focal plane chosen in step 1.
1. **Calculate integration weights** that assign a confidence score to each pixel in each view, based on how well it might contain our rubber duck.
1. **Generate the final image** by integrating the pixel values from the warped images, using the computed weights to determine the most reliable information for each pixel position.


Aside from steps 1 and 3, this approach aligns with the default image stitching pipeline. 

## Focal plane

Our rubber duck is approximately 4 cm tall, so we position the focal plane parallel to the ground but shifted upward by about 3 cm to align with the expected height of the duck.

## Integration weights
The integration weights determine the relative importance of pixels of each view during integration. By carefully choosing the weights, we increase the probability of spotting the duck. Here we compare three different approaches.

### Baseline

For the baseline weights, we use the default stitching approach, which generally assigns nearly equal weight to each view for each pixel, except at the image borders where the weights gradually decrease.

### Target Color Prior

Given that rubber ducks are predominantly yellow, a natural strategy is to assign greater importance to pixels that are closer in color to yellow, while reducing the weight of pixels that deviate from this hue. 

We measure the color distance from yellow in the `L*a*b` color space to more closely mimic perceptual color differences, as this color space is designed to align with human visual perception. 

### Outlier Prior

If we assume the rubber duck is more often obscured than visible, we can exploit the fact that the duck is underrepresented in the data. 

We compute the weights $w_{ijv}$ for pixel $i,j$ of view $v$ as
$$
\begin{aligned}
    g_{ijv} &=(I_{ijv}-\bar{I}_{ij})^2\\
    w_{ijv} &=\frac{\exp(g_{ijv}/T)}{\sum_k\exp(g_{ijk}/T)},    
\end{aligned}
$$

where $I$ is a gray scale image, $\bar{I}$ is the mean grayscale value across views, $T$ is temperature scaling that allows us to gear the sharpness of the resulting normalization.

# Evaluation



## Baseline

As a baseline we use the standard stitching approach and choose as focal plane the ground plane lifted by 3cm.

## 



## Principle

## Weights from outliers

## Weights from color

# References