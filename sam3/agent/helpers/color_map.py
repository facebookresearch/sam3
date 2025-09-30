# Copyright (c) Facebook, Inc. and its affiliates.

"""
An awesome colormap for really neat visualizations.
Copied from Detectron, and removed gray colors.
"""

import numpy as np
import random

__all__ = ["colormap", "random_color", "random_colors"]

# fmt: off
# RGB:
# _COLORS = np.array(
#     [
#         0.000, 0.447, 0.741,
#         0.850, 0.325, 0.098,
#         0.929, 0.694, 0.125,
#         0.494, 0.184, 0.556,
#         0.466, 0.674, 0.188,
#         0.301, 0.745, 0.933,
#         0.635, 0.078, 0.184,
#         0.300, 0.300, 0.300,
#         0.600, 0.600, 0.600,
#         1.000, 0.000, 0.000,
#         1.000, 0.500, 0.000,
#         0.749, 0.749, 0.000,
#         0.000, 1.000, 0.000,
#         0.000, 0.000, 1.000,
#         0.667, 0.000, 1.000,
#         0.333, 0.333, 0.000,
#         0.333, 0.667, 0.000,
#         0.333, 1.000, 0.000,
#         0.667, 0.333, 0.000,
#         0.667, 0.667, 0.000,
#         0.667, 1.000, 0.000,
#         1.000, 0.333, 0.000,
#         1.000, 0.667, 0.000,
#         1.000, 1.000, 0.000,
#         0.000, 0.333, 0.500,
#         0.000, 0.667, 0.500,
#         0.000, 1.000, 0.500,
#         0.333, 0.000, 0.500,
#         0.333, 0.333, 0.500,
#         0.333, 0.667, 0.500,
#         0.333, 1.000, 0.500,
#         0.667, 0.000, 0.500,
#         0.667, 0.333, 0.500,
#         0.667, 0.667, 0.500,
#         0.667, 1.000, 0.500,
#         1.000, 0.000, 0.500,
#         1.000, 0.333, 0.500,
#         1.000, 0.667, 0.500,
#         1.000, 1.000, 0.500,
#         0.000, 0.333, 1.000,
#         0.000, 0.667, 1.000,
#         0.000, 1.000, 1.000,
#         0.333, 0.000, 1.000,
#         0.333, 0.333, 1.000,
#         0.333, 0.667, 1.000,
#         0.333, 1.000, 1.000,
#         0.667, 0.000, 1.000,
#         0.667, 0.333, 1.000,
#         0.667, 0.667, 1.000,
#         0.667, 1.000, 1.000,
#         1.000, 0.000, 1.000,
#         1.000, 0.333, 1.000,
#         1.000, 0.667, 1.000,
#         0.333, 0.000, 0.000,
#         0.500, 0.000, 0.000,
#         0.667, 0.000, 0.000,
#         0.833, 0.000, 0.000,
#         1.000, 0.000, 0.000,
#         0.000, 0.167, 0.000,
#         0.000, 0.333, 0.000,
#         0.000, 0.500, 0.000,
#         0.000, 0.667, 0.000,
#         0.000, 0.833, 0.000,
#         0.000, 1.000, 0.000,
#         0.000, 0.000, 0.167,
#         0.000, 0.000, 0.333,
#         0.000, 0.000, 0.500,
#         0.000, 0.000, 0.667,
#         0.000, 0.000, 0.833,
#         0.000, 0.000, 1.000,
#         0.000, 0.000, 0.000,
#         0.143, 0.143, 0.143,
#         0.857, 0.857, 0.857,
#         1.000, 1.000, 1.000
#     ]
# ).astype(np.float32).reshape(-1, 3)
# fmt: on



# _COLORS = np.array(
#     [
#         # 1. Yellow (Highest luminance, very sharp)
#         1.000, 1.000, 0.000,
#         # 2. Lime Green (Peak sensitivity for the human eye)
#         0.000, 1.000, 0.000,
#         # # 3. Cyan (Very bright and high contrast)
#         # 0.000, 1.000, 1.000,
#         # 4. Magenta (Strong contrast against common backgrounds)
#         1.000, 0.000, 1.000,
#         # 5. Red (Classic high-alert, sharp color)
#         1.000, 0.000, 0.000,
#         # 6. Orange (Stimulates both red and green cones strongly)
#         1.000, 0.498, 0.000,
#         # 7. Chartreuse (A sharp, electric yellow-green)
#         0.498, 1.000, 0.000,
#         # 8. Spring Green (A bright, high-energy green-cyan)
#         0.000, 1.000, 0.498,
#         # # 9. Azure (A sharp, electric blue)
#         # 0.000, 0.498, 1.000,
#         # # 10. Blue (A primary color, sharp but with lower luminance)
#         # 0.000, 0.000, 1.000,
#     ]
# ).astype(np.float32).reshape(-1, 3)


# A list of 25 bright and sharp colors for segmentation masks,
# generated from the edges of the sRGB color space for maximum intensity.
_COLORS = np.array(
    [
        # The original 8 sharp colors
        1.000, 1.000, 0.000,  # 1. Yellow
        0.000, 1.000, 0.000,  # 2. Lime
        0.000, 1.000, 1.000,  # 3. Cyan
        1.000, 0.000, 1.000,  # 4. Magenta
        1.000, 0.000, 0.000,  # 5. Red
        1.000, 0.498, 0.000,  # 6. Orange
        0.498, 1.000, 0.000,  # 7. Chartreuse
        0.000, 1.000, 0.498,  # 8. Spring Green
        1.000, 0.000, 0.498,  # 9. Rose
        0.498, 0.000, 1.000,  # 10. Violet
        0.753, 1.000, 0.000,  # 11. Electric Lime
        1.000, 0.753, 0.000,  # 12. Vivid Orange
        0.000, 1.000, 0.753,  # 13. Turquoise
        0.753, 0.000, 1.000,  # 14. Bright Violet
        1.000, 0.000, 0.753,  # 15. Bright Pink
        1.000, 0.251, 0.000,  # 16. Fiery Orange
        0.251, 1.000, 0.000,  # 17. Bright Chartreuse
        0.000, 1.000, 0.251,  # 18. Malachite Green
        0.251, 0.000, 1.000,  # 19. Deep Violet
        1.000, 0.000, 0.251,  # 20. Hot Pink
    ]
).astype(np.float32).reshape(-1, 3)


def colormap(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a float32 array of Nx3 colors, in range [0, 255] or [0, 1]
    """
    assert maximum in [255, 1], maximum
    c = _COLORS * maximum
    if not rgb:
        c = c[:, ::-1]
    return c


def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def random_colors(N, rgb=False, maximum=255):
    """
    Args:
        N (int): number of unique colors needed
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a list of random_color
    """
    indices = random.sample(range(len(_COLORS)), N)
    ret = [_COLORS[i] * maximum for i in indices]
    if not rgb:
        ret = [x[::-1] for x in ret]
    return ret


if __name__ == "__main__":
    import cv2

    size = 100
    H, W = 10, 10
    canvas = np.random.rand(H * size, W * size, 3).astype("float32")
    for h in range(H):
        for w in range(W):
            idx = h * W + w
            if idx >= len(_COLORS):
                break
            canvas[h * size : (h + 1) * size, w * size : (w + 1) * size] = _COLORS[idx]
    cv2.imshow("a", canvas)
    cv2.waitKey(0)