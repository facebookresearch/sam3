# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

from .som_utils import ColorPalette, draw_box, draw_mask, draw_text


def render_zoom_in(
    object_data,
    image_file,
    show_box: bool = True,
    show_text: bool = False,
    show_holes: bool = True,
    mask_alpha: float = 0.15,
):
    """
    Render a two-panel visualization with a cropped original view (left/upper) and a zoomed-in
    mask overlay (right/lower), then return it as a PIL.Image along with the chosen mask color (hex).

    Parameters
    ----------
    object_data : dict
        Dict containing "labels" and COCO RLE "segmentation".
        Expected:
          object_data["labels"][0]["noun_phrase"] : str
          object_data["segmentation"] : COCO RLE (with "size": [H, W])
    image_file : PIL.Image.Image
        Source image (PIL).
    show_box : bool
        Whether to draw the bbox on the cropped original panel.
    show_text : bool
        Whether to draw the noun phrase label near the bbox.
    show_holes : bool
        Whether to render mask holes (passed through to draw_mask).
    mask_alpha : float
        Alpha for the mask overlay.

    Returns
    -------
    pil_img : PIL.Image.Image
        The composed visualization image.
    color_hex : str
        Hex string of the chosen mask color.
    """

    # ---- local constants (avoid module-level globals) ----
    _AREA_LARGE = 0.25
    _AREA_MEDIUM = 0.05

    # ---- local helpers (avoid name collisions in a larger class) ----
    def _get_shift(x, w, w_new, w_img):
        assert 0 <= w_new <= w_img
        shift = (w_new - w) / 2
        if x - shift + w_new > w_img:
            shift = x + w_new - w_img
        return min(x, shift)

    def _get_zoom_in_box(mask_box_xywh, img_h, img_w, mask_area):
        box_w, box_h = mask_box_xywh[2], mask_box_xywh[3]
        w_new = min(box_w + max(0.2 * box_w, 16), img_w)
        h_new = min(box_h + max(0.2 * box_h, 16), img_h)

        mask_relative_area = mask_area / (w_new * h_new)

        # zoom-in (larger box if mask is relatively big)
        w_new_large, h_new_large = w_new, h_new
        if mask_relative_area > _AREA_LARGE:
            ratio_large = math.sqrt(mask_relative_area / _AREA_LARGE)
            w_new_large = min(w_new * ratio_large, img_w)
            h_new_large = min(h_new * ratio_large, img_h)

        w_shift_large = _get_shift(
            mask_box_xywh[0], mask_box_xywh[2], w_new_large, img_w
        )
        h_shift_large = _get_shift(
            mask_box_xywh[1], mask_box_xywh[3], h_new_large, img_h
        )
        zoom_in_box = [
            mask_box_xywh[0] - w_shift_large,
            mask_box_xywh[1] - h_shift_large,
            w_new_large,
            h_new_large,
        ]

        # crop box for the original/cropped image
        w_new_medium, h_new_medium = w_new, h_new
        if mask_relative_area > _AREA_MEDIUM:
            ratio_med = math.sqrt(mask_relative_area / _AREA_MEDIUM)
            w_new_medium = min(w_new * ratio_med, img_w)
            h_new_medium = min(h_new * ratio_med, img_h)

        w_shift_medium = _get_shift(
            mask_box_xywh[0], mask_box_xywh[2], w_new_medium, img_w
        )
        h_shift_medium = _get_shift(
            mask_box_xywh[1], mask_box_xywh[3], h_new_medium, img_h
        )
        img_crop_box = [
            mask_box_xywh[0] - w_shift_medium,
            mask_box_xywh[1] - h_shift_medium,
            w_new_medium,
            h_new_medium,
        ]
        return zoom_in_box, img_crop_box

    # ---- main body ----
    # Input parsing
    object_label = object_data["labels"][0]["noun_phrase"]
    img = image_file.convert("RGB")
    bbox_xywh = mask_utils.toBbox(object_data["segmentation"])  # [x, y, w, h]

    # Choose a stable, visually distant color based on crop
    bbox_xyxy = [
        bbox_xywh[0],
        bbox_xywh[1],
        bbox_xywh[0] + bbox_xywh[2],
        bbox_xywh[1] + bbox_xywh[3],
    ]
    crop_img = img.crop(bbox_xyxy)
    color_palette = ColorPalette.default()
    color_obj, _ = color_palette.find_farthest_color(np.array(crop_img))
    color = np.array([color_obj.r / 255, color_obj.g / 255, color_obj.b / 255])
    color_hex = f"#{color_obj.r:02x}{color_obj.g:02x}{color_obj.b:02x}"

    # Compute zoom-in / crop boxes
    img_h, img_w = object_data["segmentation"]["size"]
    mask_area = mask_utils.area(object_data["segmentation"])
    zoom_in_box, img_crop_box = _get_zoom_in_box(bbox_xywh, img_h, img_w, mask_area)

    # Layout choice
    w, h = img_crop_box[2], img_crop_box[3]
    if w < h:
        fig, (ax1, ax2) = plt.subplots(1, 2)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1)

    # Panel 1: cropped original with optional box/text
    img_crop_box_xyxy = [
        img_crop_box[0],
        img_crop_box[1],
        img_crop_box[0] + img_crop_box[2],
        img_crop_box[1] + img_crop_box[3],
    ]
    img1 = img.crop(img_crop_box_xyxy)
    bbox_xywh_rel = [
        bbox_xywh[0] - img_crop_box[0],
        bbox_xywh[1] - img_crop_box[1],
        bbox_xywh[2],
        bbox_xywh[3],
    ]
    ax1.imshow(img1)
    ax1.axis("off")
    if show_box:
        draw_box(ax1, bbox_xywh_rel, edge_color=color)
    if show_text:
        x0, y0 = bbox_xywh_rel[0] + 2, bbox_xywh_rel[1] + 2
        draw_text(ax1, object_label, [x0, y0], color=color)

    # Panel 2: zoomed-in mask overlay
    binary_mask = mask_utils.decode(object_data["segmentation"])
    alpha = Image.fromarray((binary_mask * 255).astype("uint8"))
    img_rgba = img.convert("RGBA")
    img_rgba.putalpha(alpha)
    zoom_in_box_xyxy = [
        zoom_in_box[0],
        zoom_in_box[1],
        zoom_in_box[0] + zoom_in_box[2],
        zoom_in_box[1] + zoom_in_box[3],
    ]
    img_with_alpha_zoomin = img_rgba.crop(zoom_in_box_xyxy)
    alpha_zoomin = img_with_alpha_zoomin.split()[3]
    binary_mask_zoomin = np.array(alpha_zoomin).astype(bool)

    ax2.imshow(img_with_alpha_zoomin.convert("RGB"))
    ax2.axis("off")
    draw_mask(
        ax2, binary_mask_zoomin, color=color, show_holes=show_holes, alpha=mask_alpha
    )

    plt.tight_layout()

    # Buffer -> PIL.Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf)

    return pil_img, color_hex
