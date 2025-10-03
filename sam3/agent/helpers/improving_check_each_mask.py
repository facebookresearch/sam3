# Minimal dependencies for save_single_mask_para_visualization_zoomin

import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

from .som_utils import (
    ColorPalette,
    draw_box,
    draw_mask,
    draw_text,
)

# Color palette for mask coloring
color_palette = ColorPalette.default()

# Zoom-in helpers
area_large, area_medium = 0.25, 0.05

def get_shift(x, w, w_new, w_img):
    assert 0 <= w_new <= w_img
    shift = (w_new - w) / 2
    if x - shift + w_new > w_img:
        shift = x + w_new - w_img
    return min(x, shift)

def get_zoom_in_box(mask_box, img_h, img_w, mask_area):
    box_w, box_h = mask_box[2], mask_box[3]
    w_new, h_new = (
        min(box_w + max(0.2 * box_w, 16), img_w),
        min(box_h + max(0.2 * box_h, 16), img_h),
    )
    mask_relative_area = mask_area / (w_new * h_new)

    # zoom-in box
    w_new_large, h_new_large = w_new, h_new
    if mask_relative_area > area_large:
        expansion_raio_large = math.sqrt(mask_relative_area / area_large)
        w_new_large, h_new_large = (
            min(w_new * expansion_raio_large, img_w),
            min(h_new * expansion_raio_large, img_h),
        )
    w_shift_large = get_shift(mask_box[0], mask_box[2], w_new_large, img_w)
    h_shift_large = get_shift(mask_box[1], mask_box[3], h_new_large, img_h)
    zoom_in_box = [
        mask_box[0] - w_shift_large,
        mask_box[1] - h_shift_large,
        w_new_large,
        h_new_large,
    ]

    # crop box
    w_new_medium, h_new_medium = w_new, h_new
    if mask_relative_area > area_medium:
        expansion_raio_medium = math.sqrt(mask_relative_area / area_medium)
        w_new_medium, h_new_medium = (
            min(w_new * expansion_raio_medium, img_w),
            min(h_new * expansion_raio_medium, img_h),
        )
    w_shift_medium = get_shift(mask_box[0], mask_box[2], w_new_medium, img_w)
    h_shift_medium = get_shift(mask_box[1], mask_box[3], h_new_medium, img_h)
    img_crop_box = [
        mask_box[0] - w_shift_medium,
        mask_box[1] - h_shift_medium,
        w_new_medium,
        h_new_medium,
    ]
    return zoom_in_box, img_crop_box

def save_single_mask_para_visualization_zoomin(
    object_data,
    image_file,
    show_box=True,
    show_text=False,
    show_holes=True,
    mask_alpha=0.15,
):
    object_label = object_data["labels"][0]["noun_phrase"]
    img = image_file.convert("RGB")
    bbox_xywh = mask_utils.toBbox(object_data["segmentation"])

    # Determine mask color
    bbox_xyxy = [
        bbox_xywh[0],
        bbox_xywh[1],
        bbox_xywh[0] + bbox_xywh[2],
        bbox_xywh[1] + bbox_xywh[3],
    ]
    crop_img = img.crop(bbox_xyxy)
    color_obj, color_name = color_palette.find_farthest_color(np.array(crop_img))
    color = np.array([color_obj.r / 255, color_obj.g / 255, color_obj.b / 255])
    color_hex = f"#{color_obj.r:02x}{color_obj.g:02x}{color_obj.b:02x}"

    # Compute zoom-in / crop boxes
    img_h, img_w = object_data["segmentation"]["size"]
    mask_area = mask_utils.area(object_data["segmentation"])
    zoom_in_box, img_crop_box = get_zoom_in_box(bbox_xywh, img_h, img_w, mask_area)

    w, h = img_crop_box[2], img_crop_box[3]
    if w < h:
        fig, (ax1, ax2) = plt.subplots(1, 2)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1)

    # Left/upper: cropped original
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

    # Right/lower: zoomed mask overlay
    binary_mask = mask_utils.decode(object_data["segmentation"])
    alpha = Image.fromarray((binary_mask * 255).astype("uint8"))
    img_with_alpha = img.convert("RGBA")
    img_with_alpha.putalpha(alpha)
    zoom_in_box_xyxy = [
        zoom_in_box[0],
        zoom_in_box[1],
        zoom_in_box[0] + zoom_in_box[2],
        zoom_in_box[1] + zoom_in_box[3],
    ]
    img_with_alpha_zoomin = img_with_alpha.crop(zoom_in_box_xyxy)
    alpha_zoomin = img_with_alpha_zoomin.split()[3]
    binary_mask_zoomin = np.array(alpha_zoomin).astype(bool)

    ax2.imshow(img_with_alpha_zoomin.convert("RGB"))
    ax2.axis("off")
    draw_mask(ax2, binary_mask_zoomin, color=color, show_holes=show_holes, alpha=mask_alpha)

    plt.tight_layout()

    # Return PIL image + color hex only
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf)

    return pil_img, color_hex