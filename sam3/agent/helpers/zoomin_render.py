# Modified from https://www.internalfb.com/code/fbsource/[3bf4b3d1d890766455011441045522ab4d0f90fc]/fbcode/gen_ai/mllm/inference/llama3/scripts/mask_verify.py

import io
import logging
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_utils

from onevision.models.data_gen.llama_rater_utils.som_utils import (
    ColorPalette,
    draw_box,
    draw_mask,
    draw_text,
)
from PIL import Image


logger = logging.getLogger()


color_palette = ColorPalette.default()


system_prompt = """
You are an expert annotator of object segmentation masks. For an image and a pre-defined label, you are given a mask and asked to evaluate the quality of the mask. Follow the following rules when rating the mask. 
1. Rate the mask as "Accept" when the label accurately describes the masked object and the mask covers the object with good boundaries. We do not need masks to be pixel-perfect for this task. However the mask should cover all important parts of the object.
2. Rate the mask as "Accept as text" when the mask covers text and the label exactly matches the masked text. The mask should cover all important parts of the text (all specified letters/punctuation/etc.)
3. Rate the mask as "Flag label" when the label corresponds to Race, Ethnicity, Sexual orientation, Religion, Socio-economic status, Medical conditions, Disabilities, Derogatory terms/profanity and Animal phrases for a person. 
4. Rate the mask as "Whole image" when the label corresponds to the entire image. Description that refers to the whole image may describe setting (e.g., inside, outside, day, night), type of media (e.g., flier, screenshot, photo), type of image (e.g., close up, an aerial view) and location (e.g., an airport, the woods, a bedroom). 
5. Otherwise, rate the mask as "Reject".
Please give your rating directly without any explanation. Now let's start. 
"""

layouts = {
    "horizontal": ["left", "right"],
    "vertical": ["upper", "lower"],
}
choices = ["Accept", "Accept as text", "Flag label", "Whole image", "Reject"]
answer_string = "ABCDE"


def get_dialogue(
    label,
    layout,
    color,
    zoom_in=False,
):
    shuffled_choices = random.sample(choices, k=len(choices))
    choices_string = " ".join(
        [f"({key}). {val}." for key, val in zip(answer_string, shuffled_choices)]
    )
    layout = layouts[layout]
    if zoom_in:
        human_prompt = f'In the given figure, the {layout[0]} half shows a {color} box highlighting the region of interest in the original image, and the {layout[1]} half shows a {color} mask overlaid on a zoom-in view of that region. Rate the {color} mask for the label "{label}": {choices_string}'
    else:
        human_prompt = f'In the given image, the {layout[0]} half is the original image and the {layout[1]} half shows a {color} mask overlaid on the original image. Rate the {color} mask for the label "{label}": {choices_string}'
    prompt = system_prompt + "\n" + human_prompt
    index2ans = dict(zip(answer_string, shuffled_choices))
    return prompt, index2ans


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
    # get zoom-in box
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

    # get img_crop box
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
):
    # object_data = json.loads(object_data)
    object_label = object_data["labels"][0]["noun_phrase"]
    img = image_file.convert("RGB")
    bbox_xywh = mask_utils.toBbox(object_data["segmentation"])
    # get color of the mask
    bbox_xyxy = [
        bbox_xywh[0],
        bbox_xywh[1],
        bbox_xywh[0] + bbox_xywh[2],
        bbox_xywh[1] + bbox_xywh[3],
    ]
    crop_img = img.crop(bbox_xyxy)
    color, color_name = color_palette.find_farthest_color(np.array(crop_img))
    color = np.array([color.r / 255, color.g / 255, color.b / 255])
    # decide the zoom-in parameters
    img_h, img_w = object_data["segmentation"]["size"]
    mask_area = mask_utils.area(object_data["segmentation"])
    zoom_in_box, img_crop_box = get_zoom_in_box(bbox_xywh, img_h, img_w, mask_area)

    w, h = img_crop_box[2], img_crop_box[3]
    if w < h:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        layout = "horizontal"
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        layout = "vertical"
    # original/cropped image
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
        draw_text(
            ax1,
            object_label,
            [x0, y0],
            color=color,
        )
    # ax1.set_title(f"original image:{human_anno}")

    # masked image
    binary_mask = mask_utils.decode(object_data["segmentation"])
    alpha = Image.fromarray((binary_mask * 255).astype("uint8"))
    # Merge back the channels
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

    # ax2.set_title(f'"{object_label}" in {color_name}')
    draw_mask(ax2, binary_mask_zoomin, color=color, show_holes=show_holes)

    plt.tight_layout()
    # Save the plot as a PNG in memory.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(fig)
    buf.seek(0)
    # Load the image from the buffer
    pil_img = Image.open(buf)
    prompt, index2ans = get_dialogue(object_label, layout, color_name, zoom_in=True)
    return pil_img, prompt, index2ans
