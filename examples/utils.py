import json
import os
import pprint
import subprocess
from pathlib import Path
from typing import Any, Dict

import cv2

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_utils
import torch
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from PIL import Image
from pycocotools import mask as mask_util
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans
from torchvision.ops import masks_to_boxes
from tqdm import tqdm


def generate_colors(n_colors=256, n_samples=5000):
    # Step 1: Random RGB samples
    np.random.seed(42)
    rgb = np.random.rand(n_samples, 3)
    # Step 2: Convert to LAB for perceptual uniformity
    print(f"Converting {n_samples} RGB samples to LAB color space...")
    lab = rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)
    print("Conversion to LAB complete.")
    # Step 3: k-means clustering in LAB
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    print(f"Fitting KMeans with {n_colors} clusters on {n_samples} samples...")
    kmeans.fit(lab)
    print("KMeans fitting complete.")
    centers_lab = kmeans.cluster_centers_
    # Step 4: Convert LAB back to RGB
    colors_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    colors_rgb = np.clip(colors_rgb, 0, 1)
    return colors_rgb


COLORS = generate_colors(n_colors=128, n_samples=5000)


def show_img_tensor(img_batch, vis_img_idx=0):
    MEAN_IMG = np.array([0.485, 0.456, 0.406])
    STD_IMG = np.array([0.229, 0.224, 0.225])
    im_tensor = img_batch[vis_img_idx].detach().cpu()
    assert im_tensor.dim() == 3
    im_tensor = im_tensor.numpy().transpose((1, 2, 0))
    im_tensor = (im_tensor * STD_IMG) + MEAN_IMG
    im_tensor = np.clip(im_tensor, 0, 1)
    plt.imshow(im_tensor)


def show_points_with_labels(coords, labels, ax=None, marker_size=200):
    if ax is None:
        ax = plt.gca()
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def draw_box_on_image(image, box, color=(0, 255, 0)):
    """
    Draws a rectangle on a given PIL image using the provided box coordinates in xywh format.
    :param image: PIL.Image - The image on which to draw the rectangle.
    :param box: tuple - A tuple (x, y, w, h) representing the top-left corner, width, and height of the rectangle.
    :param color: tuple - A tuple (R, G, B) representing the color of the rectangle. Default is red.
    :return: PIL.Image - The image with the rectangle drawn on it.
    """
    # Ensure the image is in RGB mode
    image = image.convert("RGB")
    # Unpack the box coordinates
    x, y, w, h = box
    x, y, w, h = int(x), int(y), int(w), int(h)
    # Get the pixel data
    pixels = image.load()
    # Draw the top and bottom edges
    for i in range(x, x + w):
        pixels[i, y] = color
        pixels[i, y + h - 1] = color
        pixels[i, y + 1] = color
        pixels[i, y + h] = color
        pixels[i, y - 1] = color
        pixels[i, y + h - 2] = color
    # Draw the left and right edges
    for j in range(y, y + h):
        pixels[x, j] = color
        pixels[x + 1, j] = color
        pixels[x - 1, j] = color
        pixels[x + w - 1, j] = color
        pixels[x + w, j] = color
        pixels[x + w - 2, j] = color
    return image


def plot_bbox(
    img_height,
    img_width,
    box,
    box_format="XYXY",
    relative_coords=True,
    color="r",
    linestyle="solid",
    text=None,
    ax=None,
):
    if box_format == "XYXY":
        x, y, x2, y2 = box
        w = x2 - x
        h = y2 - y
    elif box_format == "XYWH":
        x, y, w, h = box
    elif box_format == "CxCyWH":
        cx, cy, w, h = box
        x = cx - w / 2
        y = cy - h / 2
    else:
        raise RuntimeError(f"Invalid box_format {box_format}")

    if relative_coords:
        x *= img_width
        w *= img_width
        y *= img_height
        h *= img_height

    if ax is None:
        ax = plt.gca()
    rect = patches.Rectangle(
        (x, y),
        w,
        h,
        linewidth=1.5,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    if text is not None:
        facecolor = "w"
        ax.text(
            x,
            y - 5,
            text,
            color=color,
            weight="bold",
            fontsize=8,
            bbox={"facecolor": facecolor, "alpha": 0.75, "pad": 2},
        )


def plot_mask(mask, color="r", ax=None):
    im_h, im_w = mask.shape
    mask_img = np.zeros((im_h, im_w, 4), dtype=np.float32)
    mask_img[..., :3] = to_rgb(color)
    mask_img[..., 3] = mask * 0.5
    # Use the provided ax or the current axis
    if ax is None:
        ax = plt.gca()
    ax.imshow(mask_img)


def normalize_bbox(bbox_xyxy, img_w, img_h):
    bbox_xyxy[0], bbox_xyxy[2] = bbox_xyxy[0] / img_w, bbox_xyxy[2] / img_w
    bbox_xyxy[1], bbox_xyxy[3] = bbox_xyxy[1] / img_h, bbox_xyxy[3] / img_h
    return bbox_xyxy


def visualize_frame_output(frame_idx, image_files, outputs, figsize=(12, 8)):
    plt.figure(figsize=figsize)
    plt.title(f"frame {frame_idx}")
    img = plt.imread(image_files[frame_idx])
    img_H, img_W, _ = img.shape
    plt.imshow(img)
    for i in range(len(outputs["out_probs"])):
        box_xywh = outputs["out_boxes_xywh"][i]
        prob = outputs["out_probs"][i]
        obj_id = outputs["out_obj_ids"][i]
        binary_mask = outputs["out_binary_masks"][i]
        color = COLORS[obj_id % len(COLORS)]
        plot_bbox(
            img_H,
            img_W,
            box_xywh,
            text=f"(id={obj_id}, {prob=:.2f})",
            box_format="XYWH",
            color=color,
        )
        plot_mask(binary_mask, color=color)


def visualize_formatted_frame_output(
    frame_idx,
    image_files,
    outputs_list,
    titles=None,
    points_list=None,
    points_labels_list=None,
    figsize=(12, 8),
):
    """Visualize up to three sets of segmentation masks on a video frame.

    outputs_list: List of {frame_idx: {obj_id: mask_tensor}}
    titles: List of titles for each set of outputs_list
    """
    num_outputs = len(outputs_list)
    if titles is None:
        titles = [f"Set {i+1}" for i in range(num_outputs)]
    assert (
        len(titles) == num_outputs
    ), "length of `titles` should match that of `outputs_list` if not None."

    fig, axes = plt.subplots(1, num_outputs, figsize=figsize)
    if num_outputs == 1:
        axes = [axes]  # Make it iterable
    img = plt.imread(image_files[frame_idx])
    img_H, img_W, _ = img.shape
    for idx in range(num_outputs):
        ax, outputs_set, ax_title = axes[idx], outputs_list[idx], titles[idx]
        ax.set_title(f"Frame {frame_idx} - {ax_title}")
        ax.imshow(img)
        _outputs = outputs_set[frame_idx]
        for obj_id, binary_mask in _outputs.items():
            box_xyxy = masks_to_boxes(torch.tensor(binary_mask).unsqueeze(0)).squeeze()
            box_xyxy = normalize_bbox(box_xyxy, img_W, img_H)
            color = COLORS[obj_id % len(COLORS)]
            plot_bbox(
                img_H,
                img_W,
                box_xyxy,
                text=f"(id={obj_id})",
                box_format="XYXY",
                color=color,
                ax=ax,
            )
            plot_mask(binary_mask, color=color, ax=ax)

        # points
        if points_list is not None and points_list[idx] is not None:
            show_points_with_labels(
                points_list[idx], points_labels_list[idx], ax=ax, marker_size=200
            )

    plt.tight_layout()
    plt.show()


def render_masklet_frame(img, outputs, frame_idx=None, alpha=0.5):
    """
    Overlays masklets and bounding boxes on a single image frame.
    Args:
        img: np.ndarray, shape (H, W, 3), uint8 or float32 in [0,255] or [0,1]
        outputs: dict with keys: out_boxes_xywh, out_probs, out_obj_ids, out_binary_masks
        frame_idx: int or None, for overlaying frame index text
        alpha: float, mask overlay alpha
    Returns:
        overlay: np.ndarray, shape (H, W, 3), uint8
    """
    if img.dtype == np.float32 or img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    img = img[..., :3]  # drop alpha if present
    height, width = img.shape[:2]
    overlay = img.copy()

    for i in range(len(outputs["out_probs"])):
        obj_id = outputs["out_obj_ids"][i]
        color = COLORS[obj_id % len(COLORS)]
        color255 = (color * 255).astype(np.uint8)
        mask = outputs["out_binary_masks"][i]
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.float32),
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_bool = mask > 0.5
        for c in range(3):
            overlay[..., c][mask_bool] = (
                alpha * color255[c] + (1 - alpha) * overlay[..., c][mask_bool]
            ).astype(np.uint8)

    # Draw bounding boxes and text
    for i in range(len(outputs["out_probs"])):
        box_xywh = outputs["out_boxes_xywh"][i]
        obj_id = outputs["out_obj_ids"][i]
        prob = outputs["out_probs"][i]
        color = COLORS[obj_id % len(COLORS)]
        color255 = tuple(int(x * 255) for x in color)
        x, y, w, h = box_xywh
        x1 = int(x * width)
        y1 = int(y * height)
        x2 = int((x + w) * width)
        y2 = int((y + h) * height)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color255, 2)
        if prob is not None:
            label = f"id={obj_id}, p={prob:.2f}"
        else:
            label = f"id={obj_id}"
        cv2.putText(
            overlay,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color255,
            1,
            cv2.LINE_AA,
        )

    # Overlay frame index at the top-left corner
    if frame_idx is not None:
        cv2.putText(
            overlay,
            f"Frame {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay


def save_masklet_video(image_files, outputs, out_path, alpha=0.5, fps=10):
    # Each outputs dict has keys: "out_boxes_xywh", "out_probs", "out_obj_ids", "out_binary_masks"
    # image_files: list of image file paths, same length as outputs_list

    # Read first frame to get size
    first_img = plt.imread(image_files[0])
    height, width = first_img.shape[:2]
    if first_img.dtype == np.float32 or first_img.max() <= 1.0:
        first_img = (first_img * 255).astype(np.uint8)
    # Use 'mp4v' for best compatibility with VSCode playback (.mp4 files)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("temp.mp4", fourcc, fps, (width, height))

    outputs_list = [
        (image_files[frame_idx], frame_idx, outputs[frame_idx])
        for frame_idx in sorted(outputs.keys())
    ]

    for img_path, frame_idx, frame_outputs in tqdm(outputs_list):
        img = plt.imread(img_path)
        overlay = render_masklet_frame(
            img, frame_outputs, frame_idx=frame_idx, alpha=alpha
        )
        writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    writer.release()

    # Re-encode the video for VSCode compatibility using ffmpeg
    subprocess.run(["ffmpeg", "-y", "-i", "temp.mp4", out_path])
    print(f"Re-encoded video saved to {out_path}")

    os.remove("temp.mp4")  # Clean up temporary file


def save_masklet_image(image_file, outputs, out_path, alpha=0.5, frame_idx=None):
    """
    Save a single image with masklet overlays.
    """
    img = plt.imread(image_file)
    overlay = render_masklet_frame(img, outputs, frame_idx=frame_idx, alpha=alpha)
    Image.fromarray(overlay).save(out_path)
    print(f"Overlay image saved to {out_path}")


def prepare_masks_for_visualization(frame_to_output):
    # frame_to_obj_masks --> {frame_idx: {'output_probs': np.array, `out_obj_ids`: np.array, `out_binary_masks`: np.array}}
    for frame_idx, out in frame_to_output.items():
        _processed_out = {}
        for idx, obj_id in enumerate(out["out_obj_ids"].tolist()):
            if out["out_binary_masks"][idx].any():
                _processed_out[obj_id] = out["out_binary_masks"][idx]
        frame_to_output[frame_idx] = _processed_out
    return frame_to_output


def convert_coco_to_masklet_format(
    annotations, img_info, is_prediction=False, score_threshold=0.5
):
    """
    Convert COCO format annotations to format expected by render_masklet_frame
    """
    outputs = {
        "out_boxes_xywh": [],
        "out_probs": [],
        "out_obj_ids": [],
        "out_binary_masks": [],
    }

    img_h, img_w = img_info["height"], img_info["width"]

    for idx, ann in enumerate(annotations):
        # Get bounding box in relative XYWH format
        if "bbox" in ann:
            bbox = ann["bbox"]
            if max(bbox) > 1.0:  # Convert absolute to relative coordinates
                bbox = [
                    bbox[0] / img_w,
                    bbox[1] / img_h,
                    bbox[2] / img_w,
                    bbox[3] / img_h,
                ]
        else:
            mask = mask_utils.decode(ann["segmentation"])
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if np.any(rows) and np.any(cols):
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                # Convert to relative XYWH
                bbox = [
                    cmin / img_w,
                    rmin / img_h,
                    (cmax - cmin + 1) / img_w,
                    (rmax - rmin + 1) / img_h,
                ]
            else:
                bbox = [0, 0, 0, 0]

        outputs["out_boxes_xywh"].append(bbox)

        # Get probability/score
        if is_prediction:
            prob = ann["score"]
        else:
            prob = 1.0  # GT has no probability
        outputs["out_probs"].append(prob)

        outputs["out_obj_ids"].append(idx)
        mask = mask_utils.decode(ann["segmentation"])
        mask = (mask > score_threshold).astype(np.uint8)

        outputs["out_binary_masks"].append(mask)

    return outputs


def save_side_by_side_visualization(img, gt_anns, pred_anns, noun_phrase):
    """
    Create side-by-side visualization of GT and predictions
    """

    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    main_title = f"Noun phrase: '{noun_phrase}'"
    fig.suptitle(main_title, fontsize=16, fontweight="bold")

    gt_overlay = render_masklet_frame(img, gt_anns, alpha=0.5)
    ax1.imshow(gt_overlay)
    ax1.set_title("Ground Truth", fontsize=14, fontweight="bold")
    ax1.axis("off")

    pred_overlay = render_masklet_frame(img, pred_anns, alpha=0.5)
    ax2.imshow(pred_overlay)
    ax2.set_title("Predictions", fontsize=14, fontweight="bold")
    ax2.axis("off")

    plt.subplots_adjust(top=0.88)
    plt.tight_layout()


def bitget(val, idx):
    return (val >> idx) & 1


def pascal_color_map():
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)
    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bitget(ind, channel) << shift
        ind >>= 3

    return colormap.astype(np.uint8)


def draw_masks_to_frame(
    frame: np.ndarray, masks: np.ndarray, colors: np.ndarray
) -> np.ndarray:
    masked_frame = frame
    for mask, color in zip(masks, colors):
        curr_masked_frame = np.where(mask[..., None], color, masked_frame)
        masked_frame = cv2.addWeighted(masked_frame, 0.75, curr_masked_frame, 0.25, 0)

        if int(cv2.__version__[0]) > 3:
            contours, _ = cv2.findContours(
                np.array(mask, dtype=np.uint8).copy(),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_NONE,
            )
        else:
            _, contours, _ = cv2.findContours(
                np.array(mask, dtype=np.uint8).copy(),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_NONE,
            )

        cv2.drawContours(
            masked_frame, contours, -1, (255, 255, 255), 7
        )  # White outer contour
        cv2.drawContours(
            masked_frame, contours, -1, (0, 0, 0), 5
        )  # Black middle contour
        cv2.drawContours(
            masked_frame, contours, -1, color.tolist(), 3
        )  # Original color inner contour
    return masked_frame


def get_annot_df(file_path: str):
    with open(file_path, "r") as f:
        data = json.load(f)

    dfs = {}

    for k, v in data.items():
        if k in ("info", "licenses"):
            dfs[k] = v
            continue
        df = pd.DataFrame(v)
        dfs[k] = df

    return dfs


def get_annot_dfs(file_list: list[str]):
    dfs = {}
    for annot_file in tqdm(file_list):
        dataset_name = Path(annot_file).stem
        dfs[dataset_name] = get_annot_df(annot_file)

    return dfs


def get_media_dir(media_dir: str, dataset: str):
    if dataset in ["saco_veval_sav_test", "saco_veval_sav_val"]:
        return os.path.join(media_dir, "saco_sav", "JPEGImages_24fps")
    elif dataset in ["saco_veval_yt1b_test", "saco_veval_yt1b_val"]:
        return os.path.join(media_dir, "saco_yt1b", "JPEGImages_6fps")
    elif dataset in ["saco_veval_smartglasses_test", "saco_veval_smartglasses_val"]:
        return os.path.join(media_dir, "saco_sg", "JPEGImages_6fps")
    elif dataset == "sa_fari_test":
        return os.path.join(media_dir, "sa_fari", "JPEGImages_6fps")
    else:
        raise ValueError(f"Dataset {dataset} not found")


def get_vis_example(
    annot_dfs: Dict[str, Any], dataset: str, row_num: int, frame_num: int, data_dir: str
):
    media_dir = os.path.join(data_dir, "media")

    # Load the annotation and video data
    annot_df = annot_dfs[dataset]["annotations"]
    video_df = annot_dfs[dataset]["videos"]

    # Get the mask
    annot_row = annot_df.iloc[row_num]
    rle = annot_row.segmentations[frame_num]
    mask = mask_util.decode(rle)

    # Get the noun phrase
    noun_phrase = annot_row.noun_phrase

    # Get the video frame
    video_df = video_df[video_df.id == annot_row.video_id]
    assert len(video_df) == 1, f"Expected 1 video row, got {len(video_df)}"

    video_row = video_df.iloc[0]
    file_name = video_row.file_names[frame_num]
    file_path = os.path.join(
        get_media_dir(media_dir=media_dir, dataset=dataset), file_name
    )
    frame = cv2.imread(file_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame, mask, noun_phrase
