import json
import os
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util
from tqdm import tqdm


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
