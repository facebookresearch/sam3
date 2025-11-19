# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
This file extracts the frames for the frame datasets in SA-CO/Gold and Silver.

Call like:
> python extract_frames.py <dataset_name>
"""

import json
import os
import shutil
import sys
from multiprocessing import Pool

from PIL import Image
from tqdm import tqdm
from utils import (
    annotation_files,
    config,
    get_frame_from_video,
    is_valid_image,
    update_annotations,
)


def extract_frame(path_video, global_frame_idx, path_frame, image_size, file_name):
    frame = get_frame_from_video(path_video, global_frame_idx)
    os.makedirs(os.path.dirname(path_frame), exist_ok=True)
    img = Image.fromarray(frame)
    if frame.shape[:2] != image_size:
        print(f"Resizing image {file_name} from {frame.shape[:2]} to {image_size}")
        height, width = image_size
        img = img.resize((width, height))  # Uses Image.NEAREST by default
    img.save(path_frame)


def process_image(args):
    image, dataset_name, config = args
    original_video, global_frame_idx, file_name, image_size = image
    extra_subpath = ""
    if dataset_name == "ego4d":
        extra_subpath = "v1/clips"
    elif dataset_name == "yt1b":
        original_video = f"video_{original_video}.mp4"
    elif dataset_name == "sav":
        extra_subpath = "videos_fps_6"
    path_video = os.path.join(
        config[f"{dataset_name}_path"],
        "downloaded_videos",
        extra_subpath,
        original_video,
    )
    path_frame = os.path.join(config[f"{dataset_name}_path"], "frames", file_name)
    to_return = file_name
    try:
        extract_frame(path_video, global_frame_idx, path_frame, image_size, file_name)
        if not is_valid_image(path_frame):
            print(f"Invalid image in {path_frame}")
            to_return = None
    except:
        print(f"Invalid image in {path_frame}")
        to_return = None
    return to_return


def main():
    assert len(sys.argv) > 1, "You have to provide the name of the dataset"
    dataset_name = sys.argv[1]
    assert (
        dataset_name in annotation_files
    ), f"The dataset can be one of {list(annotation_files.keys())}"
    all_outputs = []
    for file in annotation_files[dataset_name]:
        with open(os.path.join(config["path_annotations"], file), "r") as f:
            annotation = json.load(f)
        images = annotation["images"]
        images = set(
            (
                image["original_video"],
                image["global_frame_idx"],
                image["file_name"],
                tuple(image["image_size"]),
            )
            for image in images
        )
        args_list = [(image, dataset_name, config) for image in images]
        with Pool(os.cpu_count()) as pool:
            outputs = list(
                tqdm(pool.imap_unordered(process_image, args_list), total=len(images))
            )
        all_outputs.extend(outputs)
    if any(out is None for out in outputs):
        update_annotations(dataset_name, all_outputs, key="file_name")
    if config[f"remove_downloaded_videos_{dataset_name}"]:
        shutil.rmtree(os.path.join(config[f"{dataset_name}_path"], "downloaded_videos"))


if __name__ == "__main__":
    main()
