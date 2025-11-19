# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
import json
import os
import shutil
import subprocess
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm


annotation_files = {
    "droid": [
        "silver_droid_merged_test.json",
    ],
    "sav": [
        "silver_sav_merged_test.json",
    ],
    "yt1b": [
        "silver_yt1b_merged_test.json",
    ],
    "ego4d": [
        "silver_ego4d_merged_test.json",
    ],
}


def load_yaml(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def save_json(content, filename):
    with open(filename, "w") as f:
        json.dump(content, f)


def run_command(cmd):
    """Run a shell command and raise if it fails."""
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


config = load_yaml("CONFIG_FRAMES.yaml")


def is_valid_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        return True
    except Exception:
        return False


def get_frame_from_video(video_path, frame_id):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        # Some videos cannot be open with OpenCV
        import av

        container = av.open(video_path)
        stream = container.streams.video[0]
        for i, frame in tqdm(
            enumerate(container.decode(stream)),
            desc="Decoding with AV",
            total=frame_id + 1,
        ):
            if i == frame_id:
                img = frame.to_ndarray(format="rgb24")
                return img
        raise ValueError(
            f"Could not read frame {frame_id} from video {video_path} (out of frame)"
        )
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb


def update_annotations(dataset_name, file_names_keep, key="original_video"):
    for annotation_file in annotation_files[dataset_name]:
        path_ann = os.path.join(config["path_annotations"], annotation_file)
        path_original_ann = os.path.join(
            config["path_annotations"],
            annotation_file.replace(".json", "_original.json"),
        )
        ann = load_json(path_ann)
        shutil.copy(path_ann, path_original_ann)
        new_images = []
        image_ids_keep = set()
        for image in ann["images"]:
            if image[key].replace(".mp4", "") in file_names_keep:
                new_images.append(image)
                image_ids_keep.add(image["id"])
        new_annotations = []
        for annotation in ann["annotations"]:
            if annotation["image_id"] in image_ids_keep:
                new_annotations.append(annotation)
        ann["images"] = new_images
        ann["annotations"] = new_annotations
        save_json(ann, path_ann)


def get_filename_size_map(annotation_path):
    with open(annotation_path) as f:
        annotations = json.load(f)
    filename_size_map = {}
    for each in annotations["images"]:
        filename_size_map[each["file_name"]] = (each["width"], each["height"])
    return filename_size_map


def get_filenames(annotation_path):
    with open(annotation_path) as f:
        annotations = json.load(f)
    filenames = {Path(each["file_name"]) for each in annotations["images"]}
    return filenames


def get_image_ids(annotation_path):
    filenames = get_filenames(annotation_path)
    filestems = {Path(each).stem for each in filenames}
    return filestems


def setup(folder):
    print("Making dir", folder)
    folder.mkdir(exist_ok=True)


def copy_file(paths):
    old_path, new_path = paths
    print("Copy from", old_path, "to", new_path)
    if not Path(new_path).exists():
        shutil.copy2(old_path, new_path)
