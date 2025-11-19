# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
import json
import os
from collections import defaultdict

from tqdm import tqdm


def convert_ytbvis_to_cocovid_gt(ann_json, save_path=None):
    """Convert YouTube VIS dataset to COCO-style video instance segmentation format.

    Args:
        ann_json (str): Path to YouTube VIS annotation JSON file
        save_path (str): path to save converted COCO-style JSON
    """
    # Initialize COCO structure
    VIS = {
        "info": {},
        "images": [],
        "videos": [],
        "tracks": [],
        "annotations": [],
        "categories": [],
        "licenses": [],
    }

    # Load original annotations
    official_anns = json.load(open(ann_json))
    VIS["categories"] = official_anns["categories"]  # Direct copy categories

    # Initialize counters
    records = dict(img_id=1, ann_id=1)

    # Create video-to-annotations mapping
    vid_to_anns = defaultdict(list)
    for ann in official_anns["annotations"]:
        vid_to_anns[ann["video_id"]].append(ann)

    # Create tracks directly
    VIS["tracks"] = [
        {
            "id": ann["id"],
            "category_id": ann["category_id"],
            "video_id": ann["video_id"],
        }
        for ann in official_anns["annotations"]
    ]

    # Process videos
    for video_info in tqdm(official_anns["videos"]):
        # Create video entry
        video = {
            "id": video_info["id"],
            "name": os.path.dirname(video_info["file_names"][0]),
            "width": video_info["width"],
            "height": video_info["height"],
            "length": video_info["length"],
            "neg_category_ids": [],
            "not_exhaustive_category_ids": [],
        }
        VIS["videos"].append(video)

        # Process frames
        num_frames = len(video_info["file_names"])
        for frame_idx in range(num_frames):
            # Create image entry
            image = {
                "id": records["img_id"],
                "video_id": video_info["id"],
                "file_name": video_info["file_names"][frame_idx],
                "width": video_info["width"],
                "height": video_info["height"],
                "frame_index": frame_idx,
                "frame_id": frame_idx,
            }
            VIS["images"].append(image)

            # Process annotations for this frame
            if video_info["id"] in vid_to_anns:
                for ann in vid_to_anns[video_info["id"]]:
                    bbox = ann["bboxes"][frame_idx]
                    if bbox is None:
                        continue

                    # Create annotation entry
                    annotation = {
                        "id": records["ann_id"],
                        "video_id": video_info["id"],
                        "image_id": records["img_id"],
                        "track_id": ann["id"],
                        "category_id": ann["category_id"],
                        "bbox": bbox,
                        "area": ann["areas"][frame_idx],
                        "segmentation": ann["segmentations"][frame_idx],
                        "iscrowd": ann["iscrowd"],
                    }
                    VIS["annotations"].append(annotation)
                    records["ann_id"] += 1

            records["img_id"] += 1

    # Print summary
    print(f"Converted {len(VIS['videos'])} videos")
    print(f"Converted {len(VIS['images'])} images")
    print(f"Created {len(VIS['tracks'])} tracks")
    print(f"Created {len(VIS['annotations'])} annotations")

    if save_path is None:
        return VIS

    # Save output
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    json.dump(VIS, open(save_path, "w"))

    return VIS


def convert_ytbvis_to_cocovid_pred(
    youtubevis_pred_path: str, converted_dataset_path: str, output_path: str
) -> None:
    """
    Convert YouTubeVIS predictions to COCO format with video_id preservation

    Args:
        youtubevis_pred_path: Path to YouTubeVIS prediction JSON
        converted_dataset_path: Path to converted COCO dataset JSON
        output_path: Path to save COCO format predictions
    """

    # Load YouTubeVIS predictions
    with open(youtubevis_pred_path) as f:
        ytv_predictions = json.load(f)

    # Load converted dataset for image ID mapping
    with open(converted_dataset_path) as f:
        coco_dataset = json.load(f)

    # Create (video_id, frame_idx) -> image_id mapping
    image_id_map = {
        (img["video_id"], img["frame_index"]): img["id"]
        for img in coco_dataset["images"]
    }

    coco_annotations = []
    track_id_counter = 1  # Unique track ID generator

    for pred in tqdm(ytv_predictions):
        video_id = pred["video_id"]
        category_id = pred["category_id"]
        bboxes = pred["bboxes"]
        segmentations = pred.get("segmentations", [])  # Get segmentations if available
        areas = pred.get("areas", [])  # Get areas if available
        score = pred["score"]

        # Assign unique track ID for this prediction
        track_id = track_id_counter
        track_id_counter += 1

        # Ensure segmentations and areas have the same length as bboxes
        if len(segmentations) == 0:
            segmentations = [None] * len(bboxes)
        if len(areas) == 0:
            areas = [None] * len(bboxes)

        for frame_idx, (bbox, segmentation, area_from_pred) in enumerate(
            zip(bboxes, segmentations, areas)
        ):
            # Skip frames with missing objects (None or zero bbox)
            if bbox is None or all(x == 0 for x in bbox):
                continue

            # Get corresponding image ID from mapping
            image_id = image_id_map.get((video_id, frame_idx))
            if image_id is None:
                raise RuntimeError(
                    f"prediction {video_id=}, {frame_idx=} does not match any images in the converted COCO format"
                )

            # Extract bbox coordinates
            x, y, w, h = bbox

            # Calculate area - use area from prediction if available, otherwise from bbox
            if area_from_pred is not None and area_from_pred > 0:
                area = area_from_pred
            else:
                area = w * h

            # Create COCO annotation with video_id
            coco_annotation = {
                "image_id": int(image_id),
                "video_id": video_id,  # Added video_id field
                "track_id": track_id,
                "category_id": category_id,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(area),
                "iscrowd": 0,
                "score": float(score),
            }

            # Add segmentation if available
            if segmentation is not None:
                coco_annotation["segmentation"] = segmentation

            coco_annotations.append(coco_annotation)

    # Save output
    with open(output_path, "w") as f:
        json.dump(coco_annotations, f)

    print(f"Converted {len(coco_annotations)} predictions to COCO format with video_id")
