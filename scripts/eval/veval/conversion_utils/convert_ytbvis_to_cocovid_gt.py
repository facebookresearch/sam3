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
