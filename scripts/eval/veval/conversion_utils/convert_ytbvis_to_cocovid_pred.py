import json

from tqdm import tqdm


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

    # Build final COCO structure
    # coco_output = {
    #     "annotations": coco_annotations,
    #     "categories": coco_dataset["categories"],
    #     "info": coco_dataset.get("info", {})
    # }

    # Save output
    with open(output_path, "w") as f:
        json.dump(coco_annotations, f)

    print(f"Converted {len(coco_annotations)} predictions to COCO format with video_id")
