import json

from tqdm import tqdm


def build_video_frame_count(dataset_annotation_path: str) -> dict:
    """
    Build a dictionary mapping video_id to the total number of frames in the video.

    Args:
        dataset_annotation_path: Path to dataset annotation JSON in COCO format

    Returns:
        A dictionary where keys are video_ids and values are the total frame counts.
    """
    # Load dataset annotations
    with open(dataset_annotation_path) as f:
        dataset_annotations = json.load(f)

    # Initialize video frame count dictionary
    video_frame_counts = {}

    for image in dataset_annotations["images"]:
        video_id = image["video_id"]
        frame_index = image["frame_index"]

        if video_id not in video_frame_counts:
            video_frame_counts[video_id] = 0

        # Update the frame count to the maximum frame index + 1
        video_frame_counts[video_id] = max(
            video_frame_counts[video_id], frame_index + 1
        )

    return video_frame_counts


def convert_cocovid_to_ytbvis_pred(
    cocovid_pred_path: str,
    dataset_annotation_path: str,
    output_path: str,
    video_frame_counts: dict,
) -> None:
    """
    Convert COCO format predictions to YouTubeVIS format.

    Args:
        cocovid_pred_path: Path to COCO format prediction JSON
        dataset_annotation_path: Path to dataset annotation JSON in COCO format
        output_path: Path to save YouTubeVIS format predictions
        video_frame_counts: A dictionary mapping video_id to total frame count
    """

    # Load COCO predictions
    with open(cocovid_pred_path) as f:
        coco_predictions = json.load(f)

    # Load dataset annotations to map image_id to frame_id and video_id
    with open(dataset_annotation_path) as f:
        dataset_annotations = json.load(f)

    # Create image_id to (video_id, frame_id) mapping, frame_id is the index of the frame in the video
    image_id_map = {
        img["id"]: (img["video_id"], img["frame_id"])
        for img in dataset_annotations["images"]
    }

    ytbvis_predictions = []

    # Group predictions by track_id
    track_groups = {}
    for ann in tqdm(coco_predictions, desc="Processing COCO predictions"):
        image_id = ann["image_id"]
        track_id = ann["track_id"]
        bbox = ann["bbox"]
        score = ann["score"]
        category_id = ann["category_id"]
        segmentation = ann.get("segmentation")  # Get segmentation if available
        area = ann.get("area")  # Get area if available

        # Map image_id to video_id and frame_id
        if image_id not in image_id_map:
            raise RuntimeError(f"Image ID {image_id} not found in dataset annotations")

        video_id, frame_id = image_id_map[image_id]

        if track_id not in track_groups:
            track_groups[track_id] = {
                "video_id": video_id,
                "category_id": category_id,
                "bboxes": [None] * video_frame_counts[video_id],
                "segmentations": [None] * video_frame_counts[video_id],
                "areas": [None] * video_frame_counts[video_id],
                "scores": [],
            }

        # Assign bbox, segmentation, area and score to the correct frame
        track_groups[track_id]["bboxes"][frame_id] = bbox
        track_groups[track_id]["segmentations"][frame_id] = segmentation
        track_groups[track_id]["areas"][frame_id] = area
        track_groups[track_id]["scores"].append(score)

    # Convert grouped tracks to YouTubeVIS format
    for track_id, track_data in tqdm(
        track_groups.items(), desc="Finalizing YouTubeVIS predictions"
    ):
        # Calculate average score for the tracklet
        track_data["score"] = sum(track_data["scores"]) / len(track_data["scores"])
        del track_data["scores"]  # Remove scores list after averaging
        ytbvis_predictions.append(track_data)

    # Save output
    with open(output_path, "w") as f:
        json.dump(ytbvis_predictions, f)

    print(f"Converted {len(ytbvis_predictions)} tracks to YouTubeVIS format")
