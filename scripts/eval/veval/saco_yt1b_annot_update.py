# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
import argparse
import json
import logging
import os

import pandas as pd


logger = logging.getLogger(__name__)


def get_available_saco_yt1b_ids(yt1b_meida_dir, data):
    vdf = pd.DataFrame(data["videos"])
    expected_saco_yt1b_ids = vdf.video_name.tolist()

    yt1b_media_folders = os.listdir(yt1b_meida_dir)

    available_saco_yt1b_ids = []
    for yt1b_media_folder in yt1b_media_folders:
        if yt1b_media_folder not in expected_saco_yt1b_ids:
            continue
        jpeg_folder_dir = os.path.join(yt1b_meida_dir, yt1b_media_folder)
        jpeg_count = len(os.listdir(jpeg_folder_dir))
        if jpeg_count > 0:
            available_saco_yt1b_ids.append(yt1b_media_folder)
        else:
            logger.info(
                f"No JPEG images found for {yt1b_media_folder}. The annotation related to this video will be removed."
            )

    logger.info(
        f"Expected {len(expected_saco_yt1b_ids)} videos for {data['info']}. Found {len(available_saco_yt1b_ids)} videos available in {yt1b_meida_dir}."
    )
    return available_saco_yt1b_ids


def update_yt1b_annot_per_field(data, field, id_col, available_ids):
    field_data = data[field]
    new_field_data = []
    for data_entry in field_data:
        if data_entry[id_col] not in available_ids:
            logger.info(
                f"{field}: Removing {data_entry} due to the video being unavailable."
            )
            continue
        new_field_data.append(data_entry)

    data[field] = new_field_data
    logger.info(
        f"Updated {field} by {id_col} - Before: {len(field_data)}, After: {len(new_field_data)}, Removed: {len(field_data) - len(new_field_data)}"
    )
    return data


def update_yt1b_annot(yt1b_input_annot_path, yt1b_media_dir, yt1b_output_annot_path):
    with open(yt1b_input_annot_path, "r") as f:
        data = json.load(f)

    available_saco_yt1b_ids = get_available_saco_yt1b_ids(yt1b_media_dir, data)

    data = update_yt1b_annot_per_field(
        data=data,
        field="videos",
        id_col="video_name",
        available_ids=available_saco_yt1b_ids,
    )

    videos_data = data["videos"]
    available_video_incremental_ids = [data_entry["id"] for data_entry in videos_data]

    data = update_yt1b_annot_per_field(
        data=data,
        field="annotations",
        id_col="video_id",
        available_ids=available_video_incremental_ids,
    )
    data = update_yt1b_annot_per_field(
        data=data,
        field="video_np_pairs",
        id_col="video_id",
        available_ids=available_video_incremental_ids,
    )

    with open(yt1b_output_annot_path, "w") as f:
        json.dump(data, f)

    return data


def main():
    parser = argparse.ArgumentParser(description="Run video grounding evaluators")
    parser.add_argument(
        "--yt1b_media_dir",
        type=str,
        help="Path to the directory where the yt1b media is stored e.g media/saco_yt1b/JPEGImages_6fps",
    )
    parser.add_argument(
        "--yt1b_input_annot_path",
        type=str,
        help="Path to the saco_veval_yt1b input annotation file e.g annotation/saco_veval_yt1b_test.json or annotation/saco_veval_yt1b_val.json",
    )
    parser.add_argument(
        "--yt1b_output_annot_path",
        type=str,
        help="Path to the output annotation file e.g annotation/saco_veval_yt1b_test_updated.json or annotation/saco_veval_yt1b_val_updated.json",
    )
    parser.add_argument(
        "--yt1b_annot_update_log_path",
        type=str,
        help="Path to the yt1b annot update log file e.g annotation/yt1b_annot_update_log.log",
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.yt1b_annot_update_log_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.yt1b_output_annot_path), exist_ok=True)

    logging.basicConfig(
        filename=args.yt1b_annot_update_log_path,
        format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        filemode="w",
    )

    _ = update_yt1b_annot(
        yt1b_input_annot_path=args.yt1b_input_annot_path,
        yt1b_media_dir=args.yt1b_media_dir,
        yt1b_output_annot_path=args.yt1b_output_annot_path,
    )

    print("Done!! Check the log at", args.yt1b_annot_update_log_path)


if __name__ == "__main__":
    main()
