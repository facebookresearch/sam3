# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
import ast
import concurrent.futures
import os
import shutil
import subprocess
import sys
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path

import yt_dlp

from utils import (
    annotation_files,
    config,
    load_json,
    run_command,
    save_json,
    update_annotations,
)


def construct_gcs_path(original_video):
    """
    Convert original_video string to GCS path.
    Example:
    'AUTOLab_failure_2023-07-07_Fri_Jul__7_18:50:36_2023_recordings_MP4_22008760.mp4'
    ->
    'gs://gresearch/robotics/droid_raw/1.0.1/AUTOLab/failure/2023-07-07/Fri_Jul__7_18:50:36_2023/recordings/MP4/22008760.mp4'
    """
    parts = original_video.split("_")
    lab = parts[0]
    failure = parts[1]
    date = parts[2]
    time = "_".join(parts[3:-3])
    recordings = parts[-3]
    mp4 = parts[-2]
    file_id = parts[-1].split(".")[0]
    gcs_path = (
        f"gs://gresearch/robotics/droid_raw/1.0.1/"
        f"{lab}/{failure}/{date}/{time}/{recordings}/{mp4}/{file_id}.mp4"
    )
    return gcs_path


def download_video(args):
    gcs_path, dst_dir, json_file = args
    # Ensure subdirectory exists
    subdir = Path(dst_dir)
    os.makedirs(subdir, exist_ok=True)
    # Save file with its original name inside the subdir
    print(json_file)
    local_path = subdir / json_file
    cmd = f'gsutil cp "{gcs_path}" "{local_path}"'
    print(f"Running: {cmd}")
    try:
        run_command(cmd)
        return (gcs_path, True, None)
    except Exception as e:
        return (gcs_path, False, str(e))


def download_youtube_video(youtube_id, output_path=None):
    try:
        if output_path is None:
            output_path = os.path.join(
                config["yt1b_path"], "downloaded_videos", f"video_{youtube_id}.mp4"
            )
        url = f"https://www.youtube.com/watch?v={youtube_id}"
        if os.path.exists(output_path):
            return youtube_id, None
        format = "best[height<=720][fps<=30]/best[height<=720]/best"  # 720p or lower, max 30fps
        ydl_opts = {
            "format": format,
            "outtmpl": output_path,
            "merge_output_format": "mp4",
            "quiet": True,
            "cookiefile": config["cookies_path"],
            "socket_timeout": 60,  # Increase timeout to 60 seconds (default is 10)
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return youtube_id, None
    except Exception as e:
        return youtube_id, str(e)


def download_youtube():
    all_videos_to_download = set()
    for annotation_file in annotation_files["yt1b"]:
        ann = load_json(os.path.join(config["path_annotations"], annotation_file))
        for video_info in ann["images"]:
            youtube_id = video_info["original_video"]
            all_videos_to_download.add(youtube_id)

    videos_to_download_still = all_videos_to_download
    videos_downloaded = set()
    videos_unavailable = set()
    num_download_retries = 3
    for _ in range(num_download_retries):
        if len(videos_to_download_still) == 0:
            break
        videos_error = set()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(download_youtube_video, youtube_id)
                for youtube_id in videos_to_download_still
            ]
            for future in concurrent.futures.as_completed(futures):
                youtube_id, exception = future.result()
                if exception is None:
                    videos_downloaded.add(youtube_id)
                elif "unavailable" in exception or "members-only" in exception:
                    videos_unavailable.add(youtube_id)
                else:
                    videos_error.add(youtube_id)
        videos_to_download_still = (
            all_videos_to_download - videos_downloaded - videos_unavailable
        )
        assert videos_to_download_still == videos_error

    if len(videos_unavailable) + len(videos_to_download_still) > 0:
        message = "Some videos are either no longer available on YouTube, or are set to private, or resulted in some other error. "
        if config["update_annotation_yt1b"]:
            message += "The unavailable videos will be ***REMOVED*** from the annotation file. This will make the test results NOT DIRECTLY COMPARABLE to other reported results."
            print(message)
            update_annotations("yt1b", videos_downloaded)
        else:
            message += "You may want to either re-try the download, or remove these videos from the evaluation json"
            print(message)


def download_droid():
    ann_dir = Path(config["path_annotations"])
    dst_dir = Path(config["droid_path"]) / "downloaded_videos"
    json_files = annotation_files["droid"]

    download_tasks = []
    original_videos = set()
    for json_file in json_files:
        json_path = ann_dir / json_file
        data = load_json(json_path)
        for img in data["images"]:
            original_video = img["original_video"]
            original_videos.add(original_video)

    print(len(original_videos))
    for original_video in original_videos:
        gcs_path = construct_gcs_path(original_video)
        download_tasks.append((gcs_path, dst_dir, original_video))

    max_workers = min(16, len(download_tasks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(download_video, task): task for task in download_tasks
        }
        for future in as_completed(future_to_task):
            gcs_path, success, error = future.result()
            if not success:
                print(f"Failed to download {gcs_path}: {error}")


def download_ego4d():
    output_dir = os.path.join(config["ego4d_path"], "downloaded_videos")

    ann_dir = Path(config["path_annotations"])
    json_files = annotation_files["ego4d"]
    original_videos = set()
    for json_file in json_files:
        json_path = ann_dir / json_file
        data = load_json(json_path)
        for img in data["images"]:
            original_video = img["original_video"]
            original_videos.add(original_video)

    original_video_uids = [
        video_uid.replace(".mp4", "") for video_uid in original_videos
    ]
    video_ids_download = original_video_uids
    num_download_retries = 2
    download_correct = False
    message = ""
    for _ in range(num_download_retries):
        cmd = (
            [
                # "python", "-m", "ego4d.cli.cli",
                "ego4d",
                "--output_directory",
                output_dir,
                "--datasets",
                "clips",
                "--version",
                "v1",
                "--video_uids",
            ]
            + video_ids_download
            + ["--yes"]
        )

        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        message = result.stderr
        if (
            "RuntimeError: The following requested video UIDs could not be found in the manifest for version:"
            in result.stderr
        ):
            not_findable_videos = ast.literal_eval(result.stderr.split("\n")[-2])
            video_ids_download = [
                video_uid
                for video_uid in video_ids_download
                if video_uid not in not_findable_videos
            ]
        else:
            download_correct = True
            break

    if not download_correct:
        print(f"There was an error downloading the Ego4D data: {message}")

    if len(video_ids_download) != len(original_video_uids):
        message = "Some videos are no longer available. "
        if config["update_annotation_ego4d"]:
            message += "The unavailable videos will be ***REMOVED*** from the annotation file. This will make the test results NOT DIRECTLY COMPARABLE to other reported results."
            print(message)
            update_annotations("ego4d", video_ids_download)
        else:
            message += "You may want to either re-try the download, or remove these videos from the evaluation json"
            print(message)


def download_sav():
    tar_url = config["sav_videos_fps_6_download_path"]
    tar_file = "videos_fps_6.tar"
    sav_data_dir = os.path.join(config["sav_path"], "downloaded_videos")
    os.makedirs(sav_data_dir, exist_ok=True)

    subprocess.run(["wget", tar_url, "-O", tar_file], cwd=sav_data_dir, check=True)
    subprocess.run(["tar", "-xvf", tar_file], cwd=sav_data_dir, check=True)
    subprocess.run(["rm", tar_file], cwd=sav_data_dir, check=True)


def main():
    assert len(sys.argv) > 1, "You have to provide the name of the dataset"
    dataset_name = sys.argv[1]
    assert (
        dataset_name in annotation_files
    ), f"The dataset can be one of {list(annotation_files.keys())}"

    if dataset_name == "yt1b":
        download_youtube()
    elif dataset_name == "droid":
        download_droid()
    elif dataset_name == "ego4d":
        download_ego4d()
    elif dataset_name == "sav":
        download_sav()


if __name__ == "__main__":
    main()
