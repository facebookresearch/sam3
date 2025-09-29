import argparse
import logging
import os
import shutil
import subprocess
import sys
from glob import glob

import cv2

import pandas as pd
import yt_dlp
from tqdm import tqdm


logger = logging.getLogger(__name__)


def download_youtube_video(
    video_id, cookies_file="cookies.txt", output_dir="./downloads"
):
    video_url = f"https://youtube.com/watch?v={video_id}"

    assert os.path.exists(
        cookies_file
    ), f"Cookies file '{cookies_file}' not found. Must have it to download videos."
    os.makedirs(output_dir, exist_ok=True)
    outtmpl = os.path.join(output_dir, f"{video_id}.mp4")

    # Check if the output file already exists
    if os.path.exists(outtmpl) and os.path.isfile(outtmpl):
        logger.info(f"Video {video_id} already exists at {outtmpl}")
        return "already exists"

    ydl_opts = {
        # "format": "bestvideo[height<=720]/bestvideo",  # Video only, 720p or lower
        "format": "best[height<=720]/best",  # Best available format, 720p or lower
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "cookiefile": cookies_file,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading video from {video_url} to {outtmpl}...")
            ydl.download([video_url])
            print("Download completed successfully!")
            return "success"
    except Exception as e:
        logger.error(f"Error downloading video {video_id}: {e}")
        return f"error {e}"


def download():
    df = pd.read_json(
        "saco_veval_data/saco_id_map/yt_id_to_saco_yt1b_id_map.json", orient="records"
    )
    yt_video_ids = df.yt_video_id.unique()

    status_file = "saco_yt1b_video_download_status.csv"
    with open(status_file, "w") as f:
        f.write("yt_video_id,download_status\n")

    for video_id in tqdm(yt_video_ids):
        status = download_youtube_video(
            video_id=video_id, cookies_file="cookies.txt", output_dir="./downloads"
        )

        with open(status_file, "a") as f:
            f.write(f"{video_id},{status}\n")


def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "height": height,
        "width": width,
        "num_frames": frame_count,
        "fps": fps,
        "duration": duration,
    }


def check_downloaded_video_metadata():
    for video_path in tqdm(glob.glob("./downloads/*.mp4")):
        metadata = get_video_metadata(video_path)
        print(
            f"{os.path.basename(video_path)}: height={metadata['height']}, width={metadata['width']}, num_frames={metadata['num_frames']}, fps={metadata['fps']:.2f}, duration={metadata['duration']:.2f}s"
        )


def check_video_frame_metadata():
    jpeg_paths = glob(
        "/fsx-onevision-auto-sync/tym/sam3_video/release/media/release_09242025/*yt1b*/*/00000.jpg"
    )
    for jpeg_path in tqdm(jpeg_paths):
        # Load image to get height and width only
        img = cv2.imread(jpeg_path)
        if img is None:
            raise ValueError(f"Could not load image file: {jpeg_path}")
        height, width = img.shape[:2]
        metadata = {"height": height, "width": width}
        print(
            f"{os.path.basename(jpeg_path)}: height={metadata['height']}, width={metadata['width']}"
        )


def preprocess_video(input_video_path, output_video_path, ffmpeg_path="ffmpeg"):
    """
    Preprocess video by converting it to specified format.

    Args:
        input_video_path (str): Path to input video file
        output_video_path (str): Path to output video file
        ffmpeg_path (str): Path to ffmpeg executable (default: "ffmpeg")

    Returns:
        bool: True if successful, False otherwise
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    args = [
        "-nostdin",
        "-y",
        "-i",
        input_video_path,
        # set output video resolution to be at most 1080p
        "-vf",
        f"scale={1080}:-2",
        # specify output format
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        output_video_path,
    ]

    logger.info(f"Processing full video from {input_video_path}")
    _ = subprocess.run(
        [ffmpeg_path] + args, timeout=1000, capture_output=True, text=True
    )


def preprocess_video_to_frames(input_video_path, output_dir, ffmpeg_path="ffmpeg"):
    """
    Extract all frames from video at original frame rate and save as JPEG files.

    Args:
        input_video_path (str): Path to input video file
        output_dir (str): Directory to save output JPEG frames
        ffmpeg_path (str): Path to ffmpeg executable (default: "ffmpeg")

    Returns:
        bool: True if successful, False otherwise
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define output pattern with 5-digit zero-padded frame numbers
    output_pattern = os.path.join(output_dir, "%05d.jpg")

    args = [
        "-nostdin",
        "-y",
        "-i",
        input_video_path,
        "-q:v",
        "2",
        output_pattern,
    ]

    logger.info(f"Extracting frames from {input_video_path} to {output_dir}")
    result = subprocess.run(
        [ffmpeg_path] + args, timeout=1000, capture_output=True, text=True
    )


def preprocess_saco_yt1b_frames(yt_video_id):
    """
    Copy frames from yt_video_id folder to saco_yt1b_id folder based on frame mapping.

    Args:
        yt_video_id (str): YouTube video ID
    """
    df = pd.read_json("saco_veval_data/saco_id_map/yt_id_to_saco_yt1b_id_map.json")

    matching_rows = df[df["yt_video_id"] == yt_video_id]
    if len(matching_rows) != 1:
        raise ValueError(
            f"Expected exactly 1 mapping for yt_video_id: {yt_video_id}, found {len(matching_rows)}"
        )

    row = matching_rows.iloc[0]
    saco_yt1b_id = row["saco_yt1b_id"]
    frame_matching = row["frame_matching"]

    logger.info(f"Processing frames for {yt_video_id} -> {saco_yt1b_id}")

    source_dir = f"JPEGImages/{yt_video_id}"
    dest_dir = f"saco_veval_data/09242025/JPEGImages/{saco_yt1b_id}"
    os.makedirs(dest_dir, exist_ok=True)

    for saco_frame_num, yt_frame_num in tqdm(frame_matching, desc="Copying frames"):
        source_file = os.path.join(source_dir, f"{yt_frame_num:05d}.jpg")
        dest_filename = f"{saco_frame_num:05d}.jpg"
        dest_file = os.path.join(dest_dir, dest_filename)
        shutil.copy2(source_file, dest_file)
        
    logger.info(
        f"Successfully copied {len(frame_matching)} frames from {yt_video_id} to {saco_yt1b_id}"
    )


def main():
    logger.warning(
        "This script uses yt_dlp to donwload videos. Check the risk of account banning at https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies."
    )
    # check_downloaded_video_metadata()
    # check_video_frame_metadata()
    # download()
    # download_youtube_video()
    yt_video_id = "-06NgWyZxC0"
    preprocess_video(input_video_path=f"downloads/{yt_video_id}.mp4", output_video_path=f"processed_downloads/{yt_video_id}.mp4")
    preprocess_video_to_frames(input_video_path=f"processed_downloads/{yt_video_id}.mp4", output_dir=f"JPEGImages/{yt_video_id}")
    preprocess_saco_yt1b_frames(yt_video_id)


if __name__ == "__main__":
    main()
