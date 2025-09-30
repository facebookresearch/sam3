import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from glob import glob

import cv2

import pandas as pd
import yt_dlp
from tqdm import tqdm


class YtVideoPrep:
    def __init__(
        self,
        saco_yt1b_id: str,
        data_dir: str,
        cookies_file: str,
        id_and_frame_map_path: str,
    ):
        self.saco_yt1b_id = saco_yt1b_id  # saco_yt1b_id is like saco_yt1b_000000
        self.data_dir = data_dir
        self.cookies_file = cookies_file

        self.id_and_frame_map_df = pd.read_json(id_and_frame_map_path)
        (
            self.yt_video_id,
            self.yt_video_id_w_timestamps,
            self.start_timestamp,
            self.end_timestamp,
            self.frame_matching,
        ) = self._get_yt_video_id_map_info()

        self.raw_video_dir = os.path.join(self.data_dir, "raw_videos")
        self.raw_video_path = os.path.join(
            self.raw_video_dir, f"{self.yt_video_id}.mp4"
        )

        self.raw_frames_resized_width_1080_dir = os.path.join(
            self.data_dir, "raw_frames_resized_width_1080", self.saco_yt1b_id
        )
        self.raw_frames_resized_width_1080_pattern = os.path.join(
            self.raw_frames_resized_width_1080_dir, "%05d.jpg"
        )
        self.frames_by_frame_matching_dir = os.path.join(
            self.data_dir, "frames_by_frame_matching", self.saco_yt1b_id
        )

        self.frames_by_start_end_timestamp_one_step_dir = os.path.join(
            self.data_dir, "frames_by_start_end_timestamp_one_step", self.saco_yt1b_id
        )
        self.frames_by_start_end_timestamp_one_step_pattern = os.path.join(
            self.frames_by_start_end_timestamp_one_step_dir, "%05d.jpg"
        )

        self.video_by_start_end_timestamp_two_step_dir = os.path.join(
            self.data_dir, "video_by_start_end_timestamp_two_step", self.saco_yt1b_id
        )
        self.video_by_start_end_timestamp_two_step_path = os.path.join(
            self.video_by_start_end_timestamp_two_step_dir, f"{self.yt_video_id}.mp4"
        )
        self.frames_by_start_end_timestamp_two_step_dir = os.path.join(
            self.data_dir, "frames_by_start_end_timestamp_two_step", self.saco_yt1b_id
        )
        self.frames_by_start_end_timestamp_two_step_pattern = os.path.join(
            self.frames_by_start_end_timestamp_two_step_dir, "%05d.jpg"
        )

        os.makedirs(self.raw_video_dir, exist_ok=True)
        os.makedirs(self.raw_frames_resized_width_1080_dir, exist_ok=True)
        os.makedirs(self.frames_by_frame_matching_dir, exist_ok=True)
        os.makedirs(self.frames_by_start_end_timestamp_one_step_dir, exist_ok=True)
        os.makedirs(self.video_by_start_end_timestamp_two_step_dir, exist_ok=True)
        os.makedirs(self.frames_by_start_end_timestamp_two_step_dir, exist_ok=True)

    def _get_yt_video_id_map_info(self):
        df = self.id_and_frame_map_df[
            self.id_and_frame_map_df.saco_yt1b_id == self.saco_yt1b_id
        ]
        assert (
            len(df) == 1
        ), f"Expected exactly 1 row for saco_yt1b_id: {self.saco_yt1b_id}, found {len(df)}"
        id_and_frame_map_row = df.iloc[0]

        yt_video_id = (
            id_and_frame_map_row.yt_video_id
        )  # yt_video_id is like -06NgWyZxC0
        yt_video_id_w_timestamps = id_and_frame_map_row.yt_video_id_w_timestamps
        start_timestamp, end_timestamp = self._parse_timestamp(yt_video_id_w_timestamps)
        frame_matching = id_and_frame_map_row.frame_matching

        return (
            yt_video_id,
            yt_video_id_w_timestamps,
            start_timestamp,
            end_timestamp,
            frame_matching,
        )

    def _get_total_frame_count(self):
        """Get the total number of frames in the raw video using ffprobe for accuracy."""
        if not os.path.exists(self.raw_video_path):
            return 0

        try:
            # Use ffprobe for more accurate frame counting
            result = subprocess.run([
                "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-count_frames", "-show_entries", "stream=nb_read_frames",
                "-of", "csv=p=0", self.raw_video_path
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and result.stdout.strip():
                total_frames = int(result.stdout.strip())
                print(f"ffprobe reports {total_frames} frames")
                return total_frames
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            raise ValueError("ffprobe failed or not available")


    def _parse_timestamp(self, yt_video_id_w_timestamps):
        # In id_and_frame_map_path, we expect the pattern of {video_id}_start_{float}_end_{float} for column yt_video_id_w_timestamps
        pattern = r"^(.+)_start_(\d+(?:\.\d+)?)_end_(\d+(?:\.\d+)?)$"
        match = re.match(pattern, yt_video_id_w_timestamps)
        if not match:
            raise ValueError(
                f"Invalid format: {yt_video_id_w_timestamps}. Expected format: {{video_id}}_start_{{start_time}}_end_{{end_time}}"
            )

        # Extract start and end timestamps from the regex groups
        start_timestamp = match.group(2)
        end_timestamp = match.group(3)

        return start_timestamp, end_timestamp

    def download_youtube_video(self):
        video_url = f"https://youtube.com/watch?v={self.yt_video_id}"

        assert os.path.exists(
            self.cookies_file
        ), f"Cookies file '{self.cookies_file}' not found. Must have it to download videos."

        outtmpl = self.raw_video_path

        # Check if the output file already exists
        if os.path.exists(outtmpl) and os.path.isfile(outtmpl):
            print(f"Video {self.yt_video_id} already exists at {outtmpl}")
            return "already exists"

        ydl_opts = {
            "format": "best[height<=720]/best",  # 720p or lower
            "outtmpl": outtmpl,
            "merge_output_format": "mp4",
            "noplaylist": True,
            "quiet": True,
            "cookiefile": self.cookies_file,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print(f"Downloading video from {video_url} to {outtmpl}...")
                ydl.download([video_url])
                print("Download completed successfully!")
                return "success"
        except Exception as e:
            print(f"Error downloading video {self.yt_video_id}: {e}")
            return f"error {e}"

    def generate_all_raw_frames(self):
        """
        Extract all frames from the raw video to raw_frames_resized_width_1080_dir.
        This is the first step before frame matching.
        """
        if not os.path.exists(self.raw_video_path):
            print(f"Error: Raw video file not found at {self.raw_video_path}")
            return False

        # Check if frames already exist and match video frame count
        existing_frames = glob(
            os.path.join(self.raw_frames_resized_width_1080_dir, "*.jpg")
        )
        total_video_frames = self._get_total_frame_count()

        if existing_frames:
            existing_count = len(existing_frames)
            print(
                f"Found {existing_count} existing raw frames in {self.raw_frames_resized_width_1080_dir}"
            )
            print(f"Video has {total_video_frames} total frames")

            if existing_count == total_video_frames:
                print("Frame count matches video frame count. Using existing frames.")
                return True
            else:
                print(
                    f"Frame count mismatch ({existing_count} != {total_video_frames}). Re-generating frames."
                )
                # Remove existing frames before regenerating
                print(f"Removing {existing_count} existing frames...")
                for frame_file in existing_frames:
                    try:
                        os.remove(frame_file)
                    except OSError as e:
                        print(f"Warning: Could not remove {frame_file}: {e}")
                print("Existing frames cleared.")
                

        print(
            f"Extracting all frames from {self.raw_video_path} to {self.raw_frames_resized_width_1080_dir}"
        )

        args = [
            "-nostdin",
            "-y",
            "-i",
            self.raw_video_path,
            # set output video resolution to be at most 1080p and fps to 6
            "-vf",
            "scale=1080:-2",
            "-vsync",
            "0",  # passthrough mode - no frame duplication/dropping
            "-q:v",
            "2",  # high quality JPEG output
            "-start_number",
            "0",  # start frame numbering from 0
            self.raw_frames_resized_width_1080_pattern,
        ]

        result = subprocess.run(
            ["ffmpeg"] + args,
            timeout=300,  # 5 minute timeout
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Failed to extract raw frames: {result.stderr}")
            return False

        extracted_frames = glob(
            os.path.join(self.raw_frames_resized_width_1080_dir, "*.jpg")
        )
        print(
            f"Successfully extracted {len(extracted_frames)} frames to {self.raw_frames_resized_width_1080_dir}"
        )
        return True

    def generate_frames_by_frame_matching(self):
        """
        Copy and rename specific frames from raw_frames_resized_width_1080_dir to frames_by_frame_matching_dir
        based on the frame_matching list of [dst_frame_num, src_frame_num] pairs.
        """
        # First ensure all raw frames are extracted
        if not self.generate_all_raw_frames():
            return False

        frame_matching = self.frame_matching
        total_frames = len(frame_matching)

        print(f"Copying {total_frames} frames based on frame matching")

        success_count = 0
        for dst_frame_num, src_frame_num in tqdm(frame_matching, desc="Copying frames"):
            # Source frame file (from raw frames)
            src_file = os.path.join(
                self.raw_frames_resized_width_1080_dir, f"{src_frame_num:05d}.jpg"
            )

            # Destination frame file (renamed according to dst_frame_num)
            dst_file = os.path.join(
                self.frames_by_frame_matching_dir, f"{dst_frame_num:05d}.jpg"
            )

            # Skip if destination file already exists
            if os.path.exists(dst_file):
                success_count += 1
                continue

            # Check if source frame exists
            assert os.path.exists(
                src_file
            ), f"Source frame {src_frame_num:05d}.jpg not found"

            try:
                shutil.copy2(src_file, dst_file)
                success_count += 1
            except Exception as e:
                raise ValueError(
                    f"Error copying frame {src_frame_num} -> {dst_frame_num}: {e}"
                )

        print(
            f"Successfully copied {success_count}/{total_frames} frames to {self.frames_by_frame_matching_dir}"
        )
        return success_count == total_frames

    def generate_frames_by_start_end_timestamp_one_step(self):
        args = [
            "-nostdin",
            "-y",
            # select video segment
            "-ss",
            self.start_timestamp,
            "-to",
            self.end_timestamp,
            "-i",
            self.raw_video_path,
            # set output video resolution to be at most 1080p and fps to 6
            "-vf",
            "fps=6,scale=1080:-2",
            "-vsync",
            "0",  # passthrough mode - no frame duplication/dropping
            # high quality JPEG output
            "-q:v",
            "2",
            # start frame numbering from 0 instead of 1
            "-start_number",
            "0",
            self.frames_by_start_end_timestamp_one_step_pattern,
        ]

        result = subprocess.run(
            ["ffmpeg"] + args, timeout=1000, capture_output=True, text=True
        )

        if result.returncode != 0:
            print(
                f"Generate frames by start end timestamp one step failed - FFmpeg failed with error: {result.stderr}"
            )
            return False

        print(
            f"Successfully extracted frames to {self.frames_by_start_end_timestamp_one_step_dir}"
        )
        return True

    def generate_frames_by_start_end_timestamp_two_step(self):
        video_args = [
            "-nostdin",
            "-y",
            # select video segment
            "-ss",
            self.start_timestamp,
            "-to",
            self.end_timestamp,
            "-i",
            self.raw_video_path,
            # set output video resolution to be at most 1080p and fps to 6
            "-vf",
            "fps=6,scale=1080:-2",
            "-vsync",
            "0",  # passthrough mode - no frame duplication/dropping
            # specify output format
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            self.video_by_start_end_timestamp_two_step_path,
        ]

        result = subprocess.run(
            ["ffmpeg"] + video_args, timeout=1000, capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"Step 1 failed - FFmpeg failed with error: {result.stderr}")
            return False

        frame_args = [
            "-nostdin",
            "-y",
            "-i",
            self.video_by_start_end_timestamp_two_step_path,
            "-vsync",
            "0",  # passthrough mode - no frame duplication/dropping
            # high quality JPEG output
            "-q:v",
            "2",
            # start frame numbering from 0 instead of 1
            "-start_number",
            "0",
            self.frames_by_start_end_timestamp_two_step_pattern,
        ]

        result = subprocess.run(
            ["ffmpeg"] + frame_args, timeout=1000, capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"Step 2 failed - FFmpeg failed with error: {result.stderr}")
            return False

        print(
            f"Successfully extracted frames to {self.frames_by_start_end_timestamp_two_step_dir}"
        )
        return True


def batch_video_prep():
    id_and_frame_map_path = "/home/tym/code/git_clone/sam3_and_data/data/media/saco_yt1b/id_and_frame_map.json"
    data_dir="/home/tym/code/git_clone/sam3_and_data/data/media/saco_yt1b"
    cookies_file="/home/tym/code/git_clone/sam3_and_data/data/media/saco_yt1b/cookies.txt"

    id_map_df = pd.read_json(id_and_frame_map_path)
    saco_yt1b_ids = id_map_df.saco_yt1b_id.unique()

    for saco_yt1b_id in tqdm(saco_yt1b_ids, desc="Processing videos"):
        print(f"Processing {saco_yt1b_id}...")
        video_prep = YtVideoPrep(
            saco_yt1b_id=saco_yt1b_id,
            data_dir=data_dir,
            cookies_file=cookies_file,
            id_and_frame_map_path=id_and_frame_map_path,
        )
        video_prep.download_youtube_video()
        video_prep.generate_all_raw_frames()
        video_prep.generate_frames_by_frame_matching()
        video_prep.generate_frames_by_start_end_timestamp_one_step()
        video_prep.generate_frames_by_start_end_timestamp_two_step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saco_yt1b_id", type=str, required=True)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/tym/code/git_clone/sam3_and_data/data/media/saco_yt1b",
    )
    parser.add_argument(
        "--cookies_file",
        type=str,
        default="/home/tym/code/git_clone/sam3_and_data/data/media/saco_yt1b/cookies.txt",
    )
    parser.add_argument(
        "--id_and_frame_map_path",
        type=str,
        default="/home/tym/code/git_clone/sam3_and_data/data/media/saco_yt1b/id_and_frame_map.json",
    )
    args = parser.parse_args()

    video_prep = YtVideoPrep(
        args.saco_yt1b_id, args.data_dir, args.cookies_file, args.id_and_frame_map_path
    )
    video_prep.download_youtube_video()
    video_prep.generate_all_raw_frames()
    video_prep.generate_frames_by_frame_matching()
    video_prep.generate_frames_by_start_end_timestamp_one_step()
    video_prep.generate_frames_by_start_end_timestamp_two_step()


if __name__ == "__main__":
    batch_video_prep()
    # main()
