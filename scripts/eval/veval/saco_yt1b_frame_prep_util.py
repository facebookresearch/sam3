import argparse
import os
import re
import shutil
import subprocess
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
            self.data_dir, "JPEGImages_6fps", self.saco_yt1b_id
        )

        os.makedirs(self.raw_video_dir, exist_ok=True)
        os.makedirs(self.raw_frames_resized_width_1080_dir, exist_ok=True)
        os.makedirs(self.frames_by_frame_matching_dir, exist_ok=True)

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
        """Get the total number of frames in the raw video using cv2 for speed and reliability."""
        if not os.path.exists(self.raw_video_path):
            return 0

        try:
            # Use cv2 for fast and reliable frame counting
            cap = cv2.VideoCapture(self.raw_video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {self.raw_video_path}")

            # Get frame count from video metadata
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if total_frames > 0:
                print(f"cv2 reports {total_frames} frames")
                return total_frames
            else:
                raise ValueError("cv2 could not determine frame count")

        except Exception as e:
            print(f"Error getting frame count with cv2: {e}")
            raise ValueError("cv2 failed to get frame count")

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

    def generate_all_raw_frames(self, timeout_seconds=3600):
        """
        Extract all frames from the raw video to raw_frames_resized_width_1080_dir.
        This is the first step before frame matching.

        Args:
            timeout_seconds: Timeout for ffmpeg operation in seconds (default: 3600 = 1 hour)
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

            # Allow 1-frame tolerance buffer for frame count comparison
            frame_diff = abs(existing_count - total_video_frames)
            if frame_diff <= 1:
                print(
                    f"cv2 frame count and the already extracted frame count differ by less than 1 frame. double checking by the precise ffprobe method"
                )

                # Use ffprobe to get precise frame count
                try:
                    cmd = [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-count_frames",
                        "-show_entries",
                        "stream=nb_read_frames",
                        "-of",
                        "csv=p=0",
                        self.raw_video_path,
                    ]

                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=timeout_seconds
                    )
                    if result.returncode == 0:
                        precise_frames = int(result.stdout.strip())
                        print(f"ffprobe precise count: {precise_frames} frames")

                        # Check if existing frames match the precise count
                        precise_diff = abs(existing_count - precise_frames)
                        if precise_diff <= 1:
                            print(
                                f"Existing frames match precise count (within 1-frame tolerance: {existing_count} vs {precise_frames}). Using existing frames."
                            )
                            return True
                        else:
                            print(
                                f"Precise count mismatch ({existing_count} vs {precise_frames}, diff: {precise_diff}). Re-generating frames."
                            )
                    else:
                        print(
                            "ffprobe precise count failed, can't confirm the frame count matching, re-generating frames"
                        )
                        return False
                except Exception as e:
                    print(
                        f"ffprobe precise count error: {e}, can't confirm the frame count matching, re-generating frames"
                    )
                    return False
            else:
                print(
                    f"Frame count mismatch ({existing_count} != {total_video_frames}, diff: {frame_diff}). Re-generating frames."
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
        print(
            f"Video has {total_video_frames} frames. This may take several minutes..."
        )

        # Optimize ffmpeg args for large videos
        args = [
            "-nostdin",
            "-y",
            "-i",
            self.raw_video_path,
            # set output video resolution to be at most 1080p
            "-vf",
            "scale=1080:-2",
            "-vsync",
            "0",  # passthrough mode - no frame duplication/dropping
            "-q:v",
            "2",  # high quality JPEG output
            "-start_number",
            "0",  # start frame numbering from 0
            # Add progress reporting for long operations
            "-progress",
            "pipe:1",  # Send progress to stdout
            self.raw_frames_resized_width_1080_pattern,
        ]

        print(f"Starting ffmpeg with {timeout_seconds}s timeout...")
        result = subprocess.run(
            ["ffmpeg"] + args,
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Failed to extract raw frames: {result.stderr}")
            if "TimeoutExpired" in str(result.stderr):
                print(
                    f"Operation timed out after {timeout_seconds} seconds. Consider increasing timeout for very large videos."
                )
            return False

        extracted_frames = glob(
            os.path.join(self.raw_frames_resized_width_1080_dir, "*.jpg")
        )
        print(
            f"Successfully extracted {len(extracted_frames)} frames to {self.raw_frames_resized_width_1080_dir}"
        )

        # Verify we got the expected number of frames
        assert (
            len(extracted_frames) == total_video_frames
        ), f"Expected {total_video_frames} frames but extracted {len(extracted_frames)}"

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saco_yt1b_id", type=str, required=True)
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--cookies_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--id_map_file",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    video_prep = YtVideoPrep(
        args.saco_yt1b_id, args.data_dir, args.cookies_file, args.id_map_file
    )
    video_prep.download_youtube_video()
    # Use longer timeout for large videos (2 hours)
    video_prep.generate_all_raw_frames(timeout_seconds=7200)
    video_prep.generate_frames_by_frame_matching()


if __name__ == "__main__":
    main()
