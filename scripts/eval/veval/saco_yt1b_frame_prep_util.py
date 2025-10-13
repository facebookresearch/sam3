import argparse
import logging
import os
import re
import shutil
import subprocess
from glob import glob

import cv2

import pandas as pd
import yt_dlp

from tqdm import tqdm

logger = logging.getLogger(__name__)


class YtVideoPrep:
    def __init__(
        self,
        saco_yt1b_id: str,
        data_dir: str,
        cookies_file: str,
        id_and_frame_map_path: str,
        ffmpeg_timeout: int,
    ):
        self.saco_yt1b_id = saco_yt1b_id  # saco_yt1b_id is like saco_yt1b_000000
        self.data_dir = data_dir
        self.cookies_file = cookies_file
        self.ffmpeg_timeout = ffmpeg_timeout

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
            "format": "best[height<=720][ext=mp4][protocol^=https]/best[ext=mp4][protocol^=https]/best[height<=720]/best",  # Prefer https MP4 formats, avoid HLS/m3u8
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

    def _get_video_frame_count(self):
        cap = cv2.VideoCapture(self.raw_video_path)
        frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return frame_number

    def _generate_all_raw_frames(self):
        """
        Extract all frames from the raw video to raw_frames_resized_width_1080_dir.
        This is the first step before frame matching.
        """
        if not os.path.exists(self.raw_video_path):
            logger.warning(
                f"[frame extracting][{self.saco_yt1b_id}] Raw video file not found at {self.raw_video_path}"
            )
            os.rmdir(self.raw_frames_resized_width_1080_dir)
            return False

        already_extracted_frame_count = len(
            os.listdir(self.raw_frames_resized_width_1080_dir)
        )
        expected_frame_count = self._get_video_frame_count()
        if expected_frame_count != 0 and abs(already_extracted_frame_count - expected_frame_count) <= 1:
            # soft compare due to sometimes cv2 frame number might be 0 or off a bit
            logger.info(
                f"[frame extracting][{self.saco_yt1b_id}] all frames already exist in {self.raw_frames_resized_width_1080_dir}, skip the full extract"
            )
            return True

        print(
            f"Extracting all frames from {self.raw_video_path} to {self.raw_frames_resized_width_1080_dir}"
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

        print(f"Starting ffmpeg with {self.ffmpeg_timeout}s timeout...")
        result = subprocess.run(
            ["ffmpeg"] + args,
            timeout=self.ffmpeg_timeout,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.warning(
                f"[frame extracting][{self.saco_yt1b_id}] Failed to extract raw frames: {result.stderr}"
            )
            os.rmdir(self.raw_frames_resized_width_1080_dir)
            return False

        extracted_frames = glob(
            os.path.join(self.raw_frames_resized_width_1080_dir, "*.jpg")
        )
        logger.info(
            f"[frame extracting][{self.saco_yt1b_id}] Successfully extracted {len(extracted_frames)} frames to {self.raw_frames_resized_width_1080_dir}"
        )

        return True

    def _rm_incomplete_frames_by_frame_matching_dir(self):
        print(
            f"Removing any existing frame in {self.frames_by_frame_matching_dir} to ensure re-copy consistency"
        )
        for old_files in glob(f"{self.frames_by_frame_matching_dir}/*.jpg"):
            os.remove(old_files)
        os.rmdir(self.frames_by_frame_matching_dir)
        print("Existing frames cleared.")

    def generate_frames_by_frame_matching(self):
        """
        Copy and rename specific frames from raw_frames_resized_width_1080_dir to frames_by_frame_matching_dir
        based on the frame_matching list of [dst_frame_num, src_frame_num] pairs.
        """
        frame_matching = self.frame_matching
        total_frames = len(frame_matching)

        if len(os.listdir(self.frames_by_frame_matching_dir)) == total_frames:
            logger.info(
                f"[frame matching][{self.saco_yt1b_id}] frames already exist in {self.frames_by_frame_matching_dir}, no need to re-copy by frame matching"
            )
            return True

        # Extract full fps frames to use the frame matching map later
        if not self._generate_all_raw_frames():
            self._rm_incomplete_frames_by_frame_matching_dir()
            return False

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
            if not os.path.exists(src_file):
                logger.warning(
                    f"[frame_matching][{self.saco_yt1b_id}] Source frame {src_file} not found"
                )
                raise ValueError(f"Source frame {src_file} not found")

            try:
                shutil.copy2(src_file, dst_file)
                success_count += 1
            except Exception as e:
                self._rm_incomplete_frames_by_frame_matching_dir()
                raise ValueError(
                    f"Error copying frame {src_frame_num} -> {dst_frame_num}: {e}"
                )

        print(
            f"Successfully copied {success_count}/{total_frames} frames to {self.frames_by_frame_matching_dir}"
        )

        status = success_count == total_frames
        if status:
            logger.info(
                f"[frame matching][{self.saco_yt1b_id}] copy to {self.frames_by_frame_matching_dir} succeeded!"
            )
        else:
            self._rm_incomplete_frames_by_frame_matching_dir()
            logger.warning(
                f"[frame matching][{self.saco_yt1b_id}] failed, some frames got extracted but not match the number of frames needed extracted {success_count} != expected {total_frames}"
            )
        return status


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
    parser.add_argument(
        "--yt1b_frame_prep_log_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ffmpeg_timeout",
        type=str,
        default=7200,  # Use longer timeout in case of large videos processing timeout
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.yt1b_frame_prep_log_path,
        format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        filemode="w",
    )

    video_prep = YtVideoPrep(
        saco_yt1b_id=args.saco_yt1b_id,
        data_dir=args.data_dir,
        cookies_file=args.cookies_file,
        id_and_frame_map_path=args.id_map_file,
        ffmpeg_timeout=args.ffmpeg_timeout,
    )

    status = video_prep.download_youtube_video()
    logger.info(f"[video download][{args.saco_yt1b_id}] download status {status}")

    video_prep.generate_frames_by_frame_matching()


if __name__ == "__main__":
    main()
