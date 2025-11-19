# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
import argparse
import logging
import os
import subprocess

import pandas as pd
import yt_dlp

logger = logging.getLogger(__name__)


class YtVideoPrep:
    def __init__(
        self,
        saco_yt1b_id: str,
        data_dir: str,
        cookies_file: str,
        yt1b_start_end_time_file: str,
        ffmpeg_timeout: int,
        sleep_interval: int = 10,
        max_sleep_interval: int = 30,
    ):
        self.saco_yt1b_id = saco_yt1b_id  # saco_yt1b_id is like saco_yt1b_000000
        self.data_dir = data_dir
        self.cookies_file = cookies_file
        self.ffmpeg_timeout = ffmpeg_timeout
        self.sleep_interval = sleep_interval
        self.max_sleep_interval = max_sleep_interval

        self.yt1b_start_end_time_df = pd.read_json(yt1b_start_end_time_file)
        (
            self.yt_video_id,
            self.yt_video_id_w_timestamps,
            self.start_time,
            self.end_time,
            self.expected_num_frames,
        ) = self._get_yt_video_id_map_info()

        self.raw_video_dir = os.path.join(self.data_dir, "raw_videos")
        self.raw_video_path = os.path.join(
            self.raw_video_dir, f"{self.yt_video_id}.mp4"
        )

        self.JPEGImages_6fps_dir = os.path.join(
            self.data_dir, "JPEGImages_6fps", self.saco_yt1b_id
        )
        self.JPEGImages_6fps_pattern = os.path.join(
            self.JPEGImages_6fps_dir, "%05d.jpg"
        )

        os.makedirs(self.raw_video_dir, exist_ok=True)
        os.makedirs(self.JPEGImages_6fps_dir, exist_ok=True)

    def _get_yt_video_id_map_info(self):
        df = self.yt1b_start_end_time_df[
            self.yt1b_start_end_time_df.saco_yt1b_id == self.saco_yt1b_id
        ]
        assert (
            len(df) == 1
        ), f"Expected exactly 1 row for saco_yt1b_id: {self.saco_yt1b_id}, found {len(df)}"
        id_and_frame_map_row = df.iloc[0]

        yt_video_id = (
            id_and_frame_map_row.yt_video_id
        )  # yt_video_id is like -06NgWyZxC0
        yt_video_id_w_timestamps = id_and_frame_map_row.yt_video_id_w_timestamps
        start_time = id_and_frame_map_row.start_time
        end_time = id_and_frame_map_row.end_time
        expected_num_frames = id_and_frame_map_row.length

        return (
            yt_video_id,
            yt_video_id_w_timestamps,
            start_time,
            end_time,
            expected_num_frames,
        )

    def download_youtube_video(self):
        video_url = f"https://youtube.com/watch?v={self.yt_video_id}"

        assert os.path.exists(
            self.cookies_file
        ), f"Cookies file '{self.cookies_file}' not found. Must have it to download videos."

        outtmpl = self.raw_video_path

        # Check if the output file already exists
        if os.path.exists(outtmpl) and os.path.isfile(outtmpl):
            return "already exists"

        ydl_opts = {
            "format": "best[height<=720]/best",  # 720p or lower
            "outtmpl": outtmpl,
            "merge_output_format": "mp4",
            "noplaylist": True,
            "quiet": True,
            "cookiefile": self.cookies_file,
            "sleep_interval": self.sleep_interval,  # Sleep before each download to avoid rate limiting
            "max_sleep_interval": self.max_sleep_interval,  # Random sleep for more human-like behavior
        }

        if self.yt_video_id in ["euohdDLEMRg", "nzfAn7n4d-0"]:
            # For "euohdDLEMRg", we have to specify the https protocol or the video sometimes can't be downloaded completely
            # For "nzfAn7n4d-0", without the https protocol, the video will be downloaded as 654×480, however we need 490×360 to match the frame matching after the 1080 width resizing
            ydl_opts["format"] = (
                "best[height<=720][ext=mp4][protocol^=https]/best[ext=mp4][protocol^=https]/best[height<=720]/best"
            )

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
                return "success"
        except Exception as e:
            logger.warning(
                f"[video download][{self.saco_yt1b_id}] Error downloading video {self.yt_video_id}: {e}"
            )
            return f"error {e}"

    def extract_frames_in_6fps_and_width_1080(self):
        """
        Extract target frames in 6fps and width 1080.
        """
        if not os.path.exists(self.raw_video_path):
            logger.warning(
                f"[frame extracting][{self.saco_yt1b_id}] Raw video file not found at {self.raw_video_path}"
            )
            os.rmdir(self.JPEGImages_6fps_dir)
            return False

        if (
            os.path.exists(self.JPEGImages_6fps_dir)
            and len(os.listdir(self.JPEGImages_6fps_dir)) == self.expected_num_frames
        ):
            logger.info(
                f"[frame extracting][{self.saco_yt1b_id}] JPEGImages_6fps directory already exists at {self.JPEGImages_6fps_dir} and expected number of frames {self.expected_num_frames} matches"
            )
            return True

        # Clear the directory before extracting new frames
        for file in os.listdir(self.JPEGImages_6fps_dir):
            os.remove(os.path.join(self.JPEGImages_6fps_dir, file))

        args = [
            "-nostdin",
            "-y",
            # select video segment
            "-ss",
            str(self.start_time),
            "-to",
            str(self.end_time),
            "-i",
            self.raw_video_path,
            # set output video resolution to be 6fps and at most 1080p
            "-vf",
            "fps=6,scale=1080:-2",
            "-vsync",
            "0",  # passthrough mode - no frame duplication/dropping
            "-q:v",
            "2",  # high quality JPEG output
            "-start_number",
            "0",  # start frame numbering from 0
            self.JPEGImages_6fps_pattern,
        ]

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
            os.rmdir(self.JPEGImages_6fps_dir)
            return False

        if len(os.listdir(self.JPEGImages_6fps_dir)) != self.expected_num_frames:
            logger.warning(
                f"[frame extracting][{self.saco_yt1b_id}] Expected {self.expected_num_frames} frames but extracted {len(os.listdir(self.JPEGImages_6fps_dir))}"
            )
            # Clear the directory after failed extraction
            for file in os.listdir(self.JPEGImages_6fps_dir):
                os.remove(os.path.join(self.JPEGImages_6fps_dir, file))

            os.rmdir(self.JPEGImages_6fps_dir)
            return False

        logger.info(
            f"[frame extracting][{self.saco_yt1b_id}] Successfully extracted {self.expected_num_frames} frames to {self.JPEGImages_6fps_dir}"
        )
        return True


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
        "--yt1b_start_end_time_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--yt1b_frame_prep_log_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ffmpeg_timeout",
        type=str,
        default=7200,  # Use longer timeout in case of large videos processing timeout
    )
    parser.add_argument(
        "--sleep_interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max_sleep_interval",
        type=int,
        default=30,
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.yt1b_frame_prep_log_file,
        format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        filemode="w",
    )

    video_prep = YtVideoPrep(
        saco_yt1b_id=args.saco_yt1b_id,
        data_dir=args.data_dir,
        cookies_file=args.cookies_file,
        yt1b_start_end_time_file=args.yt1b_start_end_time_file,
        ffmpeg_timeout=args.ffmpeg_timeout,
        sleep_interval=args.sleep_interval,
        max_sleep_interval=args.max_sleep_interval,
    )

    status = video_prep.download_youtube_video()
    logger.info(f"[video download][{args.saco_yt1b_id}] download status {status}")

    status = video_prep.extract_frames_in_6fps_and_width_1080()
    logger.info(
        f"[frame extracting][{args.saco_yt1b_id}] frame extracting status {status}"
    )


if __name__ == "__main__":
    main()
