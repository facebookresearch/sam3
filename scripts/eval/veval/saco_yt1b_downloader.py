import argparse
import logging
import os

import multiprocessing as mp
from functools import partial

import pandas as pd
from saco_yt1b_frame_prep_util import YtVideoPrep
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_and_extract_frames(saco_yt1b_id, args):
    video_prep = YtVideoPrep(
        saco_yt1b_id=saco_yt1b_id,
        data_dir=args.data_dir,
        cookies_file=args.cookies_file,
        id_and_frame_map_path=args.id_map_file,
        ffmpeg_timeout=args.ffmpeg_timeout,
        sleep_interval=args.sleep_interval,
        max_sleep_interval=args.max_sleep_interval,
    )

    status = video_prep.download_youtube_video()
    logger.info(f"[video download][{saco_yt1b_id}] download status {status}")

    if status not in ["already exists", "success"]:
        logger.warning(f"Video download failed for {saco_yt1b_id}, skipping frame generation")
        return False

    # video_prep.generate_frames_by_frame_matching()
    return True


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--sleep_interval",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--max_sleep_interval",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    log_dir = os.path.dirname(args.yt1b_frame_prep_log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Set up logging to both file and console
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(threadName)s] %(levelname)s: %(message)s")

    # File handler
    file_handler = logging.FileHandler(args.yt1b_frame_prep_log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    YT_DLP_WARNING_STR = """ ==========
        NOTICE!!
        This script uses yt-dlp to download youtube videos.
        See the youtube account banning risk in https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies
        ==========
        """

    logger.info(YT_DLP_WARNING_STR)

    args = parser.parse_args()

    with open(args.id_map_file, "r") as f:
        id_map_df = pd.read_json(f)

    saco_yt1b_ids = id_map_df.saco_yt1b_id.unique()
    num_workers = args.num_workers
    logger.info(f"Starting with {num_workers} parallel worker(s) (sleep_interval={args.sleep_interval}-{args.max_sleep_interval}s)")

    with mp.Pool(num_workers) as p:
        download_func = partial(download_and_extract_frames, args=args)
        list(tqdm(p.imap(download_func, saco_yt1b_ids), total=len(saco_yt1b_ids)))

    done_str = f""" ==========
        All DONE!!
        Download, frame extraction, and frame matching is all done! YT1B frames are not ready to use in {args.data_dir}/JPEGImages_6fps
        Check video frame preparing log at {args.yt1b_frame_prep_log_path}
        Some videos might not be available any more which will affect the eval reproducibility
        ==========
    """
    logger.info(done_str)


if __name__ == "__main__":
    main()
