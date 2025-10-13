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
    )

    status = video_prep.download_youtube_video()
    logger.info(f"[video download][{saco_yt1b_id}] download status {status}")

    if status not in ["already exists", "success"]:
        print(f"Video download failed for {saco_yt1b_id}, skipping frame generation")
        return False

    video_prep.generate_frames_by_frame_matching()
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
    args = parser.parse_args()

    log_dir = os.path.dirname(args.yt1b_frame_prep_log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=args.yt1b_frame_prep_log_path,
        format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        filemode="w",
    )

    YT_DLP_WARNING_STR = """ ==========
        NOTICE!!
        This script uses yt-dlp to download youtube videos.
        See the youtube account banning risk in https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies
        ==========
        """

    print(YT_DLP_WARNING_STR)
    print(logger.info(YT_DLP_WARNING_STR))

    args = parser.parse_args()

    with open(args.id_map_file, "r") as f:
        id_map_df = pd.read_json(f)

    saco_yt1b_ids = id_map_df.saco_yt1b_id.unique()

    cpu_count = mp.cpu_count()
    print(f"Starting with {cpu_count} processes")
    logger.info(f"Starting with {cpu_count} processes")

    with mp.Pool(cpu_count) as p:
        download_func = partial(download_and_extract_frames, args=args)
        list(tqdm(p.imap(download_func, saco_yt1b_ids), total=len(saco_yt1b_ids)))

    done_str = f""" ==========
        All DONE!!
        Download, frame extraction, and frame matching is all done! YT1B frames are not ready to use in {args.data_dir}/JPEGImages_6fps
        Check video frame preparing log at {args.yt1b_frame_prep_log_path}
        Some videos might not be available any more which will affect the eval reproducibility
        ==========
    """
    print(done_str)
    logger.info(done_str)


if __name__ == "__main__":
    main()
