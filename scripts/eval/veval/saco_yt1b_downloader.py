import argparse

import multiprocessing as mp
from functools import partial

import pandas as pd
from saco_yt1b_frame_prep_util import YtVideoPrep
from tqdm import tqdm


def download_and_extract_frames(saco_yt1b_id, args):
    video_prep = YtVideoPrep(
        saco_yt1b_id=saco_yt1b_id,
        data_dir=args.data_dir,
        cookies_file=args.cookies_file,
        id_and_frame_map_path=args.id_map_file,
    )
    download_status = video_prep.download_youtube_video()

    download_status_str = f"{saco_yt1b_id}\t{download_status}\n"
    if download_status not in ["already exists", "success"]:
        print(f"Video download failed for {saco_yt1b_id}, skipping frame generation")
        return download_status_str

    # Set a large timeout number to avoid sometimes timeout issues
    video_prep.generate_all_raw_frames(timeout_seconds=7200)
    video_prep.generate_frames_by_frame_matching()
    return download_status_str


def main():
    print(
        """ ==========
        NOTICE!!
        This script uses yt-dlp to download youtube videos.
        See the youtube account banning risk in https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies
        ==========
        """
    )
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
        "--download_result",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    with open(args.id_map_file, "r") as f:
        id_map_df = pd.read_json(f)

    saco_yt1b_ids = id_map_df.saco_yt1b_id.unique()

    with open(args.download_result, "w") as f:
        print(f"Writing yt1b video downloading results to: {args.download_result}")

        cpu_count = mp.cpu_count()
        print(f"Starting with {cpu_count} processes")
        with mp.Pool(cpu_count) as p:
            download_func = partial(download_and_extract_frames, args=args)
            download_statuses = list(
                tqdm(p.imap(download_func, saco_yt1b_ids), total=len(saco_yt1b_ids))
            )

        f.write("\n".join(download_statuses))

    print(f""" ==========
        All DONE!!
        Download, frame extraction, and frame matching is all done! YT1B frames are not ready to use in {args.data_dir}/JPEGImages_6fps
        Check video downloading status at {args.download_result}
        Some videos might not be available any more which will affect the eval reproducibility
        ==========
    """)

if __name__ == "__main__":
    main()
