import argparse

import pandas as pd

from saco_yt1b_frame_prep_util import YtVideoPrep
from tqdm import tqdm


def main():
    print(
        "This script uses yt-dlp to download youtube videos. See the youtube account banning risk in https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies"
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
        for saco_yt1b_id in tqdm(saco_yt1b_ids):
            video_prep = YtVideoPrep(
                saco_yt1b_id=saco_yt1b_id,
                data_dir=args.data_dir,
                cookies_file=args.cookies_file,
                id_and_frame_map_path=args.id_map_file,
            )
            download_status = video_prep.download_youtube_video()

            f.write(f"{saco_yt1b_id}\t{download_status}\n")

            if download_status not in ["already exists", "success"]:
                print(
                    f"Video download failed for {saco_yt1b_id}, skipping frame generation"
                )
                continue

            # Use longer timeout for large videos (2 hours)
            video_prep.generate_all_raw_frames(timeout_seconds=7200)
            video_prep.generate_frames_by_frame_matching()


if __name__ == "__main__":
    main()
