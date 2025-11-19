# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import requests
from fathomnet.api import images
from tqdm import tqdm


def download_imgs(args, image_uuids):
    flag = 0
    for uuid in tqdm(image_uuids, desc="Downloading images"):
        image = images.find_by_uuid(uuid)
        file_name = (
            Path(args.processed_images_folder)
            / f"{image.uuid}.{image.url.split('.')[-1]}"
        )
        if not file_name.exists():
            try:
                resp = requests.get(image.url, stream=True)
                resp.raise_for_status()
                with open(file_name, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024):
                        f.write(chunk)
                flag += 1
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {image.url}: {e}")
    print(f"Downloaded {flag} new images to {args.processed_images_folder}")


def main():
    parser = argparse.ArgumentParser(description="Download images from FathomNet")
    parser.add_argument("--processed_images_folder", help="Path to downloaded images")
    parser.add_argument(
        "--image-uuids",
        default="fathomnet_image_uuids.json",
        help="Path to JSON file containing image uuids to download",
    )
    parser.add_argument(
        "--num-procs", type=int, default=16, help="Number of parallel processes"
    )
    args = parser.parse_args()

    with open(args.image_uuids, "r") as f:
        all_uuids = json.load(f)

    Path(args.processed_images_folder).mkdir(parents=True, exist_ok=True)

    chunk_size = len(all_uuids) // args.num_procs
    chunks = [
        all_uuids[i : i + chunk_size] for i in range(0, len(all_uuids), chunk_size)
    ]

    with Pool(processes=args.num_procs) as pool:
        pool.starmap(download_imgs, [(args, chunk) for chunk in chunks])


if __name__ == "__main__":
    main()
