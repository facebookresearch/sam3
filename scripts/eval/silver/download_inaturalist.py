# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from tqdm import tqdm


def download_archive(url, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / url.split("/")[-1]
    if not archive_path.exists():
        print(f"Downloading archive to {archive_path}...")
        result = subprocess.run(["wget", "-O", str(archive_path), url])
        if result.returncode != 0:
            print("Download failed.")
            sys.exit(1)
    else:
        print(f"Archive already exists at {archive_path}")
    return archive_path


def extract_archive(archive_path, dest_dir):
    print(f"Extracting {archive_path} to {dest_dir}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)
    print("Extraction complete.")


def copy_images(subset_json, untar_dir, output_dir):
    with open(subset_json, "r") as f:
        image_dict = json.load(f)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for target_name, rel_path in tqdm(image_dict.items(), "Copying image subset"):
        src = Path(untar_dir) / rel_path
        dst = output_dir / target_name
        if not src.exists():
            print(f"Warning: Source image {src} does not exist, skipping.")
            continue
        shutil.copy2(src, dst)
    print(f"Copied {len(image_dict)} images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download, extract, and copy subset of iNaturalist images from archive."
    )
    parser.add_argument(
        "--raw_images_folder", help="Path to downloaded and extract the archive"
    )
    parser.add_argument("--processed_images_folder", help="Path to processed images")
    parser.add_argument(
        "--subset-json",
        default="inaturalist_image_subset.json",
        help="Path to iNaturalist images subset",
    )
    parser.add_argument(
        "--archive-url",
        default="https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz",
        help="URL of the archive to download",
    )
    args = parser.parse_args()

    dest_dir = Path(args.raw_images_folder)
    images_dir = Path(args.processed_images_folder)

    archive_path = download_archive(args.archive_url, dest_dir)
    extract_archive(archive_path, dest_dir)

    untar_dir = dest_dir / "train_val_images"
    copy_images(args.subset_json, untar_dir, images_dir)


if __name__ == "__main__":
    main()
