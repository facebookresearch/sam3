# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import utils
from PIL import Image
from tqdm import tqdm

METADATA_FILE = "published_images.csv"
METADATA_URL = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/refs/heads/main/data"  # data/published_iamges.csv from https://github.com/NationalGalleryOfArt/opendata/tree/main
IMG_URL = "https://api.nga.gov/iiif/%s/full/%s/0/default.jpg"
METADATA_FOLDER = "metadata"
EXTENSION = ".jpg"


def download_metadata(annotation_folder):
    output_folder = annotation_folder / METADATA_FOLDER
    output_folder.mkdir(exist_ok=True)
    url = f"{METADATA_URL}/{METADATA_FILE}"
    print(url)
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_folder / METADATA_FILE, "wb") as f:
            f.write(response.content)


def download_url(row):
    if np.isnan(row.maxpixels) or (
        row.maxpixels > row.width and row.maxpixels > row.height
    ):
        url = IMG_URL % (row.uuid, "full")
    else:
        url = IMG_URL % (row.uuid, f"!{row.maxpixels},{row.maxpixels}")
    return url


def download_item(item, output_folder):
    uuid, url = item
    try:
        if (output_folder / f"{uuid}{EXTENSION}").exists():
            print("skipping", uuid, "already downloaded")
            return
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_folder / f"{uuid}{EXTENSION}", "wb") as f:
                f.write(response.content)
    except:
        print("errored", item)
        return


def remove_non_compliant_image(item, output_folder):
    uuid, max_pixels = item
    if np.isnan(max_pixels):
        return
    if not (output_folder / f"{uuid}{EXTENSION}").exists():
        return
    img = Image.open(output_folder / f"{uuid}{EXTENSION}")
    if img.width > max_pixels or img.height > max_pixels:
        os.remove(output_folder / f"{uuid}{EXTENSION}")  # delete image
        return uuid


def reshape_image(rel_path, filename_size_map, output_folder):
    w, h = filename_size_map[rel_path]
    path = output_folder / f"{rel_path}"
    img = Image.open(path)
    if img.width != w or img.height != h:
        new_size = (w, h)
        resized_img = img.resize(new_size)
        resized_img.save(path)


def main(args, workers=20):
    raw_folder = Path(args.raw_images_folder)
    processed_folder = Path(args.processed_images_folder)
    utils.setup(raw_folder)
    utils.setup(processed_folder)
    uuids = utils.get_image_ids(args.annotation_file)
    filename_size_map = utils.get_filename_size_map(args.annotation_file)
    if not ((raw_folder / METADATA_FOLDER) / METADATA_FILE).exists():
        download_metadata(raw_folder)

    metadata = pd.read_csv((raw_folder / METADATA_FOLDER) / METADATA_FILE)
    metadata["download_url"] = metadata.apply(download_url, axis=1)
    available_uuids = list(uuids.intersection(set(metadata["uuid"].tolist())))
    print(len(available_uuids), "available for download out of", len(uuids), "target")
    url_data = list(
        metadata.set_index("uuid")
        .loc[available_uuids]
        .to_dict()["download_url"]
        .items()
    )

    download_single = partial(download_item, output_folder=(processed_folder))

    print("Preparing to download", len(url_data), "items")
    with Pool(20) as p:
        for _ in tqdm(p.imap(download_single, url_data), total=len(url_data)):
            continue
    check_img_size = partial(
        remove_non_compliant_image, output_folder=(processed_folder)
    )
    max_pixels_dict_all = metadata.set_index("uuid").to_dict()["maxpixels"]
    max_pixels_dict = {item[0]: max_pixels_dict_all[item[0]] for item in url_data}
    print("Checking all images within size constraints")
    non_compliant = set()
    with Pool(20) as p:
        for each in tqdm(
            p.imap(check_img_size, max_pixels_dict.items()), total=len(max_pixels_dict)
        ):
            if each is not None:
                non_compliant.add(each)
    print(len(non_compliant), "not compliant size, removed")

    reshape_single = partial(
        reshape_image,
        filename_size_map=(filename_size_map),
        output_folder=(processed_folder),
    )
    rel_paths = os.listdir(args.processed_images_folder)
    print("Preparing to reshape", len(rel_paths), "items")
    with Pool(20) as p:
        for _ in tqdm(p.imap(reshape_single, rel_paths), total=len(rel_paths)):
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file", help="Path to annotation file")
    parser.add_argument("--raw_images_folder", help="Path to downloaded images")
    parser.add_argument("--processed_images_folder", help="Path to processed images")
    args = parser.parse_args()
    main(args)
