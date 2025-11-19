# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
import argparse
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import utils
from tqdm import tqdm


def main(args, n_workers=20):
    raw_folder = Path(args.raw_images_folder)
    processed_folder = Path(args.processed_images_folder)
    utils.setup(processed_folder)
    img_ids = utils.get_image_ids(args.annotation_file)
    if args.dataset_name == "geode":
        metadata = pd.read_csv(raw_folder / "index.csv")
        metadata["flat_filepath"] = metadata.file_path.apply(
            lambda x: x.replace("/", "_")
        )
        metadata["original_absolute_path"] = metadata.file_path.apply(
            lambda x: str((raw_folder / "images") / x)
        )
        metadata["new_absolute_path"] = metadata.flat_filepath.apply(
            lambda x: str(processed_folder / x)
        )
        metadata["filestem"] = metadata.new_absolute_path.apply(lambda x: Path(x).stem)
        img_id_mapping = metadata.set_index("filestem").to_dict()
        # print(img_id_mapping.keys())
        paths = [
            (
                img_id_mapping["original_absolute_path"][each],
                img_id_mapping["new_absolute_path"][each],
            )
            for each in img_ids
        ]
    elif args.dataset_name == "bdd100k":
        bdd_subfolder = "100k/train"
        img_filenames = utils.get_filenames(args.annotation_file)
        raw_folder_bdd_images = raw_folder / bdd_subfolder
        paths = [
            (raw_folder_bdd_images / each, processed_folder / each)
            for each in img_filenames
        ]
    elif args.dataset_name == "food_rec":
        food_subfolder = "public_validation_set_2.0/images"
        img_filenames = utils.get_filenames(args.annotation_file)
        raw_folder_food_images = raw_folder / food_subfolder
        paths = [
            (
                raw_folder_food_images
                / f'{Path(each).stem.split("_")[-1]}{Path(each).suffix}',
                processed_folder / each,
            )
            for each in img_filenames
        ]
    print("Preparing to copy and flatten filename for", len(paths), "images")
    with Pool(20) as p:
        for _ in tqdm(p.imap(utils.copy_file, paths), total=len(paths)):
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file", help="Path to annotation file")
    parser.add_argument("--raw_images_folder", help="Path to downloaded images")
    parser.add_argument("--processed_images_folder", help="Path to processed images")
    parser.add_argument("--dataset_name", help="Path to processed images")
    args = parser.parse_args()
    main(args)
