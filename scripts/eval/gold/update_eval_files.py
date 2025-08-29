import json
import os

from pathlib import Path

from tqdm import tqdm


gt_files_subset = {
    # Captioner
    "epsilon": [
        "gold_epsilon_merged_a_test.json",
        "gold_epsilon_merged_b_test.json",
        "gold_epsilon_merged_c_test.json",
    ],
    # FG Food
    "fg_food": [
        "gold_fg_food_merged_a_test.json",
        "gold_fg_food_merged_b_test.json",
        "gold_fg_food_merged_c_test.json",
    ],
    # FG Sports
    "fg_sports_equipment": [
        "gold_fg_sports_equipment_merged_a_test.json",
        "gold_fg_sports_equipment_merged_b_test.json",
        "gold_fg_sports_equipment_merged_c_test.json",
    ],
    # Attributes
    "attributes": [
        "gold_attributes_merged_a_test.json",
        "gold_attributes_merged_b_test.json",
        "gold_attributes_merged_c_test.json",
    ],
    # Crowded
    "crowded": [
        "gold_crowded_merged_a_test.json",
        "gold_crowded_merged_b_test.json",
        "gold_crowded_merged_c_test.json",
    ],
    # Wiki common
    "wiki_common": [
        "gold_wiki_common_merged_a_test.json",
        "gold_wiki_common_merged_b_test.json",
        "gold_wiki_common_merged_c_test.json",
    ],
}

annotators = ["a", "b", "c"]

gt_image_path = "/fsx-onevision/shoubhikdn/urls_stats/metaclip/gold_test/images/"
gt_annotations = "/fsx-onevision/shoubhikdn/release/gold_test_set/original/"
gt_annotations_updated = "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/"

for subset_name in gt_files_subset.keys():
    print("Processing subset: ", subset_name)
    gt_files = gt_files_subset[subset_name]
    for annotator, gt_rel_file in zip(annotators, gt_files):
        print("Processing annotator ", annotator)
        gt_file = os.path.join(gt_annotations, gt_rel_file)
        with open(gt_file, "r") as fr:
            data = json.load(fr)
        print("Before")
        print(len(data["images"]))
        print(len(data["annotations"]))

        new_data_images = []
        filtered_image_ids = set()
        for data_img in tqdm(data["images"]):
            rel_path = data_img["file_name"]
            image_path = f"{gt_image_path}/{rel_path}"
            if not os.path.exists(image_path):
                filtered_image_ids.add(data_img["id"])
                continue
            new_data_images.append(data_img)
        data["images"] = new_data_images

        new_data_annotations = []
        for data_annotation in tqdm(data["annotations"]):
            if data_annotation["image_id"] in filtered_image_ids:
                continue
            new_data_annotations.append(data_annotation)
        data["annotations"] = new_data_annotations

        print("After")
        print(len(data["images"]))
        print(len(data["annotations"]))

        new_gt_path = f"{gt_annotations_updated}/{gt_rel_file}"
        Path(new_gt_path).parent.mkdir(parents=True, exist_ok=True)
        with open(new_gt_path, "w") as fw:
            json.dump(data, fw)
