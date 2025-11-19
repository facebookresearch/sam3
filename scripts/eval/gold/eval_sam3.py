# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Script to run the evaluator offline given the GTs for SAC-Gold test set and SAM3 model prediction files.
It reports CGF1, IL_MCC, PM_F1 metrics for each subset of SAC-Gold test set.

Usage: python eval_sam3.py --gt-folder <folder_with_gts> --pred-folder <folder_with_predictions>
"""

import argparse
import os

from sam3.eval.cgf1_eval import CGF1Evaluator

# Relative file names for GT files for 7 SA-Co/Gold subsets

saco_gold_gts = {
    # MetaCLIP Captioner
    "metaclip_nps": [
        "gold_metaclip_merged_a_release_test.json",
        "gold_metaclip_merged_b_release_test.json",
        "gold_metaclip_merged_c_release_test.json",
    ],
    # SA-1B captioner
    "sa1b_nps": [
        "gold_sa1b_merged_a_release_test.json",
        "gold_sa1b_merged_b_release_test.json",
        "gold_sa1b_merged_c_release_test.json",
    ],
    # Crowded
    "crowded": [
        "gold_crowded_merged_a_release_test.json",
        "gold_crowded_merged_b_release_test.json",
        "gold_crowded_merged_c_release_test.json",
    ],
    # FG Food
    "fg_food": [
        "gold_fg_food_merged_a_release_test.json",
        "gold_fg_food_merged_b_release_test.json",
        "gold_fg_food_merged_c_release_test.json",
    ],
    # FG Sports
    "fg_sports_equipment": [
        "gold_fg_sports_equipment_merged_a_release_test.json",
        "gold_fg_sports_equipment_merged_b_release_test.json",
        "gold_fg_sports_equipment_merged_c_release_test.json",
    ],
    # Attributes
    "attributes": [
        "gold_attributes_merged_a_release_test.json",
        "gold_attributes_merged_b_release_test.json",
        "gold_attributes_merged_c_release_test.json",
    ],
    # Wiki common
    "wiki_common": [
        "gold_wiki_common_merged_a_release_test.json",
        "gold_wiki_common_merged_b_release_test.json",
        "gold_wiki_common_merged_c_release_test.json",
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--gt-folder",
        type=str,
        help="Path to the folder containing the ground truth json files.",
    )
    parser.add_argument(
        "-p",
        "--pred-folder",
        type=str,
        help="Path to the folder containing the predictions json files.",
    )
    args = parser.parse_args()

    results = ""

    for subset_name, gts in saco_gold_gts.items():
        print("Processing subset: ", subset_name)
        gt_paths = [os.path.join(args.gt_folder, gt) for gt in gts]
        evaluator = CGF1Evaluator(
            gt_path=gt_paths, verbose=True, iou_type="segm"
        )  # change to bbox if you want detection performance

        pred_path = os.path.join(
            args.pred_folder,
            f"gold_{subset_name}/dumps/gold_{subset_name}/coco_predictions_segm.json",
        )
        summary = evaluator.evaluate(pred_path)

        cgf1 = str(round(summary["cgF1_eval_segm_cgF1"] * 100, 2))
        il_mcc = str(round(summary["cgF1_eval_segm_IL_MCC"], 2))
        pmf1 = str(round(summary["cgF1_eval_segm_positive_micro_F1"] * 100, 2))
        final_str = f"{cgf1},{il_mcc},{pmf1}"
        results += subset_name + ": " + final_str + "\n"

    print("Subset name, CG_F1, IL_MCC, pmF1")
    print(results)


if __name__ == "__main__":
    main()
