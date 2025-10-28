# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

"""Script to run the evaluator offline given the GTs for SAC-Gold test set and SAM3 model prediction files.
It reports CGF1, IL_MCC, PM_F1, demo F1, J&F metrics for each subset of SAC-Gold test set.

Usage: python eval_sam3.py --gt-folder <folder_with_gts> --pred-folder <folder_with_predictions>
"""

import copy
import json
import os

import numpy as np

from pycocotools.coco import COCO
from sam3.train.eval.demo_eval import DemoEvaluator

all_files = {
    # MetaCLIP Captioner
    "metaclip": {
        "gt_fname": [
            "gold_metaclip_merged_a_release_test.json",
            "gold_metaclip_merged_b_release_test.json",
            "gold_metaclip_merged_c_release_test.json",
        ],
        "pred_fname": "coco_predictions_gold_metaclip_captioner.json",
    },
    # SA-1B captioner
    "sa1b": {
        "gt_fname": [
            "gold_sa1b_merged_a_release_test.json",
            "gold_sa1b_merged_b_release_test.json",
            "gold_sa1b_merged_c_release_test.json",
        ],
        "pred_fname": "coco_predictions_gold_sa1b_captioner.json",
    },
    # Crowded
    "crowded": {
        "gt_fname": [
            "gold_crowded_merged_a_release_test.json",
            "gold_crowded_merged_b_release_test.json",
            "gold_crowded_merged_c_release_test.json",
        ],
        "pred_fname": "coco_predictions_gold_crowded.json",
    },
    # FG Food
    "fg_food": {
        "gt_fname": [
            "gold_fg_food_merged_a_release_test.json",
            "gold_fg_food_merged_b_release_test.json",
            "gold_fg_food_merged_c_release_test.json",
        ],
        "pred_fname": "coco_predictions_gold_fg_food.json",
    },
    # FG Sports
    "fg_sports_equipment": {
        "gt_fname": [
            "gold_fg_sports_equipment_merged_a_release_test.json",
            "gold_fg_sports_equipment_merged_b_release_test.json",
            "gold_fg_sports_equipment_merged_c_release_test.json",
        ],
        "pred_fname": "coco_predictions_gold_fg_sports_equipment.json",
    },
    # Attributes
    "attributes": {
        "gt_fname": [
            "gold_attributes_merged_a_release_test.json",
            "gold_attributes_merged_b_release_test.json",
            "gold_attributes_merged_c_release_test.json",
        ],
        "pred_fname": "coco_predictions_gold_attr.json",
    },
    # Wiki common
    "wiki_common": {
        "gt_fname": [
            "gold_wiki_common_merged_a_release_test.json",
            "gold_wiki_common_merged_b_release_test.json",
            "gold_wiki_common_merged_c_release_test.json",
        ],
        "pred_fname": "coco_predictions_gold_wiki_common.json",
    },
}


class PostProcessorMerged:
    def __init__(self, predictions_path, img_ids):
        with open(predictions_path, "r") as f:
            self.data_anns = json.load(f)
        self.img_ids = img_ids

    def process_results(self):
        d = {
            "scores": np.array([]),
            "labels": np.array([], dtype=int),
            "masks_rle": [],
        }
        imgToAnns = {img_id: copy.deepcopy(d) for img_id in self.img_ids}
        for ann in self.data_anns:
            if ann["image_id"] not in self.img_ids:
                continue
            imgToAnns[ann["image_id"]]["scores"] = np.append(
                imgToAnns[ann["image_id"]]["scores"], ann["score"]
            )
            imgToAnns[ann["image_id"]]["labels"] = np.append(
                imgToAnns[ann["image_id"]]["labels"], ann["category_id"]
            )
            imgToAnns[ann["image_id"]]["masks_rle"].append(ann["segmentation"])

        return imgToAnns


def main():
    """
    Script that validates image urls

    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--gt-folder",
        type=str,
        default="/fsx-onevision/shoubhikdn/release/saco_gold/annotations_cleaned_v2/",
    )
    parser.add_argument(
        "-p",
        "--pred-folder",
        type=str,
        default="/fsx-onevision/shoubhikdn/release/sam3_predictions/arxiv/",
    )
    args = parser.parse_args()

    results = ""

    for subset_name, values in all_files.items():
        print("Processing subset: ", subset_name)
        gt_fnames = values["gt_fname"]
        pred_fname = values["pred_fname"]
        gt_fname_full_paths = [
            os.path.join(args.gt_folder, gt_fname) for gt_fname in gt_fnames
        ]
        pred_fname_full_path = os.path.join(args.pred_folder, pred_fname)
        coco = COCO(gt_fname_full_paths[0])

        img_ids = list(
            sorted(
                [
                    img["id"]
                    for img in coco.dataset["images"]
                    if img["is_instance_exhaustive"]
                ]
            )
        )

        evaluator = DemoEvaluator(
            coco_gt=gt_fname_full_paths,
            iou_types=["segm"],
            threshold=0.5,
            dump_dir=None,
            postprocessor=PostProcessorMerged(
                predictions_path=pred_fname_full_path, img_ids=img_ids
            ),
            average_by_rarity=False,
            gather_pred_via_filesys=False,
            exhaustive_only=True,
        )
        evaluator.update()
        summary = evaluator.compute_synced()

        cgf1 = str(round(summary["coco_eval_masks_oracle_CGF1"] * 100, 2))
        cgf1m = str(round(summary["coco_eval_masks_oracle_CGF1_micro"] * 100, 2))
        il_mcc = str(round(summary["coco_eval_masks_oracle_IL_MCC"], 2))
        pmf1 = str(round(summary["coco_eval_masks_oracle_Macro_F1"] * 100, 2))
        pmf1m = str(round(summary["coco_eval_masks_oracle_positive_micro_F1"] * 100, 2))
        demof1 = str(round(summary["coco_eval_masks_oracle_F1"] * 100, 2))
        final_str = f"{cgf1},{cgf1m},{il_mcc},{pmf1},{pmf1m},{demof1}"
        results += subset_name + ": " + final_str + "\n"

    print("Subset name, CG_F1, CG_F1_m, IL_MCC, pmF1, pmF1_m, demoF1")
    print(results)


if __name__ == "__main__":
    main()
