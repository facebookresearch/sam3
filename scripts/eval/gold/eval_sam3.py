import copy
import json

import numpy as np
from onevision.data.datasets.demo_eval import DemoEvaluator

from pycocotools.coco import COCO

all_files = {
    # Captioner: merged (55.6,0.77,72.1,53.1) --> (55.5,0.773,71.9,53.0)
    "epsilon": {
        "gt_merged_fname": [
            "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/gold_epsilon_merged_a_test.json",
            "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/gold_epsilon_merged_b_test.json",
            "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/gold_epsilon_merged_c_test.json",
        ],
        "out_merged_fname": "/fsx-onevision/shoubhikdn/test/prod_v14_eval/gt_vis/gold_metaclip_captioner_merged_v_2/prod_v14_rc3/predictions/coco_predictions_merged_dedup.json",
    },
    # SA-1B captioner (55.7,0.81,69.1,56.2)
    "sa1b": {
        "gt_merged_fname": [
            "/fsx-onevision/shoubhikdn/release/gold_test_set/original/gold_sa1b_merged_a_test.json",
            "/fsx-onevision/shoubhikdn/release/gold_test_set/original/gold_sa1b_merged_b_test.json",
            "/fsx-onevision/shoubhikdn/release/gold_test_set/original/gold_sa1b_merged_c_test.json",
        ],
        "out_merged_fname": "/fsx-onevision/shoubhikdn/test/prod_v14_eval/gt_vis/gold_sa1b_captioner_merged_v_2/prod_v14_rc3/predictions/coco_predictions_merged_dedup.json",
    },
    # Crowded (53.3,0.83,64.4,61.7)-->(53.3,0.827,64.5,61.7)
    "crowded": {
        "gt_merged_fname": [
            "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/gold_crowded_merged_a_test.json",
            "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/gold_crowded_merged_b_test.json",
            "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/gold_crowded_merged_c_test.json",
        ],
        "out_merged_fname": "/fsx-onevision/shoubhikdn/test/prod_v14_eval/gt_vis/gold_crowded_merged_v2/prod_v14_rc3/predictions/coco_predictions_merged_dedup.json",
    },
    # FG Food (65.5,0.83,78.8,66.8)-->(65.3,0.829,78.7,66.1)
    "fg_food": {
        "gt_merged_fname": [
            "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/gold_fg_food_merged_a_test.json",
            "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/gold_fg_food_merged_b_test.json",
            "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/gold_fg_food_merged_c_test.json",
        ],
        "out_merged_fname": "/fsx-onevision/shoubhikdn/test/prod_v14_eval/gt_vis/gold_fg_food_merged_v2/prod_v14_rc3/predictions/coco_predictions_merged_dedup.json",
    },
    # FG Sports (69.6,0.89,78.0,65.2) --> (69.8,0.894,78.0,65.2)
    "fg_sports_equipment": {
        "gt_merged_fname": [
            "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/gold_fg_sports_equipment_merged_a_test.json",
            "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/gold_fg_sports_equipment_merged_b_test.json",
            "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/gold_fg_sports_equipment_merged_c_test.json",
        ],
        "out_merged_fname": "/fsx-onevision/shoubhikdn/test/prod_v14_eval/gt_vis/gold_fg_sports_equipment_merged_v2/prod_v14_rc3/predictions/coco_predictions_merged_dedup.json",
    },
    # Attributes (56.6, 0.67, 84.1, 57.8) --> (56.8,0.674,84.3,58.0)
    "attributes": {
        "gt_merged_fname": [
            "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/gold_attributes_merged_a_test.json",
            "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/gold_attributes_merged_b_test.json",
            "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/gold_attributes_merged_c_test.json",
        ],
        "out_merged_fname": "/fsx-onevision/shoubhikdn/test/prod_v14_eval/gt_vis/gold_attr_merged_v2/prod_v14_rc3/predictions/coco_predictions_merged_dedup.json",
    },
    # Wiki common (46.4,0.60,77.1,51.6) --> (46.5,0.603,77.1,51.7)
    "wiki_common": {
        "gt_merged_fname": [
            "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/gold_wiki_common_merged_a_test.json",
            "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/gold_wiki_common_merged_b_test.json",
            "/fsx-onevision/shoubhikdn/release/gold_test_set/updated/gold_wiki_common_merged_c_test.json",
        ],
        "out_merged_fname": "/fsx-onevision/shoubhikdn/test/prod_v14_eval/gt_vis/gold_wiki_common_merged_v2/prod_v14_rc3/predictions/coco_predictions_merged_dedup.json",
    },
}


class PostProcessorMerged:
    def __init__(self, predictions_path: str):
        with open(predictions_path, "r") as f:
            self.data_anns = json.load(f)

    def process_results(self):
        d = {
            "scores": np.array([]),
            "labels": np.array([], dtype=int),
            "masks_rle": [],
            "boundaries": [],
            "dilated_boundaries": [],
        }
        imgToAnns = {img_id: copy.deepcopy(d) for img_id in img_ids}
        for ann in self.data_anns:
            if ann["image_id"] not in img_ids:
                continue
            imgToAnns[ann["image_id"]]["scores"] = np.append(
                imgToAnns[ann["image_id"]]["scores"], ann["score"]
            )
            imgToAnns[ann["image_id"]]["labels"] = np.append(
                imgToAnns[ann["image_id"]]["labels"], ann["category_id"]
            )
            imgToAnns[ann["image_id"]]["masks_rle"].append(ann["segmentation"])
            imgToAnns[ann["image_id"]]["boundaries"].append(ann["boundary"])
            imgToAnns[ann["image_id"]]["dilated_boundaries"].append(
                ann["dilated_boundary"]
            )

        return imgToAnns


results = {
    "epsilon": "",
    "epsilon_mosaic": ",,,,",
    "sa1b": "",
    "crowded": "",
    "fg_food": "",
    "fg_food_mosaic": ",,,,",
    "fg_sports_equipment": "",
    "fg_sports_equipment_mosaic": ",,,,",
    "attributes": "",
    "attributes_mosaic": ",,,,",
    "wiki_common": "",
    "wiki_common_mosaic": ",,,,",
}

for subset_name, values in all_files.items():
    print("Processing subset: ", subset_name)
    gt_merged_fname = values["gt_merged_fname"]
    out_merged_fname = values["out_merged_fname"]
    coco = COCO(gt_merged_fname[0])

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
        coco_gt=gt_merged_fname,
        iou_types=["segm"],
        threshold=0.5,
        dump_dir=None,
        postprocessor=PostProcessorMerged(predictions_path=out_merged_fname),
        average_by_rarity=False,
        gather_pred_via_filesys=False,
        exhaustive_only=True,
        compute_JnF=True,
    )
    evaluator.update()
    summary = evaluator.compute_synced()

    cgf1 = str(round(summary["coco_eval_masks_oracle_CGF1"] * 100, 2))
    il_mcc = str(round(summary["coco_eval_masks_oracle_IL_MCC"], 2))
    pmf1 = str(round(summary["coco_eval_masks_oracle_Macro_F1"] * 100, 2))
    demof1 = str(round(summary["coco_eval_masks_oracle_F1"] * 100, 2))
    jf = str(round(summary["coco_eval_masks_oracle_J&F"] * 100, 2))
    final_str = f"{cgf1},{il_mcc},{pmf1},{demof1},{jf}"
    print(final_str)
    results[subset_name] = final_str

print(",".join(results.values()))
