#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Copyright (c) Meta Platforms, Inc. and affiliates.


# In[3]:


import copy
import json
import os

import numpy as np

from pycocotools.coco import COCO
from sam3.train.eval.demo_eval import DemoEvaluator


# In[4]:


# Update to the directory where the GT annotation and PRED files exist
GT_DIR = (
    "/checkpoint/sam3/haithamkhedr/workspace/occhi/sam3_gold/"  # PUT YOUR PATH HERE
)
PRED_DIR = "/checkpoint/sam3/haithamkhedr/workspace/occhi/"  # PUT YOUR PATH HERE


# In[5]:


# Relative file names for GT and prediction files for 7 SA-Co/Gold subsets
saco_gold_gt_and_pred_files = {
    # MetaCLIP Captioner
    # "metaclip": {
    #     "gt_fname": [
    #         "gold_metaclip_merged_a_release_test.json",
    #         "gold_metaclip_merged_b_release_test.json",
    #         "gold_metaclip_merged_c_release_test.json",
    #     ],
    #     "pred_fname": "coco_predictions_gold_metaclip_captioner.json",
    # },
    # # SA-1B captioner
    # "sa1b": {
    #     "gt_fname": [
    #         "gold_sa1b_merged_a_release_test.json",
    #         "gold_sa1b_merged_b_release_test.json",
    #         "gold_sa1b_merged_c_release_test.json",
    #     ],
    #     "pred_fname": "coco_predictions_gold_sa1b_captioner.json",
    # },
    # # Crowded
    # "crowded": {
    #     "gt_fname": [
    #         "gold_crowded_merged_a_release_test.json",
    #         "gold_crowded_merged_b_release_test.json",
    #         "gold_crowded_merged_c_release_test.json",
    #     ],
    #     "pred_fname": "coco_predictions_gold_crowded.json",
    # },
    # # FG Food
    # "fg_food": {
    #     "gt_fname": [
    #         "gold_fg_food_merged_a_release_test.json",
    #         "gold_fg_food_merged_b_release_test.json",
    #         "gold_fg_food_merged_c_release_test.json",
    #     ],
    #     "pred_fname": "coco_predictions_gold_fg_food.json",
    # },
    # # FG Sports
    # "fg_sports_equipment": {
    #     "gt_fname": [
    #         "gold_fg_sports_equipment_merged_a_release_test.json",
    #         "gold_fg_sports_equipment_merged_b_release_test.json",
    #         "gold_fg_sports_equipment_merged_c_release_test.json",
    #     ],
    #     "pred_fname": "coco_predictions_gold_fg_sports_equipment.json",
    # },
    # Attributes
    "attributes": {
        "gt_fname": [
            "gold_attributes_merged_a_test.json",
            "gold_attributes_merged_b_test.json",
            "gold_attributes_merged_c_test.json",
        ],
        "pred_fname": "raw_result_file.json",
    },
    # # Wiki common
    # "wiki_common": {
    #     "gt_fname": [
    #         "gold_wiki_common_merged_a_release_test.json",
    #         "gold_wiki_common_merged_b_release_test.json",
    #         "gold_wiki_common_merged_c_release_test.json",
    #     ],
    #     "pred_fname": "coco_predictions_gold_wiki_common.json",
    # },
}


# In[6]:


import torch


class PostProcessorMerged:
    def __init__(self, predictions_path, img_ids):
        with open(predictions_path, "r") as f:
            self.data_anns = json.load(f)
        self.img_ids = img_ids

    def process_results(self):
        d = {
            "scores": np.array([]),
            "labels": np.array([], dtype=int),
            "boxes": [],
            # "masks_rle": [],
        }
        imgToAnns = {img_id: copy.deepcopy(d) for img_id in self.img_ids}
        for ann in self.data_anns:
            if ann["img_id"] not in self.img_ids:
                continue
            num_preds = len(ann["prediction"])
            imgToAnns[ann["img_id"]]["scores"] = np.append(
                imgToAnns[ann["img_id"]]["scores"], [ann.get("score", 1.0)] * num_preds
            )
            imgToAnns[ann["img_id"]]["labels"] = np.append(
                imgToAnns[ann["img_id"]]["labels"], [ann["category_id"]] * num_preds
            )
            imgToAnns[ann["img_id"]]["boxes"] += ann["prediction"]
            imgToAnns[ann["img_id"]]["boxes"] = torch.tensor(
                imgToAnns[ann["img_id"]]["boxes"]
            ).view(-1, 4)
            imgToAnns[ann["img_id"]]["boxes"][:, 0] /= ann["img_w"]
            imgToAnns[ann["img_id"]]["boxes"][:, 2] /= ann["img_w"]
            imgToAnns[ann["img_id"]]["boxes"][:, 1] /= ann["img_h"]
            imgToAnns[ann["img_id"]]["boxes"][:, 3] /= ann["img_h"]
        return imgToAnns


# <b>Run offline evaluation for all 7 SA-Co/Gold subsets</b>

# In[7]:


results = ""

for subset_name, values in saco_gold_gt_and_pred_files.items():
    print("Processing subset: ", subset_name)
    gt_fnames = values["gt_fname"]
    pred_fname = values["pred_fname"]
    gt_fname_full_paths = [os.path.join(GT_DIR, gt_fname) for gt_fname in gt_fnames]
    pred_fname_full_path = os.path.join(PRED_DIR, pred_fname)
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
    iou_type = "bbox"
    evaluator = DemoEvaluator(
        coco_gt=gt_fname_full_paths,
        iou_types=[iou_type],
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

    cgf1 = str(round(summary[f"coco_eval_{iou_type}_oracle_CGF1"] * 100, 2))
    cgf1m = str(round(summary[f"coco_eval_{iou_type}_oracle_CGF1_micro"] * 100, 2))
    il_mcc = str(round(summary[f"coco_eval_{iou_type}_oracle_IL_MCC"], 2))
    pmf1 = str(round(summary[f"coco_eval_{iou_type}_oracle_Macro_F1"] * 100, 2))
    pmf1m = str(
        round(summary[f"coco_eval_{iou_type}_oracle_positive_micro_F1"] * 100, 2)
    )
    demof1 = str(round(summary[f"coco_eval_{iou_type}_oracle_F1"] * 100, 2))
    final_str = f"{cgf1},{cgf1m},{il_mcc},{pmf1},{pmf1m},{demof1}"
    results += subset_name + ": " + final_str + "\n"


# In[8]:


print("Subset name, CG_F1, CG_F1_m, IL_MCC, pmF1, pmF1_m, demoF1")
print(results)


# In[83]:


import matplotlib.pyplot as plt


def show_box(boxes, ax, mode="xyxy", color="g"):

    for box in boxes:
        if mode == "xywh":
            x0, y0 = box[0], box[1]
            w, h = box[2], box[3]
        else:
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(
            plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2)
        )


# In[ ]:


gt_anns = coco.loadAnns(coco.getAnnIds(pred_id))
gt_bboxes = torch.tensor([ann["bbox"] for ann in gt_anns]).view(-1, 4)


# In[ ]:


pred_fname_full_path = (
    "/checkpoint/sam3/haithamkhedr/workspace/occhi/raw_result_file.json"
)
gt_fname_full_paths = "/checkpoint/sam3/haithamkhedr/workspace/occhi/sam3_gold/gold_attributes_merged_a_test.json"
coco = COCO(gt_fname_full_paths)
img_ids = list(
    sorted(
        [img["id"] for img in coco.dataset["images"] if img["is_instance_exhaustive"]]
    )
)
num_vis = 50
predictions_path = pred_fname_full_path
img_ids = img_ids
postprocessor = PostProcessorMerged(predictions_path=predictions_path, img_ids=img_ids)
predictions = postprocessor.process_results()
preds_ids = list(predictions.keys())
img_id_to_preds = {
    id: [ann for ann in postprocessor.data_anns if ann["img_id"] == id]
    for id in img_ids
}
img_id_to_img_path = {k: v[0]["image_path"] for k, v in img_id_to_preds.items()}
img_id_to_wh = {k: (v[0]["img_w"], v[0]["img_h"]) for k, v in img_id_to_preds.items()}
img_id_to_prompt = {
    k: v[0]["prompt"].split(":")[-1].strip() for k, v in img_id_to_preds.items()
}
fname_2_preds = {
    v[0]["image_path"]: img_id_to_preds[k] for k, v in img_id_to_preds.items()
}
start = 150
for img_idx in range(start, start + num_vis):
    pred_id = preds_ids[img_idx]
    image_path = img_id_to_img_path[pred_id]

    text = ["labels"]
    wh = img_id_to_wh[pred_id]
    preds = predictions[pred_id]
    boxes = preds["boxes"]
    boxes[:, 0] *= wh[0]
    boxes[:, 2] *= wh[0]
    boxes[:, 1] *= wh[1]
    boxes[:, 3] *= wh[1]
    gt_anns = coco.loadAnns(coco.getAnnIds(pred_id))
    gt_bboxes = torch.tensor([ann["bbox"] for ann in gt_anns]).view(-1, 4)
    gt_bboxes[:, 0] *= wh[0]
    gt_bboxes[:, 2] *= wh[0]
    gt_bboxes[:, 1] *= wh[1]
    gt_bboxes[:, 3] *= wh[1]
    if len(text) == 0:
        continue
    plt.figure(figsize=(12, 8))
    plt.imshow(plt.imread(image_path))
    ax = plt.gca()
    show_box(boxes, ax, mode="xyxy")
    # coco.showAnns(gt_anns, draw_bbox=False)
    show_box(gt_bboxes, ax, color="r", mode="xywh")
    plt.title(f"{img_id_to_prompt[pred_id]}")
    plt.axis("off")
    plt.show()


# In[31]:


for img_id, preds in predictions.items():
    if img_id != pred_id:
        continue
    else:
        print(img_id, pred_id)


# In[ ]:


# In[56]:


fname_2_preds[
    "/checkpoint/sam3/shared/data/metaclip_merged/1/100001/metaclip_1_100001_002fb0d001d06012870368b2.jpeg"
]


# In[49]:


img_id_to_ann[pred_id]


# In[51]:


for k, img2ann in img_id_to_ann.items():
    if len(img2ann) > 1:
        print(k, img2ann)
        break


# In[ ]:
