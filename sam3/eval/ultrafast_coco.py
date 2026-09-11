# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Optional native COCO evaluator construction without global import replacement."""

import copy
from collections import defaultdict


def ultrafast_tools():
    try:
        from ultrafast_pycocotools import COCO, COCOeval
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'Install the optional backend with pip install "ultrafast-pycocotools>=0.1.11,<0.2".'
        ) from exc
    return COCO, COCOeval


def create_offline_evaluator(coco_gt, coco_dt, iou_type, positive_split):
    coco_class, evaluator = ultrafast_tools()
    gt_dataset = copy.deepcopy(coco_gt.dataset)
    dt_dataset = copy.deepcopy(coco_dt.dataset)
    if positive_split:
        # Native evaluate() does not call COCOevalCustom._prepare(). Apply its
        # positive-split rule to the input annotations before native matching.
        positive_categories = defaultdict(set)
        for annotation in gt_dataset["annotations"]:
            positive_categories[annotation["image_id"]].add(annotation["category_id"])
        dt_dataset["annotations"] = [
            annotation
            for annotation in dt_dataset["annotations"]
            if annotation["category_id"] in positive_categories[annotation["image_id"]]
        ]
    return evaluator(
        coco_class(gt_dataset, verbose=False),
        coco_class(dt_dataset, verbose=False),
        iou_type,
    )
