# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

"""
This evaluator is meant for regular COCO mAP evaluation, for example on the COCO val set.
For LVIS, fixed AP use the LVIS evaluator, and for Phrase AP, use the other coco_eval.py file.

For Category mAP, we need the model to make predictions for all the categories on every single image.
In general, since the number of classes can be big, and the API model makes predictions individually for each pair (image, class),
we may need to split the inference process for a given image in several chunks. The other coco_eval.py evaluates the predictions "on the fly",
and always assume that we have all the predictions for a given image predicted at once, which is not compatible with chunking.
On the other hand, this evaluator is "offline", meaning that it will first collect all the predictions for a given image, and then evaluate them.
"""

import copy
import gc
import heapq
import json
import logging
import os
from collections import defaultdict

from pathlib import Path
from typing import Any

import numpy as np

# import onevision_cpp_ops as _CPP
import pycocotools.mask as mask_util
import torch

from iopath.common.file_io import g_pathmgr
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# from .dist import all_gather, gather_to_rank_0_via_filesys, is_main_process
from sam3.train.utils.distributed import (
    all_gather,
    gather_to_rank_0_via_filesys,
    is_main_process,
)

try:
    from tidecv import datasets, TIDE

    HAS_TIDE = True
except ImportError:
    HAS_TIDE = False
    print("WARNING: TIDE not installed. Detailed analysis will not be available.")


# the COCO detection metrics (https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L460-L471)
COCO_METRICS = [
    "AP",
    "AP_50",
    "AP_75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_maxDets@1",
    "AR_maxDets@10",
    "AR_maxDets@100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(-1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=-1)


class HeapElement:
    """Utility class to make a heap with a custom comparator"""

    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val["score"] < other.val["score"]


# # From https://github.com/facebookresearch/detectron2/blob/bcfd464d0c810f0442d91a349c0f6df945467143/detectron2/evaluation/fast_eval_api.py#L13
class COCOeval_opt(COCOeval):
    """
    This is a slightly modified version of the original COCO API, where the functions evaluateImg()
    and accumulate() are implemented in C++ to speedup evaluation
    """

    def __init__(
        self, cocoGt=None, cocoDt=None, iouType="segm", dt_only_positive=False
    ):
        super().__init__(cocoGt, cocoDt, iouType)
        self.dt_only_positive = dt_only_positive

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        """

        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann["segmentation"] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(
                self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
            dts = self.cocoDt.loadAnns(
                self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == "segm":
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if p.iouType == "keypoints":
                gt["ignore"] = (gt["num_keypoints"] == 0) or gt["ignore"]
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation

        _gts_cat_ids = defaultdict(set)  # gt for evaluation on positive split
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
            _gts_cat_ids[gt["image_id"]].add(gt["category_id"])
        for dt in dts:
            if (
                self.dt_only_positive
                and dt["category_id"] not in _gts_cat_ids[dt["image_id"]]
            ):
                continue
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results


class CocoEvaluatorOfflineWithPredFileEvaluators:
    def __init__(
        self,
        gt_path,
        tide: bool = True,
        iou_type: str = "bbox",
        positive_split=False,
    ):
        self.gt_path = gt_path
        self.tide_enabled = HAS_TIDE and tide
        self.positive_split = positive_split
        self.iou_type = iou_type

    def evaluate(self, dumped_file):
        if not is_main_process():
            return {}

        logging.info("OfflineCoco evaluator: Loading groundtruth")
        self.gt = COCO(self.gt_path)

        # Creating the result file
        logging.info("Coco evaluator: Creating the result file")
        cocoDt = self.gt.loadRes(str(dumped_file))

        # Run the evaluation
        logging.info("Coco evaluator: Running evaluation")
        coco_eval = COCOeval_opt(
            self.gt, cocoDt, iouType=self.iou_type, dt_only_positive=self.positive_split
        )
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        outs = {}
        for i, value in enumerate(coco_eval.stats):
            outs[f"coco_eval_{self.iou_type}_{COCO_METRICS[i]}"] = value

        if self.tide_enabled:
            logging.info("Coco evaluator: Loading TIDE")
            self.tide_gt = datasets.COCO(self.gt_path)
            self.tide = TIDE(mode="mask" if self.iou_type == "segm" else "bbox")

            # Run TIDE
            logging.info("Coco evaluator: Running TIDE")
            self.tide.evaluate(
                self.tide_gt, datasets.COCOResult(str(dumped_file)), name="coco_eval"
            )
            self.tide.summarize()
            for k, v in self.tide.get_main_errors()["coco_eval"].items():
                outs[f"coco_eval_{self.iou_type}_TIDE_{k}"] = v

            for k, v in self.tide.get_special_errors()["coco_eval"].items():
                outs[f"coco_eval_{self.iou_type}_TIDE_{k}"] = v

        return outs


class CocoEvaluatorOffline:
    def __init__(
        self,
        gt_path,
        postprocessor,
        dump_dir: str,
        iou_type: str = "bbox",
        tide: bool = True,
        gather_pred_via_filesys=False,
        positive_split=False,
        maxDets: int = 100,
    ):
        self.gt_path = gt_path
        self.tide_enabled = HAS_TIDE and tide
        self.postprocessor = postprocessor

        self.maxDets = maxDets

        # Check that the dump_dir exists
        if is_main_process():
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir, exist_ok=True)
                logging.info(f"Create the folder: {dump_dir}")

        self.dump_dir = dump_dir
        self.dump = []

        assert iou_type in [
            "bbox",
            "segm",
        ], f"Unsupported iou_type for coco eval: {iou_type}"
        self.iou_type = iou_type

        # Whether to gather predictions through filesystem (instead of torch
        # collective ops; requiring a shared filesystem across all ranks)
        self.gather_pred_via_filesys = gather_pred_via_filesys

        # Whether to evaluate on the positive split
        self.positive_split = positive_split

    def set_sync_device(self, device: torch.device) -> Any:
        self._sync_device = device

    def update(self, *args, **kwargs):
        predictions = self.postprocessor.process_results(*args, **kwargs)

        results = self.prepare(predictions, self.iou_type)
        self.dump.extend(results)

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def synchronize_between_processes(self):
        logging.info("OfflineCoco evaluator: Synchronizing between processes")
        # First, flatten the dict
        gc.collect()

        logging.info("OfflineCoco evaluator: Gathering predictions from all processes")

        if self.gather_pred_via_filesys:
            dump = gather_to_rank_0_via_filesys(self.dump)
        else:
            dump = all_gather(self.dump, force_cpu=True)

        # we'll combine the predictions, keeping only 100 per images
        maxDets = self.maxDets
        preds_by_image = defaultdict(list)
        seen_img_cat = set()
        for cur_dump in dump:
            cur_seen_img_cat = set()
            for p in cur_dump:
                image_id = p["image_id"]
                cat_id = p["category_id"]
                if (image_id, cat_id) in seen_img_cat:
                    # already seen this image/category pair in a previous dump, we can skip
                    continue
                cur_seen_img_cat.add((image_id, cat_id))
                if len(preds_by_image[image_id]) < maxDets:
                    heapq.heappush(preds_by_image[image_id], HeapElement(p))
                else:
                    heapq.heappushpop(preds_by_image[image_id], HeapElement(p))
            seen_img_cat.update(cur_seen_img_cat)

        self.dump = sum(
            [[h.val for h in cur_preds] for cur_preds in preds_by_image.values()], []
        )

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": [round(b, 5) for b in box],
                        "score": round(scores[k], 7),
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(
                    np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F")
                )[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def summarize(self):
        logging.info("OfflineCoco evaluator: Summarizing")
        if not is_main_process():
            return {}

        # Dump the predictions to a file
        dumped_file = Path(self.dump_dir) / "coco_predictions.json"
        logging.info(f"OfflineCoco evaluator: Dumping predictions to {dumped_file}")
        with g_pathmgr.open(str(dumped_file), "w") as f:
            json.dump(self.dump, f)

        if len(self.dump) == 0:
            logging.info("No predictions, skipping evaluator!")
            return {}

        logging.info("OfflineCoco evaluator: Loading groundtruth")
        self.gt = COCO(self.gt_path)

        # Creating the result file
        logging.info("Coco evaluator: Creating the result file")
        cocoDt = self.gt.loadRes(str(dumped_file))

        # Run the evaluation
        logging.info("Coco evaluator: Running evaluation")
        coco_eval = COCOeval_opt(
            self.gt, cocoDt, iouType=self.iou_type, dt_only_positive=self.positive_split
        )
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        outs = {}
        for i, value in enumerate(coco_eval.stats):
            outs[f"coco_eval_{self.iou_type}_{COCO_METRICS[i]}"] = value

        if self.tide_enabled:
            logging.info("Coco evaluator: Loading TIDE")
            self.tide_gt = datasets.COCO(self.gt_path)
            self.tide = TIDE(mode="mask" if self.iou_type == "segm" else "bbox")

            # Run TIDE
            logging.info("Coco evaluator: Running TIDE")
            self.tide.evaluate(
                self.tide_gt, datasets.COCOResult(str(dumped_file)), name="coco_eval"
            )
            self.tide.summarize()
            for k, v in self.tide.get_main_errors()["coco_eval"].items():
                outs[f"coco_eval_{self.iou_type}_TIDE_{k}"] = v

            for k, v in self.tide.get_special_errors()["coco_eval"].items():
                outs[f"coco_eval_{self.iou_type}_TIDE_{k}"] = v

        return outs

    def compute_synced(self):
        self.synchronize_between_processes()
        return self.summarize()

    def compute(self):
        return {"": 0.0}

    def reset(self):
        self.dump = []
