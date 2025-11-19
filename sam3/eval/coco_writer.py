# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
COCO prediction dumper for distributed training.

Handles collection and dumping of COCO-format predictions from models.
Supports distributed processing with multiple GPUs/processes.
"""

import copy
import gc
import heapq
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import pycocotools.mask as mask_utils
import torch
from iopath.common.file_io import g_pathmgr
from sam3.eval.coco_eval_offline import convert_to_xywh
from sam3.train.masks_ops import rle_encode
from sam3.train.utils.distributed import (
    all_gather,
    gather_to_rank_0_via_filesys,
    get_rank,
    is_main_process,
)


### Helper functions and classes


class HeapElement:
    """Utility class to make a heap with a custom comparator based on score."""

    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val["score"] < other.val["score"]


class PredictionDumper:
    """
    Handles collection and dumping of COCO-format predictions from a model.

    This class processes model outputs through a postprocessor, converts them to COCO format,
    and saves them to disk. It supports distributed processing with multiple GPUs/processes.
    """

    def __init__(
        self,
        dump_dir: str,
        postprocessor,
        maxdets: int,
        iou_type: str,
        gather_pred_via_filesys: bool = False,
        merge_predictions: bool = False,
        pred_file_evaluators: Optional[Any] = None,
    ):
        """
        Initialize the PredictionDumper.

        Args:
            dump_dir: Directory to dump predictions.
            postprocessor: Module to convert the model's output into COCO format.
            maxdets: Maximum number of detections per image.
            iou_type: IoU type to evaluate. Can include "bbox", "segm"
            gather_pred_via_filesys: If True, use the filesystem for collective gathers across
                processes (requires a shared filesystem). Otherwise, use torch collective ops.
            merge_predictions: If True, merge predictions from all processes and dump to a single file.
        """
        self.iou_type = iou_type
        self.maxdets = maxdets
        self.dump_dir = dump_dir
        self.postprocessor = postprocessor
        self.gather_pred_via_filesys = gather_pred_via_filesys
        self.merge_predictions = merge_predictions
        self.pred_file_evaluators = pred_file_evaluators
        if self.pred_file_evaluators is not None:
            assert (
                merge_predictions
            ), "merge_predictions must be True if pred_file_evaluators are provided"
        assert self.dump_dir is not None, "dump_dir must be provided"

        if is_main_process():
            os.makedirs(self.dump_dir, exist_ok=True)
            logging.info(f"Created prediction dump directory: {self.dump_dir}")

        # Initialize state
        self.reset()

    def update(self, *args, **kwargs):
        """
        Process and accumulate predictions from model outputs.

        Args:
            *args, **kwargs: Arguments passed to postprocessor.process_results()
        """
        predictions = self.postprocessor.process_results(*args, **kwargs)
        results = self.prepare(predictions, self.iou_type)
        self._dump(results)

    def _dump(self, results):
        """
        Add results to the dump list with precision rounding.

        Args:
            results: List of prediction dictionaries in COCO format.
        """
        dumped_results = copy.deepcopy(results)
        for r in dumped_results:
            if "bbox" in r:
                r["bbox"] = [round(coord, 5) for coord in r["bbox"]]
            r["score"] = round(r["score"], 5)
        self.dump.extend(dumped_results)

    def synchronize_between_processes(self):
        """
        Synchronize predictions across all processes and save to disk.

        If gather_pred_via_filesys is True, uses filesystem for gathering.
        Otherwise, uses torch distributed collective operations.
        Saves per-rank predictions to separate JSON files.
        """
        logging.info("Prediction Dumper: Synchronizing between processes")

        if not self.merge_predictions:
            dumped_file = (
                Path(self.dump_dir)
                / f"coco_predictions_{self.iou_type}_{get_rank()}.json"
            )
            logging.info(
                f"Prediction Dumper: Dumping local predictions to {dumped_file}"
            )
            with g_pathmgr.open(str(dumped_file), "w") as f:
                json.dump(self.dump, f)
        else:
            self.dump = self.gather_and_merge_predictions()
            dumped_file = Path(self.dump_dir) / f"coco_predictions_{self.iou_type}.json"
            if is_main_process():
                logging.info(
                    f"Prediction Dumper: Dumping merged predictions to {dumped_file}"
                )
                with g_pathmgr.open(str(dumped_file), "w") as f:
                    json.dump(self.dump, f)

        self.reset()
        return dumped_file

    def gather_and_merge_predictions(self):
        """
        Gather predictions from all processes and merge them, keeping top predictions per image.

        This method collects predictions from all processes, then keeps only the top maxdets
        predictions per image based on score. It also deduplicates predictions by (image_id, category_id).

        Returns:
            List of merged prediction dictionaries.
        """
        logging.info("Prediction Dumper: Gathering predictions from all processes")
        gc.collect()

        if self.gather_pred_via_filesys:
            dump = gather_to_rank_0_via_filesys(self.dump)
        else:
            dump = all_gather(self.dump, force_cpu=True)

        # Combine predictions, keeping only top maxdets per image
        preds_by_image = defaultdict(list)
        seen_img_cat = set()

        for cur_dump in dump:
            cur_seen_img_cat = set()
            for p in cur_dump:
                image_id = p["image_id"]
                cat_id = p["category_id"]

                # Skip if we've already seen this image/category pair in a previous dump
                if (image_id, cat_id) in seen_img_cat:
                    continue

                cur_seen_img_cat.add((image_id, cat_id))

                # Use a min-heap to keep top predictions
                if len(preds_by_image[image_id]) < self.maxdets:
                    heapq.heappush(preds_by_image[image_id], HeapElement(p))
                else:
                    heapq.heappushpop(preds_by_image[image_id], HeapElement(p))

            seen_img_cat.update(cur_seen_img_cat)

        # Flatten the heap elements back to a list
        merged_dump = sum(
            [[h.val for h in cur_preds] for cur_preds in preds_by_image.values()], []
        )

        return merged_dump

    def compute_synced(self):
        """
        Synchronize predictions across processes and compute summary.

        Returns:
            Summary dictionary from summarize().
        """
        dumped_file = self.synchronize_between_processes()
        if not is_main_process():
            return {"": 0.0}

        meters = {}
        if self.pred_file_evaluators is not None:
            for evaluator in self.pred_file_evaluators:
                results = evaluator.evaluate(dumped_file)
                meters.update(results)

        if len(meters) == 0:
            meters = {"": 0.0}
        return meters

    def compute(self):
        """
        Compute without synchronization.

        Returns:
            Empty metric dictionary.
        """
        return {"": 0.0}

    def reset(self):
        """Reset internal state for a new evaluation round."""
        self.dump = []

    def prepare(self, predictions, iou_type):
        """
        Route predictions to the appropriate preparation method based on iou_type.

        Args:
            predictions: Dictionary mapping image IDs to prediction dictionaries.
            iou_type: Type of evaluation ("bbox", "segm").

        Returns:
            List of COCO-format prediction dictionaries.
        """
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        else:
            raise ValueError(f"Unknown iou type: {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        """
        Convert predictions to COCO detection format.

        Args:
            predictions: Dictionary mapping image IDs to prediction dictionaries
                containing "boxes", "scores", and "labels".

        Returns:
            List of COCO-format detection dictionaries.
        """
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
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    @torch.no_grad()
    def prepare_for_coco_segmentation(self, predictions):
        """
        Convert predictions to COCO segmentation format.

        Args:
            predictions: Dictionary mapping image IDs to prediction dictionaries
                containing "masks" or "masks_rle", "scores", and "labels".
                Optionally includes "boundaries" and "dilated_boundaries".

        Returns:
            List of COCO-format segmentation dictionaries with RLE-encoded masks.
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            boxes = None
            if "boxes" in prediction:
                boxes = prediction["boxes"]
                boxes = convert_to_xywh(boxes).tolist()
                assert len(boxes) == len(scores)

            if "masks_rle" in prediction:
                rles = prediction["masks_rle"]
                areas = []
                for rle in rles:
                    cur_area = mask_utils.area(rle)
                    h, w = rle["size"]
                    areas.append(cur_area / (h * w))
            else:
                masks = prediction["masks"]
                masks = masks > 0.5
                h, w = masks.shape[-2:]

                areas = masks.flatten(1).sum(1) / (h * w)
                areas = areas.tolist()

                rles = rle_encode(masks.squeeze(1))

                # Memory cleanup
                del masks
                del prediction["masks"]

            assert len(areas) == len(rles) == len(scores)

            for k, rle in enumerate(rles):
                payload = {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                    "area": areas[k],
                }
                if boxes is not None:
                    payload["bbox"] = boxes[k]

                coco_results.append(payload)

        return coco_results
