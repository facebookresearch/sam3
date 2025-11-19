# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
import copy
import gc
import logging
import os
from collections import defaultdict
from operator import xor
from pathlib import Path
from typing import List, Optional

import numpy as np
import pycocotools.mask as mask_util
import torch
from pycocotools.cocoeval import COCOeval
from sam3.eval.cgf1_eval import CGF1Eval
from sam3.eval.coco_eval_offline import convert_to_xywh
from sam3.model.box_ops import box_xywh_inter_union
from sam3.train.masks_ops import rle_encode
from sam3.train.utils import distributed as dist
from typing_extensions import override

try:
    import rapidjson as json
except ModuleNotFoundError:
    import json

from iopath.common.file_io import g_pathmgr


class YTVISevalMixin:
    """
    Identical to COCOeval but adapts computeIoU to compute IoU between tracklets/masklets.
    """

    @override
    def _prepare(self):
        """
        Copied from cocoeval.py but doesn't convert masks to RLEs (we assume they already are RLEs)
        """
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

        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if p.iouType == "keypoints":
                gt["ignore"] = (gt["num_keypoints"] == 0) or gt["ignore"]
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def computeIoU(self, imgId, catId):
        """
        Compute IoU between tracklets. Copied from cocoeval.py but adapted for videos (in YT-VIS format)
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 or len(dt) == 0:
            return []

        # For class mAP and phrase AP evaluation, we sort the detections in descending order of scores (as in COCOeval).
        # For demo F1 evaluation, we DO NOT sort the detections (but match them with GTs via Hungarian matching).
        assert hasattr(self, "sort_inds_by_scores_in_iou"), (
            "subclasses that inherits YTVISevalMixin should set `self.sort_inds_by_scores_in_iou` "
            "(True for class mAP and phrase AP, False for demo F1)"
        )
        if self.sort_inds_by_scores_in_iou:
            inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
            dt = [dt[i] for i in inds]
            if len(dt) > p.maxDets[-1]:
                dt = dt[0 : p.maxDets[-1]]

        if p.iouType == "segm":
            g = [g["segmentations"] for g in gt]
            d = [d["segmentations"] for d in dt]
        elif p.iouType == "bbox":
            g = [g["bboxes"] for g in gt]
            d = [d["bboxes"] for d in dt]
        else:
            raise Exception("unknown iouType for iou computation")

        def iou_tracklets(preds, gts):
            preds = torch.tensor(preds)
            gts = torch.tensor(gts)
            inter, union = box_xywh_inter_union(
                preds.unsqueeze(1), gts.unsqueeze(0)
            )  # Num preds x Num GTS x Num frames
            inter = inter.sum(-1)
            union = union.sum(-1)
            assert (
                union > 0
            ).all(), (
                "There exists a tracklet with zero GTs across time. This is suspicious"
            )
            return inter / union

        def iou_masklets(preds, gts):
            inter = 0
            union = 0
            for p_i, gt_i in zip(preds, gts):
                if p_i and gt_i:
                    # Compute areas of intersection and union
                    inter += mask_util.area(
                        mask_util.merge([p_i, gt_i], intersect=True)
                    )
                    union += mask_util.area(
                        mask_util.merge([p_i, gt_i], intersect=False)
                    )
                elif gt_i:
                    union += mask_util.area(gt_i)
                elif p_i:
                    union += mask_util.area(p_i)
            if union > 0:
                iou = inter / union
                assert iou >= 0 and iou <= 1, "Encountered an error in IoU computation"
            else:
                assert np.isclose(inter, 0) and np.isclose(
                    union, 0
                ), "Encountered an error in IoU computation"
                iou = 1
            return iou

        if p.iouType == "segm":
            ious = [[iou_masklets(d_i, g_i) for g_i in g] for d_i in d]
        else:
            ious = iou_tracklets(d, g)
        return np.array(ious)


class YTVISeval(YTVISevalMixin, COCOeval):
    # For class mAP and phrase AP evaluation, we sort the detections in descending order of scores (as in COCOeval).
    sort_inds_by_scores_in_iou = True


class VideoDemoF1Eval(YTVISevalMixin, CGF1Eval):
    # For demo F1 evaluation, we DO NOT sort the detections (but match them with GTs via Hungarian matching).
    sort_inds_by_scores_in_iou = False


class YTVISResultsWriter:
    """
    Gather and dumps predictions in YT-VIS format.
    Expected flow of API calls: reset() -> N * update() -> compute_synced()
    """

    def __init__(
        self,
        dump_file: str,
        postprocessor,
        gather_pred_via_filesys=False,
        pred_file_evaluators: Optional[List] = None,
        save_per_frame_scores: bool = False,
        write_eval_metrics_file: bool = True,
        eval_metrics_file_suffix: str = ".sam3_eval_metrics",
    ):
        self.dump_file = dump_file
        self.dump = []
        self.postprocessor = postprocessor
        self.gather_pred_via_filesys = gather_pred_via_filesys
        if dist.is_main_process():
            dirname = os.path.dirname(self.dump_file)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
                logging.info(f"Creating folder: {dirname}")

        # the evaluation hooks to be applied to the prediction files
        self.pred_file_evaluators = pred_file_evaluators or []
        self.save_per_frame_scores = save_per_frame_scores
        # in addition to the prediction file, we also write the evaluation metrics
        # for easier debugging and analysis (stored in another eval_metrics_file
        # so that we can keep the dumped prediction file under YT-VIS format)
        self.write_eval_metrics_file = write_eval_metrics_file
        if self.write_eval_metrics_file:
            self.eval_metrics_file = self.dump_file + eval_metrics_file_suffix
            os.makedirs(os.path.dirname(self.eval_metrics_file), exist_ok=True)

    def _dump_vid_preds(self, results):
        dumped_results = copy.deepcopy(results)
        self.dump.extend(dumped_results)

    def prepare(self, predictions):
        ytvis_results = []
        for video_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            for k in ["boxes", "scores", "labels"]:
                assert (
                    k in prediction
                ), f"Expected predictions to have `{k}` key, available keys are {prediction.keys()}"
            if self.save_per_frame_scores:
                assert (
                    "per_frame_scores" in prediction
                ), f"Expected predictions to have `per_frame_scores` key, available keys are {prediction.keys()}"
            assert xor(
                "masks" in prediction, "masks_rle" in prediction
            ), f"Expected predictions to have either `masks` key or `masks_rle` key, available keys are {prediction.keys()}"

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            if "masks" in prediction:
                masks = prediction["masks"].squeeze(2)
                assert (
                    masks.ndim == 4
                ), "Expected masks to be of shape(N_preds,T_frames,H,W)"

                areas = [mask.flatten(1).sum(1).tolist() for mask in masks]
                rles = [rle_encode(masklet) for masklet in masks]

                # memory clean
                del masks
                del prediction["masks"]
            elif "masks_rle" in prediction:
                rles = prediction.pop("masks_rle")
                areas = [
                    [0 if rle is None else rle.pop("area") for rle in rles_per_obj]
                    for rles_per_obj in rles
                ]
            else:
                raise ValueError(
                    "Expected either `masks` or `masks_rle` key in the predictions."
                )

            new_results = [
                {
                    "video_id": video_id,
                    "category_id": track_label,
                    "bboxes": track_boxes,
                    "score": track_score,
                    "segmentations": track_masks,
                    "areas": track_areas,
                }
                for (
                    track_boxes,
                    track_masks,
                    track_areas,
                    track_score,
                    track_label,
                ) in zip(boxes, rles, areas, scores, labels)
            ]
            # Optionally, save per-frame scores
            if self.save_per_frame_scores:
                per_frame_scores = prediction["per_frame_scores"].tolist()
                for res, track_per_frame_scores in zip(new_results, per_frame_scores):
                    res["per_frame_scores"] = track_per_frame_scores

            ytvis_results.extend(new_results)

        return ytvis_results

    def set_sync_device(self, device: torch.device):
        self._sync_device = device

    def update(self, *args, **kwargs):
        predictions = self.postprocessor.process_results(*args, **kwargs)
        results = self.prepare(predictions)
        self._dump_vid_preds(results)

    def _dump_preds(self):
        if not dist.is_main_process():
            self.dump = []
            gc.collect()
            return
        dumped_file = Path(self.dump_file)
        logging.info(f"YTVIS evaluator: Dumping predictions to {dumped_file}")
        with g_pathmgr.open(str(dumped_file), "w") as f:
            json.dump(self.dump, f)
        self.dump = []
        gc.collect()
        return str(dumped_file)

    def synchronize_between_processes(self):
        logging.info("YT-VIS evaluator: Synchronizing between processes")
        dump_dict = self._dedup_pre_gather(self.dump)
        if self.gather_pred_via_filesys:
            dump_dict_all_gpus = dist.gather_to_rank_0_via_filesys(dump_dict)
        else:
            dump_dict_all_gpus = dist.all_gather(dump_dict, force_cpu=True)
        self.dump = self._dedup_post_gather(dump_dict_all_gpus)
        logging.info(f"Gathered all {len(self.dump)} predictions")

    def _dedup_pre_gather(self, predictions):
        """
        Organize the predictions as a dict-of-list using (video_id, category_id) as keys
        for deduplication after gathering them across GPUs.

        During evaluation, PyTorch data loader under `drop_last: False` would wrap
        around the dataset length to be a multiple of world size (GPU num) and duplicate
        the remaining batches. This causes the same test sample to appear simultaneously
        in multiple GPUs, resulting in duplicated predictions being saved into prediction
        files. These duplicates are then counted as false positives under detection mAP
        metrics (since a ground truth can be matched with only one prediction).

        For example, if there are 4 GPUs and 6 samples [A1, A2, B1, B2, C1, C2], the data
        loader (under `drop_last: False`) would load it by wrapping it around like
        `[A1, A2, B1, B2, C1, C2, *A1*, *A2*]` to make a multiple of 4 and then split it as

        - GPU 0: A1, C1
        - GPU 1: A2, C2
        - GPU 3: B1, **A1**
        - GPU 4: B2, **A2**
        (as in DistributedSampler in https://github.com/pytorch/pytorch/blob/521588519da9f4876d90ddd7a17c10d0eca89dc6/torch/utils/data/distributed.py#L116-L124)

        so the predictions on A1 and A2 will occur twice in the final gathered outputs
        in the prediction file (and counted as false positives). This also affects our
        YT-VIS official val evaluation, but to a lesser extent than YT-VIS dev since
        the latter is much smaller and more susceptible to false positives.

        So we to deduplicate this. The tricky part is that we cannot deduplicate them
        simply using video id, given that we are sharding the classes in each video
        across multiple batches (with 20 prompts per batch) in our "orig_cats" eval dbs.

        The solution is to deduplicate based on (video_id, category_id) tuple as keys.
        We organize the predictions as a dict-of-list using (video_id, category_id) as
        keys on each GPU, with the list of masklets under this (video_id, category_id)
        on this GPU as values. Then, we all-gather this dict-of-list across GPUs and
        if a key (video_id, category_id) appears in multiple GPUs, we only take the
        prediction masklet list from one GPU.
        """
        prediction_dict = defaultdict(list)
        for p in predictions:
            prediction_dict[(p["video_id"], p["category_id"])].append(p)
        return prediction_dict

    def _dedup_post_gather(self, list_of_prediction_dict):
        """
        Deduplicate the predictions from all GPUs. See `_dedup_pre_gather` for details.
        """
        dedup_prediction_dict = {}
        duplication_keys = []
        for prediction_dict in list_of_prediction_dict:
            for k, v in prediction_dict.items():
                if k not in dedup_prediction_dict:
                    dedup_prediction_dict[k] = v
                else:
                    duplication_keys.append(k)

        logging.info(
            f"skipped {len(duplication_keys)} duplicated predictions in YTVISResultsWriter "
            f"with the following (video_id, category_id) tuples: {duplication_keys}"
        )
        dedup_predictions = sum(dedup_prediction_dict.values(), [])
        return dedup_predictions

    def compute_synced(
        self,
    ):
        self.synchronize_between_processes()
        dumped_file = self._dump_preds()
        if not dist.is_main_process():
            return {"": 0.0}

        # run evaluation hooks on the prediction file
        meters = {}
        all_video_np_level_results = defaultdict(dict)
        for evaluator in self.pred_file_evaluators:
            gc.collect()
            results, video_np_level_results = evaluator.evaluate(dumped_file)
            meters.update(results)
            for (video_id, category_id), res in video_np_level_results.items():
                all_video_np_level_results[(video_id, category_id)].update(res)

        gc.collect()
        if self.write_eval_metrics_file:
            # convert the nested dict of {(video_id, category_id): per_sample_metric_dict}
            # to a list of per-sample metric dicts (with video_id and category_id) for JSON,
            # as JSON doesn't allow using tuples like (video_id, category_id) as dict keys
            video_np_level_metrics = [
                {"video_id": video_id, "category_id": category_id, **res}
                for (video_id, category_id), res in all_video_np_level_results.items()
            ]
            eval_metrics = {
                "dataset_level_metrics": meters,
                "video_np_level_metrics": video_np_level_metrics,
            }
            with g_pathmgr.open(self.eval_metrics_file, "w") as f:
                json.dump(eval_metrics, f)
            logging.info(
                f"YTVIS evaluator: Dumped evaluation metrics to {self.eval_metrics_file}"
            )

        if len(meters) == 0:
            meters = {"": 0.0}
        return meters

    def compute(self):
        return {"": 0.0}

    def reset(self, *args, **kwargs):
        self.dump = []
