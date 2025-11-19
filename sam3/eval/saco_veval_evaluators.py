# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
import json
import os
import tempfile
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pycocotools.mask
from sam3.eval.cgf1_eval import CGF1_METRICS
from sam3.eval.conversion_util import (
    convert_ytbvis_to_cocovid_gt,
    convert_ytbvis_to_cocovid_pred,
)
from sam3.eval.hota_eval_toolkit.run_ytvis_eval import run_ytvis_eval
from sam3.eval.teta_eval_toolkit import config, Evaluator, metrics
from sam3.eval.teta_eval_toolkit.datasets import COCO, TAO
from sam3.eval.ytvis_coco_wrapper import YTVIS
from sam3.eval.ytvis_eval import VideoDemoF1Eval, YTVISeval
from sam3.train.nms_helper import process_frame_level_nms, process_track_level_nms


def _get_metric_index(metric_name: str, iou_threshold: Optional[float] = None) -> int:
    """
    Find the index of a metric in CGF1_METRICS by name and IoU threshold.

    Args:
        metric_name: Name of the metric (e.g., "cgF1", "precision", "recall")
        iou_threshold: IoU threshold (None for average over 0.5:0.95, or specific value like 0.5, 0.75)

    Returns:
        Index of the metric in CGF1_METRICS

    Raises:
        ValueError: If metric not found
    """
    for idx, metric in enumerate(CGF1_METRICS):
        if metric.name == metric_name and metric.iou_threshold == iou_threshold:
            return idx
    raise ValueError(
        f"Metric '{metric_name}' with IoU threshold {iou_threshold} not found in CGF1_METRICS"
    )


class BasePredFileEvaluator:
    """A base class for evaluating a prediction file."""

    pass


class YTVISPredFileEvaluator(BasePredFileEvaluator):
    """Evaluate class mAP for YT-VIS prediction files."""

    def __init__(
        self,
        gt_ann_file: str,
        dataset_name: str = "video",
        iou_types: Optional[Sequence[str]] = None,
    ):
        self.gt_ann_file = gt_ann_file
        self.dataset_name = dataset_name
        self.iou_types = list(iou_types) if iou_types is not None else ["bbox", "segm"]
        assert all(iou_type in ["bbox", "segm"] for iou_type in self.iou_types)

    def evaluate(self, pred_file: str) -> Dict[str, float]:
        # use our internal video evaluation toolkit for YT-VIS pred file
        # (i.e. the same one we're using for video phrase AP)
        results = {}
        use_cats = True  # YT-VIS mAP evaluation uses categories
        ytvisGT = YTVIS(self.gt_ann_file, ignore_gt_cats=not use_cats)
        # the original YT-VIS GT annotations have uncompressed RLEs ("counts" is an integer list)
        # rather than compressed RLEs ("counts" is a string), so we first convert them here.
        if "segm" in self.iou_types:
            for ann in ytvisGT.dataset["annotations"]:
                ann["segmentations"] = [
                    _compress_rle(rle) for rle in ann["segmentations"]
                ]

        with open(pred_file) as f:
            dt = json.load(f)
        # Our prediction file saves "video_id" and absolute (unnormalized) boxes.
        # Note that we should use the official (original) YT-VIS annotations (i.e. the one
        # saved via "scripts/datasets/training/ytvis_split.py", instead of the one saved
        # via "scripts/api_db_to_ytvis_json.py") in this evaluator, which contain absolute
        # boxes coordinates in its GT annotations.
        for d in dt:
            d["image_id"] = d["video_id"]
        ytvisDT = ytvisGT.loadRes(dt)

        for iou_type in self.iou_types:
            ytvisEval = YTVISeval(ytvisGT, ytvisDT, iou_type)

            # set the area ranges for small, medium, and large objects (using
            # absolute pixel areas) as in the official YT-VIS evaluation toolkit:
            # https://github.com/achalddave/ytvosapi/blob/eca601117c9f86bad084cb91f1d918e9ab665a75/PythonAPI/ytvostools/ytvoseval.py#L538
            ytvisEval.params.areaRng = [
                [0**2, 1e5**2],
                [0**2, 128**2],
                [128**2, 256**2],
                [256**2, 1e5**2],
            ]
            ytvisEval.params.areaRngLbl = ["all", "small", "medium", "large"]
            ytvisEval.params.useCats = use_cats

            ytvisEval.evaluate()
            ytvisEval.accumulate()
            ytvisEval.summarize()
            result_key = f"{self.dataset_name}_{'mask' if iou_type == 'segm' else 'bbox'}_mAP_50_95"
            results[result_key] = ytvisEval.stats[0]

        # video-NP level results not supported for `YTVISPredFileEvaluator` yet
        video_np_level_results = {}
        return results, video_np_level_results


class VideoPhraseApEvaluator(BasePredFileEvaluator):
    """Evaluate Video Phrase AP with YT-VIS format prediction and GT files."""

    def __init__(
        self,
        gt_ann_file: str,
        dataset_name: str = "video",
        iou_types: Optional[Sequence[str]] = None,
    ):
        self.gt_ann_file = gt_ann_file
        self.dataset_name = dataset_name
        self.iou_types = list(iou_types) if iou_types is not None else ["bbox", "segm"]
        assert all(iou_type in ["bbox", "segm"] for iou_type in self.iou_types)

    def evaluate(self, pred_file: str) -> Dict[str, float]:
        with open(self.gt_ann_file) as f:
            gt = json.load(f)
        with open(pred_file) as f:
            dt = json.load(f)
        # For phrase AP and demo F1 evaluation, we need to remap each pair of (video_id, category_id) to
        # a new unique video_id, so that we don't mix detections from different categories under `useCat=False`
        gt, dt = remap_video_category_pairs_to_unique_video_ids(gt, dt)
        if "segm" in self.iou_types:
            for ann in gt["annotations"]:
                ann["segmentations"] = [
                    _compress_rle(rle) for rle in ann["segmentations"]
                ]
        for d in dt:
            d["image_id"] = d["video_id"]

        results = {}
        use_cats = False  # Phrase AP evaluation does not use categories
        ytvisGT = YTVIS(annotation_file=None, ignore_gt_cats=not use_cats)
        ytvisGT.dataset = gt
        ytvisGT.createIndex()
        ytvisDT = ytvisGT.loadRes(dt)

        for iou_type in self.iou_types:
            phraseApEval = YTVISeval(ytvisGT, ytvisDT, iou_type)

            # set the area ranges for small, medium, and large objects (using
            # absolute pixel areas) as in the official YT-VIS evaluation toolkit:
            # https://github.com/achalddave/ytvosapi/blob/eca601117c9f86bad084cb91f1d918e9ab665a75/PythonAPI/ytvostools/ytvoseval.py#L538
            phraseApEval.params.areaRng = [
                [0**2, 1e5**2],
                [0**2, 128**2],
                [128**2, 256**2],
                [256**2, 1e5**2],
            ]
            phraseApEval.params.areaRngLbl = ["all", "small", "medium", "large"]
            phraseApEval.params.useCats = use_cats

            phraseApEval.evaluate()
            phraseApEval.accumulate()
            phraseApEval.summarize()
            result_prefix = f"{self.dataset_name}"
            result_prefix += f"_{'mask' if iou_type == 'segm' else 'bbox'}_phrase_ap"
            # fetch Phrase AP results from the corresponding indices in `phraseApEval.stats`
            # (see `_summarizeDets` in https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
            results[result_prefix + "_50_95"] = phraseApEval.stats[0]  # IoU=0.5:0.95
            results[result_prefix + "_50"] = phraseApEval.stats[1]  # IoU=0.5
            results[result_prefix + "_75"] = phraseApEval.stats[2]  # IoU=0.75

        # video-NP level results not supported for `VideoPhraseApEvaluator` yet
        video_np_level_results = {}
        return results, video_np_level_results


class VideoCGF1Evaluator(BasePredFileEvaluator):
    """Evaluate Video Demo F1 with YT-VIS format prediction and GT files."""

    def __init__(
        self,
        gt_ann_file: str,
        dataset_name: str = "video",
        prob_thresh: float = 0.5,
        iou_types: Optional[Sequence[str]] = None,
    ):
        self.gt_ann_file = gt_ann_file
        self.dataset_name = dataset_name
        self.prob_thresh = prob_thresh
        self.iou_types = list(iou_types) if iou_types is not None else ["bbox", "segm"]
        assert all(iou_type in ["bbox", "segm"] for iou_type in self.iou_types)

    def evaluate(self, pred_file: str) -> Dict[str, float]:
        with open(self.gt_ann_file) as f:
            gt = json.load(f)
        with open(pred_file) as f:
            dt = json.load(f)
        # compute IL_MCC and CG-F1 can only be computed if we have "video_np_pairs" keys in the GT JSON
        compute_ilmcc_and_cgf1 = "video_np_pairs" in gt
        if not compute_ilmcc_and_cgf1:
            print(
                f"Warning: IL_MCC and CG-F1 are not computed for {pred_file=} as it does not have 'video_np_pairs' keys in the GT JSON"
            )
        # For phrase AP and demo F1 evaluation, we need to remap each pair of (video_id, category_id) to
        # a new unique video_id, so that we don't mix detections from different categories under `useCat=False`
        gt, dt = remap_video_category_pairs_to_unique_video_ids(
            gt, dt, add_negative_np_pairs=compute_ilmcc_and_cgf1
        )
        if "segm" in self.iou_types:
            for ann in gt["annotations"]:
                ann["segmentations"] = [
                    _compress_rle(rle) for rle in ann["segmentations"]
                ]
        for d in dt:
            d["image_id"] = d["video_id"]

        results = {}
        use_cats = False  # Demo F1 evaluation does not use categories
        ytvisGT = YTVIS(annotation_file=None, ignore_gt_cats=not use_cats)
        ytvisGT.dataset = gt
        ytvisGT.createIndex()
        ytvisDT = ytvisGT.loadRes(dt)

        video_np_level_results = {}
        for iou_type in self.iou_types:
            demoF1Eval = VideoDemoF1Eval(ytvisGT, ytvisDT, iou_type, self.prob_thresh)

            demoF1Eval.params.useCats = use_cats
            demoF1Eval.params.areaRng = [[0**2, 1e5**2]]
            demoF1Eval.params.areaRngLbl = ["all"]
            demoF1Eval.params.maxDets = [100000]

            demoF1Eval.evaluate()
            demoF1Eval.accumulate()
            demoF1Eval.summarize()
            result_prefix = f"{self.dataset_name}"
            result_prefix += f"_{'mask' if iou_type == 'segm' else 'bbox'}_demo"

            stats = demoF1Eval.stats

            if compute_ilmcc_and_cgf1:
                # Average IoU threshold (0.5:0.95)
                cgf1_micro_avg_idx = _get_metric_index("cgF1", None)
                positive_micro_f1_avg_idx = _get_metric_index("positive_micro_F1", None)
                ilmcc_avg_idx = _get_metric_index("IL_MCC", None)
                results[result_prefix + "_cgf1_micro_50_95"] = stats[cgf1_micro_avg_idx]
                results[result_prefix + "_ilmcc_50_95"] = stats[ilmcc_avg_idx]
                results[result_prefix + "_positive_micro_f1_50_95"] = stats[
                    positive_micro_f1_avg_idx
                ]

                # IoU = 0.5
                cgf1_micro_50_idx = _get_metric_index("cgF1", 0.5)
                positive_micro_f1_50_idx = _get_metric_index("positive_micro_F1", 0.5)
                results[result_prefix + "_cgf1_micro_50"] = stats[cgf1_micro_50_idx]
                results[result_prefix + "_ilmcc_50"] = float(
                    np.array(stats[cgf1_micro_50_idx])
                    / np.array(stats[positive_micro_f1_50_idx])
                )
                results[result_prefix + "_positive_micro_f1_50"] = stats[
                    positive_micro_f1_50_idx
                ]

                # IoU = 0.75
                cgf1_micro_75_idx = _get_metric_index("cgF1", 0.75)
                positive_micro_f1_75_idx = _get_metric_index("positive_micro_F1", 0.75)
                results[result_prefix + "_cgf1_micro_75"] = stats[cgf1_micro_75_idx]
                results[result_prefix + "_ilmcc_75"] = float(
                    np.array(stats[cgf1_micro_75_idx])
                    / np.array(stats[positive_micro_f1_75_idx])
                )
                results[result_prefix + "_positive_micro_f1_75"] = stats[
                    positive_micro_f1_75_idx
                ]

            self.extract_video_np_level_results(demoF1Eval, video_np_level_results)

        return results, video_np_level_results

    def extract_video_np_level_results(self, demoF1Eval, video_np_level_results):
        """Aggregate statistics for video-level metrics."""
        num_iou_thrs = len(demoF1Eval.params.iouThrs)
        iou_50_index = int(np.where(demoF1Eval.params.iouThrs == 0.5)[0])
        iou_75_index = int(np.where(demoF1Eval.params.iouThrs == 0.75)[0])

        result_prefix = "mask" if demoF1Eval.params.iouType == "segm" else "bbox"

        assert len(demoF1Eval.evalImgs) == len(demoF1Eval.cocoGt.dataset["images"])
        for i, video in enumerate(demoF1Eval.cocoGt.dataset["images"]):
            # the original video id and category id before remapping
            video_id = video["orig_video_id"]
            category_id = video["orig_category_id"]
            eval_img_dict = demoF1Eval.evalImgs[i]

            TPs = eval_img_dict.get("TPs", np.zeros(num_iou_thrs, dtype=np.int64))
            FPs = eval_img_dict.get("FPs", np.zeros(num_iou_thrs, dtype=np.int64))
            FNs = eval_img_dict.get("FNs", np.zeros(num_iou_thrs, dtype=np.int64))
            assert len(TPs) == len(FPs) == len(FNs) == num_iou_thrs
            # F1 = 2*TP / (2*TP + FP + FN), and we set F1 to 1.0 if denominator is 0
            denominator = 2 * TPs + FPs + FNs
            F1s = np.where(denominator > 0, 2 * TPs / np.maximum(denominator, 1), 1.0)
            local_results = {
                f"{result_prefix}_TP_50_95": float(TPs.mean()),
                f"{result_prefix}_FP_50_95": float(FPs.mean()),
                f"{result_prefix}_FN_50_95": float(FNs.mean()),
                f"{result_prefix}_F1_50_95": float(F1s.mean()),
                f"{result_prefix}_TP_50": float(TPs[iou_50_index]),
                f"{result_prefix}_FP_50": float(FPs[iou_50_index]),
                f"{result_prefix}_FN_50": float(FNs[iou_50_index]),
                f"{result_prefix}_F1_50": float(F1s[iou_50_index]),
                f"{result_prefix}_TP_75": float(TPs[iou_75_index]),
                f"{result_prefix}_FP_75": float(FPs[iou_75_index]),
                f"{result_prefix}_FN_75": float(FNs[iou_75_index]),
                f"{result_prefix}_F1_75": float(F1s[iou_75_index]),
            }
            if (video_id, category_id) not in video_np_level_results:
                video_np_level_results[(video_id, category_id)] = {}
            video_np_level_results[(video_id, category_id)].update(local_results)


class VideoTetaEvaluator(BasePredFileEvaluator):
    """Evaluate TETA metric using YouTubeVIS format prediction and GT files."""

    def __init__(
        self,
        gt_ann_file: str,
        dataset_name: str = "video",
        tracker_name: str = "Sam3",
        nms_threshold: float = 0.5,
        nms_strategy: str = "none",  # "track", "frame", or "none"
        prob_thresh: float = 0.5,
        is_exhaustive: bool = False,
        use_mask: bool = False,
        num_parallel_cores: int = 8,
    ):
        self.gt_ann_file = gt_ann_file
        self.dataset_name = dataset_name
        self.tracker_name = tracker_name
        self.nms_threshold = nms_threshold
        self.nms_strategy = nms_strategy.lower()  # Convert to lowercase for consistency
        self.prob_thresh = prob_thresh
        self.metric_prefix = "TETA"
        self.is_exhaustive = is_exhaustive
        self.use_mask = use_mask
        self.num_parallel_cores = num_parallel_cores

        # Verify NMS strategy is valid
        valid_strategies = ["track", "frame", "none"]
        print("current nms_strategy:", self.nms_strategy)
        if self.nms_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid NMS strategy: {self.nms_strategy}. Must be one of {valid_strategies}"
            )

        print(f"Initialized VideoTetaEvaluator with NMS strategy: {self.nms_strategy}")
        print(f"Probability threshold set to: {self.prob_thresh}")
        print(f"Dataset exhaustivity set to: {self.is_exhaustive}")
        print(f"Tracker name set to: {self.tracker_name}")
        print(f"Dataset name set to: {self.dataset_name}")
        print(f"Use mask set to: {self.use_mask}")

    def process_predictions(self, pred_file: str, tmp_dir: str) -> str:
        """Process predictions with selected NMS strategy"""
        with open(pred_file, "r") as f:
            raw_preds = json.load(f)
        print(f"Processing predictions with {self.nms_strategy} NMS strategy")

        # Filter by score threshold
        if self.prob_thresh > 0:
            raw_preds = [d for d in raw_preds if d["score"] >= self.prob_thresh]
            print(
                f"Filtered to {len(raw_preds)} predictions with score >= {self.prob_thresh}"
            )
        # Group predictions by video_id
        video_groups = defaultdict(list)
        for pred in raw_preds:
            video_groups[pred["video_id"]].append(pred)
        # Process based on NMS strategy
        if self.nms_strategy == "track":
            process_track_level_nms(video_groups, nms_threshold=self.nms_threshold)
        elif self.nms_strategy == "frame":
            process_frame_level_nms(video_groups, nms_threshold=self.nms_threshold)
        elif self.nms_strategy == "none":
            print("Skipping NMS processing as strategy is set to 'none'")
            # No processing needed for "none" strategy
        # Save processed predictions
        processed_preds = [
            track for tracks in video_groups.values() for track in tracks
        ]
        processed_path = os.path.join(tmp_dir, "processed_preds.json")
        with open(processed_path, "w") as f:
            json.dump(processed_preds, f)

        print(f"Saved processed predictions to {processed_path}")
        return processed_path

    def evaluate(self, pred_file: str) -> Tuple[Dict[str, float], Dict]:
        """Main evaluation method"""

        print(f"Evaluating TETA Metric with {self.nms_strategy.upper()} NMS strategy")
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Process predictions first
            processed_pred_file = self.process_predictions(pred_file, tmp_dir)

            # Convert GT to COCO-vid format
            gt_dir = os.path.join(tmp_dir, "gt")
            os.makedirs(gt_dir, exist_ok=True)
            gt_coco_path = os.path.join(gt_dir, "annotations.json")
            convert_ytbvis_to_cocovid_gt(self.gt_ann_file, gt_coco_path)

            # Convert processed predictions to COCO-vid format
            pred_dir = os.path.join(tmp_dir, "predictions")
            tracker_dir = os.path.join(pred_dir, self.tracker_name)
            os.makedirs(tracker_dir, exist_ok=True)
            pred_coco_path = os.path.join(tracker_dir, "track_results_cocofmt.json")
            convert_ytbvis_to_cocovid_pred(
                youtubevis_pred_path=processed_pred_file,
                converted_dataset_path=gt_coco_path,
                output_path=pred_coco_path,
            )
            # Configure TETA evaluator
            default_eval_config = config.get_default_eval_config()
            default_eval_config["PRINT_ONLY_COMBINED"] = True
            default_eval_config["DISPLAY_LESS_PROGRESS"] = True
            default_eval_config["OUTPUT_TEMP_RAW_DATA"] = True
            default_eval_config["NUM_PARALLEL_CORES"] = self.num_parallel_cores
            default_dataset_config = config.get_default_dataset_config()
            default_dataset_config["TRACKERS_TO_EVAL"] = [self.tracker_name]
            default_dataset_config["GT_FOLDER"] = gt_dir
            default_dataset_config["OUTPUT_FOLDER"] = pred_dir
            default_dataset_config["TRACKER_SUB_FOLDER"] = tracker_dir
            default_dataset_config["USE_MASK"] = self.use_mask

            evaluator = Evaluator(default_eval_config)
            if self.is_exhaustive:
                dataset_list = [COCO(default_dataset_config)]
                dataset_parsing_key = "COCO"
            else:
                dataset_list = [TAO(default_dataset_config)]
                dataset_parsing_key = "TAO"

            # Run evaluation
            eval_results, _ = evaluator.evaluate(
                dataset_list, [metrics.TETA(exhaustive=self.is_exhaustive)]
            )

            # Extract and format results
            results = {
                f"{self.dataset_name}_{'mask' if self.use_mask else 'bbox'}_teta": float(
                    eval_results[dataset_parsing_key]["TETA"][0]
                ),
                f"{self.dataset_name}_{'mask' if self.use_mask else 'bbox'}_loc_a": float(
                    eval_results[dataset_parsing_key]["TETA"][1]
                ),
                f"{self.dataset_name}_{'mask' if self.use_mask else 'bbox'}_assoc_a": float(
                    eval_results[dataset_parsing_key]["TETA"][2]
                ),
                f"{self.dataset_name}_{'mask' if self.use_mask else 'bbox'}_cls_a": float(
                    eval_results[dataset_parsing_key]["TETA"][3]
                ),
                f"{self.dataset_name}_{'mask' if self.use_mask else 'bbox'}_loc_re": float(
                    eval_results[dataset_parsing_key]["TETA"][4]
                ),
                f"{self.dataset_name}_{'mask' if self.use_mask else 'bbox'}_loc_pr": float(
                    eval_results[dataset_parsing_key]["TETA"][5]
                ),
                f"{self.dataset_name}_{'mask' if self.use_mask else 'bbox'}_assoc_re": float(
                    eval_results[dataset_parsing_key]["TETA"][6]
                ),
                f"{self.dataset_name}_{'mask' if self.use_mask else 'bbox'}_assoc_pr": float(
                    eval_results[dataset_parsing_key]["TETA"][7]
                ),
                f"{self.dataset_name}_{'mask' if self.use_mask else 'bbox'}_cls_re": float(
                    eval_results[dataset_parsing_key]["TETA"][8]
                ),
                f"{self.dataset_name}_{'mask' if self.use_mask else 'bbox'}_cls_pr": float(
                    eval_results[dataset_parsing_key]["TETA"][9]
                ),
            }

        # video-NP level results not supported for `VideoTetaEvaluator` yet
        video_np_level_results = {}
        return results, video_np_level_results


class VideoPhraseHotaEvaluator(BasePredFileEvaluator):
    """Evaluate Video Phrase HOTA with YT-VIS format prediction and GT files."""

    def __init__(
        self,
        gt_ann_file: str,
        dataset_name: str = "video",
        prob_thresh: float = 0.5,
        iou_types: Optional[Sequence[str]] = None,
        compute_video_mot_hota: bool = False,
    ):
        self.gt_ann_file = gt_ann_file
        self.dataset_name = dataset_name
        self.prob_thresh = prob_thresh
        self.metric_prefix = "phrase"
        # the list of metrics to collect from the HOTA evaluation results
        self.metric_to_collect = [
            "HOTA",
            "DetA",
            "AssA",
            "DetRe",
            "DetPr",
            "AssRe",
            "AssPr",
            "LocA",
            "OWTA",
        ]
        self.iou_types = list(iou_types) if iou_types is not None else ["bbox", "segm"]
        assert all(iou_type in ["bbox", "segm"] for iou_type in self.iou_types)

        # If True, compute video MOT HOTA, aggregating predictions/GT from all categories.
        self.compute_video_mot_hota = compute_video_mot_hota

    def evaluate(self, pred_file: str) -> Dict[str, float]:
        # use the YT-VIS evaluation toolkit in TrackEval

        with open(self.gt_ann_file) as f:
            gt = json.load(f)
        with open(pred_file) as f:
            dt = json.load(f)
        # keep only predictions with score above the probability threshold
        dt = [d for d in dt if d["score"] > self.prob_thresh]
        for d in dt:
            assert len(d["areas"]) == len(d["bboxes"])
            assert len(d["areas"]) == len(d["segmentations"])
            # remove empty boxes (otherwise they will count as false positives for during
            # per-frame detection accuracy in HOTA evaluation)
            for t in range(len(d["bboxes"])):
                bbox = d["bboxes"][t]
                if d["areas"][t] == 0 or bbox is None or all(x == 0 for x in bbox):
                    d["segmentations"][t] = None
                    d["bboxes"][t] = None
                    d["areas"][t] = None
            # check that box occurence and mask occurence are consistent
            for bbox, mask, area in zip(d["bboxes"], d["segmentations"], d["areas"]):
                assert (area is None) == (bbox is None)
                assert (area is None) == (mask is None)
            # set all scores to 1.0 for HOTA evaluation (just like Demo F1, the exact score
            # value is not used in HOTA metrics; it will be treated as a detection prediction
            # as long as its score is above the threshold)
            d["score"] = 1.0

        # remap the GT and DT annotations for phrase HOTA evaluation
        gt = _fill_in_ann_height_width(gt)
        if not self.compute_video_mot_hota:
            # remap the GT and DT annotations for phrase HOTA evaluation
            gt, dt = self._remap_gt_dt(gt, dt)
        else:
            # Compute video-level MOT HOTA
            # Apply track-level NMS
            video_groups = defaultdict(list)
            for pred in dt:
                video_groups[pred["video_id"]].append(pred)
            process_track_level_nms(video_groups, nms_threshold=0.5)
            dt = [track for tracks in video_groups.values() for track in tracks]

            # Remap GT track ids for class-agnostic HOTA
            gt, dt = remap_gt_dt_class_agnostic(gt, dt)

        # run the HOTA evaluation using TrackEval on the remapped (video_id, category_id) pairs
        out_dict = {}
        video_np_level_results = {}
        for iou_type in self.iou_types:
            output_res, _ = run_ytvis_eval(
                args=[
                    "--METRICS",
                    "HOTA",
                    "--IOU_TYPE",
                    iou_type,
                    "--DATASET_NAME",
                    self.dataset_name,
                    "--USE_PARALLEL",
                    "True",
                    "--NUM_PARALLEL_CORES",
                    "8",
                    "--PLOT_CURVES",
                    "False",
                    "--LOG_ON_ERROR",
                    "None",
                    "--PRINT_ONLY_COMBINED",
                    "True",
                    "--OUTPUT_SUMMARY",
                    "False",
                    "--OUTPUT_DETAILED",
                    "False",
                    "--TIME_PROGRESS",
                    "False",
                    "--PRINT_CONFIG",
                    "False",
                ],
                gt_json=gt,
                dt_json=dt,
            )
            self.extract_video_np_level_results(
                iou_type=iou_type,
                remapped_gt=gt,
                raw_results=output_res[self.dataset_name]["tracker"],
                video_np_level_results=video_np_level_results,
            )

            def _summarize_results(output_res, iou_type, field, suffix):
                eval_res = output_res[self.dataset_name]["tracker"][field]
                result_prefix = f"{self.dataset_name}_{'mask' if iou_type == 'segm' else 'bbox'}_{suffix}"
                for metric_name in self.metric_to_collect:
                    eval_res_hota = eval_res["cls_comb_cls_av"]["HOTA"]
                    result_key = f"{result_prefix}_{self.metric_prefix}_{metric_name}"
                    result_value = float(np.mean(eval_res_hota[metric_name]))
                    out_dict[result_key] = result_value

            _summarize_results(output_res, iou_type, "COMBINED_SEQ", "all")
            if "COMBINED_SEQ_CHALLENGING" in output_res[self.dataset_name]["tracker"]:
                _summarize_results(
                    output_res, iou_type, "COMBINED_SEQ_CHALLENGING", "challenging"
                )

        # video-NP level results not supported for `VideoPhraseHotaEvaluator` yet
        return out_dict, video_np_level_results

    def _remap_gt_dt(self, gt, dt):
        # For phrase HOTA evaluation, we need to remap each pair of (video_id, category_id) to
        # a new unique video_id, so that we don't mix detections from different categories
        gt, dt = remap_video_category_pairs_to_unique_video_ids(gt, dt)
        # We further map all the categories to category_id=1 in HOTA evaluation toolkit
        # for phrase HOTA (similar to "useCat=False" for video phrase AP)
        remapped_category_id = 1
        gt["categories"] = [
            {
                "supercategory": "object",
                "id": remapped_category_id,
                "name": "_REMAPPED_FOR_PHRASE_METRICS_",
            }
        ]
        for ann in gt["annotations"]:
            ann["category_id"] = remapped_category_id
        for d in dt:
            d["category_id"] = remapped_category_id
        # To be compatible with the TrackEval YT-VIS evaluation toolkit, we need to give
        # unique filenames to each remapped video, so we add remapped video_id as prefix.
        for video in gt["videos"]:
            new_video_id = video["id"]
            video["file_names"] = [
                f"remapped_vid_{new_video_id:012d}/{name}"
                for name in video["file_names"]
            ]
        return gt, dt

    def extract_video_np_level_results(
        self, iou_type, remapped_gt, raw_results, video_np_level_results
    ):
        """Aggregate statistics for video-level metrics."""
        result_prefix = "mask" if iou_type == "segm" else "bbox"
        for video in remapped_gt["videos"]:
            # the original video id and category id before remapping
            video_id = video["orig_video_id"]
            category_id = video["orig_category_id"]
            video_key = f"remapped_vid_{video['id']:012d}"
            results = raw_results[video_key]["_REMAPPED_FOR_PHRASE_METRICS_"]["HOTA"]

            local_results = {}
            for metric_name in self.metric_to_collect:
                result_key = f"{result_prefix}_{metric_name}"
                local_results[result_key] = float(results[metric_name].mean())
            if (video_id, category_id) not in video_np_level_results:
                video_np_level_results[(video_id, category_id)] = {}
            video_np_level_results[(video_id, category_id)].update(local_results)


class VideoClassBasedHotaEvaluator(VideoPhraseHotaEvaluator):
    def __init__(
        self,
        gt_ann_file: str,
        dataset_name: str = "video",
        prob_thresh: float = 0.5,
    ):
        super().__init__(gt_ann_file, dataset_name, prob_thresh)
        self.metric_prefix = "class"

    def _remap_gt_dt(self, gt, dt):
        return gt, dt  # no remapping needed for class-based HOTA evaluation

    def extract_video_np_level_results(self, *args, **kwargs):
        pass  # no video-NP level results for class-based HOTA evaluation


def _compress_rle(rle):
    """Convert RLEs from uncompressed (integer list) to compressed (string) format."""
    if rle is None:
        return None
    if isinstance(rle["counts"], list):
        rle = pycocotools.mask.frPyObjects(rle, rle["size"][0], rle["size"][1])
        rle["counts"] = rle["counts"].decode()
    return rle


def remap_video_category_pairs_to_unique_video_ids(
    gt_json, dt_json, add_negative_np_pairs=False
):
    """
    Remap each pair of (video_id, category_id) to a new unique video_id. This is useful
    for phrase AP and demo F1 evaluation on videos, where we have `useCat=False` and
    rely on separating different NPs (from the same video) into different new video ids,
    so that we don't mix detections from different categories in computeIoU under `useCat=False`.

    This is consistent with how do we phrase AP and demo F1 evaluation on images, where we
    use a remapped unique coco_image_id for each image-NP pair (based in its query["id"] in
    CustomCocoDetectionAPI.load_queries in modulated_detection_api.py)
    """
    # collect the unique video_id-category_id pairs
    video_id_to_video = {v["id"]: v for v in gt_json["videos"]}
    video_id_category_id_pairs = set()
    for pred in dt_json:
        video_id_category_id_pairs.add((pred["video_id"], pred["category_id"]))
    for ann in gt_json["annotations"]:
        video_id_category_id_pairs.add((ann["video_id"], ann["category_id"]))

    # assign the video_id-category_id pairs to unique video ids
    video_id_category_id_pairs = sorted(video_id_category_id_pairs)
    video_id_category_id_to_new_video_id = {
        pair: (i + 1) for i, pair in enumerate(video_id_category_id_pairs)
    }
    # also map the negative NP pairs -- this is needed for IL_MCC and CG-F1 evaluation
    if add_negative_np_pairs:
        for vnp in gt_json["video_np_pairs"]:
            pair = (vnp["video_id"], vnp["category_id"])
            if pair not in video_id_category_id_to_new_video_id:
                video_id_category_id_to_new_video_id[pair] = (
                    len(video_id_category_id_to_new_video_id) + 1
                )

    # map the "video_id" in predictions
    for pred in dt_json:
        pred["video_id"] = video_id_category_id_to_new_video_id[
            (pred["video_id"], pred["category_id"])
        ]
    # map the "video_id" in gt_json["annotations"]
    for ann in gt_json["annotations"]:
        ann["video_id"] = video_id_category_id_to_new_video_id[
            (ann["video_id"], ann["category_id"])
        ]
    # map and duplicate gt_json["videos"]
    new_videos = []
    for (
        video_id,
        category_id,
    ), new_video_id in video_id_category_id_to_new_video_id.items():
        video = video_id_to_video[video_id].copy()
        video["id"] = new_video_id
        # preserve the original video_id and category_id of each remapped video entry,
        # so that we can associate sample-level eval metrics with the original video-NP pairs
        video["orig_video_id"] = video_id
        video["orig_category_id"] = category_id
        new_videos.append(video)
    gt_json["videos"] = new_videos

    return gt_json, dt_json


def remap_gt_dt_class_agnostic(gt, dt):
    """
    For class-agnostic HOTA, merge all GT tracks for each video (across NPs),
    ensure unique track_ids, and set all category_id to 1.
    Also, add orig_video_id and orig_category_id for compatibility.
    """
    # 1. Remap all GT track_ids to be unique per video
    gt_anns_by_video = defaultdict(list)
    for ann in gt["annotations"]:
        gt_anns_by_video[ann["video_id"]].append(ann)

    # Ensure unique track ids across tracks of all videos
    next_tid = 1
    for _, anns in gt_anns_by_video.items():
        # Map old track_ids to new unique ones
        old_to_new_tid = {}
        for ann in anns:
            old_tid = ann["id"]
            if old_tid not in old_to_new_tid:
                old_to_new_tid[old_tid] = next_tid
                next_tid += 1
            ann["id"] = old_to_new_tid[old_tid]
            # Set category_id to 1 for class-agnostic
            ann["category_id"] = 1

    # Set all GT categories to a single category
    gt["categories"] = [
        {
            "supercategory": "object",
            "id": 1,
            "name": "_REMAPPED_FOR_PHRASE_METRICS_",
        }
    ]

    # Add orig_video_id and orig_category_id to each video for compatibility
    anns_by_video = defaultdict(list)
    for ann in gt["annotations"]:
        anns_by_video[ann["video_id"]].append(ann)
    for video in gt["videos"]:
        video["orig_video_id"] = video["id"]
        # Use the first annotation's original category_id if available, else None
        orig_cat = (
            anns_by_video[video["id"]][0]["category_id"]
            if anns_by_video[video["id"]]
            else None
        )
        video["orig_category_id"] = orig_cat
        video["file_names"] = [
            f"remapped_vid_{video['id']:012d}/{name}" for name in video["file_names"]
        ]

    # Set all DT category_id to 1
    for d in dt:
        d["category_id"] = 1
    return gt, dt


def _fill_in_ann_height_width(gt_json):
    """Fill in missing height/width in GT annotations from its video info."""
    video_id_to_video = {v["id"]: v for v in gt_json["videos"]}
    for ann in gt_json["annotations"]:
        if "height" not in ann or "width" not in ann:
            video = video_id_to_video[ann["video_id"]]
            if "height" not in ann:
                ann["height"] = video["height"]
            if "width" not in ann:
                ann["width"] = video["width"]

    return gt_json
