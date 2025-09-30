import json
import logging
import os
import tempfile
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pycocotools.mask
from tqdm import tqdm

from .conversion_utils.convert_ytbvis_to_cocovid_gt import convert_ytbvis_to_cocovid_gt
from .conversion_utils.convert_ytbvis_to_cocovid_pred import (
    convert_ytbvis_to_cocovid_pred,
)
from .evaluators_utils.nms_helper import (
    process_frame_level_nms,
    process_track_level_nms,
)


class BasePredFileEvaluator:
    """A base class for evaluating a prediction file."""

    pass


class YTVISPredFileEvaluator(BasePredFileEvaluator):
    """Evaluate class mAP for YT-VIS prediction files."""

    def __init__(
        self,
        gt_ann_file: str,
        dataset_name: str = "ytvis",
        iou_types: Optional[Sequence[str]] = None,
    ):
        self.gt_ann_file = gt_ann_file
        self.dataset_name = dataset_name
        self.iou_types = list(iou_types) if iou_types is not None else ["bbox", "segm"]
        assert all(iou_type in ["bbox", "segm"] for iou_type in self.iou_types)

    def evaluate(self, pred_file: str) -> Dict[str, float]:
        # use our internal video evaluation toolkit for YT-VIS pred file
        # (i.e. the same one we're using for video phrase AP)
        from onevision.data.datasets.ytvis_coco_wrapper import YTVIS
        from onevision.data.datasets.ytvis_eval import YTVISeval

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
        from onevision.data.datasets.ytvis_coco_wrapper import YTVIS
        from onevision.data.datasets.ytvis_eval import YTVISeval as VideoPhraseApEval

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
            phraseApEval = VideoPhraseApEval(ytvisGT, ytvisDT, iou_type)

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


class VideoDemoF1Evaluator(BasePredFileEvaluator):
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
        from onevision.data.datasets.demo_eval import DEMO_METRICS
        from onevision.data.datasets.ytvis_coco_wrapper import YTVIS
        from onevision.data.datasets.ytvis_eval import VideoDemoF1Eval

        with open(self.gt_ann_file) as f:
            gt = json.load(f)
        with open(pred_file) as f:
            dt = json.load(f)
        # compute IL_MCC and CG-F1 can only be computed if we have "video_np_pairs" keys in the GT JSON
        compute_ilmcc_and_cfg1 = "video_np_pairs" in gt
        if not compute_ilmcc_and_cfg1:
            print(
                f"Warning: IL_MCC and CG-F1 are not computed for {pred_file=} as it does not have 'video_np_pairs' keys in the GT JSON"
            )
        # For phrase AP and demo F1 evaluation, we need to remap each pair of (video_id, category_id) to
        # a new unique video_id, so that we don't mix detections from different categories under `useCat=False`
        gt, dt = remap_video_category_pairs_to_unique_video_ids(
            gt, dt, add_negative_np_pairs=compute_ilmcc_and_cfg1
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
            # Note that these indices must be updated if the number order of metrics in
            # `_summarizeDets` in projects/onevision/data/datasets/demo_eval.py
            # check the length and name of the metrics to guard against any changes in
            # image Demo F1 evaluation as in `_summarizeDets` of DemoEval
            demo_pr_avg_idx = 1  # IoU=0.5:0.95
            demo_re_avg_idx = 2  # IoU=0.5:0.95
            demo_f1_avg_idx = 3  # IoU=0.5:0.95
            pmf1_avg_idx = 4  # IoU=0.5:0.95
            ilmcc_avg_idx = 9  # IoU=0.5:0.95
            cfg1_avg_idx = 0  # IoU=0.5:0.95
            demo_pr_iou_50_idx = 13  # IoU=0.5
            demo_re_iou_50_idx = 14  # IoU=0.5
            demo_f1_iou_50_idx = 15  # IoU=0.5
            pmf1_iou_50_idx = 16  # IoU=0.5
            # ilmcc_iou_50_idx = None  # IoU=0.5  # (not available in current DemoEval)
            cfg1_iou_50_idx = 12  # IoU=0.5
            demo_pr_iou_75_idx = 20  # IoU=0.75
            demo_re_iou_75_idx = 21  # IoU=0.75
            demo_f1_iou_75_idx = 22  # IoU=0.75
            pmf1_iou_75_idx = 23  # IoU=0.75
            # ilmcc_iou_75_idx = None  # IoU=0.75  # (not available in current DemoEval)
            cfg1_iou_75_idx = 19  # IoU=0.75
            stats = demoF1Eval.stats
            assert len(stats) == len(DEMO_METRICS)
            assert DEMO_METRICS[demo_pr_avg_idx] == "Precision"
            assert DEMO_METRICS[demo_re_avg_idx] == "Recall"
            assert DEMO_METRICS[demo_f1_avg_idx] == "F1"
            assert DEMO_METRICS[pmf1_avg_idx] == "Macro_F1"
            assert DEMO_METRICS[ilmcc_avg_idx] == "IL_MCC"
            assert DEMO_METRICS[cfg1_avg_idx] == "CGF1"
            assert DEMO_METRICS[demo_pr_iou_50_idx] == "Precision@0.5"
            assert DEMO_METRICS[demo_re_iou_50_idx] == "Recall@0.5"
            assert DEMO_METRICS[demo_f1_iou_50_idx] == "F1@0.5"
            assert DEMO_METRICS[pmf1_iou_50_idx] == "Macro_F1@0.5"
            # assert DEMO_METRICS[ilmcc_iou_50_idx] == "IL_MCC@0.5"
            assert DEMO_METRICS[cfg1_iou_50_idx] == "CGF1@0.5"
            assert DEMO_METRICS[demo_pr_iou_75_idx] == "Precision@0.75"
            assert DEMO_METRICS[demo_re_iou_75_idx] == "Recall@0.75"
            assert DEMO_METRICS[demo_f1_iou_75_idx] == "F1@0.75"
            assert DEMO_METRICS[pmf1_iou_75_idx] == "Macro_F1@0.75"
            # assert DEMO_METRICS[ilmcc_iou_75_idx] == "IL_MCC@0.75"
            assert DEMO_METRICS[cfg1_iou_75_idx] == "CGF1@0.75"
            # fetch Demo F1 results from the corresponding indices in `demoF1Eval.stats`
            # (see `_summarizeDets` in projects/onevision/data/datasets/demo_eval.py)
            results[result_prefix + "_precision_50_95"] = stats[demo_pr_avg_idx]
            results[result_prefix + "_recall_50_95"] = stats[demo_re_avg_idx]
            results[result_prefix + "_f1_50_95"] = stats[demo_f1_avg_idx]
            results[result_prefix + "_precision_50"] = stats[demo_pr_iou_50_idx]
            results[result_prefix + "_recall_50"] = stats[demo_re_iou_50_idx]
            results[result_prefix + "_f1_50"] = stats[demo_f1_iou_50_idx]
            results[result_prefix + "_cfg1_50"] = stats[cfg1_iou_50_idx]
            results[result_prefix + "_precision_75"] = stats[demo_pr_iou_75_idx]
            results[result_prefix + "_recall_75"] = stats[demo_re_iou_75_idx]
            results[result_prefix + "_f1_75"] = stats[demo_f1_iou_75_idx]
            if compute_ilmcc_and_cfg1:
                results[result_prefix + "_pmf1_50_95"] = stats[pmf1_avg_idx]
                results[result_prefix + "_ilmcc_50_95"] = stats[ilmcc_avg_idx]
                results[result_prefix + "_cfg1_50_95"] = stats[cfg1_avg_idx]
                results[result_prefix + "_pmf1_50"] = stats[pmf1_iou_50_idx]
                results[result_prefix + "_ilmcc_50"] = float(
                    np.array(stats[cfg1_iou_50_idx]) / np.array(stats[pmf1_iou_50_idx])
                )  # IL_MCC not directly available in DemoEval, so we compute it from CFG1 and PMF1
                results[result_prefix + "_pmf1_75"] = stats[pmf1_iou_75_idx]
                results[result_prefix + "_ilmcc_75"] = float(
                    np.array(stats[cfg1_iou_75_idx]) / np.array(stats[pmf1_iou_75_idx])
                )  # IL_MCC not directly available in DemoEval, so we compute it from CFG1 and PMF1
                results[result_prefix + "_cfg1_75"] = stats[cfg1_iou_75_idx]

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
        from .teta import config, Evaluator, metrics
        from .teta.datasets import COCO, TAO

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
        from onevision.data.sam3_video_evaluators.taoow_eval_toolkit.run_ytvis_eval import (
            run_ytvis_eval,
        )

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


class LVVISPredFileEvaluator(BasePredFileEvaluator):
    """Evaluator for LVVIS prediction files."""

    def __init__(self, gt_ann_file: str, dataset_name: str = "lvvis"):
        self.gt_ann_file = gt_ann_file
        self.dataset_name = dataset_name

    def evaluate(self, pred_file: str) -> Dict[str, float]:
        # use the official evaluation toolkit for LVVIS pred file
        # copied from to https://github.com/haochenheheda/LVVIS to `lvvis_eval_toolkit`
        # note that it ONLY supports mask evaluation (bbox evaluation is not supported)
        from onevision.data.sam3_video_evaluators.lvvis_eval_toolkit.lvvis import LVVIS
        from onevision.data.sam3_video_evaluators.lvvis_eval_toolkit.lvviseval import (
            LVVISeval,
        )

        lvvisGT = LVVIS(self.gt_ann_file)
        lvvisDT = lvvisGT.loadRes(pred_file)

        lvvisEval = LVVISeval(lvvisGT, lvvisDT, "segm")
        lvvisEval.evaluate()
        lvvisEval.accumulate()
        lvvisEval.summarize()
        mask_mAP_50_95 = lvvisEval.stats[0]
        mask_mAP_50_95_base = lvvisEval.stats[1]
        mask_mAP_50_95_novel = lvvisEval.stats[2]

        out_dict = {
            f"{self.dataset_name}_mask_mAP_50_95": mask_mAP_50_95,
            f"{self.dataset_name}_mask_mAP_50_95_base": mask_mAP_50_95_base,
            f"{self.dataset_name}_mask_mAP_50_95_novel": mask_mAP_50_95_novel,
        }
        # video-NP level results not supported for `LVVISPredFileEvaluator` yet
        video_np_level_results = {}
        return out_dict, video_np_level_results


class BURSTPredFileEvaluator(BasePredFileEvaluator):
    """Evaluator for BURST prediction files."""

    def __init__(
        self,
        gt_ann_file: str,
        dataset_name: str = "burst_vis",
        prob_thresh: Optional[float] = None,
        tasks: Sequence[str] = ("class_guided",),  # "class_guided" or "open_world"
        converted_pred_file_suffix: str = ".burst_vis_format.json",
        remove_mask_overlap: bool = False,
    ):
        self.gt_ann_file = gt_ann_file
        self.dataset_name = dataset_name
        self.prob_thresh = prob_thresh
        assert all(task in ("class_guided", "open_world") for task in tasks)
        self.tasks = tasks
        self.converted_pred_file_suffix = converted_pred_file_suffix
        self.remove_mask_overlap = remove_mask_overlap

    def evaluate(self, pred_file: str) -> Dict[str, float]:
        # use the official evaluation toolkit for BURST pred file
        from onevision.data.sam3_video_evaluators.burst_eval_toolkit.run_burst_eval import (
            run_burst_eval,
        )

        logging.info("Converting prediction file to BURST format")
        converted_pred_file = self._convert_pred_file_to_burst_vis_format(pred_file)

        out_dict = {}
        for task in self.tasks:
            logging.info(f"Evaluating BURST-VIS task: {task}")
            task_out_dict = run_burst_eval(
                [
                    "--gt",
                    os.path.dirname(self.gt_ann_file),
                    "--pred",
                    converted_pred_file,
                    "--task",
                    task,
                    "--nprocs",
                    "1",  # use a single process to avoid CPU OOM errors
                ]
            )
            for metric_name, results in task_out_dict.items():
                for k, v in results.items():
                    out_dict[f"{self.dataset_name}_{task}_{metric_name}_{k}"] = v

        # video-NP level results not supported for `BURSTPredFileEvaluator` yet
        video_np_level_results = {}
        return out_dict, video_np_level_results

    def _convert_pred_file_to_burst_vis_format(self, pred_file: str) -> str:
        """
        Convert a prediction file (saved in YT-VIS format) to BURST format. Return the
        path to the converted prediction file.
        """

        # We start to build the BURST format output by loading the GT annotations
        # and then populate it with the converted predictions, since the BURST pred
        # file format is the same as the BURST GT file format. (See BURST format at
        # https://github.com/Ali2500/BURST-benchmark/blob/main/ANNOTATION_FORMAT.md)
        with open(self.gt_ann_file) as f:
            dt_json_burst_format = json.load(f)
        # clear the GT object annotations (to replace them with predicted ones later)
        for seq in dt_json_burst_format["sequences"]:
            del seq["track_category_ids"]
            del seq["segmentations"]
        # record the video dimensions and length for each video
        sequences = dt_json_burst_format["sequences"]
        video_id_to_height = {v["id"]: v["height"] for v in sequences}
        video_id_to_width = {v["id"]: v["width"] for v in sequences}
        video_id_to_length = {
            v["id"]: len(v["annotated_image_paths"]) for v in sequences
        }

        # Load the raw prediction file and organize predictions by video id.
        with open(pred_file) as f:
            dt_json_ytvis_format = json.load(f)
        preds_by_video_id = defaultdict(list)
        for pred in tqdm(dt_json_ytvis_format, desc="Converting to BURST format"):
            if self.prob_thresh is not None and pred["score"] <= self.prob_thresh:
                continue
            video_id = pred["video_id"]
            bboxes = pred["bboxes"]
            segmentations = pred["segmentations"]
            areas = pred["areas"]
            length = len(pred["bboxes"])
            assert length == video_id_to_length[video_id]
            # replace empty bbox, mask or area with None
            for t in range(length):
                if areas[t] == 0:
                    bboxes[t] = None
                    segmentations[t] = None
                    areas[t] = None
            preds_by_video_id[video_id].append(pred)

        # Potentially remove the overlap between masklets in each video
        if self.remove_mask_overlap:
            for video_id, preds_this_video in tqdm(
                preds_by_video_id.items(), desc="Removing mask overlap"
            ):
                self._remove_mask_overlap(
                    preds_this_video=preds_this_video,
                    height=video_id_to_height[video_id],
                    width=video_id_to_width[video_id],
                    length=video_id_to_length[video_id],
                )

        # Then, populate the BURST format output with the converted predictions
        for seq in sequences:
            video_id = seq["id"]
            preds_this_video = preds_by_video_id[video_id]
            length = video_id_to_length[video_id]

            # use (n_pred+1) as the track id key within this video
            track_category_ids = {
                (n_pred + 1): pred["category_id"]
                for n_pred, pred in enumerate(preds_this_video)
            }
            segmentations = [{} for _ in range(length)]
            for n_pred, pred in enumerate(preds_this_video):
                for t, segmentation in enumerate(pred["segmentations"]):
                    if segmentation is None:
                        continue
                    segmentations[t][n_pred + 1] = {
                        "rle": segmentation["counts"],
                        "score": pred["score"],  # use sequence-level score
                        "is_gt": False,
                    }

            seq["track_category_ids"] = track_category_ids
            seq["segmentations"] = segmentations

        # Finally, save the converted prediction file
        converted_pred_file = pred_file + self.converted_pred_file_suffix
        with open(converted_pred_file, "w") as f:
            json.dump(dt_json_burst_format, f)
        return converted_pred_file

    def _remove_mask_overlap(self, preds_this_video, height, width, length):
        """
        Remove overlaps between masklets in each video based on their scores:
        (high-scoring masklets fall on top of low-scoring ones).
        """
        sorted_preds = sorted(preds_this_video, key=lambda x: x["score"])
        pixel_obj_inds = np.zeros((height, width), dtype=np.int32)
        for t in range(length):
            pixel_obj_inds[...] = -1
            # write object indices to frame pixels, going from low-scoring predictions
            # to high-scoring predictions, so that high-scoring ones fall on top and
            # overwrite the low-scoring ones
            for n_pred, pred in enumerate(sorted_preds):
                obj_mask_rle = pred["segmentations"][t]
                if obj_mask_rle is not None:
                    obj_mask = pycocotools.mask.decode(obj_mask_rle) > 0
                    pixel_obj_inds[obj_mask] = n_pred

            # read out the final object index for all frame pixels and re-encode the mask
            for n_pred, pred in enumerate(sorted_preds):
                obj_mask = pixel_obj_inds == n_pred
                area = int(np.sum(obj_mask))
                if area > 0:
                    # re-encode the new mask
                    obj_mask_uint8 = np.array(obj_mask, dtype=np.uint8, order="F")
                    obj_mask_rle = pycocotools.mask.encode(obj_mask_uint8)
                    obj_mask_rle["counts"] = obj_mask_rle["counts"].decode()
                    pred["segmentations"][t] = obj_mask_rle
                    # we keep the bounding box as-is
                    pred["areas"][t] = area
                else:
                    # simply set empty mask to None
                    pred["segmentations"][t] = None
                    pred["bboxes"][t] = None
                    pred["areas"][t] = None


class TAOOWPredFileEvaluator(BasePredFileEvaluator):
    """Evaluator for TAO-OW prediction files."""

    def __init__(
        self,
        gt_dir: str,
        dataset_name: str = "taoow",
        prob_thresh: Optional[float] = None,
        subsets: Sequence[str] = ("all", "known", "unknown"),
        converted_pred_dir_suffix: str = ".taoow_format",
    ):
        self.gt_dir = gt_dir
        self.gt_ann_file = os.path.join(gt_dir, "gt.json")
        assert os.path.isfile(self.gt_ann_file), f"GT file {self.gt_ann_file} not found"
        self.dataset_name = dataset_name
        self.prob_thresh = prob_thresh
        assert all(subset in ("all", "known", "unknown") for subset in subsets)
        self.subsets = subsets
        self.converted_pred_dir_suffix = converted_pred_dir_suffix

    def evaluate(self, pred_file: str) -> Dict[str, float]:
        # use the official evaluation toolkit for TAO-OW pred file
        from onevision.data.sam3_video_evaluators.taoow_eval_toolkit.run_taoow_eval import (
            run_taoow_eval,
        )

        logging.info("Converting prediction file to TAO-OW format")
        converted_pred_dir = self._convert_pred_file_to_taoow_format(pred_file)

        out_dict = {}
        for subset in self.subsets:
            logging.info(f"Evaluating TAO-OW subset: {subset}")
            output_res, _ = run_taoow_eval(
                [
                    "--USE_PARALLEL",
                    "False",
                    "--METRICS",
                    "HOTA",
                    "--GT_FOLDER",
                    self.gt_dir,
                    "--TRACKERS_FOLDER",
                    converted_pred_dir,
                    "--TRACKERS_TO_EVAL",
                    "outputs",
                    "--SUBSET",
                    subset,
                ]
            )
            eval_res = output_res["TAO_OW"]["outputs"]["COMBINED_SEQ"]
            OWTA = float(np.mean(eval_res["cls_comb_cls_av"]["HOTA"]["OWTA"]))
            out_dict[f"{self.dataset_name}_OWTA_{subset}"] = OWTA

        # video-NP level results not supported for `TAOOWPredFileEvaluator` yet
        video_np_level_results = {}
        return out_dict, video_np_level_results

    def _convert_pred_file_to_taoow_format(self, pred_file: str) -> str:
        """
        Convert a prediction file (saved in YT-VIS format) to TAO-OW format. Return the
        path to the converted prediction dir.
        """
        # Load the prediction and GT files
        with open(pred_file) as f:
            pred_ytvis_format = json.load(f)
        with open(self.gt_ann_file) as f:
            gt_taoow_format = json.load(f)

        # Get the image ids for each video
        image_ids_per_video = defaultdict(list)
        for image_entry in gt_taoow_format["images"]:
            image_ids_per_video[image_entry["video_id"]].append(image_entry["id"])
        for v in image_ids_per_video.values():
            v.sort()

        # Then, populate a list of predictions in TAO-OW format as in
        # https://github.com/YangLiu14/Open-World-Tracking/blob/060dd193127ac1f2ae8a580ce4382d152c35e0fc/tools/track_results_conversion.py#L220-L226
        results = []
        track_id = 0
        for pred_entry in pred_ytvis_format:
            if self.prob_thresh is not None and pred_entry["score"] <= self.prob_thresh:
                continue

            track_id += 1
            video_id = pred_entry["video_id"]
            category_id = pred_entry["category_id"]
            score = pred_entry["score"]
            pred_box_list = pred_entry["bboxes"]
            pred_image_ids = image_ids_per_video[video_id]
            assert len(pred_box_list) == len(pred_image_ids)
            for bbox, image_id in zip(pred_box_list, pred_image_ids):
                if bbox is None or all(x == 0 for x in bbox):
                    continue  # skip empty box
                entry = {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": score,
                    "track_id": track_id,
                    "video_id": video_id,
                }
                results.append(entry)

        # Finally, save the converted prediction file
        converted_pred_dir = pred_file + self.converted_pred_dir_suffix
        converted_pred_file = os.path.join(
            converted_pred_dir, "outputs", "data", "tracking.json"
        )
        os.makedirs(os.path.dirname(converted_pred_file), exist_ok=True)
        with open(converted_pred_file, "w") as f:
            json.dump(results, f)
        return converted_pred_dir


class DancetrackPredFileEvaluator(BasePredFileEvaluator):
    """Evaluator for Dancetrack prediction files."""

    def __init__(
        self,
        dataset_base_dir: str,
        dataset_name: str = "dancetrack",
        prob_thresh: Optional[float] = 0.6,  # 0.6 seems to work well for Dancetrack
        use_per_frame_scores: bool = False,  # video-level scores seem to work better
        converted_pred_dir_suffix: str = ".dancetrack_format",
        split: str = "val",
    ):
        self.dataset_base_dir = dataset_base_dir
        self.split = split
        self.gt_ann_file = os.path.join(
            dataset_base_dir, "annotations", split, "instances_without_mask.json"
        )
        self.gt_dir = os.path.join(dataset_base_dir, "frames", split)
        self.gt_seqmap_file = os.path.join(
            dataset_base_dir, "frames", f"{split}_seqmap.txt"
        )
        assert os.path.isfile(self.gt_ann_file), f"GT file {self.gt_ann_file} not found"
        assert os.path.isfile(
            self.gt_seqmap_file
        ), f"GT file {self.gt_seqmap_file} not found"
        self.dataset_name = dataset_name
        self.prob_thresh = prob_thresh
        self.use_per_frame_scores = use_per_frame_scores
        self.converted_pred_dir_suffix = converted_pred_dir_suffix

    def evaluate(self, pred_file: str) -> Dict[str, float]:
        # use the official evaluation toolkit for TAO-OW pred file
        from onevision.data.sam3_video_evaluators.dancetrack_eval_toolkit.run_dancetrack_eval import (
            run_dancetrack_eval,
        )

        logging.info("Converting prediction file to Dancetrack format")
        converted_pred_dir = self._convert_pred_file_to_dancetrack_format(pred_file)

        out_dict = {}
        if self.split == "test":
            # Skip the evaluation on the test set (where we don't have GTs)
            logging.info(
                f"Skipping the evaluation on test set. Please submit the predictions in {converted_pred_dir} "
                "to the DanceTrack evaluation server (https://codalab.lisn.upsaclay.fr/competitions/5830)."
            )
            return out_dict

        logging.info(f"Evaluating Dancetrack on {self.split} split")
        output_res, _ = run_dancetrack_eval(
            [
                "--SPLIT_TO_EVAL",
                self.split,
                "--METRICS",
                "HOTA",
                "--GT_FOLDER",
                self.gt_dir,
                "--SEQMAP_FILE",
                self.gt_seqmap_file,
                "--SKIP_SPLIT_FOL",
                "True",
                "--TRACKERS_TO_EVAL",
                "tracker",
                "--TRACKER_SUB_FOLDER",
                "",
                "--USE_PARALLEL",
                "True",
                "--NUM_PARALLEL_CORES",
                "8",
                "--PLOT_CURVES",
                "False",
                "--TRACKERS_FOLDER",
                converted_pred_dir,
            ]
        )
        eval_res = output_res["MotChallenge2DBox"]["tracker"]["COMBINED_SEQ"]
        HOTA = float(np.mean(eval_res["pedestrian"]["HOTA"]["HOTA"]))
        DetA = float(np.mean(eval_res["pedestrian"]["HOTA"]["DetA"]))
        AssA = float(np.mean(eval_res["pedestrian"]["HOTA"]["AssA"]))
        out_dict[f"{self.dataset_name}_HOTA"] = HOTA
        out_dict[f"{self.dataset_name}_DetA"] = DetA
        out_dict[f"{self.dataset_name}_AssA"] = AssA

        # video-NP level results not supported for `DancetrackPredFileEvaluator` yet
        video_np_level_results = {}
        return out_dict, video_np_level_results

    def _convert_pred_file_to_dancetrack_format(self, pred_file: str) -> str:
        """
        Convert a prediction file (saved in YT-VIS format) to Dancetrack format.
        Return the path to the converted prediction dir.
        """
        # We only need the video information from the GT annotations
        with open(self.gt_ann_file) as f:
            gt_ytvis_format = json.load(f)
        video_id_to_info = {v["id"]: v for v in gt_ytvis_format["videos"]}
        del gt_ytvis_format

        # Load predictions and organize by video id
        with open(pred_file) as f:
            pred_ytvis_format = json.load(f)
        preds_by_video_id = defaultdict(list)
        for pred in pred_ytvis_format:
            preds_by_video_id[pred["video_id"]].append(pred)

        # Save the outputs in Dancetrack format, see
        # https://github.com/DanceTrack/DanceTrack?tab=readme-ov-file#evaluation
        for video_id, preds_this_video in preds_by_video_id.items():
            video_info = video_id_to_info[video_id]
            file_names = video_info["file_names"]
            _, video_name, _, _ = file_names[0].split("/")
            frame_ids = [
                int(os.path.splitext(os.path.basename(p))[0]) for p in file_names
            ]

            output_lines = []
            obj_id = 0
            for pred in preds_this_video:
                assert len(pred["bboxes"]) == video_info["length"]
                has_pred_above_thresh = False
                for frame_idx, bbox in enumerate(pred["bboxes"]):
                    if bbox is None or all(x == 0 for x in bbox):
                        continue  # skip empty box
                    score = (
                        pred["per_frame_scores"][frame_idx]
                        if self.use_per_frame_scores
                        else pred["score"]
                    )
                    if self.prob_thresh is not None and score <= self.prob_thresh:
                        continue

                    has_pred_above_thresh = True
                    x, y, w, h = bbox
                    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
                    output_line = f"{frame_ids[frame_idx]},{obj_id},{x},{y},{w},{h},{score},-1,-1,-1\n"
                    output_lines.append(output_line)

                obj_id += has_pred_above_thresh

            converted_pred_dir = pred_file + self.converted_pred_dir_suffix
            # note that the output folder name has to be "tracker" for the evaluation server
            # (see https://github.com/DanceTrack/DanceTrack?tab=readme-ov-file#competition)
            converted_pred_file = os.path.join(
                converted_pred_dir, "tracker", f"{video_name}.txt"
            )
            os.makedirs(os.path.dirname(converted_pred_file), exist_ok=True)
            with open(converted_pred_file, "w") as f:
                f.writelines(output_lines)

        return converted_pred_dir


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