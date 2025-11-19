# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
This evaluator is based upon COCO evaluation, but evaluates the model in a "demo" setting.
This means that the model's predictions are thresholded and evaluated as "hard" predictions.
"""

import logging
from typing import Optional

import numpy as np
import pycocotools.mask as maskUtils
from pycocotools.cocoeval import COCOeval

from sam3.eval.coco_eval import CocoEvaluator
from sam3.train.masks_ops import compute_F_measure
from sam3.train.utils.distributed import is_main_process

from scipy.optimize import linear_sum_assignment


class DemoEval(COCOeval):
    """
    This evaluator is based upon COCO evaluation, but evaluates the model in a "demo" setting.
    This means that the model's predictions are thresholded and evaluated as "hard" predictions.
    """

    def __init__(
        self,
        coco_gt=None,
        coco_dt=None,
        iouType="bbox",
        threshold=0.5,
        compute_JnF=False,
    ):
        """
        Args:
            coco_gt (COCO): ground truth COCO API
            coco_dt (COCO): detections COCO API
            iou_type (str): type of IoU to evaluate
            threshold (float): threshold for predictions
        """
        super().__init__(coco_gt, coco_dt, iouType)
        self.threshold = threshold

        self.params.useCats = False
        self.params.areaRng = [[0**2, 1e5**2]]
        self.params.areaRngLbl = ["all"]
        self.params.maxDets = [100000]
        self.compute_JnF = compute_JnF

    def computeIoU(self, imgId, catId):
        # Same as the original COCOeval.computeIoU, but without sorting
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []

        if p.iouType == "segm":
            g = [g["segmentation"] for g in gt]
            d = [d["segmentation"] for d in dt]
        elif p.iouType == "bbox":
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
        else:
            raise Exception("unknown iouType for iou computation")

        # compute iou between each dt and gt region
        iscrowd = [int(o["iscrowd"]) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """
        perform evaluation for single category and image
        :return: dict (single image results)
        """
        p = self.params
        assert not p.useCats, "This evaluator does not support per-category evaluation."
        assert catId == -1
        all_gts = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
        keep_gt = np.array([not g["ignore"] for g in all_gts], dtype=bool)
        gt = [g for g in all_gts if not g["ignore"]]
        all_dts = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        keep_dt = np.array([d["score"] >= self.threshold for d in all_dts], dtype=bool)
        dt = [d for d in all_dts if d["score"] >= self.threshold]
        if len(gt) == 0 and len(dt) == 0:
            # This is a "true negative" case, where there are no GTs and no predictions
            # The box-level metrics are ill-defined, so we don't add them to this dict
            return {
                "image_id": imgId,
                "IL_TP": 0,
                "IL_TN": 1,
                "IL_FP": 0,
                "IL_FN": 0,
                "IL_perfect_neg": np.ones((len(p.iouThrs),), dtype=np.int64),
                "num_dt": len(dt),
            }

        if len(gt) > 0 and len(dt) == 0:
            # This is a "false negative" case, where there are GTs but no predictions
            return {
                "image_id": imgId,
                "IL_TP": 0,
                "IL_TN": 0,
                "IL_FP": 0,
                "IL_FN": 1,
                "TPs": np.zeros((len(p.iouThrs),), dtype=np.int64),
                "FPs": np.zeros((len(p.iouThrs),), dtype=np.int64),
                "FNs": np.ones((len(p.iouThrs),), dtype=np.int64) * len(gt),
                "local_F1s": np.zeros((len(p.iouThrs),), dtype=np.int64),
                "local_positive_F1s": np.zeros((len(p.iouThrs),), dtype=np.int64),
                "IL_perfect_pos": np.zeros((len(p.iouThrs),), dtype=np.int64),
                "num_dt": len(dt),
            }

        # Load pre-computed ious
        ious = self.ious[(imgId, catId)]

        # compute matching
        if len(ious) == 0:
            ious = np.zeros((len(dt), len(gt)))
        else:
            ious = ious[keep_dt, :][:, keep_gt]
        assert ious.shape == (len(dt), len(gt))

        matched_dt, matched_gt = linear_sum_assignment(-ious)

        match_scores = ious[matched_dt, matched_gt]

        if self.compute_JnF and len(match_scores) > 0:
            j_score = match_scores.mean()
            f_measure = 0
            for dt_id, gt_id in zip(matched_dt, matched_gt):
                f_measure += compute_F_measure(
                    gt_boundary_rle=gt[gt_id]["boundary"],
                    gt_dilated_boundary_rle=gt[gt_id]["dilated_boundary"],
                    dt_boundary_rle=dt[dt_id]["boundary"],
                    dt_dilated_boundary_rle=dt[dt_id]["dilated_boundary"],
                )
            f_measure /= len(match_scores) + 1e-9
            JnF = (j_score + f_measure) * 0.5
        else:
            j_score = f_measure = JnF = -1

        TPs, FPs, FNs = [], [], []
        IL_perfect = []
        for thresh in p.iouThrs:
            TP = (match_scores >= thresh).sum()
            FP = len(dt) - TP
            FN = len(gt) - TP
            assert (
                FP >= 0 and FN >= 0
            ), f"FP: {FP}, FN: {FN}, TP: {TP}, match_scores: {match_scores}, len(dt): {len(dt)}, len(gt): {len(gt)}, ious: {ious}"
            TPs.append(TP)
            FPs.append(FP)
            FNs.append(FN)

            if FP == FN and FP == 0:
                IL_perfect.append(1)
            else:
                IL_perfect.append(0)

        TPs = np.array(TPs, dtype=np.int64)
        FPs = np.array(FPs, dtype=np.int64)
        FNs = np.array(FNs, dtype=np.int64)
        IL_perfect = np.array(IL_perfect, dtype=np.int64)

        # compute precision recall and F1
        precision = TPs / (TPs + FPs + 1e-4)
        assert np.all(precision <= 1)
        recall = TPs / (TPs + FNs + 1e-4)
        assert np.all(recall <= 1)
        F1 = 2 * precision * recall / (precision + recall + 1e-4)

        result = {
            "image_id": imgId,
            "TPs": TPs,
            "FPs": FPs,
            "FNs": FNs,
            "local_F1s": F1,
            "IL_TP": (len(gt) > 0) and (len(dt) > 0),
            "IL_FP": (len(gt) == 0) and (len(dt) > 0),
            "IL_TN": (len(gt) == 0) and (len(dt) == 0),
            "IL_FN": (len(gt) > 0) and (len(dt) == 0),
            ("IL_perfect_pos" if len(gt) > 0 else "IL_perfect_neg"): IL_perfect,
            "F": f_measure,
            "J": j_score,
            "J&F": JnF,
            "num_dt": len(dt),
        }
        if len(gt) > 0 and len(dt) > 0:
            result["local_positive_F1s"] = F1
        return result

    def accumulate(self, p=None):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        """
        if not self.evalImgs:
            print("Please run evaluate() first")
        # allows input customized parameters
        if p is None:
            p = self.params

        setImgIds = set(p.imgIds)

        # TPs, FPs, FNs
        TPs = np.zeros((len(p.iouThrs),), dtype=np.int64)
        FPs = np.zeros((len(p.iouThrs),), dtype=np.int64)
        pmFPs = np.zeros((len(p.iouThrs),), dtype=np.int64)
        FNs = np.zeros((len(p.iouThrs),), dtype=np.int64)
        local_F1s = np.zeros((len(p.iouThrs),), dtype=np.float64)

        # Image level metrics
        IL_TPs = 0
        IL_FPs = 0
        IL_TNs = 0
        IL_FNs = 0
        IL_perfects_neg = np.zeros((len(p.iouThrs),), dtype=np.int64)
        IL_perfects_pos = np.zeros((len(p.iouThrs),), dtype=np.int64)

        # JnF metric
        total_J = 0
        total_F = 0
        total_JnF = 0

        valid_img_count = 0
        total_pos_count = 0
        total_neg_count = 0
        valid_J_count = 0
        valid_F1_count = 0
        valid_F1_count_w0dt = 0
        for res in self.evalImgs:
            if res["image_id"] not in setImgIds:
                continue
            IL_TPs += res["IL_TP"]
            IL_FPs += res["IL_FP"]
            IL_TNs += res["IL_TN"]
            IL_FNs += res["IL_FN"]
            if "IL_perfect_neg" in res:
                IL_perfects_neg += res["IL_perfect_neg"]
                total_neg_count += 1
            else:
                assert "IL_perfect_pos" in res
                IL_perfects_pos += res["IL_perfect_pos"]
                total_pos_count += 1

            if "TPs" not in res:
                continue

            TPs += res["TPs"]
            FPs += res["FPs"]
            FNs += res["FNs"]
            valid_img_count += 1

            if "local_positive_F1s" in res:
                local_F1s += res["local_positive_F1s"]
                pmFPs += res["FPs"]
                valid_F1_count_w0dt += 1
                if res["num_dt"] > 0:
                    valid_F1_count += 1

            if "J" in res and res["J"] > -1e-9:
                total_J += res["J"]
                total_F += res["F"]
                total_JnF += res["J&F"]
                valid_J_count += 1

        # compute precision recall and F1
        precision = TPs / (TPs + FPs + 1e-4)
        positive_micro_precision = TPs / (TPs + pmFPs + 1e-4)
        assert np.all(precision <= 1)
        recall = TPs / (TPs + FNs + 1e-4)
        assert np.all(recall <= 1)
        F1 = 2 * precision * recall / (precision + recall + 1e-4)
        positive_micro_F1 = (
            2
            * positive_micro_precision
            * recall
            / (positive_micro_precision + recall + 1e-4)
        )

        IL_rec = IL_TPs / (IL_TPs + IL_FNs + 1e-6)
        IL_prec = IL_TPs / (IL_TPs + IL_FPs + 1e-6)
        IL_F1 = 2 * IL_prec * IL_rec / (IL_prec + IL_rec + 1e-6)
        IL_FPR = IL_FPs / (IL_FPs + IL_TNs + 1e-6)
        IL_MCC = float(IL_TPs * IL_TNs - IL_FPs * IL_FNs) / (
            (
                float(IL_TPs + IL_FPs)
                * float(IL_TPs + IL_FNs)
                * float(IL_TNs + IL_FPs)
                * float(IL_TNs + IL_FNs)
            )
            ** 0.5
            + 1e-6
        )
        IL_perfect_pos = IL_perfects_pos / (total_pos_count + 1e-9)
        IL_perfect_neg = IL_perfects_neg / (total_neg_count + 1e-9)

        total_J = total_J / (valid_J_count + 1e-9)
        total_F = total_F / (valid_J_count + 1e-9)
        total_JnF = total_JnF / (valid_J_count + 1e-9)

        self.eval = {
            "params": p,
            "TPs": TPs,
            "FPs": FPs,
            "positive_micro_FPs": pmFPs,
            "FNs": FNs,
            "precision": precision,
            "positive_micro_precision": positive_micro_precision,
            "recall": recall,
            "F1": F1,
            "positive_micro_F1": positive_micro_F1,
            "positive_macro_F1": local_F1s / valid_F1_count,
            "positive_w0dt_macro_F1": local_F1s / valid_F1_count_w0dt,
            "IL_recall": IL_rec,
            "IL_precision": IL_prec,
            "IL_F1": IL_F1,
            "IL_FPR": IL_FPR,
            "IL_MCC": IL_MCC,
            "IL_perfect_pos": IL_perfect_pos,
            "IL_perfect_neg": IL_perfect_neg,
            "J": total_J,
            "F": total_F,
            "J&F": total_JnF,
        }
        self.eval["CGF1"] = self.eval["positive_macro_F1"] * self.eval["IL_MCC"]
        self.eval["CGF1_w0dt"] = (
            self.eval["positive_w0dt_macro_F1"] * self.eval["IL_MCC"]
        )
        self.eval["CGF1_micro"] = self.eval["positive_micro_F1"] * self.eval["IL_MCC"]

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """
        if not self.eval:
            raise Exception("Please run accumulate() first")

        def _summarize(iouThr=None, metric=""):
            p = self.params
            iStr = " {:<18} @[ IoU={:<9}] = {:0.3f}"
            titleStr = "Average " + metric
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            s = self.eval[metric]
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, iouStr, mean_s))
            return mean_s

        def _summarize_single(metric=""):
            titleStr = "Average " + metric
            iStr = " {:<35} = {:0.3f}"
            s = self.eval[metric]
            print(iStr.format(titleStr, s))
            return s

        def _summarizeDets():
            # note: the index of these metrics are also used in video Demo F1 evaluation
            # when adding new metrics, please update the index in video Demo F1 evaluation
            # in "evaluate" method of the "VideoDemoF1Evaluator" class
            stats = np.zeros((len(DEMO_METRICS),))
            stats[0] = _summarize(metric="CGF1")
            stats[1] = _summarize(metric="precision")
            stats[2] = _summarize(metric="recall")
            stats[3] = _summarize(metric="F1")
            stats[4] = _summarize(metric="positive_macro_F1")
            stats[5] = _summarize_single(metric="IL_precision")
            stats[6] = _summarize_single(metric="IL_recall")
            stats[7] = _summarize_single(metric="IL_F1")
            stats[8] = _summarize_single(metric="IL_FPR")
            stats[9] = _summarize_single(metric="IL_MCC")
            stats[10] = _summarize(metric="IL_perfect_pos")
            stats[11] = _summarize(metric="IL_perfect_neg")
            stats[12] = _summarize(iouThr=0.5, metric="CGF1")
            stats[13] = _summarize(iouThr=0.5, metric="precision")
            stats[14] = _summarize(iouThr=0.5, metric="recall")
            stats[15] = _summarize(iouThr=0.5, metric="F1")
            stats[16] = _summarize(iouThr=0.5, metric="positive_macro_F1")
            stats[17] = _summarize(iouThr=0.5, metric="IL_perfect_pos")
            stats[18] = _summarize(iouThr=0.5, metric="IL_perfect_neg")
            stats[19] = _summarize(iouThr=0.75, metric="CGF1")
            stats[20] = _summarize(iouThr=0.75, metric="precision")
            stats[21] = _summarize(iouThr=0.75, metric="recall")
            stats[22] = _summarize(iouThr=0.75, metric="F1")
            stats[23] = _summarize(iouThr=0.75, metric="positive_macro_F1")
            stats[24] = _summarize(iouThr=0.75, metric="IL_perfect_pos")
            stats[25] = _summarize(iouThr=0.75, metric="IL_perfect_neg")
            stats[26] = _summarize_single(metric="J")
            stats[27] = _summarize_single(metric="F")
            stats[28] = _summarize_single(metric="J&F")
            stats[29] = _summarize(metric="CGF1_micro")
            stats[30] = _summarize(metric="positive_micro_precision")
            stats[31] = _summarize(metric="positive_micro_F1")
            stats[32] = _summarize(iouThr=0.5, metric="CGF1_micro")
            stats[33] = _summarize(iouThr=0.5, metric="positive_micro_precision")
            stats[34] = _summarize(iouThr=0.5, metric="positive_micro_F1")
            stats[35] = _summarize(iouThr=0.75, metric="CGF1_micro")
            stats[36] = _summarize(iouThr=0.75, metric="positive_micro_precision")
            stats[37] = _summarize(iouThr=0.75, metric="positive_micro_F1")
            stats[38] = _summarize(metric="CGF1_w0dt")
            stats[39] = _summarize(metric="positive_w0dt_macro_F1")
            stats[40] = _summarize(iouThr=0.5, metric="CGF1_w0dt")
            stats[41] = _summarize(iouThr=0.5, metric="positive_w0dt_macro_F1")
            stats[42] = _summarize(iouThr=0.75, metric="CGF1_w0dt")
            stats[43] = _summarize(iouThr=0.75, metric="positive_w0dt_macro_F1")
            return stats

        summarize = _summarizeDets
        self.stats = summarize()


DEMO_METRICS = [
    "CGF1",
    "Precision",
    "Recall",
    "F1",
    "Macro_F1",
    "IL_Precision",
    "IL_Recall",
    "IL_F1",
    "IL_FPR",
    "IL_MCC",
    "IL_perfect_pos",
    "IL_perfect_neg",
    "CGF1@0.5",
    "Precision@0.5",
    "Recall@0.5",
    "F1@0.5",
    "Macro_F1@0.5",
    "IL_perfect_pos@0.5",
    "IL_perfect_neg@0.5",
    "CGF1@0.75",
    "Precision@0.75",
    "Recall@0.75",
    "F1@0.75",
    "Macro_F1@0.75",
    "IL_perfect_pos@0.75",
    "IL_perfect_neg@0.75",
    "J",
    "F",
    "J&F",
    "CGF1_micro",
    "positive_micro_Precision",
    "positive_micro_F1",
    "CGF1_micro@0.5",
    "positive_micro_Precision@0.5",
    "positive_micro_F1@0.5",
    "CGF1_micro@0.75",
    "positive_micro_Precision@0.75",
    "positive_micro_F1@0.75",
    "CGF1_w0dt",
    "positive_w0dt_macro_F1",
    "CGF1_w0dt@0.5",
    "positive_w0dt_macro_F1@0.5",
    "CGF1_w0dt@0.75",
    "positive_w0dt_macro_F1@0.75",
]


class DemoEvaluator(CocoEvaluator):
    def __init__(
        self,
        coco_gt,
        iou_types,
        dump_dir: Optional[str],
        postprocessor,
        threshold=0.5,
        average_by_rarity=False,
        gather_pred_via_filesys=False,
        exhaustive_only=False,
        all_exhaustive_only=True,
        compute_JnF=False,
        metrics_dump_dir: Optional[str] = None,
    ):
        self.iou_types = iou_types
        self.threshold = threshold
        super().__init__(
            coco_gt=coco_gt,
            iou_types=iou_types,
            useCats=False,
            dump_dir=dump_dir,
            postprocessor=postprocessor,
            # average_by_rarity=average_by_rarity,
            gather_pred_via_filesys=gather_pred_via_filesys,
            exhaustive_only=exhaustive_only,
            all_exhaustive_only=all_exhaustive_only,
            metrics_dump_dir=metrics_dump_dir,
        )

        self.use_self_evaluate = True
        self.compute_JnF = compute_JnF

    def _lazy_init(self):
        if self.initialized:
            return
        super()._lazy_init()
        self.use_self_evaluate = True
        self.reset()

    def select_best_scoring(self, scorings):
        # This function is used for "oracle" type evaluation.
        # It accepts the evaluation results with respect to several ground truths, and picks the best
        if len(scorings) == 1:
            return scorings[0]

        assert (
            scorings[0].ndim == 3
        ), f"Expecting results in [numCats, numAreas, numImgs] format, got {scorings[0].shape}"
        assert (
            scorings[0].shape[0] == 1
        ), f"Expecting a single category, got {scorings[0].shape[0]}"

        for scoring in scorings:
            assert (
                scoring.shape == scorings[0].shape
            ), f"Shape mismatch: {scoring.shape}, {scorings[0].shape}"

        selected_imgs = []
        for img_id in range(scorings[0].shape[-1]):
            best = scorings[0][:, :, img_id]

            for scoring in scorings[1:]:
                current = scoring[:, :, img_id]
                if "local_F1s" in best[0, 0] and "local_F1s" in current[0, 0]:
                    # we were able to compute a F1 score for this particular image in both evaluations
                    # best["local_F1s"] contains the results at various IoU thresholds. We simply take the average for comparision
                    best_score = best[0, 0]["local_F1s"].mean()
                    current_score = current[0, 0]["local_F1s"].mean()
                    if current_score > best_score:
                        best = current

                else:
                    # If we're here, it means that in that in some evaluation we were not able to get a valid local F1
                    # This happens when both the predictions and targets are empty. In that case, we can assume it's a perfect prediction
                    if "local_F1s" not in current[0, 0]:
                        best = current
            selected_imgs.append(best)
        result = np.stack(selected_imgs, axis=-1)
        assert result.shape == scorings[0].shape
        return result

    def summarize(self):
        self._lazy_init()
        logging.info("Demo evaluator: Summarizing")
        if not is_main_process():
            return {}
        outs = {}
        prefix = "oracle_" if len(self.coco_evals) > 1 else ""
        # if self.rarity_buckets is None:
        self.accumulate(self.eval_img_ids)
        for iou_type, coco_eval in self.coco_evals[0].items():
            print("Demo metric, IoU type={}".format(iou_type))
            coco_eval.summarize()

        if "bbox" in self.coco_evals[0]:
            for i, value in enumerate(self.coco_evals[0]["bbox"].stats):
                outs[f"coco_eval_bbox_{prefix}{DEMO_METRICS[i]}"] = value
        if "segm" in self.coco_evals[0]:
            for i, value in enumerate(self.coco_evals[0]["segm"].stats):
                outs[f"coco_eval_masks_{prefix}{DEMO_METRICS[i]}"] = value
        # else:
        #     total_stats = {}
        #     for bucket, img_list in self.rarity_buckets.items():
        #         self.accumulate(imgIds=img_list)
        #         bucket_name = RARITY_BUCKETS[bucket]
        #         for iou_type, coco_eval in self.coco_evals[0].items():
        #             print(
        #                 "Demo metric, IoU type={}, Rarity bucket={}".format(
        #                     iou_type, bucket_name
        #                 )
        #             )
        #             coco_eval.summarize()

        #         if "bbox" in self.coco_evals[0]:
        #             if "bbox" not in total_stats:
        #                 total_stats["bbox"] = np.zeros_like(
        #                     self.coco_evals[0]["bbox"].stats
        #                 )
        #             total_stats["bbox"] += self.coco_evals[0]["bbox"].stats
        #             for i, value in enumerate(self.coco_evals[0]["bbox"].stats):
        #                 outs[
        #                     f"coco_eval_bbox_{bucket_name}_{prefix}{DEMO_METRICS[i]}"
        #                 ] = value
        #         if "segm" in self.coco_evals[0]:
        #             if "segm" not in total_stats:
        #                 total_stats["segm"] = np.zeros_like(
        #                     self.coco_evals[0]["segm"].stats
        #                 )
        #             total_stats["segm"] += self.coco_evals[0]["segm"].stats
        #             for i, value in enumerate(self.coco_evals[0]["segm"].stats):
        #                 outs[
        #                     f"coco_eval_masks_{bucket_name}_{prefix}{DEMO_METRICS[i]}"
        #                 ] = value

        #     if "bbox" in total_stats:
        #         total_stats["bbox"] /= len(self.rarity_buckets)
        #         for i, value in enumerate(total_stats["bbox"]):
        #             outs[f"coco_eval_bbox_{prefix}{DEMO_METRICS[i]}"] = value
        #     if "segm" in total_stats:
        #         total_stats["segm"] /= len(self.rarity_buckets)
        #         for i, value in enumerate(total_stats["segm"]):
        #             outs[f"coco_eval_masks_{prefix}{DEMO_METRICS[i]}"] = value

        return outs

    def accumulate(self, imgIds=None):
        self._lazy_init()
        logging.info(
            f"demo evaluator: Accumulating on {len(imgIds) if imgIds is not None else 'all'} images"
        )
        if not is_main_process():
            return

        if imgIds is not None:
            for coco_eval in self.coco_evals[0].values():
                coco_eval.params.imgIds = list(imgIds)

        for coco_eval in self.coco_evals[0].values():
            coco_eval.accumulate()

    def reset(self):
        self.coco_evals = [{} for _ in range(len(self.coco_gts))]
        for i, coco_gt in enumerate(self.coco_gts):
            for iou_type in self.iou_types:
                self.coco_evals[i][iou_type] = DemoEval(
                    coco_gt=coco_gt,
                    iouType=iou_type,
                    threshold=self.threshold,
                    compute_JnF=self.compute_JnF,
                )
                self.coco_evals[i][iou_type].useCats = False
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}
        if self.dump is not None:
            self.dump = []
