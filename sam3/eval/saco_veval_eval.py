# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
import argparse
import json
import os
from collections import defaultdict

from iopath.common.file_io import g_pathmgr
from sam3.eval.saco_veval_evaluators import (
    VideoCGF1Evaluator,
    VideoPhraseApEvaluator,
    VideoPhraseHotaEvaluator,
    VideoTetaEvaluator,
    YTVISPredFileEvaluator,
)


class VEvalEvaluator:
    def __init__(self, gt_annot_file: str, eval_res_file: str):
        self.gt_annot_file = gt_annot_file
        self.eval_res_file = eval_res_file
        self.evaluators = [
            # mAP
            YTVISPredFileEvaluator(gt_annot_file),
            # Phrase AP
            VideoPhraseApEvaluator(gt_annot_file),
            # TETA
            VideoTetaEvaluator(gt_annot_file, use_mask=True, is_exhaustive=True),
            # HOTA
            VideoPhraseHotaEvaluator(gt_annot_file),
            # cgF1
            VideoCGF1Evaluator(gt_annot_file),
        ]

    def run_eval(self, pred_file: str):
        dataset_results = {}
        video_np_results = defaultdict(dict)
        for evaluator in self.evaluators:
            d_res, v_np_res = evaluator.evaluate(pred_file)
            dataset_results.update(d_res)
            for (video_id, category_id), res in v_np_res.items():
                video_np_results[(video_id, category_id)].update(res)

        if len(dataset_results) == 0:
            dataset_results = {"": 0.0}

        formatted_video_np_results = [
            {"video_id": video_id, "category_id": category_id, **res}
            for (video_id, category_id), res in video_np_results.items()
        ]
        eval_metrics = {
            "dataset_results": dataset_results,
            "video_np_results": formatted_video_np_results,
        }

        with g_pathmgr.open(self.eval_res_file, "w") as f:
            json.dump(eval_metrics, f)

        return eval_metrics


def run_main_all(dataset_name, args):
    gt_annot_file = os.path.join(args.gt_annot_dir, dataset_name + ".json")
    pred_file = os.path.join(args.pred_dir, dataset_name + "_preds.json")
    eval_res_file = os.path.join(args.eval_res_dir, dataset_name + "_eval_res.json")
    print(f"=== Running evaluation for Pred {pred_file} vs GT {gt_annot_file} ===")
    veval_evaluator = VEvalEvaluator(
        gt_annot_file=gt_annot_file, eval_res_file=eval_res_file
    )
    _ = veval_evaluator.run_eval(pred_file=pred_file)

    print(f"=== Results saved to {eval_res_file} ===")


def main_all(args):
    saco_veval_dataset_names = [
        "saco_veval_sav_test",
        "saco_veval_sav_val",
        "saco_veval_yt1b_test",
        "saco_veval_yt1b_val",
        "saco_veval_smartglasses_test",
        "saco_veval_smartglasses_val",
    ]

    # multiprocessing may not really work as inner evaluator also using multiprocessing
    # so we just for loop
    for dataset_name in saco_veval_dataset_names:
        print(f"=== Running evaluation for dataset {dataset_name} ===")
        run_main_all(dataset_name=dataset_name, args=args)


def main_one(args):
    gt_annot_file = args.gt_annot_file
    pred_file = args.pred_file
    eval_res_file = args.eval_res_file

    print(f"=== Running evaluation for Pred {pred_file} vs GT {gt_annot_file} ===")
    veval_evaluator = VEvalEvaluator(
        gt_annot_file=gt_annot_file, eval_res_file=eval_res_file
    )
    _ = veval_evaluator.run_eval(pred_file=pred_file)

    print(f"=== Results saved to {eval_res_file} ===")


def main():
    parser = argparse.ArgumentParser(description="Run video grounding evaluators")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run evaluation for all datasets
    all_parser = subparsers.add_parser("all", help="Run evaluation for all datasets")
    all_parser.add_argument(
        "--gt_annot_dir",
        type=str,
        help="Directory that contains the ground truth annotation files",
    )
    all_parser.add_argument(
        "--pred_dir",
        type=str,
        help="Directory that contains the prediction files",
    )
    all_parser.add_argument(
        "--eval_res_dir",
        type=str,
        help="Directory that contains the eval results files",
    )
    all_parser.set_defaults(func=main_all)

    # Run evaluation for one dataset
    one_parser = subparsers.add_parser("one", help="Run evaluation for one dataset")
    one_parser.add_argument(
        "--gt_annot_file",
        type=str,
        help="Path to the ground truth annotation file",
    )
    one_parser.add_argument(
        "--pred_file",
        type=str,
        help="Path to the prediction file",
    )
    one_parser.add_argument(
        "--eval_res_file",
        type=str,
        help="Path to the eval results file",
    )
    one_parser.set_defaults(func=main_one)

    # Parse and dispatch
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
