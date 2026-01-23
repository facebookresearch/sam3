# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Simple script to run the CGF1 evaluator given a prediction file and GT file(s).

Usage: python standalone_cgf1.py --pred_file <path_to_prediction_file> --gt_files <path_to_gt_file1> <path_to_gt_file2> ...
"""

import argparse

from sam3.eval.cgf1_eval import CGF1Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_file",
        type=str,
        required=True,
        help="Path to the prediction file in COCO format.",
    )
    parser.add_argument(
        "--gt_files",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the ground truth files in COCO format.",
    )
    args = parser.parse_args()
    if len(args.gt_files) == 0:
        raise ValueError("At least one GT file must be provided.")

    is_gold = args.gt_files[0].split("_")[-1].startswith("gold_")
    if is_gold and len(args.gt_files) < 3:
        print(
            "WARNING: based on the name, it seems you are using gold GT files. Typically, there should be 3 GT files for gold subsets (a, b, c)."
        )

    evaluator = CGF1Evaluator(
        gt_path=args.gt_files, verbose=True, iou_type="segm"
    )  # change to bbox if you want detection performance

    results = evaluator.evaluate(args.pred_file)

    print(results)


if __name__ == "__main__":
    main()
