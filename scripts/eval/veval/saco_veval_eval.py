import argparse
import json
import os
from pathlib import Path

from sam3_video_evaluators.external_evaluators import (
    VideoDemoF1Evaluator,
    VideoPhraseHotaEvaluator,
    VideoTetaEvaluator,
)


def run_eval(gt_ann_file: str, pred_file: str):
    results, _ = VideoTetaEvaluator(gt_ann_file, dataset_name="video").evaluate(
        pred_file
    )
    print(f"\nVideo Phrase TETA eval results:\n{results}")
    teta_results = results

    results, _ = VideoPhraseHotaEvaluator(
        gt_ann_file, dataset_name="video", prob_thresh=0.5
    ).evaluate(pred_file)
    print(f"\nVideo Phrase HOTA eval results:\n{results}")
    hota_results = results

    results, _ = VideoDemoF1Evaluator(
        gt_ann_file, dataset_name="video", prob_thresh=0.5
    ).evaluate(pred_file)
    print(f"\nVideo Demo F1 eval results:\n{results}")
    demof1_results = results

    print("\nEvaluation completed successfully!")
    print(f"gt_annot:{gt_ann_file}")
    print("\nVideo Phrase TETA eval results:")
    print(teta_results)
    print("\nVideo Phrase HOTA eval results:")
    print(hota_results)
    print("\nVideo Demo F1 eval results:")
    print(demof1_results)

    print(f"gt_annot:{gt_ann_file}")

    res_dict = {
        "GT Annotations": Path(gt_ann_file).stem,
        "Predictions": Path(pred_file).stem,
        "TETA": teta_results["video_bbox_teta"],
        "TETA LocA": teta_results["video_bbox_loc_a"],
        "TETA AssA": teta_results["video_bbox_assoc_a"],
        "TETA ClsA": teta_results["video_bbox_cls_a"],
        "Demo F1": demof1_results["video_mask_demo_f1_50_95"],
        "Demo Recall": demof1_results["video_mask_demo_recall_50_95"],
        "Demo Precision": demof1_results["video_mask_demo_precision_50_95"],
        "HOTA": hota_results["video_mask_all_phrase_HOTA"],
        "HOTA DetA": hota_results["video_mask_all_phrase_DetA"],
        "HOTA AssA": hota_results["video_mask_all_phrase_AssA"],
        "TETA Results": teta_results,
        "HOTA Results": hota_results,
        "Demo F1 Results": demof1_results,
    }

    return res_dict


def main():
    parser = argparse.ArgumentParser(description="Run video grounding evaluators")
    parser.add_argument(
        "--gt_ann_file",
        type=str,
        default="/fsx-onevision/tym/sam3_and_data/sam3/scripts/eval/veval/toy_gt_and_pred/toy_saco_sav_test_gt.json",
        help="Path to the ground truth annotation file",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        default="/fsx-onevision/tym/sam3_and_data/sam3/scripts/eval/veval/toy_gt_and_pred/toy_saco_sav_test_preds.json",
        help="Path to the prediction file",
    )

    args = parser.parse_args()
    gt_ann_file = args.gt_ann_file
    pred_file = args.pred_file

    print(f"=== Running evaluation for Pred {pred_file} vs GT {gt_ann_file} ===")

    res_dict = run_eval(gt_ann_file, pred_file)

    # Save results to a JSON file
    output_json_file = os.path.splitext(pred_file)[0] + "_res.json"
    with open(output_json_file, "w") as f:
        json.dump(res_dict, f, indent=4)

    print(f"=== Results saved to {output_json_file} ===")


if __name__ == "__main__":
    main()
