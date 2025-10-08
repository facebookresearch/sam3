import argparse
import json
import os

from saco_veval_metrics import DATASET_LEVEL_METRICS, VIDEO_NP_LEVEL_METRICS
from sam3.train.eval.veval_evaluators import (
    VideoDemoF1Evaluator,
    VideoPhraseHotaEvaluator,
    VideoTetaEvaluator,
)


def run_eval(gt_ann_file: str, pred_file: str):
    print(f"Evaluating Pred {pred_file} on GT {gt_ann_file}")
    teta_dataset_level_results, teta_video_np_level_results = VideoTetaEvaluator(gt_ann_file, dataset_name="video").evaluate(
        pred_file
    )

    hota_dataset_level_results, hota_video_np_level_results = VideoPhraseHotaEvaluator(
        gt_ann_file, dataset_name="video", prob_thresh=0.5
    ).evaluate(pred_file)

    demof1_dataset_level_results, demof1_video_np_level_results = VideoDemoF1Evaluator(
        gt_ann_file, dataset_name="video", prob_thresh=0.5
    ).evaluate(pred_file)
    
    return {
        "teta_dataset_level_results": teta_dataset_level_results,
        "teta_video_np_level_results": teta_video_np_level_results,
        "hota_dataset_level_results": hota_dataset_level_results,
        "hota_video_np_level_results": hota_video_np_level_results,
        "demof1_dataset_level_results": demof1_dataset_level_results,
        "demof1_video_np_level_results": demof1_video_np_level_results,
    }
    

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
