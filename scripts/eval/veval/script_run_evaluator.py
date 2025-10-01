import argparse
import json

gtpath2name = {
    "/fsx-onevision/tym/sam3_and_data/sam3/scripts/eval/veval/toy_gt_and_pred/toy_saco_sav_test_gt.json": "toy_saco_sav_test",
}

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
parser.add_argument(
    "--output_file",
    type=str,
    default="eval_results.txt",
    help="Path to the output file where evaluation results will be saved",
)
args = parser.parse_args()
gt_ann_file = args.gt_ann_file
pred_file = args.pred_file

pred = json.load(open(pred_file, "r"))
if "score" not in pred[0]:
    for p in pred:
        p["score"] = 1.0  # Add a default score if not present
    with open(pred_file, "w") as f:
        json.dump(pred, f)


from sam3_video_evaluators.external_evaluators import VideoTetaEvaluator

results, video_np_level_results = VideoTetaEvaluator(
    gt_ann_file, dataset_name="video"
).evaluate(pred_file)
print(f"\nVideo Phrase TETA eval results:\n{results}")
teta_results = results

from sam3_video_evaluators.external_evaluators import (
    VideoPhraseHotaEvaluator,
)

results, video_np_level_results = VideoPhraseHotaEvaluator(
    gt_ann_file, dataset_name="video", prob_thresh=0.5
).evaluate(pred_file)
print(f"\nVideo Phrase HOTA eval results:\n{results}")
hota_results = results

from sam3_video_evaluators.external_evaluators import (
    VideoDemoF1Evaluator,
)

results, video_np_level_results = VideoDemoF1Evaluator(
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
    "GT Annotations": gtpath2name[gt_ann_file],
    "Predictions": pred_file.split("/")[-1],
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

# Save results to a JSON file
output_json_file = args.pred_file.replace("_eval.json", "_eval_res.json")
with open(output_json_file, "w") as f:
    json.dump(res_dict, f, indent=4)

# Prepare headers and formatted values
headers = list(res_dict.keys())
values = [
    f"{value:.3f}" if isinstance(value, float) else f"{value}"
    for value in res_dict.values()
]
# Construct the Markdown table
markdown_table = "| " + " | ".join(headers) + " |\n"
markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
markdown_table += "| " + " | ".join(values) + " |\n"
# Print the Markdown table
print(markdown_table)

f = open(args.output_file, "a")
# f.write(f"gt_annot:{gt_ann_file}, pred:{pred_file}\n")
f.write(markdown_table + "\n")
f.close()
print(f"Results saved to {args.output_file}")
