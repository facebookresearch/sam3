import argparse
import json

from external_evaluators import (
    VideoDemoF1Evaluator,
    VideoPhraseHotaEvaluator,
    VideoTetaEvaluator,
)


gt_file_dict = {
    "dai1.3a": "/fsx-onevision-auto-sync/ythu/sam3/video_grounding/release/20250317/ytvis/all/eval_set_v1.3a_rebalanced_pos_ratio_0.25_20250318_ytvis_format_w_tags.json",
    "dai2.0": "/fsx-onevision-auto-sync/ythu/sam3/video_grounding/release/20250315/ytvis/all/20250315_eval_v2.0.json",
    "yt1b_tc": "/fsx-onevision-auto-sync/ythu/sam3/video_grounding/release/20250512/ytvis/all/20250512_eval_YT1Bv1.0_tracking_challenging_subset_no_cleaning_w_tags.json",
    "yt1b_crowded": "/fsx-onevision-auto-sync/ythu/sam3/video_grounding/release/20250512/ytvis/all/20250512_eval_YT1Bv1.0_crowded_subset_no_cleaning_w_tags.json",
    "yt1b_human220": "/fsx-onevision-auto-sync/ythu/sam3/video_grounding/release/20250613/ytvis/human_performance_eval_at_least_3_agreement/yt1b_human_performance_set_group_1_no_auto_dedup.json",
    "liveai1.0": "/fsx-onevision-auto-sync/ythu/sam3/video_grounding/release/20250617/ytvis/all/20250617_eval_LiveAIv1.0_w_tags.json",
    "ytvis_dev": "/fsx-onevision-auto-sync/ythu/sam3/video_grounding/oss/ytvis_dev/instances_w_tags_w_negatives.json",
    "burst_val": "/fsx-onevision-auto-sync/ythu/sam3/video_grounding/oss/burst_vis_val/instances_w_tags_w_negatives.json",
    "cxl_val": "/fsx-onevision-auto-sync/ythu/sam3/video_grounding/release/20250729/ytvis/all/20250729_eval_CXL_w_tags_exhaustive.json",
    "cxl_val_negatives": "/fsx-onevision/didac/data/20250729_eval_CXL_w_tags_exhaustive_with_negatives.json",
    "cxl_val_negatives_parents": "/fsx-onevision/didac/data/20250729_eval_CXL_w_tags_exhaustive_with_negatives_parents.json",
}
gtpath2name = {v: k for k, v in gt_file_dict.items()}

parser = argparse.ArgumentParser(description="Run video grounding evaluators")
parser.add_argument(
    "--gt_ann_file",
    type=str,
    # default="/fsx-onevision-auto-sync/ythu/sam3/video_grounding/release/20250315/ytvis/all/20250315_eval_v2.0.json",
    # default="/fsx-onevision-auto-sync/ythu/sam3/video_grounding/release/20250512/ytvis/all/20250512_eval_YT1Bv1.0_crowded_subset_no_cleaning_w_tags.json",
    default="/fsx-onevision-auto-sync/ythu/sam3/video_grounding/release/20250512/ytvis/all/20250512_eval_YT1Bv1.0_tracking_challenging_subset_no_cleaning_w_tags.json",
    help="Path to the ground truth annotation file",
)
parser.add_argument(
    "--pred_file",
    type=str,
    # default="/fsx-onevision-auto-sync/ythu/sam3/video_grounding/pseudo_labeling/sav_pl_v1_on_dai2_eval_20250620/vg_output_json_ytvis_eval.json",
    # default="/fsx-onevision-auto-sync/ythu/sam3/video_grounding/pseudo_labeling/sav_pl_v1_on_yt1b_eval_crowded_20250620/vg_output_json_ytvis_eval.json",
    default="/fsx-onevision-auto-sync/ythu/sam3/video_grounding/pseudo_labeling/sav_pl_v1_on_yt1b_eval_tc_20250620/vg_output_json_ytvis_eval.json",
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



results, video_np_level_results = VideoTetaEvaluator(
    gt_ann_file, dataset_name="video"
).evaluate(pred_file)
print(f"\nVideo Phrase TETA eval results:\n{results}")
teta_results = results

results, video_np_level_results = VideoPhraseHotaEvaluator(
    gt_ann_file, dataset_name="video", prob_thresh=0.5
).evaluate(pred_file)
print(f"\nVideo Phrase HOTA eval results:\n{results}")
hota_results = results

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
