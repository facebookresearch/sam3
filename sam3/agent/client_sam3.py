# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import json
import os

import torch
from PIL import Image

from sam3.model.box_ops import box_xyxy_to_xywh
from sam3.train.masks_ops import rle_encode

from .helpers.mask_overlap_removal import remove_overlapping_masks
from .viz import visualize


def sam3_inference(processor, image_path, text_prompt):
    """Run SAM 3 image inference with text prompts and format the outputs"""
    image = Image.open(image_path)
    orig_img_w, orig_img_h = image.size

    # model inference
    inference_state = processor.set_image(image)
    inference_state = processor.set_text_prompt(
        state=inference_state, prompt=text_prompt
    )

    # format and assemble outputs
    pred_boxes_xyxy = torch.stack(
        [
            inference_state["boxes"][:, 0] / orig_img_w,
            inference_state["boxes"][:, 1] / orig_img_h,
            inference_state["boxes"][:, 2] / orig_img_w,
            inference_state["boxes"][:, 3] / orig_img_h,
        ],
        dim=-1,
    )  # normalized in range [0, 1]
    pred_boxes_xywh = box_xyxy_to_xywh(pred_boxes_xyxy).tolist()
    pred_masks = rle_encode(inference_state["masks"].squeeze(1))
    pred_masks = [m["counts"] for m in pred_masks]
    outputs = {
        "orig_img_h": orig_img_h,
        "orig_img_w": orig_img_w,
        "pred_boxes": pred_boxes_xywh,
        "pred_masks": pred_masks,
        "pred_scores": inference_state["scores"].tolist(),
    }
    return outputs


def call_sam_service(
    sam3_processor,
    image_path: str,
    text_prompt: str,
    output_folder_path: str = "sam3_output",
):
    """
    Loads an image, sends it with a text prompt to the service,
    saves the results, and renders the visualization.
    """
    print(f"📞 Loading image '{image_path}' and sending with prompt '{text_prompt}'...")

    text_prompt_for_save_path = (
        text_prompt.replace("/", "_") if "/" in text_prompt else text_prompt
    )

    os.makedirs(
        os.path.join(output_folder_path, image_path.replace("/", "-")), exist_ok=True
    )
    output_json_path = os.path.join(
        output_folder_path,
        image_path.replace("/", "-"),
        rf"{text_prompt_for_save_path}.json",
    )
    output_image_path = os.path.join(
        output_folder_path,
        image_path.replace("/", "-"),
        rf"{text_prompt_for_save_path}.png",
    )

    try:
        # Send the image and text prompt as a multipart/form-data request
        serialized_response = sam3_inference(sam3_processor, image_path, text_prompt)

        # 1. Prepare the response dictionary
        serialized_response = remove_overlapping_masks(serialized_response)
        serialized_response = {
            "original_image_path": image_path,
            "output_image_path": output_image_path,
            **serialized_response,
        }

        # 2. Reorder predictions by scores (highest to lowest) if scores are available
        if "pred_scores" in serialized_response and serialized_response["pred_scores"]:
            # Create indices sorted by scores in descending order
            score_indices = sorted(
                range(len(serialized_response["pred_scores"])),
                key=lambda i: serialized_response["pred_scores"][i],
                reverse=True,
            )

            # Reorder all three lists based on the sorted indices
            serialized_response["pred_scores"] = [
                serialized_response["pred_scores"][i] for i in score_indices
            ]
            serialized_response["pred_boxes"] = [
                serialized_response["pred_boxes"][i] for i in score_indices
            ]
            serialized_response["pred_masks"] = [
                serialized_response["pred_masks"][i] for i in score_indices
            ]

        # 3. Remove any invalid RLE masks that is too short (shorter than 5 characters)
        valid_masks = []
        valid_boxes = []
        valid_scores = []
        for i, rle in enumerate(serialized_response["pred_masks"]):
            if len(rle) > 4:
                valid_masks.append(rle)
                valid_boxes.append(serialized_response["pred_boxes"][i])
                valid_scores.append(serialized_response["pred_scores"][i])
        serialized_response["pred_masks"] = valid_masks
        serialized_response["pred_boxes"] = valid_boxes
        serialized_response["pred_scores"] = valid_scores

        with open(output_json_path, "w") as f:
            json.dump(serialized_response, f, indent=4)
        print(f"✅ Raw JSON response saved to '{output_json_path}'")

        # 4. Render and save visualizations on the image and save it in the SAM3 output folder
        print("🔍 Rendering visualizations on the image ...")
        viz_image = visualize(serialized_response)
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        viz_image.save(output_image_path)
        print("✅ Saved visualization at:", output_image_path)
    except Exception as e:
        print(f"❌ Error calling service: {e}")

    return output_json_path
