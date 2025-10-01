import io
import os
import argparse
import json
from typing import List
import numpy as np
import requests
from .viz import visualize
import pycocotools.mask as mask_utils
from .helpers.mask_overlap_removal import remove_overlapping_masks
import cv2
import getpass


USER = getpass.getuser()
SAM_OUTPUT_DIR = f"/fsx-onevision/{USER}/code/out/sam_out"


# --- Main API Call Function ---
SAM3_SERVICE_URL = "http://localhost:8000/segment"
# SAM3_SERVICE_URL = "http://h100-023-012:8000/segment"

def call_sam_service(image_path: str, text_prompt: str, output_folder_path: str = SAM_OUTPUT_DIR, threshold: float = 0.5, selected_masks: List[int]=None, server_url=SAM3_SERVICE_URL):
    """
    Loads an image, sends it with a text prompt to the service,
    saves the results, and renders the visualization.
    """
    print(f"📞 Loading image '{image_path}' and sending with prompt '{text_prompt}'...")
    
    text_prompt_for_save_path = text_prompt.replace("/", "_") if "/" in text_prompt else text_prompt
    
    os.makedirs(os.path.join(output_folder_path, image_path.replace("/", "-")), exist_ok=True)
    output_json_path = os.path.join(output_folder_path, image_path.replace("/", "-"), rf"{text_prompt_for_save_path}.json")
    output_image_path = os.path.join(output_folder_path, image_path.replace("/", "-"), rf"{text_prompt_for_save_path}.png")


    try:
        # Send the image and text prompt as a multipart/form-data request
        with open(image_path, "rb") as f:
            data = {'image_path': image_path, 'find_input_text': text_prompt, 'threshold': threshold}
            response = requests.post(server_url, data=data)

        response.raise_for_status()
        
        # 1. Get the raw JSON response from SAM3 Server
        serialized_response = response.json()
        
        # add remove duplicate masks
        serialized_response = remove_overlapping_masks(serialized_response)
        serialized_response = {"original_image_path": image_path, **serialized_response}
        serialized_response = {"output_image_path": output_image_path, **serialized_response}
        
    
        # 2. Reorder predictions by scores (highest to lowest) if scores are available
        if 'pred_scores' in serialized_response and serialized_response['pred_scores']:
            # Create indices sorted by scores in descending order
            score_indices = sorted(range(len(serialized_response['pred_scores'])), 
                                 key=lambda i: serialized_response['pred_scores'][i], reverse=True)
            
            # Reorder all three lists based on the sorted indices
            serialized_response['pred_scores'] = [serialized_response['pred_scores'][i] for i in score_indices]
            serialized_response['pred_boxes'] = [serialized_response['pred_boxes'][i] for i in score_indices]
            serialized_response['pred_masks'] = [serialized_response['pred_masks'][i] for i in score_indices]
        
        # 3. Remove any invalid RLE masks that is too short (shorter than 5 characters)
        valid_masks = []
        valid_boxes = []
        valid_scores = []
        for i, rle in enumerate(serialized_response['pred_masks']):
            if len(rle) > 4:
                valid_masks.append(rle)
                valid_boxes.append(serialized_response['pred_boxes'][i])
                valid_scores.append(serialized_response['pred_scores'][i])
        serialized_response['pred_masks'] = valid_masks
        serialized_response['pred_boxes'] = valid_boxes
        serialized_response['pred_scores'] = valid_scores

        with open(output_json_path, 'w') as f:
            json.dump(serialized_response, f, indent=4)
        print(f"✅ Raw JSON response saved to '{output_json_path}'")
        
        
        # 4. Render and save visualizations on the image and save it in the SAM3 output folder
        print("🔍 Rendering visualizations on the image...")
        # pil_image = np.array(Image.open(image_path).convert('RGB'))
        cv2_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        boxes_array = np.array(serialized_response['pred_boxes'])
        coco_rle_masks = [{'size': (serialized_response["orig_img_h"], serialized_response["orig_img_w"]), 'counts': rle} for rle in serialized_response['pred_masks']]
        binary_masks = [mask_utils.decode(i) for i in coco_rle_masks]
        viz_image = visualize(cv2_img, boxes_array, coco_rle_masks, binary_masks)
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        viz_image.save(output_image_path)
        print("✅ Saved visualization at:", output_image_path)

    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling service: {e}")
    
    return output_json_path
