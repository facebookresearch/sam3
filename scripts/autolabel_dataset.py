# scripts/autolabel_dataset.py
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_utils

# SAM 3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def parse_args():
    parser = argparse.ArgumentParser(description="SAM 3 Auto-Labeling Data Engine")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save COCO JSON annotation file")
    parser.add_argument("--concepts", type=str, nargs='+', required=True, help="List of text concepts to label (e.g. 'person' 'car' 'dog')")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--box_threshold", type=float, default=0.5, help="Confidence threshold for detection")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for faster inference (requires PyTorch 2.0+)")
    return parser.parse_args()

def binary_mask_to_rle(binary_mask):
    """Converts a binary mask to COCO RLE format."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def main():
    args = parse_args()
    
    # 1. Setup & Model Loading
    print(f"Loading SAM 3 Model on {args.device}...")
    model = build_sam3_image_model()
    model.to(args.device)
    
    if args.compile:
        print("Compiling model backbone with torch.compile()...")
        model.vision_encoder = torch.compile(model.vision_encoder, mode="max-autotune")
    
    processor = Sam3Processor(model)
    
    # 2. Prepare COCO Structure
    categories = [{"id": i+1, "name": name, "supercategory": "object"} for i, name in enumerate(args.concepts)]
    concept_map = {name: i+1 for i, name in enumerate(args.concepts)}
    
    coco_output = {
        "info": {
            "description": "Auto-labeled dataset using SAM 3",
            "year": datetime.now().year,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories
    }

    image_files = sorted([
        f for f in Path(args.image_dir).iterdir() 
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
    ])
    
    print(f"Found {len(image_files)} images. Starting annotation engine...")
    
    annotation_id = 1
    image_id = 1

    # 3. Batch Processing Loop
    # Note: SAM 3 Processor typically handles one image at a time for setting state, 
    # but we can pipeline the concept prompts.
    
    for img_path in tqdm(image_files):
        try:
            # Load Image
            pil_image = Image.open(img_path).convert("RGB")
            w, h = pil_image.size
            
            coco_output["images"].append({
                "id": image_id,
                "width": w,
                "height": h,
                "file_name": img_path.name
            })
            
            # Set Image State (Encoder Run)
            # This is the heavy lifting.
            inference_state = processor.set_image(pil_image)
            
            # Iterate through all concepts for this image
            # In a true batch scenario, you'd batch concepts, but SAM 3 processes prompt-by-prompt efficiently
            for concept in args.concepts:
                # Run Inference for Concept
                output = processor.set_text_prompt(
                    state=inference_state, 
                    prompt=concept
                )
                
                masks = output["masks"] # [N, H, W]
                scores = output["scores"] # [N]
                boxes = output["boxes"] # [N, 4]
                
                # Filter by threshold
                valid_indices = torch.where(scores > args.box_threshold)[0]
                
                if len(valid_indices) > 0:
                    # Move to CPU for serialization
                    masks = masks[valid_indices].cpu().numpy()
                    boxes = boxes[valid_indices].cpu().numpy()
                    scores = scores[valid_indices].cpu().numpy()
                    
                    for i in range(len(valid_indices)):
                        mask = masks[i] > 0.5
                        box = boxes[i].tolist() # xyxy
                        score = float(scores[i])
                        
                        # Calculate Area
                        area = float(np.sum(mask))
                        
                        # Convert Box to COCO format (x, y, w, h)
                        coco_box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                        
                        # Convert Mask to RLE
                        segmentation = binary_mask_to_rle(mask)
                        
                        ann = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": concept_map[concept],
                            "bbox": coco_box,
                            "area": area,
                            "segmentation": segmentation,
                            "iscrowd": 0,
                            "score": score
                        }
                        coco_output["annotations"].append(ann)
                        annotation_id += 1
            
            image_id += 1
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

    # 4. Save Output
    print(f"Saving annotations to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        json.dump(coco_output, f)
    
    print("Done! Data Engine complete.")

if __name__ == "__main__":
    main()
