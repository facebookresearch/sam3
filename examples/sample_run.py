
import os
import torch
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def run_sample():
    print("Setting up SAM3...")
    
    # Check if assets exist
    # Updated to look one directory up for assets since this is in examples/
    image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets/images/test_image.jpg")
    
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found. Please ensure the assets directory exists.")
        return

    try:
        # Load the model
        print("Loading SAM3 model (this may trigger checkpoint download)...")
        model = build_sam3_image_model()
        # Lower confidence threshold from default 0.5 to detect more objects
        processor = Sam3Processor(model, confidence_threshold=0.1)
        
        # Load an image
        print(f"Loading image from {image_path}...")
        image = Image.open(image_path)
        
        # Run inference
        print("Running inference...")
        inference_state = processor.set_image(image)
        
        # Prompt the model
        prompt_text = "kid wearing a red bib" 
        print(f"Prompting with: '{prompt_text}'")
        output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)
        
        # Get results
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        print("\nSuccess!")
        print(f"Found {len(masks)} masks.")
        print(f"Scores: {scores[:5]} (showing top 5)")
        print(f"Masks shape: {masks.shape}")
        
    except Exception as e:
        print("\nError occurred:")
        print(str(e))
        if "huggingface" in str(e).lower() or "unauthorized" in str(e).lower() or "login" in str(e).lower():
            print("\nIt looks like an authentication error.")
            print("Please run: uv run huggingface-cli login")
            print("And paste your token from https://huggingface.co/settings/tokens")

if __name__ == "__main__":
    run_sample()
