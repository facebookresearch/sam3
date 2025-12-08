"""
Minimal test for text-based inference with SAM3.

This test demonstrates how to:
1. Load a SAM3 model with appropriate configuration
2. Run text prompt inference on images
3. Extract masks, boxes, and scores from the results
4. Visualize the results by overlaying masks on the original image

This serves as both a test and documentation for basic SAM3 inference.
"""
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor

checkpoint_path = "/checkpoint/sam3/weights.pt"
# checkpoint_path = "/checkpoint/sam3/weights-finetuned2.pt"


def build_model():
    """Build and load the SAM3 model."""
    from sam3 import build_sam3_image_model

    # Enable bfloat16 for better performance
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    # Paths
    test_dir = Path(__file__).parent
    bpe_vocab = str(test_dir.parent / "assets" /
                    "bpe_simple_vocab_16e6.txt.gz")

    # Verify paths exist
    assert os.path.exists(bpe_vocab), f"BPE vocab not found at {bpe_vocab}"
    assert os.path.exists(
        checkpoint_path), f"Checkpoint not found at {checkpoint_path}"

    model = build_sam3_image_model(
        bpe_path=bpe_vocab,
        checkpoint_path=checkpoint_path,
    )

    return model


def run_inference(model, image_path, text_prompt, output_path):
    """Run inference with the model and save visualization.

    This function demonstrates the complete inference pipeline:
    1. Load image and create processor
    2. Initialize inference state with the image
    3. Add text prompt to the inference state
    4. Run model inference
    5. Post-process outputs to get masks, boxes, and scores
    6. Visualize results

    Args:
        model: The SAM3 model loaded with build_sam3_image_model
        image_path: Path to the input image
        text_prompt: Text description of what to segment
        output_path: Path where to save the visualization
    """
    assert os.path.exists(image_path), f"Test image not found at {image_path}"

    print(f"Running inference on {image_path}...")

    # Step 1: Initialize processor and create inference state from image
    processor = Sam3Processor(model)
    image = Image.open(image_path)
    state = processor.set_image(image)

    # Step 2: Add text prompt and run inference
    print(f"Using text prompt: '{text_prompt}'")
    state = processor.set_text_prompt(text_prompt, state)

    # Extract results
    masks = state.get("masks", [])
    boxes = state.get("boxes", [])
    scores = state.get("scores", [])

    # Print statistics
    print("\nResults:")
    if len(masks) > 0:
        print(f"  Number of detected objects: {len(masks)}")
        print(f"  Scores: {[f'{s.item():.3f}' for s in scores]}")
        if len(boxes) > 0:
            print(f"  Boxes (xyxy): {[box.cpu().tolist() for box in boxes]}")
    else:
        print("  No objects found above confidence threshold")

    # Create visualization with mask overlay
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    if len(masks) > 0:
        # Generate random colors for each mask
        np.random.seed(1000)
        random_colors = np.random.rand(len(masks), 3)

        # Create overlay
        overlay = img_array.copy()
        for idx, mask in enumerate(masks):
            color = (random_colors[idx] * 255).astype(np.uint8)
            # Convert mask to numpy array - masks are already in the right format
            mask_np = mask[0].cpu().numpy() > 0.5
            # Apply color where mask is True
            overlay[mask_np] = overlay[mask_np] * 0.5 + color * 0.5
    else:
        overlay = img_array.copy()

    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_img = Image.fromarray(overlay.astype(np.uint8))
    result_img.save(output_path)
    print(f"Visualization saved to: {output_path}")

    # Results are available - visualization has been saved

    return state


test_dir = Path(__file__).parent


def _testout(s):
    return str(test_dir/"testdata"/"out"/s)


def main():
    """Run SAM3 text-based inference examples.

    This demonstrates how to use SAM3 for text-based segmentation
    on various types of images and prompts.
    """
    # Load the model once and reuse for all inference calls
    model = build_model()

    # Example 1: SAM3 test image with people
    # This demonstrates segmentation of people and objects in a scene
    image_path = str(test_dir.parent / "assets" / "images" / "test_image.jpg")
    run_inference(model, image_path, "shoe", _testout("test_image_shoe.jpg"))
    run_inference(model, image_path, "boy", _testout("test_image_boy.jpg"))
    run_inference(model, image_path, "girl", _testout("test_image_girl.jpg"))
    run_inference(model, image_path, "kid", _testout("test_image_kid.jpg"))

    # Example 2: Trash classification test image
    # This demonstrates multi-class segmentation of different material types
    image_path = str(test_dir / "testdata" / "trash_test_image.jpg")
    classes = ["aggregate", "cardboard", "hard plastic",
               "soft plastic", "metal", "timber"]
    for class_ in classes:
        run_inference(model, image_path, class_, _testout(
            f"trash_test_image_{class_}.jpg"))

    print("\nAll inference examples completed successfully!")


if __name__ == "__main__":
    main()
