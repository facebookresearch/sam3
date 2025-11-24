import argparse
import os
from PIL import Image
import numpy as np
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def list_images(root):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    items = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(exts):
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, root).replace("\\", "/")
                items.append((full, rel))
    return items


def save_mask(mask_tensor, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mask = (mask_tensor.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8) * 255
    Image.fromarray(mask).save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("prompt", type=str, nargs="?", default="defect")
    parser.add_argument("--bpe_path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model = build_sam3_image_model(
        bpe_path=args.bpe_path,
        checkpoint_path=args.checkpoint,
        enable_segmentation=True,
        eval_mode=True,
    )
    processor = Sam3Processor(model)
    items = list_images(args.dataset_root)
    for full, rel in items:
        try:
            img = Image.open(full).convert("RGB")
        except Exception:
            continue
        state = processor.set_image(img, state={})
        processor.set_confidence_threshold(args.threshold, state)
        state = processor.set_text_prompt(args.prompt, state)
        if "masks" not in state:
            continue
        masks = state["masks"]
        if masks.shape[0] == 0:
            continue
        out_path = os.path.join(args.output_dir, os.path.splitext(rel)[0] + ".png")
        save_mask(masks[0], out_path)


if __name__ == "__main__":
    main()

