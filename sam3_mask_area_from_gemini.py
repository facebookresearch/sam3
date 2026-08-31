import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

ROOT = Path("/home/users/ntu/gwang016/scratch/sam3_test_same_object")
RANK_PATH = ROOT / "gemini_ranked_candidates.json"
OUT_ROOT = ROOT / "sam3_mask_results"

TOP_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def to_numpy_masks(masks):
    if masks is None:
        return None

    if isinstance(masks, torch.Tensor):
        arr = masks.detach().float().cpu().numpy()
    else:
        arr = np.asarray(masks)

    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :]

    return arr


def to_numpy_scores(scores, n_masks):
    if scores is None:
        return np.ones(n_masks, dtype=np.float32)

    if isinstance(scores, torch.Tensor):
        arr = scores.detach().float().cpu().numpy()
    else:
        arr = np.asarray(scores)

    arr = np.squeeze(arr)

    if arr.ndim == 0:
        arr = np.array([float(arr)])

    return arr.astype(np.float32)


def binarize_mask(mask):
    if mask.dtype == np.bool_:
        return mask

    mask = mask.astype(np.float32)

    # if logits or signed scores
    if mask.min() < 0 or mask.max() > 1:
        return mask > 0

    # if already probabilities in [0,1]
    return mask > 0.5


def save_mask_png(mask_bool: np.ndarray, out_path: Path):
    mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255))
    mask_img.save(out_path)


def save_overlay(image: Image.Image, mask_bool: np.ndarray, out_path: Path):
    img = np.array(image).astype(np.uint8)
    overlay = img.copy()

    red = np.array([255, 0, 0], dtype=np.uint8)

    alpha = 0.45
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool].astype(np.float32)
        + alpha * red.astype(np.float32)
    ).astype(np.uint8)

    Image.fromarray(overlay).save(out_path)


def pick_first_existing(output_dict, keys):
    for k in keys:
        if isinstance(output_dict, dict) and k in output_dict and output_dict[k] is not None:
            return output_dict[k], k
    return None, None


def select_best_mask(output_dict):
    if output_dict is None:
        raise RuntimeError("SAM3 output is None.")

    if not isinstance(output_dict, dict):
        raise RuntimeError(f"SAM3 output is not dict. type={type(output_dict)}")

    print("SAM3 output keys:", list(output_dict.keys()), flush=True)

    masks_raw, mask_key = pick_first_existing(
        output_dict,
        ["masks", "pred_masks", "mask_logits", "logits_masks"]
    )

    scores_raw, score_key = pick_first_existing(
        output_dict,
        ["scores", "pred_scores", "iou_scores", "object_scores", "confidence_scores"]
    )

    print("Using mask key:", mask_key, flush=True)
    print("Using score key:", score_key, flush=True)

    if masks_raw is None:
        # 打印每个 key 的类型，方便判断真实输出结构
        print("No mask-like field found. Output summary:", flush=True)
        for k, v in output_dict.items():
            if isinstance(v, torch.Tensor):
                print(k, "Tensor", tuple(v.shape), v.dtype, flush=True)
            else:
                print(k, type(v), flush=True)

        raise RuntimeError("SAM3 returned no masks. Need to inspect output keys or prompt failed.")

    if isinstance(masks_raw, torch.Tensor):
        print("Raw masks shape:", tuple(masks_raw.shape), "dtype:", masks_raw.dtype, flush=True)
    else:
        print("Raw masks type:", type(masks_raw), flush=True)

    if isinstance(scores_raw, torch.Tensor):
        print("Raw scores shape:", tuple(scores_raw.shape), "dtype:", scores_raw.dtype, flush=True)
    else:
        print("Raw scores type:", type(scores_raw), flush=True)

    masks = to_numpy_masks(masks_raw)

    if masks is None or len(masks) == 0:
        raise RuntimeError("SAM3 masks are empty after conversion.")

    scores = to_numpy_scores(scores_raw, len(masks))

    mask_infos = []
    for i in range(len(masks)):
        mask_bool = binarize_mask(masks[i])
        area_pixels = int(mask_bool.sum())
        area_ratio = float(area_pixels / mask_bool.size)

        score = float(scores[i]) if i < len(scores) else 0.0

        mask_infos.append({
            "mask_index": i,
            "score": score,
            "area_pixels": area_pixels,
            "area_ratio": area_ratio,
            "mask_bool": mask_bool,
        })

    mask_infos_sorted = sorted(
        mask_infos,
        key=lambda x: (x["score"], x["area_ratio"]),
        reverse=True,
    )

    best = mask_infos_sorted[0]
    return best, mask_infos_sorted


def make_prompt_variants(target_prompt: str):
    base = target_prompt.strip()
    lower = base.lower()

    variants = [
        base,
        lower,
    ]

    # 针对 Basil oil bottle 这种目标，增加更容易被模型识别的通用 prompt
    if "bottle" in lower:
        variants += [
            "oil bottle",
            "bottle",
            "green bottle",
        ]

    # 去重但保持顺序
    seen = set()
    final = []
    for v in variants:
        if v and v not in seen:
            final.append(v)
            seen.add(v)

    return final


def run_one_image(processor, image_path: Path, target_prompt: str):
    image = Image.open(image_path).convert("RGB")

    prompt_variants = make_prompt_variants(target_prompt)
    last_error = None

    for prompt in prompt_variants:
        print(f"Trying prompt: {prompt}", flush=True)

        try:
            with torch.inference_mode():
                if DEVICE == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        state = processor.set_image(image)
                        output = processor.set_text_prompt(state=state, prompt=prompt)
                else:
                    state = processor.set_image(image)
                    output = processor.set_text_prompt(state=state, prompt=prompt)

            best_mask, all_masks = select_best_mask(output)

            h, w = best_mask["mask_bool"].shape

            return {
                "image": image,
                "image_path": str(image_path),
                "image_name": image_path.name,
                "height": h,
                "width": w,
                "used_prompt": prompt,
                "best_mask": best_mask,
                "all_masks": [
                    {
                        "mask_index": x["mask_index"],
                        "score": x["score"],
                        "area_pixels": x["area_pixels"],
                        "area_ratio": x["area_ratio"],
                    }
                    for x in all_masks
                ],
            }

        except Exception as e:
            last_error = e
            print(f"[WARN] Prompt failed: {prompt}")
            print(f"[WARN] Error: {e}", flush=True)

    raise RuntimeError(
        f"All prompt variants failed for image {image_path.name}. "
        f"Last error: {last_error}"
    )

def main():
    ensure_dir(OUT_ROOT)

    with open(RANK_PATH, "r") as f:
        rank_data = json.load(f)

    target_prompt = rank_data["target_prompt"]

    print("Using device:", DEVICE)
    print("Target prompt:", target_prompt)

    # ===== Build model =====
    # If your local SAM3 repo uses a slightly different init function,
    # only this block may need adjustment.
    model = build_sam3_image_model()

    if hasattr(model, "to"):
        model = model.to(DEVICE)

        # Do NOT convert the whole SAM3 model to bfloat16.
        # Some text/decoder tensors remain float32, and converting all weights to bf16
        # can cause dtype mismatch: Float input vs BFloat16 weight.
        if DEVICE == "cuda":
            model = model.float()

    if hasattr(model, "eval"):
        model.eval()

    processor = Sam3Processor(model)

    final_output = {
        "target_prompt": target_prompt,
        "selection_rule": "For each image, choose the mask with highest SAM3 score (tie-break by larger area_ratio). Then rank images by area_ratio.",
        "views": {}
    }

    for view_name, view_data in rank_data["views"].items():
        print("\n" + "=" * 80)
        print("Processing view:", view_name)

        view_out_dir = OUT_ROOT / view_name
        ensure_dir(view_out_dir)
        ensure_dir(view_out_dir / "masks")
        ensure_dir(view_out_dir / "overlays")

        top_candidates = view_data["top_candidates"][:TOP_K]
        results = []

        for i, cand in enumerate(top_candidates, 1):
            image_path = Path(cand["image_path"])
            print(f"[{i}/{len(top_candidates)}] {image_path.name}")

            result = run_one_image(
                processor=processor,
                image_path=image_path,
                target_prompt=target_prompt,
            )

            best_mask_bool = result["best_mask"]["mask_bool"]

            mask_png_path = view_out_dir / "masks" / f"{image_path.stem}_mask.png"
            overlay_png_path = view_out_dir / "overlays" / f"{image_path.stem}_overlay.png"

            save_mask_png(best_mask_bool, mask_png_path)
            save_overlay(result["image"], best_mask_bool, overlay_png_path)

            results.append({
                "image_name": result["image_name"],
                "image_path": result["image_path"],
                "height": result["height"],
                "width": result["width"],
                "used_prompt": result["used_prompt"],
                "best_mask_index": result["best_mask"]["mask_index"],
                "best_mask_score": result["best_mask"]["score"],
                "mask_area_pixels": result["best_mask"]["area_pixels"],
                "area_ratio": result["best_mask"]["area_ratio"],
                "mask_png": str(mask_png_path),
                "overlay_png": str(overlay_png_path),
                "all_masks": result["all_masks"],
            })

        ranked_by_area = sorted(results, key=lambda x: x["area_ratio"], reverse=True)
        best_image = ranked_by_area[0]

        final_output["views"][view_name] = {
            "num_processed": len(results),
            "ranked_results": ranked_by_area,
            "best_image": best_image,
        }

        print(f"\nBest image for {view_name}:")
        print(json.dumps(best_image, indent=2, ensure_ascii=False))

    out_json = OUT_ROOT / "sam3_mask_area_results.json"
    with open(out_json, "w") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Done.")
    print("Saved results to:")
    print(out_json)


if __name__ == "__main__":
    main()
