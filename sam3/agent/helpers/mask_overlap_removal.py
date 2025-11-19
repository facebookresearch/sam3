# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from typing import Dict, List

import numpy as np
import torch

try:
    from pycocotools import mask as mask_utils
except Exception:
    mask_utils = None


def mask_intersection(
    masks1: torch.Tensor, masks2: torch.Tensor, block_size: int = 16
) -> torch.Tensor:
    assert masks1.shape[1:] == masks2.shape[1:]
    assert masks1.dtype == torch.bool and masks2.dtype == torch.bool
    N, M = masks1.shape[0], masks2.shape[0]
    out = torch.zeros(N, M, device=masks1.device, dtype=torch.long)
    for i in range(0, N, block_size):
        for j in range(0, M, block_size):
            a = masks1[i : i + block_size]
            b = masks2[j : j + block_size]
            inter = (a[:, None] & b[None, :]).flatten(-2).sum(-1)
            out[i : i + block_size, j : j + block_size] = inter
    return out


def mask_iom(masks1: torch.Tensor, masks2: torch.Tensor) -> torch.Tensor:
    assert masks1.shape[1:] == masks2.shape[1:]
    assert masks1.dtype == torch.bool and masks2.dtype == torch.bool
    inter = mask_intersection(masks1, masks2)
    area1 = masks1.flatten(-2).sum(-1)  # (N,)
    area2 = masks2.flatten(-2).sum(-1)  # (M,)
    min_area = torch.min(area1[:, None], area2[None, :]).clamp_min(1)
    return inter.float() / (min_area.float() + 1e-8)


def _decode_single_mask(mask_repr, h: int, w: int) -> np.ndarray:
    if isinstance(mask_repr, (list, tuple, np.ndarray)):
        arr = np.array(mask_repr)
        if arr.ndim != 2:
            raise ValueError("Mask array must be 2D (H, W).")
        return (arr > 0).astype(np.uint8)

    if mask_utils is None:
        raise ImportError(
            "pycocotools is required to decode RLE mask strings. pip install pycocotools"
        )

    if not isinstance(mask_repr, (str, bytes)):
        raise ValueError("Unsupported mask representation type for RLE decode.")

    rle = {
        "counts": mask_repr if isinstance(mask_repr, (str, bytes)) else str(mask_repr),
        "size": [h, w],
    }
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return (decoded > 0).astype(np.uint8)


def _decode_masks_to_torch_bool(pred_masks: List, h: int, w: int) -> torch.Tensor:
    bin_masks = [_decode_single_mask(m, h, w) for m in pred_masks]
    masks_np = np.stack(bin_masks, axis=0).astype(np.uint8)  # (N, H, W)
    return torch.from_numpy(masks_np > 0)


def remove_overlapping_masks(sample: Dict, iom_thresh: float = 0.3) -> Dict:
    """
    Greedy keep: sort by score desc; keep a mask if IoM to all kept masks <= threshold.
    If pred_masks has length 0 or 1, returns sample unchanged (no extra keys).
    """
    # Basic presence checks
    if "pred_masks" not in sample or not isinstance(sample["pred_masks"], list):
        return sample  # nothing to do / preserve as-is

    pred_masks = sample["pred_masks"]
    N = len(pred_masks)

    # --- Early exit: 0 or 1 mask -> do NOT modify the JSON at all ---
    if N <= 1:
        return sample

    # From here on we have at least 2 masks
    h = int(sample["orig_img_h"])
    w = int(sample["orig_img_w"])
    pred_scores = sample.get("pred_scores", [1.0] * N)  # fallback if scores missing
    pred_boxes = sample.get("pred_boxes", None)

    assert N == len(pred_scores), "pred_masks and pred_scores must have same length"
    if pred_boxes is not None:
        assert N == len(pred_boxes), "pred_masks and pred_boxes must have same length"

    masks_bool = _decode_masks_to_torch_bool(pred_masks, h, w)  # (N, H, W)

    order = sorted(range(N), key=lambda i: float(pred_scores[i]), reverse=True)
    kept_idx: List[int] = []
    kept_masks: List[torch.Tensor] = []

    for i in order:
        cand = masks_bool[i].unsqueeze(0)  # (1, H, W)
        if len(kept_masks) == 0:
            kept_idx.append(i)
            kept_masks.append(masks_bool[i])
            continue

        kept_stack = torch.stack(kept_masks, dim=0)  # (K, H, W)
        iom_vals = mask_iom(cand, kept_stack).squeeze(0)  # (K,)
        if torch.any(iom_vals > iom_thresh):
            continue  # overlaps too much with a higher-scored kept mask
        kept_idx.append(i)
        kept_masks.append(masks_bool[i])

    kept_idx_sorted = sorted(kept_idx)

    # Build filtered JSON (this *does* modify fields; only for N>=2 case)
    out = dict(sample)
    out["pred_masks"] = [pred_masks[i] for i in kept_idx_sorted]
    out["pred_scores"] = [pred_scores[i] for i in kept_idx_sorted]
    if pred_boxes is not None:
        out["pred_boxes"] = [pred_boxes[i] for i in kept_idx_sorted]
    out["kept_indices"] = kept_idx_sorted
    out["removed_indices"] = [i for i in range(N) if i not in set(kept_idx_sorted)]
    out["iom_threshold"] = float(iom_thresh)
    return out
