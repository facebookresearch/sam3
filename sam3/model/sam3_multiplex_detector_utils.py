import logging

import numpy as np
import torch
from sam3 import perflib

try:
    # Ronghang's generic GPU NMS implementation; install via
    # pip uninstall -y torch_generic_nms; TORCH_CUDA_ARCH_LIST="8.0 9.0" pip install git+https://github.com/ronghanghu/torch_generic_nms
    from torch_generic_nms import generic_nms

    GENERIC_NMS_AVAILABLE = True
except ImportError:
    GENERIC_NMS_AVAILABLE = False

from sam3.perflib.masks_ops import mask_iou
from sam3.train.masks_ops import mask_iom


def nms_masks(
    pred_probs: torch.Tensor,
    pred_masks: torch.Tensor,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
    do_compile: bool = False,
    running_in_prod: bool = False,
) -> torch.Tensor:
    """
    Args:
      - pred_probs: (num_det,) or (B, num_det) float Tensor, containing the score (probability) of each detection
      - pred_masks: (num_det, H_mask, W_mask) or (B, num_det, H_mask, W_mask) float Tensor, containing the binary segmentation mask of each detection
      - prob_threshold: float, score threshold to prefilter detections (NMS is performed on detections above threshold)
      - iou_threshold: float, mask IoU threshold for NMS (it would also be used as IoM threshold if `nms_use_iom` is True)
      - nms_use_iom: bool, if True, use IoM instead of IoU for NMS
      - do_compile: bool, whether to compile the function for optimization
      - running_in_prod: bool, whether the function is running in production (ie, in Instagram)

    Returns:
     - keep: (num_det,) or (B, num_det) bool Tensor, indicating whether each detection is kept after score thresholding + NMS
    """
    if do_compile and perflib.is_enabled:
        # Apply torch.compile with the same settings as before
        compiled_fn = torch.compile(
            _nms_masks_core,
            mode="max-autotune",
            fullgraph=True,
            # dynamic=False,
        )
        return compiled_fn(
            pred_probs, pred_masks, prob_threshold, iou_threshold, nms_use_iom
        )
    else:
        return _nms_masks_core(
            pred_probs, pred_masks, prob_threshold, iou_threshold, nms_use_iom
        )


def _nms_masks_core(
    pred_probs: torch.Tensor,
    pred_masks: torch.Tensor,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
) -> torch.Tensor:
    """Core NMS implementation without compilation.

    Supports both single-frame and batched inputs:
      - Single-frame: pred_probs (num_det,), pred_masks (num_det, H, W)
      - Batched: pred_probs (B, num_det), pred_masks (B, num_det, H, W)

    Returns:
      - keep: bool Tensor with same leading dimensions as input, indicating kept detections
    """
    # Check if input is batched (has batch dimension)
    is_batched = pred_probs.dim() == 2

    if is_batched:
        return _nms_masks_core_batched(
            pred_probs, pred_masks, prob_threshold, iou_threshold, nms_use_iom
        )
    else:
        # Single-frame input: use original logic
        return _nms_masks_core_single(
            pred_probs, pred_masks, prob_threshold, iou_threshold, nms_use_iom
        )


def _nms_masks_core_batched(
    pred_probs: torch.Tensor,
    pred_masks: torch.Tensor,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
) -> torch.Tensor:
    """Core NMS implementation for batched inputs using vectorized operations.

    Args:
      - pred_probs: (B, num_det) float Tensor
      - pred_masks: (B, num_det, H_mask, W_mask) float Tensor
      - prob_threshold: float, score threshold to prefilter detections
      - iou_threshold: float, mask IoU/IoM threshold for NMS
      - nms_use_iom: bool, if True, use IoM instead of IoU for NMS

    Returns:
      - keep: (B, num_det) bool Tensor
    """
    B, num_det, H, W = pred_masks.shape
    device = pred_masks.device

    is_valid = pred_probs > prob_threshold  # (B, num_det)
    masks_binary = pred_masks > 0  # (B, num_det, H, W)

    if perflib.is_enabled:
        # Compute batched pairwise IoU/IoM
        if nms_use_iom:
            overlaps = _batched_mask_iom(masks_binary)  # (B, num_det, num_det)
        else:
            overlaps = _batched_mask_iou(masks_binary)  # (B, num_det, num_det)
        keep = _batched_generic_nms_mask(overlaps, pred_probs, is_valid, iou_threshold)
        return keep

    # Non-perflib path: compute batched IoU/IoM
    if nms_use_iom:
        overlaps = _batched_mask_iom(masks_binary)  # (B, num_det, num_det)
    else:
        overlaps = _batched_mask_iou(masks_binary)  # (B, num_det, num_det)

    # Apply batched NMS
    keep = _batched_generic_nms_mask(overlaps, pred_probs, is_valid, iou_threshold)
    return keep


def _batched_mask_iou(masks: torch.Tensor) -> torch.Tensor:
    """Compute batched pairwise IoU for masks.

    Args:
      - masks: (B, N, H, W) bool Tensor

    Returns:
      - ious: (B, N, N) float Tensor
    """
    B, N, H, W = masks.shape
    # Flatten spatial dims: (B, N, H*W)
    masks_flat = masks.reshape(B, N, -1).float()

    # Compute intersection via batched matrix multiplication: (B, N, N)
    intersection = torch.bmm(masks_flat, masks_flat.transpose(1, 2))

    # Compute areas: (B, N)
    areas = masks_flat.sum(dim=-1)

    # Compute union: (B, N, N)
    union = areas.unsqueeze(2) + areas.unsqueeze(1) - intersection

    return intersection / (union + 1e-8)


def _batched_mask_iom(masks: torch.Tensor) -> torch.Tensor:
    """Compute batched pairwise IoM (Intersection over Minimum) for masks.

    Args:
      - masks: (B, N, H, W) bool Tensor

    Returns:
      - ioms: (B, N, N) float Tensor
    """
    B, N, H, W = masks.shape
    # Flatten spatial dims: (B, N, H*W)
    masks_flat = masks.reshape(B, N, -1).float()

    # Compute intersection via batched matrix multiplication: (B, N, N)
    intersection = torch.bmm(masks_flat, masks_flat.transpose(1, 2))

    # Compute areas: (B, N)
    areas = masks_flat.sum(dim=-1)

    # Compute min area: (B, N, N)
    min_area = torch.minimum(areas.unsqueeze(2), areas.unsqueeze(1))

    return intersection / (min_area + 1e-8)


def _batched_generic_nms_mask(
    ious: torch.Tensor,
    scores: torch.Tensor,
    is_valid: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Batched NMS using vectorized operations.

    Args:
      - ious: (B, N, N) float Tensor, pairwise IoU/IoM matrix
      - scores: (B, N) float Tensor, detection scores
      - is_valid: (B, N) bool Tensor, valid detections mask
      - iou_threshold: float, threshold for suppression

    Returns:
      - keep: (B, N) bool Tensor
    """
    B, N = scores.shape
    device = scores.device

    # Sort by score descending for each batch: (B, N)
    order = scores.argsort(dim=-1, descending=True)

    # Create batch indices for advanced indexing
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)

    # Reorder IoU matrix according to sorted scores: (B, N, N)
    # ious_sorted[b, i, j] = ious[b, order[b, i], order[b, j]]
    ious_sorted = ious[batch_idx.unsqueeze(2), order.unsqueeze(2), order.unsqueeze(1)]

    # Create threshold mask: (B, N, N)
    threshold_mask = ious_sorted > iou_threshold

    # Initialize keep mask with valid detections in sorted order: (B, N)
    keep = is_valid[batch_idx, order]

    # Upper triangular mask to avoid double processing: (N, N)
    triu = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)

    # Vectorized NMS - iterate through detections
    for i in range(N):
        # For each position i, suppress later detections with high overlap
        # Only suppress if current detection is kept
        suppress = (
            threshold_mask[:, i, :] & triu[i].unsqueeze(0) & keep[:, i].unsqueeze(1)
        )
        keep = keep & ~suppress

    # Return keep mask in original order: (B, N)
    original_keep = torch.zeros_like(keep)
    original_keep[batch_idx, order] = keep
    return original_keep


def _nms_masks_core_single(
    pred_probs: torch.Tensor,
    pred_masks: torch.Tensor,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
) -> torch.Tensor:
    """Core NMS implementation for a single frame (no batch dimension).

    Args:
      - pred_probs: (num_det,) float Tensor
      - pred_masks: (num_det, H_mask, W_mask) float Tensor
      - prob_threshold: float, score threshold to prefilter detections
      - iou_threshold: float, mask IoU/IoM threshold for NMS
      - nms_use_iom: bool, if True, use IoM instead of IoU for NMS

    Returns:
      - keep: (num_det,) bool Tensor
    """
    is_valid = pred_probs > prob_threshold  # (num_det,)

    if perflib.is_enabled:
        masks_binary = pred_masks > 0  # (num_det, H_mask, W_mask)
        if nms_use_iom:
            ious = perf_mask_iom(masks_binary, masks_binary)  # (num_det, num_det)
        else:
            ious = perf_mask_iou(masks_binary, masks_binary)  # (num_det, num_det)
        kept_mask = generic_nms_mask(ious, pred_probs, is_valid, iou_threshold)
        return kept_mask
    # prefilter the detections with prob_threshold ("valid" are those above prob_threshold)
    probs = pred_probs[is_valid]  # (num_valid,)
    masks_binary = pred_masks[is_valid] > 0  # (num_valid, H_mask, W_mask)
    if probs.numel() == 0:
        return is_valid  # no valid detection, return empty keep mask

    if nms_use_iom:
        overlaps = mask_iom(masks_binary, masks_binary)  # (num_valid, num_valid)
    else:
        overlaps = mask_iou(masks_binary, masks_binary)  # (num_valid, num_valid)
    # kept_inds are the indices among `probs` of those kept detections after NMS
    if GENERIC_NMS_AVAILABLE:
        kept_inds = generic_nms(overlaps, probs, iou_threshold, use_iou_matrix=True)
    else:
        logging.warning(
            "Falling back to CPU mask NMS implementation -- please install `torch_generic_nms` via\n\t"
            'pip uninstall -y torch_generic_nms; TORCH_CUDA_ARCH_LIST="8.0 9.0" pip install git+https://github.com/ronghanghu/torch_generic_nms'
        )
        kept_inds = generic_nms_cpu(overlaps, probs, iou_threshold)

    # valid_inds are the indices among `probs` of valid detections before NMS (or -1 for invalid)
    valid_inds = torch.where(is_valid, is_valid.cumsum(dim=0) - 1, -1)  # (num_det,)
    keep = torch.isin(valid_inds, kept_inds)  # (num_det,)
    return keep


def generic_nms_cpu(
    ious: torch.Tensor, scores: torch.Tensor, iou_threshold=0.5
) -> torch.Tensor:
    """
    A generic version of `torchvision.ops.nms` that takes a pairwise IoU matrix. (CPU implementation
    based on https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/model/nms/nms_cpu.py)
    """
    ious_np = ious.float().detach().cpu().numpy()
    scores_np = scores.float().detach().cpu().numpy()
    order = scores_np.argsort()[::-1]
    kept_inds = []
    while order.size > 0:
        i = order.item(0)
        kept_inds.append(i)
        inds = np.where(ious_np[i, order[1:]] <= iou_threshold)[0]
        order = order[inds + 1]

    return torch.tensor(kept_inds, dtype=torch.int64, device=scores.device)


def generic_nms_mask(
    ious: torch.Tensor, scores: torch.Tensor, is_valid: torch.Tensor, iou_threshold=0.5
) -> torch.Tensor:
    """
    A generic version of `torchvision.ops.nms` that takes a pairwise IoU matrix. (CPU implementation
    using vectorized operations similar to nms_masks_kernel)
    """
    # Sort by score descending
    order = scores.argsort(descending=True)

    # Reorder IoU matrix according to sorted scores
    ious_sorted = ious[order][:, order]

    # Create threshold mask
    threshold_mask = ious_sorted > iou_threshold

    # Initialize keep mask
    # keep = torch.ones(len(scores), device=ious.device, dtype=torch.bool)
    keep = is_valid[order]

    # Upper triangular mask to avoid double processing
    tr = torch.triu(torch.ones_like(threshold_mask), diagonal=1)

    # Vectorized NMS
    for i in range(len(scores)):
        # Suppress all boxes that have high IoU with current box
        m = threshold_mask[i]
        keep = torch.where(m & tr[i], torch.zeros_like(keep), keep)

    # Return keep mask in original order
    original_keep = torch.zeros_like(keep)
    original_keep[order] = keep
    return original_keep


def perf_mask_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the IoU (Intersection over Union) between predicted masks and ground truth masks.

    Args:
      - pred_masks: (N, H, W) bool Tensor, containing binary predicted segmentation masks
      - gt_masks: (M, H, W) bool Tensor, containing binary ground truth segmentation masks

    Returns:
      - ious: (N, M) float Tensor, containing IoUs for each pair of predicted and ground truth masks
    """
    assert pred_masks.dtype == gt_masks.dtype == torch.bool
    from sam3.perflib.iou import pairwise_iou

    return pairwise_iou(pred_masks, gt_masks, eps=None)


def perf_mask_iom(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    assert pred_masks.dtype == gt_masks.dtype == torch.bool
    from sam3.perflib.iou import pairwise_iom

    return pairwise_iom(pred_masks, gt_masks)
