# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import logging

import numpy as np
import torch

from sam3.perflib.masks_ops import mask_iou


try:
    from torch_generic_nms import generic_nms as generic_nms_cuda

    GENERIC_NMS_AVAILABLE = True
except ImportError:
    logging.debug(
        "Falling back to triton or CPU mask NMS implementation -- please install `torch_generic_nms` via\n\t"
        'pip uninstall -y torch_generic_nms; TORCH_CUDA_ARCH_LIST="8.0 9.0" pip install git+https://github.com/ronghanghu/torch_generic_nms'
    )
    GENERIC_NMS_AVAILABLE = False


def nms_masks(
    pred_probs: torch.Tensor,
    pred_masks: torch.Tensor,
    prob_threshold: float,
    iou_threshold: float,
) -> torch.Tensor:
    """
    Args:
      - pred_probs: (num_det,) float Tensor, containing the score (probability) of each detection
      - pred_masks: (num_det, H_mask, W_mask) float Tensor, containing the binary segmentation mask of each detection
      - prob_threshold: float, score threshold to prefilter detections (NMS is performed on detections above threshold)
      - iou_threshold: float, mask IoU threshold for NMS

    Returns:
     - keep: (num_det,) bool Tensor, indicating whether each detection is kept after score thresholding + NMS
    """
    # prefilter the detections with prob_threshold ("valid" are those above prob_threshold)
    is_valid = pred_probs > prob_threshold  # (num_det,)
    probs = pred_probs[is_valid]  # (num_valid,)
    masks_binary = pred_masks[is_valid] > 0  # (num_valid, H_mask, W_mask)
    if probs.numel() == 0:
        return is_valid  # no valid detection, return empty keep mask

    ious = mask_iou(masks_binary, masks_binary)  # (num_valid, num_valid)
    kept_inds = generic_nms(ious, probs, iou_threshold)

    # valid_inds are the indices among `probs` of valid detections before NMS (or -1 for invalid)
    valid_inds = torch.where(is_valid, is_valid.cumsum(dim=0) - 1, -1)  # (num_det,)
    keep = torch.isin(valid_inds, kept_inds)  # (num_det,)
    return keep


def generic_nms(
    ious: torch.Tensor, scores: torch.Tensor, iou_threshold=0.5
) -> torch.Tensor:
    """A generic version of `torchvision.ops.nms` that takes a pairwise IoU matrix."""

    assert ious.dim() == 2 and ious.size(0) == ious.size(1)
    assert scores.dim() == 1 and scores.size(0) == ious.size(0)

    if ious.is_cuda:
        if GENERIC_NMS_AVAILABLE:
            return generic_nms_cuda(ious, scores, iou_threshold, use_iou_matrix=True)
        else:
            from sam3.perflib.triton.nms import nms_triton

            return nms_triton(ious, scores, iou_threshold)

    return generic_nms_cpu(ious, scores, iou_threshold)


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
