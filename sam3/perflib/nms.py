# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

import torch

from sam3.perflib.iou import pairwise_iou


def nms_masks_kernel(
    masks: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
):
    keep = [True for _ in range(len(masks))]

    # Sort by score descending
    order = scores.argsort(descending=True)
    masks = masks[order]
    scores = scores[order]
    labels = labels[order]

    # Compute pairwise IoU matrix
    iou_matrix = pairwise_iou(masks, masks)

    # Branchy code with small matrices is much faster on the CPU
    threshold_mask = (iou_matrix > iou_threshold).cpu().numpy().tolist()
    labels = labels.tolist()

    for i in range(len(masks)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(masks)):
            if not keep[j]:
                continue
            # Only suppress if labels match AND IoU is high
            if labels[i] == labels[j] and threshold_mask[i][j]:
                keep[j] = False

    return torch.tensor(keep).pin_memory().to(device=masks.device, non_blocking=True)


def nms_masks(det_out, iou_threshold=0.5):
    with torch.autograd.profiler.record_function("perflib: nms_masks"):
        if det_out["mask"].shape[0] == 0:
            keep = torch.ones(0, dtype=torch.bool, device=det_out["mask"].device)
            return det_out, keep
        for key in ["mask", "scores", "labels"]:
            if key not in det_out:
                raise ValueError(f"Expected key {key} in input det_out")
            if not (isinstance(det_out[key], torch.Tensor) and det_out[key].is_cuda):
                raise ValueError(f"Expected det_out[{key}] to be a CUDA Tensor.")
        masks = det_out["mask"] > 0
        scores = det_out["scores"]
        labels = det_out["labels"]
        if not (
            (masks.shape[0] == scores.shape[0]) and (scores.shape[0] == labels.shape[0])
        ):
            raise ValueError(
                f"Expected leading size of masks, scores and labels to match"
            )
        keep = nms_masks_kernel(masks, scores, labels, iou_threshold)
        # Reduces the number of nonzero (sync) calls to 1
        # index_select doesn't cause a sync, because shapes are known
        keep_idx = torch.nonzero(keep, as_tuple=True)[0]
        for k, v in det_out.items():
            # Assumes all values in det_out are tensors. Otherwise, it's good to raise an error here.
            det_out[k] = torch.index_select(v, 0, keep_idx)
        return det_out, keep
