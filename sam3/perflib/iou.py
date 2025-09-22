# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

import torch


def pairwise_iou(pred_masks, gt_masks, eps=1e-6):
    N, H, W = pred_masks.shape
    M = gt_masks.shape[0]
    # Flatten and convert to float for matmul
    pred_flat = pred_masks.reshape(N, -1).float()
    gt_flat = gt_masks.reshape(M, -1).float()
    # Intersection: (N, M)
    intersection = torch.matmul(pred_flat, gt_flat.t())
    # Areas
    area_pred = pred_flat.sum(dim=1, keepdim=True)  # (N, 1)
    area_gt = gt_flat.sum(dim=1, keepdim=True)  # (M, 1)
    # Union: (N, M)
    union = area_pred + area_gt.t() - intersection
    iou = intersection / (union + eps)
    return iou  # shape: (N, M)
