"""Utilities for masks manipulation"""

import numpy as np
import pycocotools.mask as maskUtils
import torch


def instance_masks_to_semantic_masks(
    instance_masks: torch.Tensor, num_instances: torch.Tensor
) -> torch.Tensor:
    """This function converts instance masks to semantic masks.
    It accepts a collapsed batch of instances masks (ie all instance masks are concatenated in a single tensor) and
    the number of instances in each image of the batch.
    It returns a mask with the same spatial dimensions as the input instance masks, where for each batch element the
    semantic mask is the union of all the instance masks in the batch element.

    If for a given batch element there are no instances (ie num_instances[i]==0), the corresponding semantic mask will be a tensor of zeros.

    Args:
        instance_masks (torch.Tensor): A tensor of shape (N, H, W) where N is the number of instances in the batch.
        num_instances (torch.Tensor): A tensor of shape (B,) where B is the batch size. It contains the number of instances
            in each image of the batch.

    Returns:
        torch.Tensor: A tensor of shape (B, H, W) where B is the batch size and H, W are the spatial dimensions of the
            input instance masks.
    """

    masks_per_query = torch.split(instance_masks, num_instances.tolist())

    return torch.stack([torch.any(masks, dim=0) for masks in masks_per_query], dim=0)


def mask_iou(masks1, masks2):
    """
    Compute the intersection over union of two sets of masks.
    The function is symmetric.

    Args:
        masks1: torch.Tensor of shape (N, H, W) where N is the number of masks.
        masks2: torch.Tensor of shape (M, H, W) where N is the number of masks.

    Returns:
        torch.Tensor of shape (N, M) where the i-th row and j-th column contains the IoU between masks1[i] and masks2[j].
    """
    assert masks1.shape[1:] == masks2.shape[1:]
    assert masks1.dtype == torch.bool and masks2.dtype == torch.bool

    intersection = (masks1[:, None] * masks2[None]).flatten(-2).sum(-1)
    area1 = masks1.flatten(-2).sum(-1)
    area2 = masks2.flatten(-2).sum(-1)

    union = area1[:, None] + area2[None] - intersection
    return intersection / (union + 1e-8)


def mask_intersection(masks1, masks2, block_size=16):
    """Compute the intersection of two sets of masks, without blowing the memory"""

    assert masks1.shape[1:] == masks2.shape[1:]
    assert masks1.dtype == torch.bool and masks2.dtype == torch.bool

    result = torch.zeros(
        masks1.shape[0], masks2.shape[0], device=masks1.device, dtype=torch.long
    )
    for i in range(0, masks1.shape[0], block_size):
        for j in range(0, masks2.shape[0], block_size):
            intersection = (
                (masks1[i : i + block_size, None] * masks2[None, j : j + block_size])
                .flatten(-2)
                .sum(-1)
            )
            result[i : i + block_size, j : j + block_size] = intersection
    return result


def mask_iom(masks1, masks2):
    """
    Similar to IoU, except the denominator is the area of the smallest mask
    """
    assert masks1.shape[1:] == masks2.shape[1:]
    assert masks1.dtype == torch.bool and masks2.dtype == torch.bool

    # intersection = (masks1[:, None] * masks2[None]).flatten(-2).sum(-1)
    intersection = mask_intersection(masks1, masks2)
    area1 = masks1.flatten(-2).sum(-1)
    area2 = masks2.flatten(-2).sum(-1)
    min_area = torch.min(area1[:, None], area2[None, :])
    return intersection / (min_area + 1e-8)


def compute_boundary(seg):
    """
    Adapted from https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/j_and_f.py#L148
    Return a 1pix wide boundary of the given mask
    """
    assert seg.ndim >= 2
    e = torch.zeros_like(seg)
    s = torch.zeros_like(seg)
    se = torch.zeros_like(seg)

    e[..., :, :-1] = seg[..., :, 1:]
    s[..., :-1, :] = seg[..., 1:, :]
    se[..., :-1, :-1] = seg[..., 1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[..., -1, :] = seg[..., -1, :] ^ e[..., -1, :]
    b[..., :, -1] = seg[..., :, -1] ^ s[..., :, -1]
    b[..., -1, -1] = 0
    return b


def dilation(mask, kernel_size):
    """
    Implements the dilation operation. If the input is on cpu, we call the cv2 version.
    Otherwise, we implement it using a convolution

    The kernel is assumed to be a square kernel

    """

    assert mask.ndim == 3
    kernel_size = int(kernel_size)
    assert (
        kernel_size % 2 == 1
    ), f"Dilation expects a odd kernel size, got {kernel_size}"

    if mask.is_cuda:
        m = mask.unsqueeze(1).to(torch.float16)
        k = torch.ones(1, 1, kernel_size, 1, dtype=m.dtype, device=m.device)

        result = torch.nn.functional.conv2d(m, k, padding="same")
        result = torch.nn.functional.conv2d(result, k.transpose(-1, -2), padding="same")
        return result.view_as(mask) > 0

    all_masks = mask.view(-1, mask.size(-2), mask.size(-1)).numpy().astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    import cv2

    processed = [torch.from_numpy(cv2.dilate(m, kernel)) for m in all_masks]
    return torch.stack(processed).view_as(mask).to(mask)


def compute_F_measure(
    gt_boundary_rle, gt_dilated_boundary_rle, dt_boundary_rle, dt_dilated_boundary_rle
):
    """Adapted from https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/j_and_f.py#L207

    Assumes the boundary and dilated boundaries have already been computed and converted to RLE
    """
    gt_match = maskUtils.merge([gt_boundary_rle, dt_dilated_boundary_rle], True)
    dt_match = maskUtils.merge([dt_boundary_rle, gt_dilated_boundary_rle], True)

    n_dt = maskUtils.area(dt_boundary_rle)
    n_gt = maskUtils.area(gt_boundary_rle)
    # % Compute precision and recall
    if n_dt == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_dt > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_dt == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = maskUtils.area(dt_match) / float(n_dt)
        recall = maskUtils.area(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        f_val = 0
    else:
        f_val = 2 * precision * recall / (precision + recall)

    return f_val


def mask_iom_nms(scores, masks, score_thresh=0.5, mask_threshold=0.5, iom_thresh=0.5):
    """
    scores: [N]
    masks: [N, K, K]
    N is number of queries
    """
    assert scores.ndim == 1, "please check scores input shape"
    assert masks.ndim == 3, "please check masks input shape"
    keep = scores.squeeze(-1) >= score_thresh
    if keep.sum() < 1:
        return scores[keep], masks[keep]
    maskids = torch.where(keep)[0]

    scores_subset = scores[keep]
    masks_subset = masks[keep].sigmoid() >= mask_threshold

    N = scores_subset.size(0)

    # sort by score (desc), and compute IoM matrix on the sorted boxes
    order = scores_subset.argsort(descending=True)
    masks_sorted = masks_subset[order]
    pairwise_iou = mask_iom(masks_sorted, masks_sorted)  # (N, N)

    keep_mask = torch.zeros(N, dtype=torch.bool, device=scores_subset.device)
    suppressed = torch.zeros(N, dtype=torch.bool, device=scores_subset.device)

    for i in range(N):
        if suppressed[i]:
            continue
        keep_mask[i] = True

        # suppress all j>i that have IoU > threshold with box i
        sup = pairwise_iou[i] > iom_thresh
        # only affect the *future* candidates
        if i + 1 < N:
            suppressed[i + 1 :] |= sup[i + 1 :]

    # map back to original indices
    kept_sorted_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
    maskids_keep_order = order[kept_sorted_idx]
    maskids_keep = torch.tensor([maskids[i] for i in maskids_keep_order])

    new_scores, new_masks = scores[maskids_keep], masks[maskids_keep]

    return new_scores, new_masks
