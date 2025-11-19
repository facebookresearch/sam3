# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# Adapted from https://github.com/stackav-oss/conch/blob/main/conch/kernels/vision/nms.py

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"cxpr_block_size": 128}),
        triton.Config({"cxpr_block_size": 256}),
        triton.Config({"cxpr_block_size": 512}),
        triton.Config({"cxpr_block_size": 1024}),
        triton.Config({"cxpr_block_size": 2048}),
        triton.Config({"cxpr_block_size": 4096}),
        triton.Config({"cxpr_block_size": 8192}),
    ],
    key=["num_boxes"],
)
@triton.jit
def _nms_suppression_kernel(
    # Tensors
    iou_mask_ptr: tl.tensor,  # [N, N]
    keep_mask_ptr: tl.tensor,  # [N]
    # Scalars
    num_boxes: tl.int32,
    # Strides
    iou_mask_stride: tl.int32,
    # Constexprs
    cxpr_block_size: tl.constexpr,
) -> None:
    """NMS suppression kernel.

    Args:
        iou_mask_ptr: Pointer to precomputed IoU mask, shape: (N, N).
        keep_mask_ptr: Pointer to keep mask tensor, shape: (N,).
        num_boxes: Number of boxes.
        iou_mask_stride: Stride for IoU mask tensor.
        cxpr_block_size: Block size for processing.
    """
    # Sequential NMS: for each box in sorted order, suppress later boxes
    for current_box_idx in range(num_boxes - 1):
        # Check if current box is still kept
        is_kept = tl.load(keep_mask_ptr + current_box_idx)
        if is_kept:
            # IoU mask row offset for the current box
            # Because the IoU mask is sorted by score, we will only consider boxes that come after the current box.
            # This means we only need to read the upper triangular part of the IoU mask.
            iou_row_offset = current_box_idx * iou_mask_stride

            # Only process boxes that come after the current box
            next_box_idx = current_box_idx + 1
            remaining_boxes = num_boxes - next_box_idx

            # Iterate blockwise through the columns
            for block_idx in range(tl.cdiv(remaining_boxes, cxpr_block_size)):
                # Masked load of indices for the target boxes in the current block
                block_start = next_box_idx + block_idx * cxpr_block_size
                target_box_offsets = block_start + tl.arange(0, cxpr_block_size)
                target_box_mask = target_box_offsets < num_boxes

                # Suppress boxes with lower scores that have high IoU
                suppression_mask = tl.load(
                    iou_mask_ptr + iou_row_offset + target_box_offsets,
                    mask=target_box_mask,
                    other=False,
                )
                suppression_mask = tl.cast(suppression_mask, tl.int1)

                # Conditionally store suppression result for high-IoU boxes
                tl.store(
                    keep_mask_ptr + target_box_offsets, False, mask=suppression_mask
                )

            # Potential race condition: we need to ensure all threads complete the store before the next
            # iteration otherwise we may load stale data for whether or not a box has been suppressed.
            tl.debug_barrier()


def nms_triton(
    ious: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Perform NMS given the iou matrix, the scores and the iou threshold

    Args:
        ious: Pairwise IoU tensor of shape (N, N).
        scores: Scores tensor of shape (N,).
        iou_threshold: IoU threshold for suppression.

    Returns:
        Tensor: Indices of kept boxes, sorted by decreasing score.
    """
    assert scores.dim() == 1, "Scores must be 1D"
    iou_mask = ious > iou_threshold
    assert iou_mask.dim() == 2
    assert iou_mask.shape[0] == iou_mask.shape[1] == scores.shape[0]
    assert iou_mask.device == scores.device
    assert iou_mask.dtype == torch.bool

    num_boxes = scores.size(0)
    keep_mask = torch.ones(len(scores), device=scores.device, dtype=torch.bool)

    # Sort boxes by scores in descending order
    _, sorted_indices = torch.sort(scores, dim=0, stable=True, descending=True)
    iou_mask = iou_mask[sorted_indices][:, sorted_indices].contiguous()

    # For the suppression stage, we need to process sequentially, but we'll still take
    # advantage of parallelism by processing in blocks in one program.
    stage2_grid = (1,)
    _nms_suppression_kernel[stage2_grid](
        # Tensors
        iou_mask_ptr=iou_mask,
        keep_mask_ptr=keep_mask,
        # Scalars
        num_boxes=num_boxes,
        # Strides
        iou_mask_stride=iou_mask.stride(0),
    )
    # Extract indices of kept boxes
    return sorted_indices[keep_mask]
