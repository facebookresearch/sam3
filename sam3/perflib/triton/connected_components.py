# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _any_combine(a, b):
    return a | b


@triton.jit
def tl_any(a, dim=0):
    return tl.reduce(a, dim, _any_combine)


# ==============================================================================
# ## Phase 1: Initialization Kernel
# ==============================================================================
# Each foreground pixel (value > 0) gets a unique label equal to its
# linear index. Background pixels (value == 0) get a sentinel label of -1.
# Note that the indexing is done across batch boundaries for simplicity
# (i.e., the first pixel of image 1 gets label H*W, etc.)


@triton.jit
def _init_labels_kernel(
    input_ptr, labels_ptr, numel: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    input_values = tl.load(input_ptr + offsets, mask=mask, other=0)

    indices = tl.where((input_values != 0), offsets, -1)
    tl.store(labels_ptr + offsets, indices, mask=mask)


# ==============================================================================
# ## Phase 2: Local merging
# ==============================================================================
# Each pixel tries to merge with its 8-connected neighbors (up, down, left, right)
# if they have the same value. This is done using a disjoint-set union operation.


@triton.jit
def find(labels_ptr, indices, mask):
    current_pids = indices

    # 'is_done' tracks lanes that have finished their work.
    # A lane is initially "done" if it's not active (mask is False).
    is_done = ~mask

    # Loop as long as there is at least one lane that is NOT done.
    while tl_any(~is_done):
        # The work_mask is for lanes that are still active and seeking their root.
        work_mask = ~is_done
        parents = tl.load(labels_ptr + current_pids, mask=work_mask, other=-1)
        # A lane is now done if its parent is itself (it's a root)
        # or if it hits a -1 sentinel (a safe exit condition).
        is_root = parents == current_pids
        is_sentinel = parents == -1
        is_done |= is_root | is_sentinel

        # For lanes that are not yet done, update their pid to their parent to continue traversal.
        current_pids = tl.where(is_done, current_pids, parents)
    # We could add the following line to do path compression, but experimentally it's slower
    # tl.atomic_min(labels_ptr + indices, current_pids, mask=mask)
    return current_pids


@triton.jit
def union(labels_ptr, a, b, process_mask):
    # This function implements a disjoint-set union
    # As an invariant, we use the fact that the roots have the lower id. That helps parallelization
    # However, that is not sufficient by itself. Suppose two threads want to do union(0,2) and union(1,2) at the same time
    # Then if we do a naive atomic_min, 0 and 1 will compete to be the new parent of 2 and min(0, 1) will win.
    # However, 1 still needs to be merged with the new {0, 2} component.
    # To ensure that merge is also done, we need to detect whether the merge was successful, and if not retry until it is

    current_a = a
    current_b = b

    final_root = a
    # A mask to track which lanes have successfully completed their union.
    done_mask = ~process_mask  # tl.zeros_like(a) == 1  # Init with all False

    while tl_any(~done_mask):
        # Define the mask for lanes that still need work in this iteration
        work_mask = process_mask & ~done_mask

        # Find the roots for the current a and b values in the active lanes
        root_a = find(labels_ptr, current_a, work_mask)
        tl.debug_barrier()
        root_b = find(labels_ptr, current_b, work_mask)

        # 7. Merge logic
        # If roots are already the same, the sets are already merged. Mark as done.
        are_equal = root_a == root_b
        final_root = tl.where(are_equal & work_mask & ~done_mask, root_a, final_root)
        done_mask |= are_equal & work_mask

        # Define masks for the two merge cases (a < b or b < a)
        a_is_smaller = root_a < root_b

        # Case 1: root_a < root_b. Attempt to set parent[root_b] = root_a
        merge_mask_a_smaller = work_mask & a_is_smaller & ~are_equal
        ptr_b = labels_ptr + root_b
        old_val_b = tl.atomic_min(ptr_b, root_a, mask=merge_mask_a_smaller)

        # A lane is done if its atomic op was successful (old value was what we expected)
        success_b = old_val_b == root_b
        final_root = tl.where(success_b & work_mask & ~done_mask, root_a, final_root)
        done_mask |= success_b & merge_mask_a_smaller

        # *** Crucial Retry Logic ***
        # If the update failed (old_val_b != root_b), another thread interfered.
        # We update `current_b` to this new root (`old_val_b`) and will retry in the next loop iteration.
        current_b = tl.where(success_b | ~merge_mask_a_smaller, current_b, old_val_b)

        # Case 2: root_b < root_a. Attempt to set parent[root_a] = root_b
        merge_mask_b_smaller = work_mask & ~a_is_smaller & ~are_equal
        ptr_a = labels_ptr + root_a
        old_val_a = tl.atomic_min(ptr_a, root_b, mask=merge_mask_b_smaller)

        success_a = old_val_a == root_a
        final_root = tl.where(success_a & work_mask & ~done_mask, root_b, final_root)
        done_mask |= success_a & merge_mask_b_smaller

        # *** Crucial Retry Logic ***
        # Similarly, update `current_a` if the atomic operation failed.
        current_a = tl.where(success_a | ~merge_mask_b_smaller, current_a, old_val_a)

    return final_root


@triton.jit
def _merge_helper(
    input_ptr,
    labels_ptr,
    base_offset,
    offsets_h,
    offsets_w,
    mask_2d,
    valid_current,
    current_values,
    current_labels,
    H,
    W,
    dx: tl.constexpr,
    dy: tl.constexpr,
):
    # Helper functions to compute merge with a specific neighbor offset (dx, dy)

    neighbor_h = offsets_h + dy
    neighbor_w = offsets_w + dx
    # Proper bounds checking: all four bounds must be satisfied
    mask_n = (
        mask_2d
        & (neighbor_h[:, None] >= 0)
        & (neighbor_h[:, None] < H)
        & (neighbor_w[None, :] >= 0)
        & (neighbor_w[None, :] < W)
    )

    offsets_neighbor = neighbor_h[:, None] * W + neighbor_w[None, :]
    neighbor_values = tl.load(
        input_ptr + base_offset + offsets_neighbor, mask=mask_n, other=-1
    )

    mask_n = tl.ravel(mask_n)
    neighbor_labels = tl.load(
        labels_ptr + tl.ravel(base_offset + offsets_neighbor), mask=mask_n, other=-1
    )

    to_merge = (
        mask_n & (neighbor_labels != -1) & tl.ravel(current_values == neighbor_values)
    )
    valid_write = valid_current & to_merge

    # returns new parents for the pixels that were merged (otherwise keeps current labels)
    return tl.where(
        valid_write,
        union(labels_ptr, current_labels, neighbor_labels, valid_write),
        current_labels,
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_H": 4, "BLOCK_SIZE_W": 16}, num_stages=1, num_warps=2
        ),
        triton.Config(
            {"BLOCK_SIZE_H": 4, "BLOCK_SIZE_W": 32}, num_stages=2, num_warps=4
        ),
    ],
    key=["H", "W"],
    restore_value=["labels_ptr"],
)
@triton.jit
def _local_prop_kernel(
    labels_ptr,
    input_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # This is the meat of the Phase 2 to do local merging
    # It will be launched with a 2D grid:
    # - dim 0: batch index
    # - dim 1: block index over HxW image (2D tiling)
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)

    # Calculate offsets for the core block
    offsets_h = (pid_hw // tl.cdiv(W, BLOCK_SIZE_W)) * BLOCK_SIZE_H + tl.arange(
        0, BLOCK_SIZE_H
    )
    offsets_w = (pid_hw % tl.cdiv(W, BLOCK_SIZE_W)) * BLOCK_SIZE_W + tl.arange(
        0, BLOCK_SIZE_W
    )

    base_offset = pid_b * H * W
    offsets_2d = offsets_h[:, None] * W + offsets_w[None, :]
    mask_2d = (offsets_h[:, None] < H) & (offsets_w[None, :] < W)
    mask_1d = tl.ravel(mask_2d)

    # Load the current labels for the block - these are parent pointers
    current_labels = tl.load(
        labels_ptr + tl.ravel(base_offset + offsets_2d), mask=mask_1d, other=-1
    )
    current_values = tl.load(
        input_ptr + base_offset + offsets_2d, mask=mask_2d, other=-1
    )
    valid_current = mask_1d & (current_labels != -1)

    # Horizontal merge
    current_labels = _merge_helper(
        input_ptr,
        labels_ptr,
        base_offset,
        offsets_h,
        offsets_w,
        mask_2d,
        valid_current,
        current_values,
        current_labels,
        H,
        W,
        -1,
        0,
    )
    # Vertical merge
    current_labels = _merge_helper(
        input_ptr,
        labels_ptr,
        base_offset,
        offsets_h,
        offsets_w,
        mask_2d,
        valid_current,
        current_values,
        current_labels,
        H,
        W,
        0,
        -1,
    )
    # Diagonal merges
    current_labels = _merge_helper(
        input_ptr,
        labels_ptr,
        base_offset,
        offsets_h,
        offsets_w,
        mask_2d,
        valid_current,
        current_values,
        current_labels,
        H,
        W,
        -1,
        -1,
    )
    current_labels = _merge_helper(
        input_ptr,
        labels_ptr,
        base_offset,
        offsets_h,
        offsets_w,
        mask_2d,
        valid_current,
        current_values,
        current_labels,
        H,
        W,
        -1,
        1,
    )

    # This actually does some path compression, in a lightweight but beneficial way
    tl.atomic_min(
        labels_ptr + tl.ravel(base_offset + offsets_2d), current_labels, mask=mask_1d
    )


# ==============================================================================
# ## Phase 3: Pointer Jumping Kernel
# ==============================================================================
# This kernel performs pointer jumping to ensure that all pixels point directly to their root labels.
# This is done in a loop until convergence.


@triton.jit
def _pointer_jump_kernel(
    labels_in_ptr, labels_out_ptr, numel: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """
    Pointer jumping kernel with double buffering to avoid race conditions.
    Reads from labels_in_ptr and writes to labels_out_ptr.
    """
    # This kernel is launched with a 1D grid, and does not care about batching explicitly.
    # By construction, the labels are global indices across the batch, and we never perform
    # cross-batch merges, so this is safe.

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Load current labels from input buffer
    current_labels = tl.load(labels_in_ptr + offsets, mask=mask, other=-1)
    valid_mask = mask & (current_labels != -1)

    # A mask to track which lanes have successfully completed their union.
    done_mask = ~valid_mask
    while tl_any(~(done_mask | ~valid_mask)):
        parent_labels = tl.load(
            labels_in_ptr + current_labels, mask=valid_mask, other=-1
        )

        are_equal = current_labels == parent_labels
        done_mask |= are_equal & valid_mask

        current_labels = tl.where(
            ~done_mask, tl.minimum(current_labels, parent_labels), current_labels
        )

    # Write to output buffer (safe because we're not reading from it)
    tl.store(labels_out_ptr + offsets, current_labels, mask=mask)


# ==============================================================================
# ## Phase 4: Kernels for Computing Component Sizes
# ==============================================================================


# Step 4.1: Count occurrences of each root label using atomic adds.
@triton.jit
def _count_labels_kernel(labels_ptr, sizes_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Load the final, converged labels
    labels = tl.load(labels_ptr + offsets, mask=mask, other=-1)
    valid_mask = mask & (labels != -1)

    # Atomically increment the counter for each label. This builds a histogram.
    tl.atomic_add(sizes_ptr + labels, 1, mask=valid_mask)


# Step 4.2: Broadcast the computed sizes back to the output tensor.
@triton.jit
def _broadcast_sizes_kernel(
    labels_ptr, sizes_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Load the final labels
    labels = tl.load(labels_ptr + offsets, mask=mask, other=-1)
    valid_mask = mask & (labels != -1)

    # Look up the size for each label from the histogram
    component_sizes = tl.load(sizes_ptr + labels, mask=valid_mask, other=0)

    # Write the size to the final output tensor. Background pixels get size 0.
    tl.store(out_ptr + offsets, component_sizes, mask=mask)


def connected_components_triton(input_tensor: torch.Tensor):
    """
    Computes connected components labeling on a batch of 2D integer tensors using Triton.

    Args:
        input_tensor (torch.Tensor): A BxHxW integer tensor or Bx1xHxW. Non-zero values are considered foreground. Bool tensor also accepted

    Returns:
        Tuple[torch.Tensor, int]: A tuple containing:
            - A BxHxW output tensor with dense labels. Background is 0.
            - A BxHxW tensor with the size of the connected component for each pixel.
    """
    assert (
        input_tensor.is_cuda and input_tensor.is_contiguous()
    ), "Input tensor must be a contiguous CUDA tensor."
    out_shape = input_tensor.shape
    if input_tensor.dim() == 4 and input_tensor.shape[1] == 1:
        input_tensor = input_tensor.squeeze(1)
    else:
        assert (
            input_tensor.dim() == 3
        ), "Input tensor must be (B, H, W) or (B, 1, H, W)."

    B, H, W = input_tensor.shape
    numel = B * H * W
    device = input_tensor.device

    # --- Allocate Tensors ---
    labels = torch.empty_like(input_tensor, dtype=torch.int32)
    output = torch.empty_like(input_tensor, dtype=torch.int32)

    # --- Phase 1 ---
    BLOCK_SIZE = 256
    grid_init = (triton.cdiv(numel, BLOCK_SIZE),)
    _init_labels_kernel[grid_init](
        input_tensor,
        labels,
        numel,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # --- Phase 2 ---
    grid_local_prop = lambda meta: (
        B,
        triton.cdiv(H, meta["BLOCK_SIZE_H"]) * triton.cdiv(W, meta["BLOCK_SIZE_W"]),
    )
    _local_prop_kernel[grid_local_prop](labels, input_tensor, H, W)

    # --- Phase 3 ---
    BLOCK_SIZE = 256
    grid_jump = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    _pointer_jump_kernel[grid_jump](labels, output, numel, BLOCK_SIZE=BLOCK_SIZE)

    # --- Phase 4 ---
    # Allocate tensor to store the final output sizes
    component_sizes_out = torch.empty_like(input_tensor, dtype=torch.int32)

    # Allocate a temporary 1D tensor to act as the histogram
    # Size is numel because labels can be up to numel-1
    sizes_histogram = torch.zeros(numel, dtype=torch.int32, device=device)

    # 4.1: Count the occurrences of each label
    grid_count = (triton.cdiv(numel, BLOCK_SIZE),)
    _count_labels_kernel[grid_count](
        output, sizes_histogram, numel, BLOCK_SIZE=BLOCK_SIZE
    )

    # 2.2: Broadcast the counts to the final output tensor
    grid_broadcast = (triton.cdiv(numel, BLOCK_SIZE),)
    _broadcast_sizes_kernel[grid_broadcast](
        output, sizes_histogram, component_sizes_out, numel, BLOCK_SIZE=BLOCK_SIZE
    )
    return output.view(out_shape) + 1, component_sizes_out.view(out_shape)
