# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Triton kernel for euclidean distance transform (EDT)"""

import torch
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    import cv2
    import numpy as np

if HAS_TRITON:
    @triton.jit
    def edt_kernel(inputs_ptr, outputs_ptr, v, z, height, width, horizontal: tl.constexpr):
        # This is a somewhat verbatim implementation of the efficient 1D EDT algorithm described above
        # It can be applied horizontally or vertically depending if we're doing the first or second stage.
        # It's parallelized across batch+row (or batch+col if horizontal=False)
        # TODO: perhaps the implementation can be revisited if/when local gather/scatter become available in triton
        batch_id = tl.program_id(axis=0)
        if horizontal:
            row_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + row_id * width
            length = width
            stride = 1
        else:
            col_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + col_id
            length = height
            stride = width

        # This will be the index of the right most parabola in the envelope ("the top of the stack")
        k = 0
        for q in range(1, length):
            # Read the function value at the current location. Note that we're doing a singular read, not very efficient
            cur_input = tl.load(inputs_ptr + block_start + (q * stride))
            # location of the parabola on top of the stack
            r = tl.load(v + block_start + (k * stride))
            # associated boundary
            z_k = tl.load(z + block_start + (k * stride))
            # value of the function at the parabola location
            previous_input = tl.load(inputs_ptr + block_start + (r * stride))
            # intersection between the two parabolas
            s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            # we'll pop as many parabolas as required
            while s <= z_k and k - 1 >= 0:
                k = k - 1
                r = tl.load(v + block_start + (k * stride))
                z_k = tl.load(z + block_start + (k * stride))
                previous_input = tl.load(inputs_ptr + block_start + (r * stride))
                s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            # Store the new one
            k = k + 1
            tl.store(v + block_start + (k * stride), q)
            tl.store(z + block_start + (k * stride), s)
            if k + 1 < length:
                tl.store(z + block_start + ((k + 1) * stride), 1e9)

        # Last step, we read the envelope to find the min in every location
        k = 0
        for q in range(length):
            while (
                k + 1 < length
                and tl.load(
                    z + block_start + ((k + 1) * stride), mask=(k + 1) < length, other=q
                )
                < q
            ):
                k += 1
            r = tl.load(v + block_start + (k * stride))
            d = q - r
            old_value = tl.load(inputs_ptr + block_start + (r * stride))
            tl.store(outputs_ptr + block_start + (q * stride), old_value + d * d)


def edt_triton(data: torch.Tensor):
    """
    Computes the Euclidean Distance Transform (EDT) of a batch of binary images.

    Args:
        data: A tensor of shape (B, H, W) representing a batch of binary images.

    Returns:
        A tensor of the same shape as data containing the EDT.
        It should be equivalent to a batched version of cv2.distanceTransform(input, cv2.DIST_L2, 0)
    """
    assert data.dim() == 3
    
    if HAS_TRITON and data.is_cuda:
        B, H, W = data.shape
        data = data.contiguous()

        # Allocate the "function" tensor. Implicitly the function is 0 if data[i,j]==0 else +infinity
        output = torch.where(data, 1e18, 0.0)
        assert output.is_contiguous()

        # Scratch tensors for the parabola stacks
        parabola_loc = torch.zeros(B, H, W, dtype=torch.uint32, device=data.device)
        parabola_inter = torch.empty(B, H, W, dtype=torch.float, device=data.device)
        parabola_inter[:, :, 0] = -1e18
        parabola_inter[:, :, 1] = 1e18

        # Grid size (number of blocks)
        grid = (B, H)

        # Launch initialization kernel
        edt_kernel[grid](
            output.clone(),
            output,
            parabola_loc,
            parabola_inter,
            H,
            W,
            horizontal=True,
        )

        # reset the parabola stacks
        parabola_loc.zero_()
        parabola_inter[:, :, 0] = -1e18
        parabola_inter[:, :, 1] = 1e18

        grid = (B, W)
        edt_kernel[grid](
            output.clone(),
            output,
            parabola_loc,
            parabola_inter,
            H,
            W,
            horizontal=False,
        )
        # don't forget to take sqrt at the end
        return output.sqrt()
    else:
        # Fallback using OpenCV or SciPy
        device = data.device
        data_cpu = data.detach().cpu().float().numpy()
        B, H, W = data_cpu.shape
        output_cpu = np.zeros_like(data_cpu, dtype=np.float32)

        for b in range(B):
            # cv2.distanceTransform computes distance to nearest zero pixel
            # data is binary: 0 or 1 (or boolean). 
            # We assume non-zero pixels are the ones we want distance FOR, to the nearest zero.
            # Convert to uint8 for cv2
            img = (data_cpu[b] > 0).astype(np.uint8)
            
            # If all pixels are > 0, distance is something large? No, if no zeros, distance might differ.
            # But normally masks have some zeros.
            
            # Using cv2.DIST_MASK_PRECISE (0) for exact, or 5/3 for approx
            dist = cv2.distanceTransform(img, cv2.DIST_L2, 5) 
            output_cpu[b] = dist

        return torch.from_numpy(output_cpu).to(device)
