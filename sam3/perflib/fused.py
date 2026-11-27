# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch

addmm_act_op = torch.ops.aten._addmm_activation


def _fused_dtype(device_type):
    # fp16 autocast is used on GPUs without native bf16 (e.g. T4), so follow it there.
    if (
        torch.is_autocast_enabled(device_type)
        and torch.get_autocast_dtype(device_type) == torch.float16
    ):
        return torch.float16
    return torch.bfloat16


def addmm_act(activation, linear, mat1):
    if torch.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")
    dtype = _fused_dtype(mat1.device.type)
    self = linear.bias.detach()
    mat2 = linear.weight.detach()
    self = self.to(dtype)
    mat1 = mat1.to(dtype)
    mat2 = mat2.to(dtype)
    mat1_flat = mat1.view(-1, mat1.shape[-1])
    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=False)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))
    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=True)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))
    raise ValueError(f"Unexpected activation {activation}")
