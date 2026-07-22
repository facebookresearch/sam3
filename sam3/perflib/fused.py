# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch

addmm_act_op = torch.ops.aten._addmm_activation


def addmm_act(activation, linear, mat1):
    if torch.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")
    bias = linear.bias.detach()
    weight = linear.weight.detach()
    mat1_flat = mat1.view(-1, mat1.shape[-1])

    if not torch.cuda.is_available():
        # aten::_addmm_activation is a CUDA-only fused kernel.
        # Cast everything to the weight dtype (float32 after model.float()) so
        # the computation stays consistent regardless of the input's dtype.
        target_dtype = weight.dtype
        x = mat1_flat.to(target_dtype)
        w = weight.to(target_dtype)
        b = bias.to(target_dtype)
        y = torch.nn.functional.linear(x, w, b)
        if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
            y = torch.relu(y)
        elif activation in [torch.nn.functional.gelu, torch.nn.GELU]:
            y = torch.nn.functional.gelu(y)
        else:
            raise ValueError(f"Unexpected activation {activation}")
        return y.view(mat1.shape[:-1] + (y.shape[-1],))

    # CUDA path: use bfloat16 fused kernel
    self = bias.to(torch.bfloat16)
    mat1_flat = mat1_flat.to(torch.bfloat16)
    mat2 = weight.to(torch.bfloat16)
    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=False)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))
    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=True)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))
    raise ValueError(f"Unexpected activation {activation}")
