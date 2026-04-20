# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Various utility models"""

import copy
import math
import warnings
import weakref
from collections.abc import Iterator
from contextlib import AbstractContextManager
from enum import auto, Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.overrides import handle_torch_function, has_torch_function
from typing_extensions import override

try:
    import xformers
except ImportError:
    xformers = None


def inverse_sigmoid(x, eps=1e-3):
    """
    The inverse function for sigmoid activation function.
    Note: It might face numberical issues with fp16 small eps.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def get_sdpa_settings():
    if torch.cuda.is_available():
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        # only use Flash Attention on Ampere (8.0) or newer GPUs
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        if not use_flash_attn:
            warnings.warn(
                "Flash Attention is disabled as it requires a GPU with Ampere (8.0) CUDA capability.",
                category=UserWarning,
                stacklevel=2,
            )
        # keep math kernel for PyTorch versions before 2.2 (Flash Attention v2 is only
        # available on PyTorch 2.2+, while Flash Attention v1 cannot handle all cases)
        pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2])
        if pytorch_version < (2, 2):
            warnings.warn(
                f"You are using PyTorch {torch.__version__} without Flash Attention v2 support. "
                "Consider upgrading to PyTorch 2.2+ for Flash Attention v2 (which could be faster).",
                category=UserWarning,
                stacklevel=2,
            )
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
    else:
        old_gpu = True
        use_flash_attn = False
        math_kernel_on = True

    return old_gpu, use_flash_attn, math_kernel_on


OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()


class AttentionType:
    """Type of attention"""

    # Simple dot product attention
    Vanilla = "Vanilla"

    # Efficient attention from xformers
    Xformer = "Xformer"

    # Sparse attention
    Sparse = "Sparse"

    # Deformable attention (not compatible with text)
    Deformable = "Deformable"


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    attn_type: AttentionType = AttentionType.Vanilla,
    attn_sparsity: float = 0.0,
    attn_bias: Optional[Tensor] = None,
    use_fa3: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    tens_ops = (
        query,
        key,
        value,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        out_proj_weight,
        out_proj_bias,
    )
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            is_causal=is_causal,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
            use_fa3=use_fa3,
        )

    is_batched = True

    if is_causal:
        raise NotImplementedError("is_causal is not supported in this implem")
        attn_mask = None

    if not is_batched:
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    if key_padding_mask is not None:
        _kpm_dtype = key_padding_mask.dtype
        if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
            raise AssertionError(
                "only bool and floating types of key_padding_mask are supported"
            )
    assert embed_dim == embed_dim_to_check, (
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    )
    if isinstance(embed_dim, torch.Tensor):
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, (
        f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    )
    if use_separate_proj_weight:
        assert key.shape[:2] == value.shape[:2], (
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
        )
    else:
        assert key.shape == value.shape, (
            f"key shape {key.shape} does not match value shape {value.shape}"
        )

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, (
            "use_separate_proj_weight is False but in_proj_weight is None"
        )
        q, k, v = F._in_projection_packed(
            query, key, value, in_proj_weight, in_proj_bias
        )
    else:
        assert q_proj_weight is not None, (
            "use_separate_proj_weight is True but q_proj_weight is None"
        )
        assert k_proj_weight is not None, (
            "use_separate_proj_weight is True but k_proj_weight is None"
        )
        assert v_proj_weight is not None, (
            "use_separate_proj_weight is True but v_proj_weight is None"
        )
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = F._in_projection(
            query,
            key,
            value,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            b_q,
            b_k,
            b_v,
        )

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, (
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
            )
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.dim()} is not supported"
            )

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        assert static_k.size(0) == bsz * num_heads, (
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        )
        assert static_k.size(2) == head_dim, (
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        )
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        assert static_v.size(0) == bsz * num_heads, (
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        )
        assert static_v.size(2) == head_dim, (
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        )
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat(
            [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
        )
        v = torch.cat(
            [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
        )
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (
            bsz,
            src_len,
        ), (
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        )
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    if attn_mask is not None:
        if attn_mask.size(0) == 1:
            attn_mask = attn_mask.unsqueeze(0)
        else:
            attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

    if attn_bias is not None:
        assert attn_bias.shape == (
            bsz,
            num_heads,
            tgt_len,
            src_len,
        ), (
            f"expecting attn_bias shape of {(bsz, num_heads, tgt_len, src_len)}, but got {attn_bias.shape}"
        )
        if attn_mask is None:
            attn_mask = attn_bias
        else:
            attn_mask = attn_mask + attn_bias

    q = q.view(bsz, num_heads, tgt_len, head_dim)
    k = k.view(bsz, num_heads, src_len, head_dim)
    v = v.view(bsz, num_heads, src_len, head_dim)

    if attn_type == AttentionType.Vanilla:
        if attn_mask is None and not is_causal and use_fa3:
            from sam3.perflib.fa3 import flash_attn_func

            assert dropout_p == 0.0
            attn_output = flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2)
        else:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask, dropout_p, is_causal
            )

        attn_output = (
            attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        )
    elif attn_type == AttentionType.Xformer:
        attn_output_weights = None
        assert not need_weights, "need_weights is not supported in efficient mode"
        attn_output = xformers.ops.memory_efficient_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_bias=attn_mask,
            p=dropout_p,
        )
        attn_output = attn_output.permute(1, 0, 2, 3).reshape(bsz * tgt_len, embed_dim)
    elif attn_type == AttentionType.Sparse:
        attn_output_weights = None
        assert not need_weights, "need_weights is not supported in efficient mode"
        # Need to collapse heads and batch dimensions
        q = q.reshape(bsz * num_heads, tgt_len, head_dim).contiguous()
        k = k.reshape(bsz * num_heads, src_len, head_dim).contiguous()
        v = v.reshape(bsz * num_heads, src_len, head_dim).contiguous()
        row_offsets, column_indices = xformers.ops.find_locations_new(
            q, k, attn_sparsity, True
        )
        attn_output = xformers.ops.sparse_memory_efficient_attention(
            q, k, v, row_offsets, column_indices, attn_bias=attn_mask
        ).reshape(bsz, num_heads, tgt_len, head_dim)
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(bsz * tgt_len, embed_dim)
    else:
        raise ValueError(f"Unsupported attention type {attn_type}")

    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    if need_weights:
        attn_output_weights = (q * head_dim**-0.5) @ k.transpose(-2, -1)
        attn_output_weights = attn_output_weights.softmax(dim=-1)
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

        if not is_batched:
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        attn_output_weights = None
        if not is_batched:
            attn_output = attn_output.squeeze(1)
        return attn_output, None


class MultiheadAttention(nn.Module):
    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
        attn_type: AttentionType = AttentionType.Vanilla,
        sparsity: float = 0.0,
        use_act_checkpoint: bool = False,
        use_fa3: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.use_act_checkpoint = use_act_checkpoint
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        assert attn_type == AttentionType.Sparse or sparsity == 0.0, (
            "sparsity is only supported for sparse attention"
        )

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = nn.Parameter(
                torch.empty((embed_dim, embed_dim), **factory_kwargs)
            )
            self.k_proj_weight = nn.Parameter(
                torch.empty((embed_dim, self.kdim), **factory_kwargs)
            )
            self.v_proj_weight = nn.Parameter(
                torch.empty((embed_dim, self.vdim), **factory_kwargs)
            )
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(
                torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = nn.Parameter(
                torch.empty(3 * embed_dim, **factory_kwargs)
            )
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.modules.linear.NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.attn_type = attn_type
        self.sparsity = sparsity
        self.use_fa3 = use_fa3

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        attn_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        is_batched = query.dim() == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(
                key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            if self.use_act_checkpoint:
                attn_output, attn_output_weights = torch.utils.checkpoint.checkpoint(
                    multi_head_attention_forward,
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.bias_k,
                    self.bias_v,
                    self.add_zero_attn,
                    self.dropout,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    use_reentrant=False,
                    training=self.training,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj_weight,
                    k_proj_weight=self.k_proj_weight,
                    v_proj_weight=self.v_proj_weight,
                    average_attn_weights=average_attn_weights,
                    attn_type=self.attn_type,
                    attn_sparsity=self.sparsity,
                    attn_bias=attn_bias,
                    use_fa3=self.use_fa3,
                )
            else:
                attn_output, attn_output_weights = multi_head_attention_forward(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.bias_k,
                    self.bias_v,
                    self.add_zero_attn,
                    self.dropout,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj_weight,
                    k_proj_weight=self.k_proj_weight,
                    v_proj_weight=self.v_proj_weight,
                    average_attn_weights=average_attn_weights,
                    attn_type=self.attn_type,
                    attn_sparsity=self.sparsity,
                    attn_bias=attn_bias,
                    use_fa3=self.use_fa3,
                )
        else:
            if self.use_act_checkpoint:
                attn_output, attn_output_weights = torch.utils.checkpoint.checkpoint(
                    multi_head_attention_forward,
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.bias_k,
                    self.bias_v,
                    self.add_zero_attn,
                    self.dropout,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    use_reentrant=False,
                    training=self.training,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    average_attn_weights=average_attn_weights,
                    attn_type=self.attn_type,
                    attn_sparsity=self.sparsity,
                    attn_bias=attn_bias,
                )
            else:
                attn_output, attn_output_weights = multi_head_attention_forward(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.bias_k,
                    self.bias_v,
                    self.add_zero_attn,
                    self.dropout,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    average_attn_weights=average_attn_weights,
                    attn_type=self.attn_type,
                    attn_sparsity=self.sparsity,
                    attn_bias=attn_bias,
                )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


# Keep backward compatibility alias
MultiheadAttentionWrapper = MultiheadAttention


class DotProductScoring(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_proj,
        prompt_mlp=None,
        clamp_logits=True,
        clamp_max_val=12.0,
    ):
        super().__init__()
        self.d_proj = d_proj
        assert isinstance(prompt_mlp, torch.nn.Module) or prompt_mlp is None
        self.prompt_mlp = prompt_mlp  # an optional MLP projection for prompt
        self.prompt_proj = torch.nn.Linear(d_model, d_proj)
        self.hs_proj = torch.nn.Linear(d_model, d_proj)
        self.scale = float(1.0 / np.sqrt(d_proj))
        self.clamp_logits = clamp_logits
        if self.clamp_logits:
            self.clamp_max_val = clamp_max_val

    def mean_pool_text(self, prompt, prompt_mask):
        # is_valid has shape (seq, bs, 1), where 1 is valid and 0 is padding
        is_valid = (~prompt_mask).float().permute(1, 0)[..., None]
        # num_valid has shape (bs, 1)
        num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
        # mean pool over all the valid tokens -- pooled_prompt has shape (bs, proj_dim)
        pooled_prompt = (prompt * is_valid).sum(dim=0) / num_valid
        return pooled_prompt

    def forward(self, hs, prompt, prompt_mask):
        # hs has shape (num_layer, bs, num_query, d_model)
        # prompt has shape (seq, bs, d_model)
        # prompt_mask has shape (bs, seq), where 1 is valid and 0 is padding
        assert hs.dim() == 4 and prompt.dim() == 3 and prompt_mask.dim() == 2

        # apply MLP on prompt if specified
        if self.prompt_mlp is not None:
            prompt = self.prompt_mlp(prompt)

        # first, get the mean-pooled version of the prompt
        pooled_prompt = self.mean_pool_text(prompt, prompt_mask)

        # then, project pooled_prompt and hs to d_proj dimensions
        proj_pooled_prompt = self.prompt_proj(pooled_prompt)  # (bs, d_proj)
        proj_hs = self.hs_proj(hs)  # (num_layer, bs, num_query, d_proj)

        # finally, get dot-product scores of shape (num_layer, bs, num_query, 1)
        scores = torch.matmul(proj_hs, proj_pooled_prompt.unsqueeze(-1))
        scores *= self.scale

        # clamp scores to a max value to avoid numerical issues in loss or matcher
        if self.clamp_logits:
            scores.clamp_(min=-self.clamp_max_val, max=self.clamp_max_val)

        return scores


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class TransformerWrapper(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        d_model: int,
        two_stage_type="none",  # ["none"] only for now
        pos_enc_at_input_dec=True,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = decoder.num_queries if decoder is not None else None
        self.pos_enc_at_input_dec = pos_enc_at_input_dec

        # for two stage
        assert two_stage_type in ["none"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        self.two_stage_type = two_stage_type

        self._reset_parameters()
        self.d_model = d_model

    def _reset_parameters(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                if (
                    "box_embed" not in n
                    and "query_embed" not in n
                    and "reference_points" not in n
                ):
                    nn.init.xavier_uniform_(p)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        residual: bool = False,
        out_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # whether to add the output as a residual connection to the input
        if residual and input_dim != output_dim:
            raise ValueError("residual is only supported if input_dim == output_dim")
        self.residual = residual
        # whether to apply a normalization layer to the output
        assert isinstance(out_norm, nn.Module) or out_norm is None
        self.out_norm = out_norm or nn.Identity()

    def forward(self, x):
        orig_x = x
        for i, layer in enumerate(self.layers):
            x = self.drop(F.relu(layer(x))) if i < self.num_layers - 1 else layer(x)
        if self.residual:
            x = x + orig_x
        x = self.out_norm(x)
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_clones_seq(module, N):
    return nn.Sequential(*[copy.deepcopy(module) for i in range(N)])


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_activation_module(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_valid_ratio(mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


def gen_sineembed_for_position(pos_tensor, num_feats=256):
    assert num_feats % 2 == 0
    num_feats = num_feats // 2
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / num_feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class SAM3Output(list):
    """
    A class representing the output of a SAM3 model.
    It provides an iterable interface that supports different iteration modes, including iterating over all steps per stage,
    last step per stage, and flattened output.
    Attributes:
        output: The output of the SAM3 model, represented as a list of lists.
        iter_mode: The current iteration mode.
    Example:
        >>> output = [[1, 2], [3, 4], [5, 6]]
        >>> sam3_output = SAM3Output(output)
        >>> for step in sam3_output:
        ...     print(step)
        [1, 2]
        [3, 4]
        [5, 6]
        >>> with SAM3Output.iteration_mode(SAM3Output.IterMode.LAST_STEP_PER_STAGE) as sam3_last_step_out:
        ...     for step in sam3_last_step_out:
        ...         print(step)
        [2]
        [4]
        [6]
        >>> with SAM3Output.iteration_mode(SAM3Output.IterMode.FLATTENED) as sam3_flattened_out:
        ...     for step in sam3_flattened_out:
        ...         print(step)
        1
        2
        3
        4
        5
        6
    """

    class IterMode(Enum):
        # Defines the type of iterator over ouptuts.
        ALL_STEPS_PER_STAGE = auto()
        LAST_STEP_PER_STAGE = auto()
        FLATTENED = auto()  # Returns each interactivity step as if it is a separate stage (this is used in SAM3Image model)

    def __init__(
        self,
        output: List[List[Dict]] = None,
        iter_mode: IterMode = IterMode.ALL_STEPS_PER_STAGE,
        loss_stages: Optional[List[int]] = None,
    ):
        if output is not None:
            assert (
                isinstance(output, list)
                and len(output) > 0
                and isinstance(output[0], list)
            ), "Expected output to be a list of lists"
            self.output = output
        else:
            self.output = []
        assert isinstance(iter_mode, SAM3Output.IterMode), (
            f"iter_mode shoulf be of enum type 'SAM3Output.IterMode'. Got {type(iter_mode)}"
        )

        self.iter_mode = iter_mode
        # We create a weak reference to self to be used in the lambda functions.
        # This is to avoid cyclic references and let SAM3Output be garabge collected.
        self_ref = weakref.ref(self)
        self._mode2iter = {
            SAM3Output.IterMode.ALL_STEPS_PER_STAGE: lambda: iter(self_ref().output),
            SAM3Output.IterMode.LAST_STEP_PER_STAGE: lambda: (
                inner_list[-1] for inner_list in self_ref().output
            ),
            SAM3Output.IterMode.FLATTENED: lambda: (
                element for inner_list in self_ref().output for element in inner_list
            ),
        }
        self.loss_stages = loss_stages

    @override
    def __iter__(self) -> Iterator:
        return self._mode2iter[self.iter_mode]()

    def __getitem__(self, index):
        """
        Returns the item at the specified index.
        Args:
            index (int): The index of the item to return.
        Returns:
            list or element: The item at the specified index.
        """
        assert isinstance(index, int), f"index should be an integer. Got {type(index)}"
        if self.iter_mode == SAM3Output.IterMode.ALL_STEPS_PER_STAGE:
            return self.output[index]
        elif self.iter_mode == SAM3Output.IterMode.LAST_STEP_PER_STAGE:
            return self.output[index][-1]
        elif self.iter_mode == SAM3Output.IterMode.FLATTENED:
            if index == -1:
                return self.self.output[-1][-1]
            else:
                flattened_output = sum(self.output, [])
                return flattened_output[index]

    class _IterationMode(AbstractContextManager):
        """
        A context manager that temporarily changes the iteration mode of a SAM3Output object.
        This class is used internally by the SAM3Output.iteration_mode method.
        """

        def __init__(
            self, model_output: "SAM3Output", iter_mode: "SAM3Output.IterMode"
        ):
            self._model_output = model_output
            self._orig_iter_mode = model_output.iter_mode
            self._new_iter_mode = iter_mode

        @override
        def __enter__(self) -> "SAM3Output":
            self._model_output.iter_mode = self._new_iter_mode
            return self._model_output

        @override
        def __exit__(self, exc_type, exc_value, traceback):
            self._model_output.iter_mode = self._orig_iter_mode
            return super().__exit__(exc_type, exc_value, traceback)

    @staticmethod
    def iteration_mode(
        model_output: "SAM3Output", iter_mode: IterMode
    ) -> _IterationMode:
        """
        Returns a context manager that allows you to temporarily change the iteration mode of the SAM3Output object.
        Args:
            model_output: The SAM3Output object.
            iter_mode: The new iteration mode.
        Returns:
            SAM3Output._IterationMode: A context manager that changes the iteration mode of the SAM3Output object.
        """
        return SAM3Output._IterationMode(model_output=model_output, iter_mode=iter_mode)

    def append(self, item: list):
        assert isinstance(item, list), (
            f"Only list items are supported. Got {type(item)}"
        )
        self.output.append(item)

    def __repr__(self):
        return self.output.__repr__()

    def __len__(self):
        if self.iter_mode in [
            SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
            SAM3Output.IterMode.LAST_STEP_PER_STAGE,
        ]:
            return len(self.output)
        elif self.iter_mode == SAM3Output.IterMode.FLATTENED:
            flattened_output = sum(self.output, [])
            return len(flattened_output)
