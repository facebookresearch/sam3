# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
Device selection helpers so inference runs on CUDA, Apple Silicon (MPS), or CPU.

Model code resolves its device through ``DEVICE`` and ``bf16_autocast()``
rather than hardcoding ``.cuda()``, ``device="cuda"`` or
``torch.autocast(device_type="cuda")``.

Environment overrides:
  SAM3_DEVICE=cuda|mps|cpu   force a device (default: cuda > mps > cpu)
"""

import os

import torch


def _resolve_device() -> torch.device:
    override = os.environ.get("SAM3_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _resolve_device()
IS_CUDA = DEVICE.type == "cuda"


def bf16_autocast():
    """bf16 autocast on the active device.

    Required on every device, not just CUDA: the SAM 3.1 multiplex path casts
    backbone and memory features to bf16 explicitly and relies on autocast to
    mix them with fp32 weights. Usable as a context manager or a decorator.
    """
    return torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16)


def cuda_supports_tf32() -> bool:
    return IS_CUDA and torch.cuda.get_device_properties(0).major >= 8


def empty_cache() -> None:
    if IS_CUDA:
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()
