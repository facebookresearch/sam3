# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from sam3.utils.device import (
    get_device,
    get_device_str,
    is_cuda_available,
    is_gpu_available,
    is_mps_available,
    move_model_to_device,
    setup_device_optimizations,
    tensor_is_on_cuda,
    tensor_is_on_gpu,
    tensor_is_on_mps,
    to_device,
)

__all__ = [
    "get_device",
    "get_device_str",
    "is_cuda_available",
    "is_mps_available",
    "is_gpu_available",
    "to_device",
    "setup_device_optimizations",
    "tensor_is_on_gpu",
    "tensor_is_on_cuda",
    "tensor_is_on_mps",
    "move_model_to_device",
]
