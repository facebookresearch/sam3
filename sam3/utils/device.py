# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Device utilities for supporting CUDA, MPS (Apple Silicon), and CPU backends.
"""

import logging
from functools import lru_cache
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    """
    Get the best available device for computation.

    Priority: CUDA > MPS > CPU

    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_str() -> str:
    """
    Get the best available device as a string.

    Returns:
        str: Device string ("cuda", "mps", or "cpu")
    """
    return str(get_device())


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if MPS (Apple Silicon GPU) is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def is_gpu_available() -> bool:
    """Check if any GPU (CUDA or MPS) is available."""
    return is_cuda_available() or is_mps_available()


def to_device(
    tensor: torch.Tensor,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> torch.Tensor:
    """
    Move tensor to the specified device, or to the best available device if not specified.

    Args:
        tensor: The tensor to move
        device: Target device. If None, uses get_device()
        non_blocking: Whether to perform the transfer asynchronously

    Returns:
        torch.Tensor: Tensor on the target device
    """
    if device is None:
        device = get_device()
    return tensor.to(device=device, non_blocking=non_blocking)


def setup_device_optimizations() -> None:
    """
    Setup device-specific optimizations.

    - For CUDA Ampere+ GPUs: Enable TensorFloat-32
    - For MPS: Enable high water mark ratio for memory management
    - For CPU: Currently no special optimizations
    """
    if torch.cuda.is_available():
        try:
            device_props = torch.cuda.get_device_properties(0)
            if device_props.major >= 8:
                # Enable TF32 for Ampere GPUs (compute capability >= 8.0)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.debug("Enabled TensorFloat-32 for Ampere GPU")
        except Exception as e:
            logger.debug(f"Could not set up CUDA optimizations: {e}")
    elif is_mps_available():
        # MPS optimizations for Apple Silicon
        try:
            # Set high water mark ratio to allow more GPU memory usage
            # This can improve performance by reducing memory pressure
            torch.mps.set_per_process_memory_fraction(0.0)  # No limit
            logger.debug("Using MPS (Apple Silicon GPU) with optimizations")
        except Exception as e:
            logger.debug(f"MPS optimization setup: {e}")
    else:
        logger.debug("Using CPU")


def mps_synchronize() -> None:
    """
    Synchronize MPS operations.

    Call this when you need to ensure all MPS operations are complete,
    such as before timing or when switching between GPU and CPU operations.
    """
    if is_mps_available():
        torch.mps.synchronize()


def empty_cache() -> None:
    """
    Empty the GPU cache to free memory.

    Works for both CUDA and MPS backends.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif is_mps_available():
        torch.mps.empty_cache()


def get_device_for_tensor(tensor: torch.Tensor) -> torch.device:
    """Get the device of a tensor."""
    return tensor.device


def tensor_is_on_gpu(tensor: torch.Tensor) -> bool:
    """Check if tensor is on a GPU (CUDA or MPS)."""
    device_type = tensor.device.type
    return device_type in ("cuda", "mps")


def tensor_is_on_cuda(tensor: torch.Tensor) -> bool:
    """Check if tensor is specifically on CUDA."""
    return tensor.device.type == "cuda"


def tensor_is_on_mps(tensor: torch.Tensor) -> bool:
    """Check if tensor is specifically on MPS."""
    return tensor.device.type == "mps"


def move_model_to_device(
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.nn.Module:
    """
    Move a model to the specified device.

    Args:
        model: The model to move
        device: Target device. If None, uses get_device()

    Returns:
        The model on the target device
    """
    if device is None:
        device = get_device()
    return model.to(device)
