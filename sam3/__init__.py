# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from .model_builder import build_sam3_image_model
from .platform import check_platform_compatibility, get_platform_info, is_jetson

__version__ = "0.1.0"

__all__ = [
    "build_sam3_image_model",
    "is_jetson",
    "get_platform_info",
    "check_platform_compatibility",
]

# Check platform compatibility on import (emits warning if misconfigured)
check_platform_compatibility(warn=True)
