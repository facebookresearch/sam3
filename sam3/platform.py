# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Platform detection utilities for SAM3.

Provides automatic detection of NVIDIA Jetson hardware and platform-specific
configuration recommendations.
"""

import os
import sys
import warnings
from typing import Optional


def is_jetson() -> bool:
    """Check if running on NVIDIA Jetson platform.

    Returns:
        True if running on Jetson hardware, False otherwise.
    """
    return os.path.exists("/etc/nv_tegra_release")


def get_platform_info() -> dict:
    """Get detailed platform information.

    Returns:
        Dictionary containing:
        - is_jetson: Whether running on Jetson
        - l4t_release: L4T version string (Jetson only)
        - device_model: Device model name (Jetson only)
        - python_version: Current Python version
        - platform_machine: CPU architecture (e.g., 'aarch64', 'x86_64')
    """
    import platform

    info = {
        "is_jetson": is_jetson(),
        "l4t_release": None,
        "device_model": None,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform_machine": platform.machine(),
    }

    if info["is_jetson"]:
        try:
            with open("/etc/nv_tegra_release", "r") as f:
                info["l4t_release"] = f.readline().strip()
        except (IOError, OSError):
            pass

        try:
            with open("/proc/device-tree/model", "r") as f:
                info["device_model"] = f.read().strip("\x00")
        except (IOError, OSError):
            pass

    return info


def get_recommended_python() -> str:
    """Get recommended Python version for current platform.

    Returns:
        Recommended Python version string (e.g., "3.10" or "3.12").
    """
    if is_jetson():
        return "3.10"  # NVIDIA PyTorch for Jetson only supports 3.10
    return "3.12"  # Recommended for x86


def get_pytorch_index_url() -> str:
    """Get the recommended PyTorch index URL for current platform.

    Returns:
        PyTorch package index URL.
    """
    if is_jetson():
        return "https://pypi.jetson-ai-lab.io/jp6/cu126"
    return "https://download.pytorch.org/whl/cu126"


def check_platform_compatibility(warn: bool = True) -> Optional[str]:
    """Check if current Python version is compatible with platform.

    Args:
        warn: If True, emit a warning for incompatible configurations.

    Returns:
        Warning message if incompatible, None if compatible.
    """
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    recommended = get_recommended_python()

    message = None

    if is_jetson():
        # On Jetson, Python 3.10 is required due to NVIDIA PyTorch constraints
        if sys.version_info[:2] != (3, 10):
            message = (
                f"SAM3 on Jetson requires Python 3.10 (NVIDIA PyTorch constraint), "
                f"but you're using Python {current_version}. "
                f"Performance may be affected or imports may fail. "
                f"Reinstall with: python3.10 -m venv .venv && pip install -e '.[jetson]'"
            )
    else:
        # On x86, Python 3.12 is recommended but 3.9+ should work
        if sys.version_info[:2] < (3, 9):
            message = (
                f"SAM3 requires Python 3.9 or higher, "
                f"but you're using Python {current_version}."
            )

    if message and warn:
        warnings.warn(message, UserWarning, stacklevel=2)

    return message


def print_platform_info() -> None:
    """Print platform information to stdout."""
    info = get_platform_info()

    print("SAM3 Platform Information")
    print("=" * 40)

    if info["is_jetson"]:
        print(f"Platform: NVIDIA Jetson")
        if info["device_model"]:
            print(f"Device: {info['device_model']}")
        if info["l4t_release"]:
            print(f"L4T: {info['l4t_release']}")
    else:
        print(f"Platform: {info['platform_machine']}")

    print(f"Python: {info['python_version']}")
    print(f"Recommended Python: {get_recommended_python()}")
    print(f"PyTorch Index: {get_pytorch_index_url()}")


if __name__ == "__main__":
    print_platform_info()
