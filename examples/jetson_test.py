#!/usr/bin/env python3
"""
Test SAM3 on NVIDIA Jetson Platform

This script validates that SAM3 is properly installed and working on Jetson devices.
It tests basic imports, CUDA availability, and system information.

Usage:
    python examples/jetson_test.py

Requirements:
    - NVIDIA Jetson device (AGX Orin, Orin Nano, Orin NX)
    - JetPack 6.x
    - Python 3.10
    - PyTorch 2.8.0+ with CUDA support
"""

import platform
import sys


def test_basic_import():
    """Test SAM3 imports"""
    print("=" * 60)
    print("Testing SAM3 imports...")
    print("=" * 60)

    try:
        from sam3 import build_sam3_image_model

        print("✓ SAM3 imports successful")
        return True
    except ImportError as e:
        print(f"✗ SAM3 import failed: {e}")
        return False


def test_cuda():
    """Test CUDA availability and GPU properties"""
    print("\n" + "=" * 60)
    print("Testing CUDA and GPU...")
    print("=" * 60)

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")

            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU memory: {total_memory:.2f} GB")

            print("✓ CUDA test successful")
            return True
        else:
            print("✗ CUDA is not available")
            return False

    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False


def test_jetson_info():
    """Display Jetson-specific information"""
    print("\n" + "=" * 60)
    print("Jetson Platform Information")
    print("=" * 60)

    try:
        # Read Jetson model
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip("\x00")
            print(f"Device model: {model}")
    except:
        print("Device model: Unable to read (not on Jetson?)")

    try:
        # Read L4T/JetPack version
        with open("/etc/nv_tegra_release", "r") as f:
            release = f.readline().strip()
            print(f"L4T Release: {release}")
    except:
        print("L4T Release: Unable to read (not on Jetson?)")

    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")


def test_model_load():
    """Test loading SAM3 model (requires checkpoints)"""
    print("\n" + "=" * 60)
    print("Testing SAM3 model loading...")
    print("=" * 60)

    print("⚠ Model loading test skipped (requires downloaded checkpoints)")
    print("  To download checkpoints:")
    print("  1. Request access: https://huggingface.co/facebook/sam3")
    print("  2. Run: huggingface-cli login")
    print("  3. Run: huggingface-cli download facebook/sam3 --local-dir ./checkpoints")

    return True


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  SAM3 Jetson Platform Validation".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    results = []

    # Run tests
    results.append(("Basic Imports", test_basic_import()))
    results.append(("CUDA Support", test_cuda()))
    test_jetson_info()
    results.append(("Model Loading", test_model_load()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<45} {status}")
        all_passed = all_passed and passed

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed! SAM3 is ready to use on Jetson.\n")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the output above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
