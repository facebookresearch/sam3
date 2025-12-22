# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Tests for CPU and MPS (Apple Silicon) device support.

Run with: pytest tests/test_device_support.py -v
"""

import pytest
import torch


class TestDeviceUtilities:
    """Test the device utility module."""

    def test_device_module_imports(self):
        """Test that device utilities can be imported."""
        from sam3.utils.device import (
            get_device,
            get_device_str,
            is_cuda_available,
            is_gpu_available,
            is_mps_available,
            setup_device_optimizations,
            tensor_is_on_cuda,
            tensor_is_on_gpu,
            tensor_is_on_mps,
            to_device,
        )

        # All functions should be callable
        assert callable(get_device)
        assert callable(get_device_str)
        assert callable(is_cuda_available)
        assert callable(is_mps_available)
        assert callable(is_gpu_available)
        assert callable(to_device)
        assert callable(setup_device_optimizations)

    def test_get_device_returns_valid_device(self):
        """Test that get_device returns a valid torch.device."""
        from sam3.utils.device import get_device

        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cuda", "mps", "cpu")

    def test_get_device_str_returns_string(self):
        """Test that get_device_str returns a string."""
        from sam3.utils.device import get_device_str

        device_str = get_device_str()
        assert isinstance(device_str, str)
        assert device_str in ("cuda", "mps", "cpu")

    def test_device_detection_consistency(self):
        """Test that device detection functions are consistent."""
        from sam3.utils.device import (
            get_device,
            is_cuda_available,
            is_gpu_available,
            is_mps_available,
        )

        device = get_device()

        # If CUDA is available, device should be CUDA
        if is_cuda_available():
            assert device.type == "cuda"
            assert is_gpu_available()
        # If MPS is available and CUDA is not, device should be MPS
        elif is_mps_available():
            assert device.type == "mps"
            assert is_gpu_available()
        # Otherwise, device should be CPU
        else:
            assert device.type == "cpu"

    def test_to_device_moves_tensor(self):
        """Test that to_device correctly moves tensors."""
        from sam3.utils.device import get_device, to_device

        tensor = torch.randn(3, 3)
        moved_tensor = to_device(tensor)

        expected_device = get_device()
        assert moved_tensor.device.type == expected_device.type

    def test_tensor_device_checks(self):
        """Test tensor device check functions."""
        from sam3.utils.device import (
            tensor_is_on_cuda,
            tensor_is_on_gpu,
            tensor_is_on_mps,
        )

        cpu_tensor = torch.randn(3, 3, device="cpu")
        assert not tensor_is_on_cuda(cpu_tensor)
        assert not tensor_is_on_mps(cpu_tensor)
        assert not tensor_is_on_gpu(cpu_tensor)


class TestCPUSupport:
    """Test that operations work on CPU."""

    def test_sigmoid_focal_loss_cpu(self):
        """Test sigmoid focal loss works on CPU."""
        from sam3.train.loss.sigmoid_focal_loss import (
            sigmoid_focal_loss,
            sigmoid_focal_loss_reduce,
        )

        inputs = torch.randn(10, 5, device="cpu", requires_grad=True)
        targets = torch.rand(10, 5, device="cpu")

        # Test unreduced version
        loss = sigmoid_focal_loss(inputs, targets)
        assert loss.device.type == "cpu"
        assert loss.shape == inputs.shape

        # Test reduced version
        loss_reduced = sigmoid_focal_loss_reduce(inputs, targets)
        assert loss_reduced.device.type == "cpu"
        assert loss_reduced.dim() == 0  # scalar

        # Test backward pass
        loss_reduced.backward()
        assert inputs.grad is not None
        assert inputs.grad.shape == inputs.shape

    def test_edt_cpu(self):
        """Test EDT (Euclidean Distance Transform) works on CPU."""
        from sam3.model.edt import edt

        # Create a batch of binary masks
        data = torch.zeros(2, 64, 64, device="cpu")
        data[:, 20:40, 20:40] = 1  # Square in the middle

        result = edt(data)
        assert result.device.type == "cpu"
        assert result.shape == data.shape
        # EDT of zeros should be zero
        assert (result[data == 0] == 0).all()

    def test_nms_cpu(self):
        """Test NMS works on CPU."""
        from sam3.perflib.nms import generic_nms

        n = 10
        # Create a symmetric IoU matrix
        ious = torch.rand(n, n, device="cpu")
        ious = (ious + ious.T) / 2  # Make symmetric
        ious.fill_diagonal_(1.0)  # Diagonal should be 1

        scores = torch.rand(n, device="cpu")

        kept = generic_nms(ious, scores, iou_threshold=0.5)
        assert kept.device.type == "cpu"
        assert kept.dim() == 1
        assert len(kept) <= n

    def test_connected_components_cpu(self):
        """Test connected components works on CPU."""
        from sam3.perflib.connected_components import connected_components

        # Create a batch of binary masks with distinct components
        data = torch.zeros(2, 1, 64, 64, device="cpu", dtype=torch.uint8)
        data[0, 0, 10:20, 10:20] = 1  # Component 1
        data[0, 0, 40:50, 40:50] = 1  # Component 2
        data[1, 0, 5:15, 5:15] = 1  # Component in second batch

        labels, counts = connected_components(data)
        assert labels.device.type == "cpu"
        assert counts.device.type == "cpu"
        assert labels.shape == data.shape
        assert counts.shape == data.shape


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)
class TestMPSSupport:
    """Test that operations work on MPS (Apple Silicon)."""

    def test_sigmoid_focal_loss_mps(self):
        """Test sigmoid focal loss works on MPS."""
        from sam3.train.loss.sigmoid_focal_loss import (
            sigmoid_focal_loss,
            sigmoid_focal_loss_reduce,
        )

        inputs = torch.randn(10, 5, device="mps", requires_grad=True)
        targets = torch.rand(10, 5, device="mps")

        # Test unreduced version
        loss = sigmoid_focal_loss(inputs, targets)
        assert loss.device.type == "mps"
        assert loss.shape == inputs.shape

        # Test reduced version
        loss_reduced = sigmoid_focal_loss_reduce(inputs, targets)
        assert loss_reduced.device.type == "mps"

    def test_edt_mps(self):
        """Test EDT works on MPS (falls back to CPU internally)."""
        from sam3.model.edt import edt

        # Create a batch of binary masks on MPS
        data = torch.zeros(2, 64, 64, device="mps")
        data[:, 20:40, 20:40] = 1

        result = edt(data)
        # Result should be on MPS (moved back after CPU computation)
        assert result.device.type == "mps"
        assert result.shape == data.shape

    def test_nms_mps(self):
        """Test NMS works on MPS (falls back to CPU internally)."""
        from sam3.perflib.nms import generic_nms

        n = 10
        ious = torch.rand(n, n, device="mps")
        ious = (ious + ious.T) / 2
        ious.fill_diagonal_(1.0)
        scores = torch.rand(n, device="mps")

        kept = generic_nms(ious, scores, iou_threshold=0.5)
        # Result should be on MPS
        assert kept.device.type == "mps"

    def test_connected_components_mps(self):
        """Test connected components works on MPS."""
        from sam3.perflib.connected_components import connected_components

        data = torch.zeros(2, 1, 64, 64, device="mps", dtype=torch.uint8)
        data[0, 0, 10:20, 10:20] = 1
        data[0, 0, 40:50, 40:50] = 1

        labels, counts = connected_components(data)
        # Results should be on MPS
        assert labels.device.type == "mps"
        assert counts.device.type == "mps"

    def test_device_detection_mps(self):
        """Test that MPS is detected when available."""
        from sam3.utils.device import get_device, is_gpu_available, is_mps_available

        assert is_mps_available()
        assert is_gpu_available()
        # If CUDA is not available, MPS should be the default
        if not torch.cuda.is_available():
            assert get_device().type == "mps"


class TestModelBuilderDeviceSupport:
    """Test model builder device handling."""

    def test_device_parameter_accepted(self):
        """Test that build functions accept device parameter."""
        from sam3.model_builder import build_sam3_image_model, build_sam3_video_model
        import inspect

        # Check that device parameter exists
        image_sig = inspect.signature(build_sam3_image_model)
        video_sig = inspect.signature(build_sam3_video_model)

        assert "device" in image_sig.parameters
        assert "device" in video_sig.parameters

        # Check defaults are None (auto-detect)
        assert image_sig.parameters["device"].default is None
        assert video_sig.parameters["device"].default is None


class TestTransformerDeviceSupport:
    """Test transformer module device handling."""

    def test_rope_attention_cpu(self):
        """Test RoPEAttention works on CPU."""
        from sam3.sam.transformer import RoPEAttention

        attention = RoPEAttention(
            embedding_dim=256,
            num_heads=8,
            downsample_rate=1,
            feat_sizes=(8, 8),
        )
        attention = attention.to("cpu")

        # Create dummy inputs
        batch_size = 2
        seq_len = 64
        q = torch.randn(batch_size, seq_len, 256, device="cpu")
        k = torch.randn(batch_size, seq_len, 256, device="cpu")
        v = torch.randn(batch_size, seq_len, 256, device="cpu")

        output = attention(q, k, v)
        assert output.device.type == "cpu"
        assert output.shape == (batch_size, seq_len, 256)

    def test_attention_cpu(self):
        """Test base Attention works on CPU."""
        from sam3.sam.transformer import Attention

        attention = Attention(
            embedding_dim=256,
            num_heads=8,
        )
        attention = attention.to("cpu")

        batch_size = 2
        seq_len = 64
        q = torch.randn(batch_size, seq_len, 256, device="cpu")
        k = torch.randn(batch_size, seq_len, 256, device="cpu")
        v = torch.randn(batch_size, seq_len, 256, device="cpu")

        output = attention(q, k, v)
        assert output.device.type == "cpu"
        assert output.shape == (batch_size, seq_len, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
