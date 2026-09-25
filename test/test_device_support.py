# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import importlib
import os
import unittest
from unittest import mock

import sam3.device_utils as device_utils
import torch
from sam3.perflib.fused import addmm_act


class TestDeviceSelection(unittest.TestCase):
    def setUp(self) -> None:
        with mock.patch.dict(os.environ, {"SAM3_DEVICE": "cpu"}):
            importlib.reload(device_utils)

    def tearDown(self) -> None:
        importlib.reload(device_utils)

    def test_env_override_selects_device(self) -> None:
        self.assertEqual(torch.device("cpu"), device_utils.DEVICE)
        self.assertFalse(device_utils.IS_CUDA)

    def test_bf16_autocast_runs_on_selected_device(self) -> None:
        a = torch.randn(4, 4)
        with device_utils.bf16_autocast():
            self.assertEqual(torch.bfloat16, torch.mm(a, a).dtype)


class TestFusedLinearActivationFallback(unittest.TestCase):
    """Off CUDA, addmm_act must match the _addmm_activation reference kernel."""

    def _check(self, activation, use_gelu: bool) -> None:
        torch.manual_seed(0)
        linear = torch.nn.Linear(16, 8).double()
        x = torch.randn(2, 3, 16, dtype=torch.float64)
        with torch.no_grad():
            expected = torch.ops.aten._addmm_activation(
                linear.bias, x.view(-1, 16), linear.weight.t(), use_gelu=use_gelu
            ).view(2, 3, 8)
            actual = addmm_act(activation, linear, x)
        torch.testing.assert_close(actual, expected)

    def test_gelu_matches_reference(self) -> None:
        self._check(torch.nn.GELU, use_gelu=True)

    def test_relu_matches_reference(self) -> None:
        self._check(torch.nn.ReLU, use_gelu=False)


if __name__ == "__main__":
    unittest.main()
