# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Tests for _load_checkpoint handling of the various checkpoint key layouts."""

import os
import tempfile
import unittest

import torch

from sam3.model_builder import _load_checkpoint


class _DummyImageModel(torch.nn.Module):
    """Minimal stand-in for the SAM3 image model."""

    inst_interactive_predictor = None

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4, bias=False)
        self.head = torch.nn.Linear(4, 2, bias=False)


class TestLoadCheckpoint(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = _DummyImageModel()
        self.reference = _DummyImageModel()
        # Give the reference model distinct weights so we can tell whether
        # loading actually happened.
        with torch.no_grad():
            for p in self.reference.parameters():
                p.add_(1.0)
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _save(self, ckpt: dict) -> str:
        path = os.path.join(self._tmpdir.name, "ckpt.pt")
        torch.save(ckpt, path)
        return path

    def _assert_loaded(self) -> None:
        for (name, p), (_, p_ref) in zip(
            self.model.named_parameters(), self.reference.named_parameters()
        ):
            self.assertTrue(torch.equal(p, p_ref), f"parameter {name} was not loaded")

    def test_detector_prefixed_checkpoint(self) -> None:
        """Official layout: keys prefixed with 'detector.'."""
        ckpt = {f"detector.{k}": v for k, v in self.reference.state_dict().items()}
        _load_checkpoint(self.model, self._save(ckpt))
        self._assert_loaded()

    def test_model_wrapped_checkpoint(self) -> None:
        """Training layout: state dict nested under a 'model' key."""
        ckpt = {
            "model": {
                f"detector.{k}": v for k, v in self.reference.state_dict().items()
            }
        }
        _load_checkpoint(self.model, self._save(ckpt))
        self._assert_loaded()

    def test_ddp_module_prefixed_checkpoint(self) -> None:
        """DDP training layout: keys prefixed with 'module.detector.'."""
        ckpt = {
            f"module.detector.{k}": v for k, v in self.reference.state_dict().items()
        }
        _load_checkpoint(self.model, self._save(ckpt))
        self._assert_loaded()

    def test_plain_image_model_checkpoint(self) -> None:
        """Checkpoint saved directly from the image model (no 'detector.' prefix)."""
        _load_checkpoint(self.model, self._save(dict(self.reference.state_dict())))
        self._assert_loaded()

    def test_mismatched_checkpoint_raises(self) -> None:
        """A checkpoint with no matching key must raise instead of silently
        leaving the model randomly initialized."""
        ckpt = {"some.other.model.weight": torch.zeros(2, 2)}
        with self.assertRaises(ValueError):
            _load_checkpoint(self.model, self._save(ckpt))


if __name__ == "__main__":
    unittest.main()
