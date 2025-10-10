# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved
"""
Tests for the model_builder module.
"""

import unittest
from unittest.mock import patch

import torch


class TestModelBuilder(unittest.TestCase):
    """Test cases for the model_builder module."""

    def test_build_sam3_image_model(self):
        """Test that build_sam3_image_model creates a model with expected structure."""
        from sam3.model.sam3_image import Sam3Image
        from sam3.model_builder import build_sam3_image_model

        # Test that the function exists
        self.assertTrue(callable(build_sam3_image_model))

        # Test model creation without checkpoint
        bpe_path = "assets/bpe_simple_vocab_16e6.txt.gz"
        model = build_sam3_image_model(
            bpe_path=bpe_path, checkpoint_path=None, device="cpu", eval_mode=True
        )
        self.assertIsInstance(model, Sam3Image)

        # Test that model has expected attributes
        self.assertTrue(hasattr(model, "backbone"))
        self.assertTrue(hasattr(model, "transformer"))
        self.assertTrue(hasattr(model, "segmentation_head"))

    def test_build_sam3_video_model(self):
        """Test that build_sam3_video_model creates a model with expected structure."""
        from sam3.model.sam3_video_inference import (
            Sam3VideoInferenceWithInstanceInteractivity,
        )
        from sam3.sam3_video_model_builder import build_sam3_video_model

        # Test that the function exists
        self.assertTrue(callable(build_sam3_video_model))

        # Test model creation without checkpoint
        bpe_path = "assets/bpe_simple_vocab_16e6.txt.gz"
        model = build_sam3_video_model(
            bpe_path=bpe_path, checkpoint_path=None, device="cpu"
        )
        self.assertIsInstance(model, Sam3VideoInferenceWithInstanceInteractivity)

        # Test that model has expected attributes
        self.assertTrue(hasattr(model, "sam2_model"))
        self.assertTrue(hasattr(model, "sam3_model"))


if __name__ == "__main__":
    unittest.main()
