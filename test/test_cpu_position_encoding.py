# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Regression test for CPU-only construction of PositionEmbeddingSine.

Previously, PositionEmbeddingSine precomputed its cache with a hardcoded
``device="cuda"``, which raised a RuntimeError on CPU-only machines when a
``precompute_resolution`` was provided (see GitHub issue #587).
"""

import unittest
from unittest.mock import patch

import torch
from sam3.model.position_encoding import PositionEmbeddingSine


class TestPositionEmbeddingSineCpu(unittest.TestCase):
    """PositionEmbeddingSine should build without CUDA when none is available."""

    def test_precompute_builds_on_cpu_when_cuda_unavailable(self) -> None:
        """Constructing with precompute_resolution must not require CUDA."""
        with patch("torch.cuda.is_available", return_value=False):
            pos_enc = PositionEmbeddingSine(
                num_pos_feats=256,
                precompute_resolution=112,
            )

        # The cache is populated and lives on CPU.
        self.assertGreater(len(pos_enc.cache), 0)
        for cached in pos_enc.cache.values():
            self.assertEqual(cached.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
