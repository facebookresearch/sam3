# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

# The module queries CUDA capability at import time, but these output-merging
# tests need only CPU tensors and never construct a model or load weights.
with patch.object(
    torch.cuda, "get_device_properties", return_value=SimpleNamespace(major=7)
):
    from sam3.model.sam3_multiplex_tracking import (
        Sam3MultiplexTrackingWithInteractivity,
    )


class TestMultiplexTrackingOutput(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = Sam3MultiplexTrackingWithInteractivity.__new__(
            Sam3MultiplexTrackingWithInteractivity
        )
        self.mask = torch.tensor([[[False, True], [True, False]]])

    def test_refinement_without_cached_frame_is_returned(self) -> None:
        state = {"cached_frame_outputs": {}}

        output = self.tracker._build_sam2_output(state, 50, {1: self.mask})

        self.assertEqual(set(output), {1})
        self.assertIs(output[1], self.mask)
        self.assertEqual(state["cached_frame_outputs"], {})

    def test_refinement_preserves_other_objects_and_cached_masks(self) -> None:
        old_mask = torch.zeros_like(self.mask)
        other_mask = torch.ones_like(self.mask)
        cached = {1: old_mask, 2: other_mask}
        state = {"cached_frame_outputs": {50: cached}}

        output = self.tracker._build_sam2_output(state, 50, {1: self.mask})

        self.assertEqual(set(output), {1, 2})
        self.assertIs(output[1], self.mask)
        self.assertIs(output[2], other_mask)
        self.assertIsNot(output, cached)
        self.assertIs(cached[1], old_mask)
        self.assertIs(cached[2], other_mask)

    def test_absent_or_empty_refinement_preserves_cache_behavior(self) -> None:
        for refinement in (None, {}):
            with self.subTest(refinement=refinement):
                fresh = {"cached_frame_outputs": {}}
                self.assertEqual(
                    self.tracker._build_sam2_output(fresh, 50, refinement), {}
                )
                self.assertEqual(fresh["cached_frame_outputs"], {})
                cached = {"cached_frame_outputs": {50: {1: self.mask}}}
                output = self.tracker._build_sam2_output(cached, 50, refinement)
                self.assertIs(output[1], self.mask)

    def test_supplied_masks_are_not_filtered_and_none_is_rejected(self) -> None:
        empty_mask = torch.zeros_like(self.mask)
        output = self.tracker._build_sam2_output(
            {"cached_frame_outputs": {}}, 50, {1: empty_mask}
        )
        self.assertIs(output[1], empty_mask)
        for cached in ({}, {50: {1: self.mask}}):
            with self.subTest(cached=bool(cached)):
                with self.assertRaisesRegex(AssertionError, "obj_id 1"):
                    self.tracker._build_sam2_output(
                        {"cached_frame_outputs": cached}, 50, {1: None}
                    )


if __name__ == "__main__":
    unittest.main()
