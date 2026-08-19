# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import types
import unittest

import torch
from sam3.model.sam3_video_predictor import Sam3VideoPredictor


class TestVideoPredictorShutdown(unittest.TestCase):
    def test_exits_tracker_autocast_context(self) -> None:
        predictor = Sam3VideoPredictor.__new__(Sam3VideoPredictor)
        context = torch.amp.autocast("cpu", dtype=torch.bfloat16)
        context.__enter__()
        tracker = types.SimpleNamespace(bf16_context=context)
        predictor.model = types.SimpleNamespace(tracker=tracker)
        predictor._all_inference_states = {}
        self.assertTrue(torch.is_autocast_enabled("cpu"))

        predictor.shutdown()
        predictor.shutdown()

        self.assertFalse(torch.is_autocast_enabled("cpu"))
        self.assertIsNone(tracker.bf16_context)


if __name__ == "__main__":
    unittest.main()
