# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import types
import unittest

import torch
from sam3.model.sam3_base_predictor import Sam3BasePredictor
from sam3.model.sam3_video_predictor import Sam3VideoPredictor


class _StreamingModel:
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        del start_frame_idx, max_frame_num_to_track, reverse
        for frame_idx in range(2):
            inference_state["cached_frame_outputs"][frame_idx] = {
                7: torch.ones((1, 4, 4))
            }
            yield frame_idx, {"frame": frame_idx}


class TestVideoPredictorStreaming(unittest.TestCase):
    def _predictor(self):
        predictor = Sam3BasePredictor()
        predictor.model = _StreamingModel()
        state = {"cached_frame_outputs": {}}
        predictor._all_inference_states["test"] = {
            "state": state,
            "last_use_time": 0,
        }
        return predictor, state

    def test_streaming_eviction_releases_each_yielded_frame(self) -> None:
        predictor, state = self._predictor()
        responses = predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": "test",
                "propagation_direction": "forward",
                "evict_cached_frame_outputs": True,
            }
        )

        self.assertEqual(0, next(responses)["frame_index"])
        self.assertNotIn(0, state["cached_frame_outputs"])
        self.assertEqual(1, next(responses)["frame_index"])
        self.assertNotIn(1, state["cached_frame_outputs"])

    def test_streaming_cache_is_retained_by_default(self) -> None:
        predictor, state = self._predictor()
        list(
            predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": "test",
                    "propagation_direction": "forward",
                }
            )
        )

        self.assertEqual([0, 1], sorted(state["cached_frame_outputs"]))


class _MultiplexInitModel:
    """Like the SAM 3.1 multiplex model, init_state() has no offload_state_to_cpu."""

    def init_state(self, resource_path, offload_video_to_cpu=False):
        return {"resource_path": resource_path}


class TestStartSession(unittest.TestCase):
    def test_drops_kwargs_the_model_does_not_accept(self) -> None:
        predictor = Sam3BasePredictor()
        predictor.model = _MultiplexInitModel()

        response = predictor.handle_request(
            {"type": "start_session", "resource_path": "video.mp4", "session_id": "s"}
        )

        self.assertEqual("s", response["session_id"])
        state = predictor._all_inference_states["s"]["state"]
        self.assertEqual("video.mp4", state["resource_path"])


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
