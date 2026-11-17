# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Tests that batched grounding covers every frame scheduled by propagation (#618)."""

import unittest
from functools import partial
from unittest.mock import MagicMock

from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
from sam3.model.sam3_multiplex_tracking import (
    Sam3MultiplexTracking,
    Sam3MultiplexTrackingProd,
)


def _processing_order(start_frame_idx, max_frame_num_to_track, num_frames, reverse):
    inference_state = {
        "num_frames": num_frames,
        "previous_stages_out": [None] * num_frames,
    }
    processing_order, _ = Sam3MultiplexTracking._get_processing_order(
        MagicMock(),
        inference_state,
        start_frame_idx,
        max_frame_num_to_track,
        reverse,
    )
    return processing_order


def _grounded_chunks(
    processing_order,
    start_frame_idx,
    max_frame_num_to_track,
    num_frames,
    reverse,
    batch_size,
):
    """Run the batched grounding chunking logic with the model calls stubbed out."""
    detector = MagicMock()
    chunk_keys = []

    def process_chunk(chunk_start, chunk_end, **kwargs):
        chunk_keys.append((chunk_start, chunk_end))
        return list(range(chunk_start, chunk_end))

    detector._process_grounding_chunk_batched.side_effect = process_chunk
    detector._slice_batched_output.side_effect = lambda outputs, idx, _: outputs[idx]

    grounding_cache = {}
    for frame_idx in processing_order:
        out, _ = Sam3MultiplexDetector.forward_video_grounding_batched_multigpu(
            detector,
            backbone_out={},
            find_inputs=[],
            geometric_prompt=None,
            frame_idx=frame_idx,
            num_frames=num_frames,
            grounding_cache=grounding_cache,
            track_in_reverse=reverse,
            max_frame_num_to_track=max_frame_num_to_track,
            propagate_in_video_start_frame_idx=start_frame_idx,
            batch_size=batch_size,
        )
        assert out == frame_idx, f"got output for frame {out}, expected {frame_idx}"
    return chunk_keys


class _StopPropagation(Exception):
    pass


def _propagate_and_ground(
    tracker_cls, prompt_frame, max_frame_num_to_track, num_frames, reverse, batch_size
):
    """Call `propagate_in_video` without a start frame, then ground every scheduled
    frame using the tracking bounds it stored for the detector."""
    tracker = MagicMock()
    tracker.rank = 0
    tracker.masklet_confirmation_consecutive_det_thresh = 1
    tracker._get_processing_order = partial(
        Sam3MultiplexTracking._get_processing_order, tracker
    )
    tracker._default_start_frame_idx = partial(
        Sam3MultiplexTracking._default_start_frame_idx, tracker
    )
    previous_stages_out = [None] * num_frames
    previous_stages_out[prompt_frame] = object()
    inference_state = {
        "num_frames": num_frames,
        "previous_stages_out": previous_stages_out,
        "feature_cache": {},
    }
    order, _ = tracker._get_processing_order(
        inference_state, None, max_frame_num_to_track, reverse
    )

    def run_frames(state, frame_idx, reverse, **kwargs):
        bounds = state["feature_cache"]["tracking_bounds"]
        _grounded_chunks(
            order,
            bounds["propagate_in_video_start_frame_idx"],
            bounds["max_frame_num_to_track"],
            num_frames,
            reverse,
            batch_size,
        )
        raise _StopPropagation

    tracker._run_single_frame_inference.side_effect = run_frames
    frames = tracker_cls.propagate_in_video(
        tracker,
        inference_state,
        max_frame_num_to_track=max_frame_num_to_track,
        reverse=reverse,
    )
    try:
        next(frames)
    except _StopPropagation:
        pass


class TestBatchedGroundingBounds(unittest.TestCase):
    def test_forward_bound_has_no_empty_chunk(self) -> None:
        # Repro from #618: 16 frames, batch size 1, max_frame_num_to_track=15
        order = _processing_order(0, 15, num_frames=16, reverse=False)
        self.assertEqual(list(order), list(range(16)))
        chunks = _grounded_chunks(
            order, 0, 15, num_frames=16, reverse=False, batch_size=1
        )
        self.assertTrue(all(end > start for start, end in chunks))

    def test_bounds_match_processing_order(self) -> None:
        num_frames = 20
        for reverse in (False, True):
            for start in (0, 5, 19):
                for max_frames in (1, 4, 7, 30):
                    for batch_size in (1, 3, 16):
                        with self.subTest(
                            reverse=reverse,
                            start=start,
                            max_frames=max_frames,
                            batch_size=batch_size,
                        ):
                            order = _processing_order(
                                start, max_frames, num_frames, reverse
                            )
                            chunks = _grounded_chunks(
                                order,
                                start,
                                max_frames,
                                num_frames,
                                reverse,
                                batch_size,
                            )
                            for chunk_start, chunk_end in chunks:
                                self.assertLess(chunk_start, chunk_end)
                                self.assertLessEqual(chunk_end, num_frames)

    def test_default_start_frame_is_used_for_grounding(self) -> None:
        # Prompt on a later frame and propagate without `start_frame_index`:
        # grounding must use the same start frame as the tracker.
        for tracker_cls in (Sam3MultiplexTracking, Sam3MultiplexTrackingProd):
            for reverse in (False, True):
                for batch_size in (1, 16):
                    with self.subTest(
                        tracker_cls=tracker_cls.__name__,
                        reverse=reverse,
                        batch_size=batch_size,
                    ):
                        _propagate_and_ground(
                            tracker_cls,
                            prompt_frame=10,
                            max_frame_num_to_track=5,
                            num_frames=20,
                            reverse=reverse,
                            batch_size=batch_size,
                        )


if __name__ == "__main__":
    unittest.main()
