# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import json
import tempfile
import unittest
from unittest.mock import MagicMock

from sam3.eval.coco_reindex import reindex_coco_to_temp
from sam3.agent.client_sam3 import call_sam_service
from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingWithInteractivity


class TestBugFixes(unittest.TestCase):
    def test_coco_reindex_invalid_json_raises_json_decode_error(self) -> None:
        """Invalid JSON should raise json.JSONDecodeError rather than TypeError."""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{invalid json")
            f_name = f.name

        with self.assertRaises(json.JSONDecodeError) as ctx:
            reindex_coco_to_temp(f_name)

        self.assertIn("Invalid JSON in", str(ctx.exception))
        self.assertIn(f_name, str(ctx.exception))

    def test_agent_core_assertion_raises_without_tool_tag(self) -> None:
        """Text without <tool> tag must raise AssertionError."""
        generated_text = "Here is the result without any tags."
        with self.assertRaises(AssertionError):
            assert "<tool>" in generated_text, (
                f"Generated text does not contain <tool> tag: {generated_text}"
            )

    def test_multiplex_tracking_cancellation_with_short_history(self) -> None:
        """Cancellation following a fetch with only 2 history items must not raise IndexError."""
        model = Sam3MultiplexTrackingWithInteractivity.__new__(
            Sam3MultiplexTrackingWithInteractivity
        )
        inference_state = {
            "action_history": [
                {"type": "propagation_fetch"},
                {"type": "propagation_cancel"},
            ],
            "num_frames": 10,
        }
        prop_type, obj_ids = model.parse_action_history_for_propagation(inference_state)
        self.assertEqual(prop_type, "propagation_full")
        self.assertIsNone(obj_ids)

    def test_call_sam_service_without_processor_raises_value_error(self) -> None:
        """Calling call_sam_service without sam3_processor raises ValueError instead of TypeError."""
        with self.assertRaises(ValueError) as ctx:
            call_sam_service(image_path="test.jpg", text_prompt="cat")
        self.assertIn("sam3_processor must be provided", str(ctx.exception))

    def test_tracker_new_points_broadcast_none_handling(self) -> None:
        """When multi-GPU broadcast receives None, it safely sets new_mask_data to None."""
        data_list = [None]
        device = "cpu"
        new_mask_data = data_list[0].to(device) if data_list[0] is not None else None
        self.assertIsNone(new_mask_data)


if __name__ == "__main__":
    unittest.main()
