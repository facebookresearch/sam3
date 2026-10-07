# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import types
import unittest

import torch

from sam3.model.geometry_encoders import Prompt
from sam3.model.sam3_video_inference import Sam3VideoInference


class TestVideoInferenceBoxPrompts(unittest.TestCase):
    def _predictor(self):
        return types.SimpleNamespace(device=torch.device("cpu"))

    def _state(self):
        return {
            "per_frame_visual_prompt": [None],
            "previous_stages_out": [None],
        }

    def test_single_box_preserves_visual_prompt_behavior(self):
        predictor = self._predictor()
        state = self._state()

        boxes = torch.tensor(
            [[0.50, 0.50, 0.20, 0.20]],
            dtype=torch.float32,
        )
        labels = torch.tensor([1], dtype=torch.long)

        boxes_out, labels_out, prompt = Sam3VideoInference._get_visual_prompt(
            predictor,
            state,
            frame_idx=0,
            boxes_cxcywh=boxes,
            box_labels=labels,
        )

        self.assertIs(prompt, state["per_frame_visual_prompt"][0])
        self.assertEqual(tuple(prompt.box_embeddings.shape), (1, 1, 4))
        self.assertEqual(tuple(prompt.box_labels.shape), (1, 1))

        torch.testing.assert_close(prompt.box_embeddings[:, 0], boxes)
        torch.testing.assert_close(prompt.box_labels[:, 0], labels)

        self.assertEqual(tuple(boxes_out.shape), (0, 4))
        self.assertEqual(tuple(labels_out.shape), (0,))

    def test_multiple_boxes_are_packed_into_one_visual_prompt(self):
        predictor = self._predictor()
        state = self._state()

        boxes = torch.tensor(
            [
                [0.20, 0.20, 0.10, 0.10],
                [0.50, 0.50, 0.15, 0.20],
                [0.75, 0.40, 0.12, 0.18],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([1, 1, 0], dtype=torch.long)

        with self.assertLogs(level="WARNING"):
            boxes_out, labels_out, prompt = Sam3VideoInference._get_visual_prompt(
                predictor,
                state,
                frame_idx=0,
                boxes_cxcywh=boxes,
                box_labels=labels,
            )

        self.assertIs(prompt, state["per_frame_visual_prompt"][0])
        self.assertEqual(tuple(prompt.box_embeddings.shape), (3, 1, 4))
        self.assertEqual(tuple(prompt.box_labels.shape), (3, 1))
        self.assertEqual(tuple(prompt.box_mask.shape), (1, 3))

        torch.testing.assert_close(prompt.box_embeddings[:, 0], boxes)
        torch.testing.assert_close(prompt.box_labels[:, 0], labels)

        self.assertFalse(prompt.box_mask.any().item())

        # All input boxes were consumed by the visual prompt.
        self.assertEqual(tuple(boxes_out.shape), (0, 4))
        self.assertEqual(tuple(labels_out.shape), (0,))

    def test_existing_visual_prompt_does_not_create_another_prompt(self):
        predictor = self._predictor()
        state = self._state()

        existing_boxes = torch.tensor(
            [[[0.50, 0.50, 0.20, 0.20]]],
            dtype=torch.float32,
        )
        existing_labels = torch.tensor([[1]], dtype=torch.long)

        existing_prompt = Prompt(
            box_embeddings=existing_boxes,
            box_labels=existing_labels,
            box_mask=None,
            point_embeddings=None,
            point_mask=None,
        )
        state["per_frame_visual_prompt"][0] = existing_prompt

        boxes = torch.tensor(
            [[0.30, 0.30, 0.10, 0.10]],
            dtype=torch.float32,
        )
        labels = torch.tensor([1], dtype=torch.long)

        boxes_out, labels_out, prompt = Sam3VideoInference._get_visual_prompt(
            predictor,
            state,
            frame_idx=0,
            boxes_cxcywh=boxes,
            box_labels=labels,
        )

        self.assertIsNone(prompt)
        self.assertIs(state["per_frame_visual_prompt"][0], existing_prompt)

        # Preserve existing behavior: one box is stripped when a visual prompt already exists.
        self.assertEqual(tuple(boxes_out.shape), (0, 4))
        self.assertEqual(tuple(labels_out.shape), (0,))


if __name__ == "__main__":
    unittest.main()
