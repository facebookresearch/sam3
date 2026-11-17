# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Tests for collate_fn_api with mixed point prompts in a batch (#458)."""

import unittest
from typing import Optional

import torch
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image,
    InferenceMetadata,
    Object,
)


def _datapoint(input_points: Optional[torch.Tensor]) -> Datapoint:
    obj = Object(bbox=torch.tensor([0.5, 0.5, 0.2, 0.2]), area=0.04, object_id=0)
    query = FindQueryLoaded(
        query_text="cat",
        image_id=0,
        object_ids_output=[0],
        is_exhaustive=True,
        input_points=input_points,
        inference_metadata=InferenceMetadata(
            coco_image_id=0,
            original_image_id=0,
            original_category_id=1,
            original_size=(8, 8),
            object_id=0,
            frame_index=0,
        ),
    )
    image = Image(data=torch.zeros(3, 8, 8), objects=[obj], size=(8, 8))
    return Datapoint(find_queries=[query], images=[image])


class TestCollateMixedPoints(unittest.TestCase):
    def test_batch_with_and_without_points(self) -> None:
        # Raw (x, y, label) points with a trivial batch dim, as built by the dataset
        points = torch.tensor([[[0.1, 0.2, 1.0], [0.3, 0.4, 0.0]]])
        batch = [_datapoint(points), _datapoint(None)]

        out = collate_fn_api(batch, dict_key="all")["all"]

        find_input = out.find_inputs[0]
        self.assertEqual(tuple(find_input.input_points.shape[-1:]), (3,))
        self.assertEqual(find_input.input_points.numel(), 2 * 2 * 3)
        self.assertEqual(find_input.input_points_mask.numel(), 2 * 2)


if __name__ == "__main__":
    unittest.main()
