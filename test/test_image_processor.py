# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Image processor geometry tests using a dummy model and CPU tensors."""

import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor


class TestSam3ProcessorImageDimensions(unittest.TestCase):
    def setUp(self) -> None:
        self.model = MagicMock()
        self.model.inst_interactive_predictor = None
        self.model.backbone.forward_image.return_value = {}
        self.model.backbone.forward_text.return_value = {}
        self.processor = Sam3Processor(self.model, resolution=14, device="cpu")

    def test_set_image_records_original_dimensions(self) -> None:
        image = np.zeros((6, 10, 3), dtype=np.uint8)
        inputs = {
            "numpy_hwc": image,
            "pil": Image.fromarray(image),
            "tensor_chw": torch.from_numpy(image).permute(2, 0, 1),
        }
        for name, value in inputs.items():
            with self.subTest(input_format=name):
                state = self.processor.set_image(value)
                self.assertEqual(state["original_height"], 6)
                self.assertEqual(state["original_width"], 10)
                encoder_input = self.model.backbone.forward_image.call_args.args[0]
                self.assertEqual(tuple(encoder_input.shape), (1, 3, 14, 14))

    def test_numpy_output_boxes_and_masks_use_original_dimensions(self) -> None:
        self.model.forward_grounding.return_value = {
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]]),
            "pred_logits": torch.tensor([[[10.0]]]),
            "pred_masks": torch.ones(1, 1, 2, 3),
            "presence_logit_dec": torch.tensor([[10.0]]),
        }
        state = self.processor.set_image(np.zeros((6, 10, 3), dtype=np.uint8))

        result = self.processor.set_text_prompt("object", state)

        torch.testing.assert_close(
            result["boxes"], torch.tensor([[2.5, 1.5, 7.5, 4.5]])
        )
        self.assertEqual(tuple(result["masks"].shape), (1, 1, 6, 10))
        self.assertEqual(tuple(result["masks_logits"].shape), (1, 1, 6, 10))
