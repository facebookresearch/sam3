# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import os

import numpy as np
import pytest
import torch
from PIL import Image
from sam3.perflib import fused
from sam3.perflib.masks_ops import masks_to_boxes


class TestMasksToBoxes:
    def test_masks_box(self):
        def masks_box_check(masks, expected, atol=1e-4):
            out = masks_to_boxes(masks, [1 for _ in range(masks.shape[0])])
            assert out.dtype == torch.float
            print("out: ", out)
            print("expected: ", expected)
            torch.testing.assert_close(
                out, expected, rtol=0.0, check_dtype=True, atol=atol
            )

        # Check for int type boxes.
        def _get_image():
            assets_directory = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "assets"
            )
            mask_path = os.path.join(assets_directory, "masks.tiff")
            image = Image.open(mask_path)
            return image

        def _create_masks(image, masks):
            for index in range(image.n_frames):
                image.seek(index)
                frame = np.array(image)
                masks[index] = torch.tensor(frame)

            return masks

        expected = torch.tensor(
            [
                [127, 2, 165, 40],
                [2, 50, 44, 92],
                [56, 63, 98, 100],
                [139, 68, 175, 104],
                [160, 112, 198, 145],
                [49, 138, 99, 182],
                [108, 148, 152, 213],
            ],
            dtype=torch.float,
        )

        image = _get_image()
        for dtype in [torch.float16, torch.float32, torch.float64]:
            masks = torch.zeros(
                (image.n_frames, image.height, image.width), dtype=dtype
            )
            masks = _create_masks(image, masks)
            masks_box_check(masks, expected)


class TestAddmmAct:
    def test_follows_fp16_autocast(self, monkeypatch):
        # On CPU, autocast would recast _addmm_activation itself, so record
        # the dtypes addmm_act passes to the kernel instead.
        dtypes = []

        def fake_addmm_act_op(bias, mat1, mat2, **kwargs):
            dtypes.append((bias.dtype, mat1.dtype, mat2.dtype))
            return torch.zeros(mat1.shape[0], mat2.shape[1], dtype=mat1.dtype)

        monkeypatch.setattr(fused, "addmm_act_op", fake_addmm_act_op)
        linear = torch.nn.Linear(8, 16)
        x = torch.randn(2, 3, 8)
        with torch.no_grad():
            with torch.autocast("cpu", dtype=torch.float16):
                fused.addmm_act(torch.nn.GELU, linear, x)
            fused.addmm_act(torch.nn.GELU, linear, x)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                fused.addmm_act(torch.nn.GELU, linear, x)

        assert dtypes == [
            (torch.float16,) * 3,
            (torch.bfloat16,) * 3,
            (torch.bfloat16,) * 3,
        ]
