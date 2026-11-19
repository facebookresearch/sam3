# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Tests for SAM 3.1 multiplex checkpoint loading (#506)."""

import unittest
from unittest.mock import patch

from sam3 import model_builder
from sam3.model.vitdet import ViT


class _StopBuild(Exception):
    pass


class TestMultiplexCheckpointLoading(unittest.TestCase):
    def test_tracker_does_not_load_merged_checkpoint(self) -> None:
        # The merged checkpoint is loaded once into the full demo model; loading it
        # into the bare tracker first matches no keys and only prints warnings.
        with patch.object(
            model_builder,
            "build_sam3_multiplex_video_model",
            side_effect=_StopBuild,
        ) as build_tracker:
            with self.assertRaises(_StopBuild):
                model_builder.build_sam3_multiplex_video_predictor(
                    checkpoint_path="sam3.1_multiplex.pt"
                )
        self.assertIsNone(build_tracker.call_args.kwargs["checkpoint_path"])
        self.assertFalse(build_tracker.call_args.kwargs["load_from_HF"])

    def test_real_rope_buffers_are_not_persistent(self) -> None:
        def make_vit(use_rope_real):
            return ViT(
                img_size=56,
                pretrain_img_size=56,
                patch_size=14,
                embed_dim=32,
                depth=1,
                num_heads=2,
                use_rope=True,
                use_abs_pos=False,
                tile_abs_pos=False,
                window_size=0,
                global_att_blocks=(0,),
                rel_pos_blocks=(),
                retain_cls_token=False,
                use_act_checkpoint=False,
                use_rope_real=use_rope_real,
            )

        vit = make_vit(use_rope_real=True)
        attn = vit.blocks[0].attn
        self.assertIsNotNone(attn.freqs_cis_real)
        self.assertIsNotNone(attn.freqs_cis_imag)

        keys = vit.state_dict().keys()
        self.assertFalse(
            any(k.endswith(("freqs_cis_real", "freqs_cis_imag")) for k in keys)
        )
        # A checkpoint saved without real-valued RoPE loads strictly
        vit.load_state_dict(make_vit(use_rope_real=False).state_dict())


if __name__ == "__main__":
    unittest.main()
