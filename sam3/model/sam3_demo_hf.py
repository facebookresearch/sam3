# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

import gc
import logging

import numpy as np

import torch
from PIL import Image

from .act_ckpt_utils import clone_output_wrapper

from .box_ops import box_xywh_to_cxcywh, box_xyxy_to_xywh

from .data_misc import (
    BatchedDatapoint,
    BatchedPointer,
    convert_my_tensors,
    FindStage,
    recursive_to,
)

from .geometry_encoders import Prompt
# from .model_misc import NestedTensor
from .sam3_image import Sam3Image





class Sam3ImageInteractiveDemo(Sam3Image):
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2

    def __init__(
        self,
        
        
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        
    

    @torch.inference_mode()
    def run_inference(
        self,
        inference_state,
        
        instance_prompt=False,
    ):

        # 3) run inference on this frame
        inference_state["backbone_out"] = self._init_backbone_out(inference_state)
        new_visual_prompt = inference_state["new_visual_prompt"]
        frame_idx = inference_state["frame_idx"]
        if new_visual_prompt is not None:
            # currently we do not allow simultaneously adding text prompt and visual
            # prompt both as initial prompt (since visual prompt uses the text "visual")
            if inference_state["text_prompt"] is not None:
                raise RuntimeError(
                    "Text and visual prompts (box as an initial prompt) cannot be used together. "
                    "Please reset the session."
                )

            # add the visual prompt into the input batch and encode it (currently the added
            # visual prompt is applied to *all* frames, i.e. not just this prompted frame)
            for t in range(inference_state["num_frames"]):
                text_id = self.TEXT_ID_FOR_VISUAL
                inference_state["input_batch"].find_inputs[t].text_ids[...] = text_id
            # currently visual prompt is encoded the same way (`_encode_prompt`) as geometric prompt
            visual_prompt_embed, visual_prompt_mask, _backbone_out = (
                self._encode_prompt(
                    backbone_out=inference_state["backbone_out"],
                    find_input=inference_state["input_batch"].find_inputs[frame_idx],
                    geometric_prompt=new_visual_prompt,
                    encode_text=False,
                )
            )
            inference_state["visual_prompt_embed"] = visual_prompt_embed
            inference_state["visual_prompt_mask"] = visual_prompt_mask

        
        out = self._run_single_frame_inference(
            inference_state,
            frame_idx,
            
            is_instance_processing=instance_prompt,
        )

        inference_state["model_out"] = out
        


    def _init_backbone_out(self, inference_state):
        """
        Initialize a backbone_out dictionary and extract the text features.

        Note that the visual features of each frame are not extracted here. They will be
        extracted on the fly when running inference on each frame.
        """
        input = inference_state["input_batch"]
        device = self.device
        backbone_out = {"img_batch_all_stages": input.img_batch}
        text_outputs = self.backbone.forward_text(input.find_text_batch, device=device)
        backbone_out.update(text_outputs)
        return backbone_out

    def _run_single_frame_inference(
        self, inference_state, frame_idx, is_instance_processing=False
    ):
        """
        Perform inference on a single frame and get its inference results. This would
        also update `inference_state`.
        """
        input = inference_state["input_batch"]
        find_input = input.find_inputs[frame_idx]
        # find_target = None
        # num_frames = inference_state["num_frames"]
        # is_video_batch = num_frames > 1

        backbone_out = inference_state["backbone_out"]
        geometric_prompt = inference_state["per_frame_geometric_prompt"][frame_idx]
        if geometric_prompt is None:
            geometric_prompt = inference_state["constants"]["empty_geometric_prompt"]
        previous_stages_out = inference_state["previous_stages_out"]
        prev_encoder_out = None
        if previous_stages_out[frame_idx] is not None:
            prev_encoder_out = previous_stages_out[frame_idx].get("prev_encoder_out")
        cur_step = inference_state["per_frame_cur_step"][frame_idx]


        prev_mask_pred = None
        if (
            inference_state["previous_stages_out"][frame_idx]
            # and self.use_prev_mask
            and is_instance_processing
        ):
            prev_mask_pred = self._get_best_mask(
                inference_state["previous_stages_out"][frame_idx]
            )

        out, _ = self.forward_video_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            frame_idx=frame_idx,
            # num_frames=num_frames,
            previous_stages_out=previous_stages_out,
            geometric_prompt=geometric_prompt.clone(),
            run_encoder= cur_step == 0,
            prev_encoder_out=prev_encoder_out,
            visual_prompt=inference_state["visual_prompt_embed"],
            visual_prompt_mask=inference_state["visual_prompt_mask"],
            is_instance_prompt=is_instance_processing,
            # track_in_reverse=reverse,
            prev_mask_pred=prev_mask_pred,
        )
        inference_state["previous_stages_out"][frame_idx] = out
        inference_state["per_frame_cur_step"][frame_idx] = cur_step + 1

        return out

    