# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

import os
from copy import deepcopy
from typing import Dict, Optional

import torch

from sam3.model.model_misc import SAM3Output

from sam3.train.data.collator import BatchedDatapoint, FindStage

from .act_ckpt_utils import activation_ckpt_wrapper, clone_output_wrapper

from .box_ops import box_cxcywh_to_xyxy

from .geometry_encoders import Prompt
from .model_misc import inverse_sigmoid, NestedTensor


def _update_out(out, out_name, out_value, auxiliary=True):
    out[out_name] = out_value[-1] if auxiliary else out_value
    if auxiliary:
        if "aux_outputs" not in out:
            out["aux_outputs"] = [{} for _ in range(len(out_value) - 1)]
        assert len(out["aux_outputs"]) == len(out_value) - 1
        for aux_output, aux_value in zip(out["aux_outputs"], out_value[:-1]):
            aux_output[out_name] = aux_value


class Sam3Image(torch.nn.Module):
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2

    def __init__(
        self,
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head=None,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=None,
        use_instance_query: bool = True,
        multimask_output: bool = True,
        use_act_checkpoint_seg_head: bool = True,
        interactivity_in_encoder: bool = True,
        matcher=None,
        use_dot_prod_scoring=True,
        supervise_joint_box_scores: bool = False,  # only relevant if using presence token/score
        detach_presence_in_joint_score: bool = False,  # only relevant if using presence token/score
        separate_scorer_for_instance: bool = False,
        num_interactive_steps_val: int = 0,  # TODO: Add support back for this.
        **kwargs,  # TODO: Kalyan, Remove this!
    ):
        super().__init__()
        self.backbone = backbone
        self.geometry_encoder = input_geometry_encoder
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.segmentation_head = segmentation_head

        self.o2m_mask_predict = o2m_mask_predict

        self.dot_prod_scoring = dot_prod_scoring
        self.use_act_checkpoint_seg_head = use_act_checkpoint_seg_head
        self.interactivity_in_encoder = interactivity_in_encoder
        self.matcher = matcher

        self.num_interactive_steps_val = num_interactive_steps_val
        self.use_dot_prod_scoring = use_dot_prod_scoring

        if self.use_dot_prod_scoring:
            assert dot_prod_scoring is not None
            self.dot_prod_scoring = dot_prod_scoring
            self.instance_dot_prod_scoring = None
            if separate_scorer_for_instance:
                self.instance_dot_prod_scoring = deepcopy(dot_prod_scoring)
        else:
            self.class_embed = torch.nn.Linear(self.hidden_dim, 1)
            self.instance_class_embed = None
            if separate_scorer_for_instance:
                self.instance_class_embed = deepcopy(self.class_embed)

        self.supervise_joint_box_scores = supervise_joint_box_scores
        self.detach_presence_in_joint_score = detach_presence_in_joint_score

        # verify the number of queries for O2O and O2M
        num_o2o_static = self.transformer.decoder.num_queries
        num_o2m_static = self.transformer.decoder.num_o2m_queries
        assert num_o2m_static == (num_o2o_static if self.transformer.decoder.dac else 0)
        self.dac = self.transformer.decoder.dac

        self.use_instance_query = use_instance_query
        self.multimask_output = multimask_output

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        # clear cached _device in case the model is moved to a different device
        self._device = None
        return super().to(*args, **kwargs)

    def _get_img_feats(self, backbone_out, img_ids):
        """Retrieve correct image features from backbone output."""
        if "backbone_fpn" in backbone_out:
            if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
                img_ids = backbone_out["id_mapping"][img_ids]
                # If this assert fails, it likely means we're requesting different img_ids (perhaps a different frame?)
                # We currently don't expect this to happen. We could technically trigger a recompute here,
                # but likely at the cost of a cpu<->gpu sync point, which would deteriorate perf
                torch._assert_async((img_ids >= 0).all())

            vis_feats = backbone_out["backbone_fpn"][-self.num_feature_levels :]
            vis_pos_enc = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
            vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]  # (H, W) shapes
            # index and flatten visual features NxCxHxW => HWxNxC (batch-first => seq-first)
            img_feats = [
                x.tensors[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats
            ]
            img_masks = [
                None if x.mask is None else x.mask[img_ids].flatten(1)
                for x in vis_feats
            ]
            img_pos_embeds = [
                x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc
            ]
            return backbone_out, img_feats, img_masks, img_pos_embeds, vis_feat_sizes

        # Image features not available in backbone output, so we compute them on the fly
        # This case likely occurs for video. In that case, we want to forward only the current frame
        img_batch = backbone_out["img_batch_all_stages"]
        if img_ids.numel() > 1:
            # Only forward backbone on unique image ids to avoid repetitive computation
            unique_ids, _ = torch.unique(img_ids, return_inverse=True)
        else:
            unique_ids, _ = img_ids, slice(None)
        # Compute the image features on those unique image ids
        # note: we allow using a list (or other indexable types) of tensors as img_batch.tensors
        # (e.g. for async frame loading in demo). In this case we index img_batch.tensors directly
        if isinstance(img_batch.tensors, torch.Tensor):
            image = img_batch.tensors[unique_ids]
        elif unique_ids.numel() == 1:
            image = img_batch.tensors[unique_ids.item()].unsqueeze(0)
        else:
            image = torch.stack([img_batch.tensors[i] for i in unique_ids.tolist()])
        # `img_batch` might be fp16 and offloaded to CPU
        image = image.to(dtype=torch.float32, device=self.device)
        image_mask = img_batch.mask[unique_ids] if img_batch.mask is not None else None
        image_tensors = NestedTensor(tensors=image, mask=image_mask)
        # Next time we call this function, we want to remember which indices we computed
        id_mapping = torch.full(
            (len(img_batch.tensors),), -1, dtype=torch.long, device=self.device
        )
        id_mapping[unique_ids] = torch.arange(len(unique_ids), device=self.device)
        backbone_out = {
            **backbone_out,
            **self.backbone.forward_image(image_tensors),
            "id_mapping": id_mapping,
        }
        assert "backbone_fpn" in backbone_out
        return self._get_img_feats(backbone_out, img_ids=img_ids)

    def _encode_prompt(
        self,
        backbone_out,
        find_input,
        geometric_prompt,
        visual_prompt_embed=None,
        visual_prompt_mask=None,
        encode_text=True,
        prev_mask_pred=None,
    ):
        # index text features (note that regardless of early or late fusion, the batch size of
        # `txt_feats` is always the number of *prompts* in the encoder)
        txt_ids = find_input.text_ids
        txt_feats = backbone_out["language_features"][:, txt_ids]
        txt_masks = backbone_out["language_mask"][txt_ids]

        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_masks, img_pos_embeds, vis_feat_sizes = feat_tuple

        if prev_mask_pred is not None:
            # TODO: Support Multi-scale? for now, mutli-scale will break other things (like decoder boxRPB), so it won't go silently.
            img_feats = [img_feats[-1] + prev_mask_pred]
        # Encode geometry
        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )
        if visual_prompt_embed is None:
            visual_prompt_embed = torch.zeros(
                (0, *geo_feats.shape[1:]), device=geo_feats.device
            )
            visual_prompt_mask = torch.zeros(
                (*geo_masks.shape[:-1], 0),
                device=geo_masks.device,
                dtype=geo_masks.dtype,
            )
        if encode_text:
            prompt = torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([txt_masks, geo_masks, visual_prompt_mask], dim=1)
        else:
            prompt = torch.cat([geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([geo_masks, visual_prompt_mask], dim=1)
        return prompt, prompt_mask, backbone_out

    def _run_encoder(
        self,
        backbone_out,
        find_input,
        prompt,
        prompt_mask,
        encoder_extra_kwargs: Optional[Dict] = None,
    ):
        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_masks, img_pos_embeds, vis_feat_sizes = feat_tuple

        # Run the encoder
        prompt_pos_embed = torch.zeros_like(prompt)
        # make a copy of the image feature lists since the encoder may modify these lists in-place
        memory = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=img_masks.copy(),
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )
        encoder_out = {
            # encoded image features
            "encoder_hidden_states": memory["memory"],
            "pos_embed": memory["pos_embed"],
            "padding_mask": memory["padding_mask"],
            "level_start_index": memory["level_start_index"],
            "spatial_shapes": memory["spatial_shapes"],
            "valid_ratios": memory["valid_ratios"],
            "vis_feat_sizes": vis_feat_sizes,
            # encoded text features (or other prompts)
            "prompt_before_enc": prompt,
            "prompt_after_enc": memory.get("memory_text", prompt),
            "prompt_mask": prompt_mask,
        }
        return backbone_out, encoder_out, feat_tuple

    def _run_decoder(
        self,
        pos_embed,
        memory,
        src_mask,
        out,
        prompt,
        prompt_mask,
        encoder_out,
    ):
        bs = memory.shape[1]
        query_embed = self.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)

        hs, reference_boxes, dec_presence_out, dec_presence_feats = (
            self.transformer.decoder(
                tgt=tgt,
                memory=memory,
                memory_key_padding_mask=src_mask,
                pos=pos_embed,
                reference_boxes=None,
                level_start_index=encoder_out["level_start_index"],
                spatial_shapes=encoder_out["spatial_shapes"],
                valid_ratios=encoder_out["valid_ratios"],
                tgt_mask=None,
                memory_text=prompt,
                text_attention_mask=prompt_mask,
            )
        )
        hs = hs.transpose(1, 2)  # seq-first to batch-first
        reference_boxes = reference_boxes.transpose(1, 2)  # seq-first to batch-first
        if dec_presence_out is not None:
            # seq-first to batch-first
            dec_presence_out = dec_presence_out.transpose(1, 2)

        out["presence_feats"] = dec_presence_feats
        self._update_scores_and_boxes(
            out,
            hs,
            reference_boxes,
            prompt,
            prompt_mask,
            dec_presence_out=dec_presence_out,
        )
        return out, hs

    def _update_scores_and_boxes(
        self,
        out,
        hs,
        reference_boxes,
        prompt,
        prompt_mask,
        apply_dac=None,
        dec_presence_out=None,
        is_instance_prompt=False,
    ):
        apply_dac = apply_dac if apply_dac is not None else self.transformer.decoder.dac
        num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)
        num_o2m = hs.size(2) - num_o2o
        assert num_o2m == (num_o2o if apply_dac else 0)
        out["queries"] = hs[-1][:, :num_o2o]  # remove o2m queries if there are any
        # score prediction
        if self.use_dot_prod_scoring:
            dot_prod_scoring_head = self.dot_prod_scoring
            if is_instance_prompt and self.instance_dot_prod_scoring is not None:
                dot_prod_scoring_head = self.instance_dot_prod_scoring
            outputs_class = dot_prod_scoring_head(hs, prompt, prompt_mask)
        else:
            class_embed_head = self.class_embed
            if is_instance_prompt and self.instance_class_embed is not None:
                class_embed_head = self.instance_class_embed
            outputs_class = class_embed_head(hs)

        # box prediction
        box_head = self.transformer.decoder.bbox_embed
        if (
            is_instance_prompt
            and self.transformer.decoder.instance_bbox_embed is not None
        ):
            box_head = self.transformer.decoder.instance_bbox_embed
        anchor_box_offsets = box_head(hs)
        reference_boxes_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid()
        outputs_boxes_xyxy = box_cxcywh_to_xyxy(outputs_coord)

        if dec_presence_out is not None:
            _update_out(out, "presence_logit_dec", dec_presence_out)

        if self.supervise_joint_box_scores:
            assert dec_presence_out is not None
            prob_dec_presence_out = dec_presence_out.clone().sigmoid()
            if self.detach_presence_in_joint_score:
                prob_dec_presence_out = prob_dec_presence_out.detach()

            outputs_class = inverse_sigmoid(
                outputs_class.sigmoid() * prob_dec_presence_out.unsqueeze(2)
            ).clamp(min=-10.0, max=10.0)

        _update_out(out, "pred_logits", outputs_class[:, :, :num_o2o])
        _update_out(out, "pred_boxes", outputs_coord[:, :, :num_o2o])
        _update_out(out, "pred_boxes_xyxy", outputs_boxes_xyxy[:, :, :num_o2o])
        if num_o2m > 0:
            _update_out(out, "pred_logits_o2m", outputs_class[:, :, num_o2o:])
            _update_out(out, "pred_boxes_o2m", outputs_coord[:, :, num_o2o:])
            _update_out(out, "pred_boxes_xyxy_o2m", outputs_boxes_xyxy[:, :, num_o2o:])

    def _run_segmentation_heads(
        self,
        out,
        backbone_out,
        img_ids,
        vis_feat_sizes,
        encoder_hidden_states,
        prompt,
        prompt_mask,
        hs,
        apply_dac=None,
    ):
        apply_dac = apply_dac if apply_dac is not None else self.transformer.decoder.dac
        if self.segmentation_head is not None:
            num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)
            num_o2m = hs.size(2) - num_o2o
            obj_queries = hs if self.o2m_mask_predict else hs[:, :, :num_o2o]
            seg_head_outputs = activation_ckpt_wrapper(self.segmentation_head)(
                backbone_feats=backbone_out["backbone_fpn"],
                obj_queries=obj_queries,
                image_ids=img_ids,
                encoder_hidden_states=encoder_hidden_states,
                act_ckpt_enable=self.training and self.use_act_checkpoint_seg_head,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )
            aux_masks = False  # self.aux_loss and self.segmentation_head.aux_masks
            for k, v in seg_head_outputs.items():
                if k in self.segmentation_head.instance_keys:
                    _update_out(out, k, v[:, :num_o2o], auxiliary=aux_masks)
                    if (
                        self.o2m_mask_predict and num_o2m > 0
                    ):  # handle o2m mask prediction
                        _update_out(
                            out, f"{k}_o2m", v[:, num_o2o:], auxiliary=aux_masks
                        )
                else:
                    out[k] = v

    def _get_best_mask(self, out):
        prev_mask_idx = out["pred_logits"].argmax(dim=1).squeeze(1)
        batch_idx = torch.arange(
            out["pred_logits"].shape[0], device=prev_mask_idx.device
        )
        prev_mask_pred = out["pred_masks"][batch_idx, prev_mask_idx][:, None]
        # Downsample mask to match image resolution.
        prev_mask_pred = self.geometry_encoder.mask_encoder.mask_downsampler(
            prev_mask_pred
        )
        prev_mask_pred = prev_mask_pred.flatten(-2).permute(2, 0, 1)

        return prev_mask_pred

    def forward_grounding(
        self,
        backbone_out,
        find_input,
        geometric_prompt: Prompt,
    ):
        with torch.profiler.record_function("SAM3Image._encode_prompt"):
            prompt, prompt_mask, backbone_out = self._encode_prompt(
                backbone_out, find_input, geometric_prompt
            )
        # Run the encoder
        with torch.profiler.record_function("SAM3Image._run_encoder"):
            backbone_out, encoder_out, _ = self._run_encoder(
                backbone_out, find_input, prompt, prompt_mask
            )
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }

        # Run the decoder
        with torch.profiler.record_function("SAM3Image._run_decoder"):
            out, hs = self._run_decoder(
                memory=out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )

        # Run segmentation heads
        with torch.profiler.record_function("SAM3Image._run_segmentation_heads"):
            self._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=find_input.img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"][img_feat_inds],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )

        # TODO (Nico / Kalyan): Add support back for interactive in evals.
        # matcher is only used during training or for interactive prompts
        # if self.training or self.num_interactive_steps_val > 0:
        #     self._compute_matching(out, self.back_convert(find_target))
        return out

    def _postprocess_out(self, out: Dict, multimask_output: bool = False):
        # TODO: Drop some keys to save memory

        # For multimask output, during eval we return the single best mask with the dict keys expected by the evaluators, but also return the multimasks output with new keys.
        num_mask_boxes = out["pred_boxes"].size(1)
        if not self.training and multimask_output and num_mask_boxes > 1:
            out["multi_pred_logits"] = out["pred_logits"]
            if "pred_masks" in out:
                out["multi_pred_masks"] = out["pred_masks"]
            out["multi_pred_boxes"] = out["pred_boxes"]
            out["multi_pred_boxes_xyxy"] = out["pred_boxes_xyxy"]

            best_mask_idx = out["pred_logits"].argmax(1).squeeze(1)
            batch_idx = torch.arange(len(best_mask_idx), device=best_mask_idx.device)

            out["pred_logits"] = out["pred_logits"][batch_idx, best_mask_idx].unsqueeze(
                1
            )
            if "pred_masks" in out:
                out["pred_masks"] = out["pred_masks"][
                    batch_idx, best_mask_idx
                ].unsqueeze(1)
            out["pred_boxes"] = out["pred_boxes"][batch_idx, best_mask_idx].unsqueeze(1)
            out["pred_boxes_xyxy"] = out["pred_boxes_xyxy"][
                batch_idx, best_mask_idx
            ].unsqueeze(1)

        return out


    # Methods / objects to import BatchedDatapoint, SAM3Output, FindStage
    # Methods to add to the class: _get_geo_prompt_from_find_input, _get_dummy_prompt

    def _get_geo_prompt_from_find_input(self, find_input: FindStage):
        """Construct an initial geometric prompt from the find input."""
        point_embeddings, point_mask, point_labels = None, None, None
        if find_input.input_points_before_embed is not None:
            # Point embeddings are batch first, switch to seq first
            point_embeddings = find_input.input_points_before_embed.transpose(0, 1)

            # they are stored as (x,y,label), so we unpack
            point_labels = point_embeddings[..., -1]
            point_embeddings = point_embeddings[..., :-1]
            point_mask = find_input.input_points_mask

        geometric_prompt = Prompt(
            box_embeddings=find_input.input_boxes_before_embed,
            box_mask=find_input.input_boxes_mask,
            box_labels=find_input.input_boxes_label,
            point_embeddings=point_embeddings,
            point_mask=point_mask,
            point_labels=point_labels,
        )
        return geometric_prompt

    def _get_dummy_prompt(self, find_input: FindStage):
        num_prompts = find_input.img_ids.size(0)
        device = self.device
        geometric_prompt = Prompt(
            box_embeddings=torch.zeros(0, num_prompts, 4, device=device),
            box_mask=torch.zeros(num_prompts, 0, device=device, dtype=torch.bool),
        )
        return geometric_prompt

    def forward(self, input: BatchedDatapoint):
        device = self.device
        backbone_out = {"img_batch_all_stages": input.img_batch}
        backbone_out.update(self.backbone.forward_image(input.img_batch))
        num_frames = len(input.find_inputs)
        assert num_frames == 1

        text_outputs = self.backbone.forward_text(input.find_text_batch, device=device)
        backbone_out.update(text_outputs)

        previous_stages_out = SAM3Output(
            iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
        )

        find_input = input.find_inputs[0]
        find_target = input.find_targets[0]

        num_interactive_steps = 0 if self.training else self.num_interactive_steps_val
        geometric_prompt = self._get_geo_prompt_from_find_input(find_input)

        # Init vars that are shared across the loop.
        stage_outs = []
        for cur_step in range(num_interactive_steps + 1):
            if cur_step > 0:
                # We sample interactive geometric prompts (boxes, points)
                geometric_prompt, _ = self.interactive_prompt_sampler.sample(
                    geo_prompt=geometric_prompt,
                    find_target=find_target,
                    previous_out=stage_outs[-1],
                )
            out= self.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                geometric_prompt=geometric_prompt.clone(),
            )
            stage_outs.append(out)

        previous_stages_out.append(stage_outs)
        return previous_stages_out

    def back_convert(self, targets):
        batched_targets = {
            "boxes": targets.boxes.view(-1, 4),
            "boxes_xyxy": box_cxcywh_to_xyxy(targets.boxes.view(-1, 4)),
            "boxes_padded": targets.boxes_padded,
            "positive_map": targets.boxes.new_ones(len(targets.boxes), 1),
            "num_boxes": targets.num_boxes,
            "masks": targets.segments,
            "semantic_masks": targets.semantic_segments,
            "is_valid_mask": targets.is_valid_segment,
            "is_exhaustive": targets.is_exhaustive,
            "object_ids_packed": targets.object_ids,
            "object_ids_padded": targets.object_ids_padded,
        }
        return batched_targets


    ## Everything below is only used in inference.
    @torch.inference_mode()
    def run_inference(
        self,
        inference_state,
        # instance_prompt=False,
    ):

        # 3) run inference on this frame
        instance_prompt = inference_state["instance_prompt"]
        inference_state["backbone_out"] = self._init_backbone_out_inference(
            inference_state
        )
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

    def _init_backbone_out_inference(self, inference_state):
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
            # run_encoder= cur_step == 0, # TODO: Check this. Currently interactivity_in_encoder is always True?
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

    def compile_model(self):
        """Compile the SAM model with torch.compile for speedup."""
        is_compiled = getattr(self, "_model_is_compiled", False)
        if is_compiled or not self.compile_model:
            return

        import torch._dynamo

        # a larger cache size to hold varying number of shapes for torch.compile
        # see https://github.com/pytorch/pytorch/blob/v2.5.1/torch/_dynamo/config.py#L42-L49
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True

        self.backbone.vision_backbone.forward = clone_output_wrapper(
            torch.compile(
                self.backbone.vision_backbone.forward,
                fullgraph=True,
                mode="max-autotune",
            )
        )
        self.transformer.encoder.forward = clone_output_wrapper(
            torch.compile(
                self.transformer.encoder.forward,
                fullgraph=True,
                mode="max-autotune",
            )
        )
        self.transformer.decoder.forward = clone_output_wrapper(
            torch.compile(
                self.transformer.decoder.forward,
                fullgraph=True,
                mode="max-autotune",
                dynamic=True,  # the decoder uses dynamic shapes
            )
        )
        self._model_is_compiled = True

    ## Everything below is only used in inference.
    @torch.inference_mode()
    def run_inference(
        self,
        inference_state,
        # instance_prompt=False,
    ):

        # 3) run inference on this frame
        instance_prompt = inference_state["instance_prompt"]
        inference_state["backbone_out"] = self._init_backbone_out_inference(
            inference_state
        )
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

    def _init_backbone_out_inference(self, inference_state):
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

        out, _ = self.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            #previous_stages_out=previous_stages_out,
            geometric_prompt=geometric_prompt.clone(),
            # prev_encoder_out=prev_encoder_out,
            # visual_prompt=inference_state["visual_prompt_embed"],
            # visual_prompt_mask=inference_state["visual_prompt_mask"],
            # is_instance_prompt=is_instance_processing,
            # # track_in_reverse=reverse,
            # prev_mask_pred=prev_mask_pred,
        )
        inference_state["previous_stages_out"][frame_idx] = out
        inference_state["per_frame_cur_step"][frame_idx] = cur_step + 1

        return out
