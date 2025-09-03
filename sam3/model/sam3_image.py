# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

from copy import deepcopy
from typing import Dict, Optional

import torch

from sam3.model.model_misc import SAM3Output

# TODO: Kalyan, imports to be fixed!
from sam3.train.data.collator import BatchedDatapoint, FindStage

from .act_ckpt_utils import activation_ckpt_wrapper

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
        apply_dac_on_initial_frame: bool = True,
        interactivity_in_encoder: bool = True,
        matcher=None,
        use_dot_prod_scoring=True,
        supervise_joint_box_scores: bool = False,  # only relevant if using presence token/score
        detach_presence_in_joint_score: bool = False,  # only relevant if using presence token/score
        separate_scorer_for_instance: bool = False,
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
        self.apply_dac_on_initial_frame = apply_dac_on_initial_frame
        self.interactivity_in_encoder = interactivity_in_encoder
        self.matcher = matcher
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
                # this is for the video case. We only have a partial forward of all features
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

    def _get_feat_tuple(self, backbone_out, find_input):
        img_ids, img_feat_inds = find_input.img_ids, slice(None)
        return self._get_img_feats(backbone_out, img_ids), img_feat_inds

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

        feat_tuple, _ = self._get_feat_tuple(backbone_out, find_input)
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
        prev_mask_pred=None,
        encoder_extra_kwargs: Optional[Dict] = None,
    ):
        feat_tuple, img_feat_inds = self._get_feat_tuple(backbone_out, find_input)
        backbone_out, img_feats, img_masks, img_pos_embeds, vis_feat_sizes = feat_tuple
        if prev_mask_pred is not None:
            # TODO: Support Multi-scale? for now, mutli-scale will break other things (like decoder boxRPB), so it won't go silently.
            img_feats = [img_feats[-1] + prev_mask_pred]
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
            "img_feat_inds": img_feat_inds,
            # encoded text features (or other prompts)
            "prompt_before_enc": prompt,
            "prompt_after_enc": memory.get("memory_text", prompt),
            "prompt_mask": prompt_mask,
        }
        return backbone_out, encoder_out, feat_tuple

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
        # Apply segmentation head (w/ bfloat16 autocast just like the rest of the model)
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

    def forward_video_grounding(
        self,
        backbone_out,
        find_input,
        find_target,
        frame_idx,
        previous_stages_out,
        geometric_prompt: Prompt = None,
        run_encoder: bool = True,
        prev_encoder_out: dict = None,
        visual_prompt=None,
        visual_prompt_mask=None,
        is_instance_prompt=False,
        act_ckpt_enable: bool = False,
        track_in_reverse: bool = False,  # track in reverse time order (for demo usage)
        multimask_output: bool = False,
        prev_mask_pred: torch.Tensor = None,
    ):
        """Only activation checkpointing the inner part of video grounding forward."""
        num_prompts = find_input.img_ids.size(0)
        prev_frame_idx = frame_idx + 1 if track_in_reverse else frame_idx - 1

        prev_tracking_queries = self._init_tracking_queries(
            B=num_prompts,
            is_instance_prompt=is_instance_prompt,
            multimask_output=multimask_output,
        )

        # apply DAC on the current frame if specified. If using instance query, the matching is Dummy and we don't need dac.
        apply_dac = (
            self.training  # DAC is only applied during training
            and self.apply_dac_on_initial_frame
            and (frame_idx == 0)
            and not (is_instance_prompt and self.use_instance_query)
        )

        prompt, prompt_mask, backbone_out = self._encode_prompt(
            backbone_out,
            find_input,
            geometric_prompt,
            visual_prompt_embed=visual_prompt,
            visual_prompt_mask=visual_prompt_mask,
            prev_mask_pred=prev_mask_pred,
        )
        # Run the encoder
        if run_encoder:
            backbone_out, encoder_out, _ = self._run_encoder(
                backbone_out,
                find_input,
                prompt,
                prompt_mask,
                prev_mask_pred=prev_mask_pred,
            )
        else:
            assert (
                prev_encoder_out is not None
            ), "Something went wrong. If `run_encoder` is False, encoder outputs from previous step should be passed."
            backbone_out, encoder_out = (
                prev_encoder_out["backbone_out"],
                prev_encoder_out["encoder_out"],
            )

        img_feat_inds = encoder_out["img_feat_inds"]
        # (Note that here we directly index the encoder output visual features into per-query
        # feature maps, so the batch size in decoder will always be the number of text prompts
        # rather than the number of images even under late fusion. This is because for video
        # tracking, the tracking queries will have batch size equal to the number of text prompts
        # anyway. So there is no way to reduce the decoder batch size to be the number of images
        # under late fusion, which is unlike the case of image grounding.)
        encoder_hidden_states = encoder_out["encoder_hidden_states"][:, img_feat_inds]
        pos_embed = encoder_out["pos_embed"][:, img_feat_inds]
        assert encoder_hidden_states.size(1) == num_prompts
        assert pos_embed.size(1) == num_prompts
        src_mask = None
        if encoder_out["padding_mask"] is not None:
            src_mask = encoder_out["padding_mask"][img_feat_inds]  # mask is batch-first
            assert src_mask.size(0) == num_prompts

        out = {
            "encoder_hidden_states": encoder_hidden_states,
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }
        # Run the decoder
        out, hs = self._run_decoder_for_tracking(
            memory=encoder_hidden_states,
            pos_embed=pos_embed,
            src_mask=src_mask,
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
            tracking_queries=prev_tracking_queries,
            is_instance_prompt=is_instance_prompt,
            is_multimask_output=multimask_output,
            apply_dac=apply_dac,
        )

        # Run segmentation heads
        self._run_segmentation_heads(
            out=out,
            backbone_out=backbone_out,
            img_ids=find_input.img_ids,
            vis_feat_sizes=encoder_out["vis_feat_sizes"],
            encoder_hidden_states=encoder_hidden_states,
            prompt=prompt,
            prompt_mask=prompt_mask,
            hs=hs,
            apply_dac=apply_dac,
        )

        if self.training:
            self._compute_matching_for_tracking(out, self.back_convert(find_target))

        out = self._postprocess_out(out, multimask_output=multimask_output)
        return out, backbone_out

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

    def _run_decoder_for_tracking(
        self,
        pos_embed,
        memory,
        src_mask,
        out,
        prompt,
        prompt_mask,
        encoder_out,
        tracking_queries,
        apply_dac=None,
        is_instance_prompt=False,
        **kwargs,
    ):
        # In Video OWL-ViT style tracking, we directly feed previous frame's decoder
        # output embeddings from as current frame's decoder inputs.
        tgt = tracking_queries["embed"]
        reference_boxes = tracking_queries["reference_boxes"]

        hs, reference_boxes, dec_presence_out, dec_presence_feats = (
            self.transformer.decoder(
                tgt=tgt,
                memory=memory,
                memory_key_padding_mask=src_mask,
                pos=pos_embed,
                reference_boxes=reference_boxes,
                level_start_index=encoder_out["level_start_index"],
                spatial_shapes=encoder_out["spatial_shapes"],
                valid_ratios=encoder_out["valid_ratios"],
                tgt_mask=None,
                memory_text=prompt,
                text_attention_mask=prompt_mask,
                apply_dac=apply_dac,
                is_instance_prompt=is_instance_prompt,
            )
        )
        hs = hs.transpose(1, 2)  # seq-first to batch-first
        reference_boxes = reference_boxes.transpose(1, 2)  # seq-first to batch-first
        if dec_presence_out is not None:
            dec_presence_out = dec_presence_out.transpose(
                1, 2
            )  # seq-first to batch-first
        elif self.transformer.decoder.presence_token is not None and is_instance_prompt:
            dec_presence_out = 10 * torch.ones(
                hs.shape[0], hs.shape[1], 1, device=tgt.device
            )
        out["presence_feats"] = dec_presence_feats

        self._update_scores_and_boxes(
            out,
            hs,
            reference_boxes,
            prompt,
            prompt_mask,
            apply_dac,
            dec_presence_out=dec_presence_out,
            is_instance_prompt=is_instance_prompt,
        )
        # in Video OWL-ViT style tracking, all output queries are valid
        scores = out["pred_logits"].squeeze(-1)
        out["pred_is_valid"] = torch.ones_like(scores, dtype=torch.bool)  # (B, Q) shape
        # the previously tracked object ids for all (det + track) output queries
        out["pred_old_obj_ids"] = tracking_queries["object_ids"]
        return out, hs

    def _init_tracking_queries(self, B, is_instance_prompt, multimask_output=False):
        """Initialize the tracking queries for the first frame of a video."""
        # Following Video OWL-ViT, on the first frame, the tracking queries are initialized
        # using the learned detection queries.
        if is_instance_prompt and self.use_instance_query:
            query_embed = self.transformer.decoder.instance_query_embed.weight
            query_embed = query_embed[1:] if multimask_output else query_embed[:1]
            reference_boxes = self.transformer.decoder.instance_reference_points.weight
            reference_boxes = (
                reference_boxes[1:] if multimask_output else reference_boxes[:1]
            )
        else:
            query_embed = self.transformer.decoder.query_embed.weight
            reference_boxes = self.transformer.decoder.reference_points.weight

        reference_boxes = reference_boxes.unsqueeze(1).expand(-1, B, -1).sigmoid()
        device = query_embed.device
        init_embed = query_embed.unsqueeze(1).expand(-1, B, -1)  # (Q, B, D), seq-first
        # Initial object ids are all -1, meaning that they are not tracking any objects yet
        Q = query_embed.size(0)
        init_obj_ids = -torch.ones(B, Q, dtype=torch.long, device=device)
        # Initialize the keep-alive countdown for each query. If the tracked object of a query
        # goes out of frame or gets occluded, we maintain its tracking object id for this countdown
        # number of frames before resetting its object id to -1.
        keep_alive_countdown = -torch.ones_like(init_obj_ids)  # (B, Q) shape
        tracking_queries = {
            "embed": init_embed,  # (Q, B, D) shape, seq-first
            "reference_boxes": reference_boxes,  # (Q, B, D) shape, seq-first
            "object_ids": init_obj_ids,  # (B, Q) shape
            "keep_alive_countdown": keep_alive_countdown,  # (B, Q) shape
            # the maximum object id assigned so far (to assign new ids during inference)
            "max_object_id": torch.zeros(B, 1, dtype=torch.long, device=device),
        }
        return tracking_queries

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

    def forward(self, input: BatchedDatapoint, is_inference=False):
        device = self.device
        backbone_out = {"img_batch_all_stages": input.img_batch}
        # if self.training or not self.forward_backbone_per_frame_for_eval:
        backbone_out.update(self.backbone.forward_image(input.img_batch))
        num_frames = len(input.find_inputs)
        assert num_frames == 1
        # NOTE: This sampler can change the input. In case we sample a visual prompt on the fly.
        # initial_prompt, initial_sampler_meta = self.initial_prompt_sampler.sample(
        #     geo_prompt=None,
        #     find_target=None,
        #     previous_out=None,
        #     batch=input,
        #     strategy="targets",  # sample an input from the targets
        # )
        # initial_prompt_type = initial_sampler_meta.sampled_prompt_type
        # is_instance_prompt = initial_prompt_type.is_instance_prompt()
        # if is_instance_prompt and self.use_instance_query:
        #     assert (
        #         self.transformer.decoder.instance_query_embed is not None
        #     ), "If use_instance_query is True, the transformer decoder should have an extra object query, consider setting `instance_query=True` in the transformer decoder"
        # NOTE: text input can be modified on the fly in the initial prompt sampler. We run the text backbone after sampling the first prompt.
        text_outputs = self.backbone.forward_text(input.find_text_batch, device=device)
        backbone_out.update(text_outputs)

        # Store visual prompt to be used in all frames
        visual_prompt, visual_prompt_mask = None, None
        # if initial_prompt_type is not SampledPromptType.TEXT:
        #     visual_prompt, visual_prompt_mask, _ = self._encode_prompt(
        #         backbone_out=backbone_out,
        #         find_input=input.find_inputs[initial_sampler_meta.frame_idx],
        #         geometric_prompt=initial_prompt,
        #         encode_text=False,
        #     )
        # Sample frames to add correction geometric prompts
        # frames_to_correct = self._get_frames_to_correct(num_frames)
        # In the model, we always want to access the final output of the previous stage
        # NOTE: In the loss computation, we still use all steps per stage
        previous_stages_out = SAM3Output(
            iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
        )
        # loop over all frames in the video
        for frame_idx in range(num_frames):
            find_input = input.find_inputs[frame_idx]
            find_target = input.find_targets[frame_idx]

            num_interactive_steps = 0
            is_instance_prompt = False
            # num_interactive_steps = 0 self._get_num_interactive_steps(
            #     initial_prompt_type, frames_to_correct, frame_idx
            # )
            # Each frame starts with an initial geometric prompt
            if visual_prompt is None:
                # Load initial geometric prompt from the batch only if we didn't sample one in the model.
                geometric_prompt = self._get_geo_prompt_from_find_input(find_input)
            else:
                # Init the geometric prompt to an empty sequence
                geometric_prompt = self._get_dummy_prompt(find_input)
            # Init vars that are shared across the loop.
            prev_encoder_out, stage_outs, is_multimask_out, prev_mask_pred = (
                None,
                [],
                False,
                None,
            )
            for cur_step in range(num_interactive_steps + 1):
                # if cur_step > 0:
                #     # We sample interactive geometric prompts (boxes, points)
                #     geometric_prompt, _ = self.interactive_prompt_sampler.sample(
                #         geo_prompt=geometric_prompt,
                #         find_target=find_target,
                #         previous_out=stage_outs[-1],
                #         is_multimask_out=is_multimask_out,
                #     )
                # is_multimask_out = (
                #     self.multimask_output
                #     and is_instance_prompt
                #     and initial_prompt_type == SampledPromptType.INSTANCE_POINT
                #     and self.use_instance_query
                #     and (
                #         frame_idx == 0 and cur_step == 0
                #     )  # For now, limit to frame 0, TODO: allow cond. frames
                # )
                vis_prompt_with_mem = visual_prompt
                vis_prompt_with_mem_mask = visual_prompt_mask
                # Don't encode first frame spatial memory for instance mask prompts (as its added separately as a visual prompt)
                # We can encode its object pointers though.
                # skip_spatial_mem_frames = (
                #     {0}
                #     if initial_prompt_type is SampledPromptType.INSTANCE_MASK
                #     else set()
                # )
                # memory, memory_mask = self.memory_bank.get_memories(
                #     frame_idx, skip_spatial_frames=skip_spatial_mem_frames
                # )
                # if memory is not None and not self.image_only:
                #     vis_prompt_with_mem = torch.cat(
                #         (visual_prompt, memory)
                #         if visual_prompt is not None
                #         else (memory,),
                #         dim=0,
                #     )
                #     vis_prompt_with_mem_mask = torch.cat(
                #         (visual_prompt_mask, memory_mask)
                #         if visual_prompt_mask is not None
                #         else (memory_mask,),
                #         dim=1,
                #     )
                out, backbone_out_with_frame_feats = self.forward_video_grounding(
                    backbone_out=backbone_out,
                    find_input=find_input,
                    find_target=find_target,
                    frame_idx=frame_idx,
                    previous_stages_out=previous_stages_out,
                    geometric_prompt=geometric_prompt.clone(),
                    run_encoder=self.interactivity_in_encoder or cur_step == 0,
                    prev_encoder_out=prev_encoder_out,
                    visual_prompt=vis_prompt_with_mem,
                    visual_prompt_mask=vis_prompt_with_mem_mask,
                    is_instance_prompt=is_instance_prompt,
                    multimask_output=is_multimask_out,
                    prev_mask_pred=prev_mask_pred,
                )
                # save additional metadata for loss computation
                is_instance_processing = is_instance_prompt and self.use_instance_query
                is_video_grounding_batch = num_frames > 1 and not is_instance_processing
                for o in [out] + out["aux_outputs"]:
                    # video grounding batch (more than 1 frames; not VOS)
                    o["is_video_grounding_batch"] = is_video_grounding_batch
                    o["is_video_grounding_batch_o2m"] = is_video_grounding_batch
                prev_encoder_out = out.pop("prev_encoder_out")
                # Using previous mask in the next interactivity step is currently only supported for instance tasks.
                # if self.use_prev_mask and (
                #     is_instance_prompt and self.use_instance_query
                # ):
                #     prev_mask_pred = self._get_best_mask(out)
                stage_outs.append(out)

            # HACK: Recompute mask visual prompt for box/point instance prompts, to act as init conditioning frame in VOS
            # TODO: Remove this part once we have a proper memory bank
            # if (
            #     frame_idx == 0
            #     and num_frames > 1
            #     and initial_sampler_meta.sampled_prompt_type
            #     in [SampledPromptType.INSTANCE_POINT, SampledPromptType.INSTANCE_BOX]
            # ):
            #     visual_prompt, visual_prompt_mask = self._get_mask_vis_prompt(
            #         input, backbone_out_with_frame_feats, initial_prompt, frame_idx, out
            #     )

            if self.training:
                previous_stages_out.append(stage_outs)
            else:
                if num_frames == 1:
                    # Keep all steps for image evals
                    previous_stages_out.append(stage_outs)
                else:
                    # For videos, we save memory by only keeping the final out per stage
                    previous_stages_out.append([stage_outs[-1]])
            # Optionally trim outputs or offload to CPU during eval to save GPU memory
            # if self.trim_outputs_for_eval and not self.training:
            #     self._trim_outputs(previous_stages_out, frame_idx)
            # if self.offload_outputs_to_cpu_for_eval and not self.training:
            #     self._offload_outputs_to_cpu(previous_stages_out, frame_idx)
        # if DEBUG:
        #     previous_stages_out[0]["initial_prompt"] = initial_prompt
        get_queries = None
        # self.memory_bank.reset()
        return previous_stages_out, get_queries

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

    def _compute_matching_for_tracking(self, out, converted_targets):
        """
        Compute matching between the decoder outputs and the ground-truth objects for tracking.

        Note that to avoid confusion, we use the term "prompt" to refer to the text (or any
        other input) prompts in the batch dimension, and the term "query" to refer to the DETR-
        style "query embedding" (decoder input or output hidden states) of each text prompt.

        We assume we have B text prompts (i.e. the decoder batch size is B) and Q queries for
        each text prompt (i.e. the decoder hidden state has shape B x Q x dim, some of which
        might be already tracking an object in the previous frame), and also the ground-truth
        objects (including those invisible in the current frame) for the B text prompts in
        padded format (see comments in the code below).

        During training, we split the predictions and GT objects in two groups (locked vs free)
        and match them separately:
        - locked queries and GTs: for a query that is already tracking an object in previous
          frames (i.e. those with tracking_queries["object_ids"] >= 0), it will be matched with
          the same object if it's still visible in the current frame; otherwise (if the tracked
          object is no longer visible in the current frame), it will not be matched to any GTs.
        - free queries and GTs: for a free query not tracking a existing object (i.e. those with
          tracking_queries["object_ids"] < 0), it will just be matched with the free GTs (i.e.
          GTs not already matched to a locked query in the step above) via bipartie matching.
        """
        # Step A: prepare the GT boxes and object ids for matching.
        # We first figure out those GT boxes that are actually visible in the current frame
        # (note that for easy model implementation, in the video dataset we assume the same
        # list of objects on all frames regardless of whether they are visible or not; so
        # we need to filter out those invisible objects).
        pred_old_obj_ids = out["pred_old_obj_ids"]  # (B, Q) shape
        pred_is_valid = out["pred_is_valid"]  # (B, Q) shape
        B, Q = pred_old_obj_ids.shape
        # Here `num_boxes` contains the number of GT boxes for each of the B text prompts.
        num_boxes = converted_targets["num_boxes"]  # (B,) shape
        device = num_boxes.device
        # *padded* GT boxes, with target boxes for each of the B text prompt's padded into
        # (B, H, 4) shape in (cx, cy, w, h) format, where H is the maximum number of GT boxes
        # among all B text prompts (including invisible ones). It's generated via `packed_to_padded_naive`
        # in projects/onevision/utils/misc.py)
        gt_padded_boxes = converted_targets["boxes_padded"]  # (B, H, 4) shape, CxCyWH
        _, H, _ = gt_padded_boxes.shape
        # The object ids for the find targets in padded format.
        gt_padded_object_ids = converted_targets["object_ids_padded"]  # (B, H) shape
        # *packed* GT object ids, where G is the *total* number of GT boxes (including
        # invisible ones) for all B text prompts.
        gt_packed_object_ids = converted_targets["object_ids_packed"]  # (G,) shape

        # Step B: first, match those locked queries and GTs based on tracking query index
        pred_is_locked = pred_is_valid & (pred_old_obj_ids >= 0)  # (B, Q) shape
        # Find the visilble ground-truth boxes based on their width and height
        gt_padded_is_visible = (
            (gt_padded_object_ids >= 0)
            & (gt_padded_boxes[..., 2] > 0)  # width > 0
            & (gt_padded_boxes[..., 3] > 0)  # height > 0
        )  # (B, H) shape
        # Match all the queries with GTs based on the previously tracked object ids
        locked_match_padded = (
            pred_is_locked[:, :, None]
            & (pred_old_obj_ids[:, :, None] == gt_padded_object_ids[:, None, :])
            & gt_padded_is_visible[:, None, :]
        )  # (B, Q, H) shape
        locked_batch_idx, locked_src_idx, locked_tgt_idx_padded = torch.nonzero(
            locked_match_padded, as_tuple=True
        )
        # Convert target box index from padded format to packed format `locked_tgt_idx`
        # (`all_tgt_inds` has shape (B, H) and contains a target box's packed index, computed as
        # GT index within each prompt + packed offset of the prompt)
        all_tgt_inds = torch.arange(H, device=device)[None, :].repeat(B, 1)
        all_tgt_inds[1:] += num_boxes.cumsum(dim=0)[:-1][:, None]
        locked_tgt_idx = all_tgt_inds[locked_batch_idx, locked_tgt_idx_padded]

        # Step C: then, match the remaining free queries and free GTs via bipartite matching
        # note that we don't match those with `pred_old_obj_ids == -2` (which are false positives
        # added to the tracking queries during training in the `Sam3TrackFormer` model class)
        pred_is_free = pred_is_valid & (pred_old_obj_ids == -1)  # (B, Q) shape
        gt_padded_is_free = gt_padded_is_visible & torch.logical_not(
            torch.any(locked_match_padded, dim=1)
        )  # (B, H) shape
        # Match the top-layer free queries to the free GTs
        matcher_inds = self.matcher(
            out,
            converted_targets,
            out_is_valid=pred_is_free,
            target_is_valid_padded=gt_padded_is_free,
        )
        # save the output and target validity (for potential o2m matcher in DAC)
        out["o2m_out_is_valid"] = pred_is_valid
        out["o2m_target_is_valid_padded"] = gt_padded_is_visible

        batch_idx = torch.cat([locked_batch_idx, matcher_inds[0]])
        src_idx = torch.cat([locked_src_idx, matcher_inds[1]])
        tgt_idx = torch.cat([locked_tgt_idx, matcher_inds[2]])
        out["indices"] = batch_idx, src_idx, tgt_idx
        # Match the aux-layer free queries to the free GTs
        for aux_out in out.get("aux_outputs", []):
            aux_matcher_inds = self.matcher(
                aux_out,
                converted_targets,
                out_is_valid=pred_is_free,
                target_is_valid_padded=gt_padded_is_free,
            )
            aux_batch_idx = torch.cat([locked_batch_idx, aux_matcher_inds[0]])
            aux_src_idx = torch.cat([locked_src_idx, aux_matcher_inds[1]])
            aux_tgt_idx = torch.cat([locked_tgt_idx, aux_matcher_inds[2]])
            aux_out["indices"] = aux_batch_idx, aux_src_idx, aux_tgt_idx

        # Step D: assign new object ids to the queries based on the matching results
        pred_matched_object_ids = -torch.ones_like(pred_old_obj_ids)
        pred_matched_object_ids[batch_idx, src_idx] = gt_packed_object_ids[tgt_idx]
        out["matched_object_ids"] = pred_matched_object_ids
