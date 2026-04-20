import os

import torch
from sam3.model.vl_combiner import SAM3VLBackbone

try:
    from sam3.model.vl_combiner import SAM3VLBackboneTri
except ImportError:
    SAM3VLBackboneTri = None
from typing import Dict, List, Optional

import numpy as np
from sam3.model.data_misc import BatchedDatapoint, FindStage
from sam3.model.geometry_encoders import Prompt
from sam3.model.model_misc import SAM3Output
from sam3.model.sam3_image import Sam3Image
from sam3.model.sam3_multiplex_detector_utils import nms_masks


class Sam3MultiplexImageBase(Sam3Image):
    """A wrapper class to run Sam3Image on videos for per-frame detection (no tracking)."""

    def __init__(
        self,
        *args,
        tracking_score_thresh: float = 0.0,
        offload_outputs_to_cpu_for_eval: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tracking_score_thresh = tracking_score_thresh
        self.offload_outputs_to_cpu_for_eval = offload_outputs_to_cpu_for_eval
        self.trim_outputs_for_eval = True  # dummy option -- it doesn't do anything

    def forward(
        self,
        input: BatchedDatapoint,
        is_inference=False,  # (a dummy parameter not used anymore)
    ):
        assert not self.training, (
            "Sam3MultiplexImageBase should only be used in eval mode."
        )

        device = self.device
        backbone_out = {"img_batch_all_stages": input.img_batch}
        text_outputs = self.backbone.forward_text(input.find_text_batch, device=device)
        backbone_out.update(text_outputs)
        num_frames = len(input.find_inputs)

        previous_stages_out = SAM3Output(
            iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
        )
        for frame_idx in range(num_frames):
            find_input = input.find_inputs[frame_idx]
            find_target = input.find_targets[frame_idx]
            geometric_prompt = self._get_geo_prompt_from_find_input(find_input)
            cur_out, _ = self.forward_video_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=find_target,
                geometric_prompt=geometric_prompt,
            )
            # offload model outputs to CPU (to save GPU memory) for evaluation
            if self.offload_outputs_to_cpu_for_eval:
                cur_out = {k: v.cpu() for k, v in cur_out.items()}

            previous_stages_out.append([cur_out])

        get_queries = None
        return previous_stages_out, get_queries

    def forward_video_grounding(
        self,
        backbone_out,
        find_input,
        find_target,
        geometric_prompt: Prompt,
        **kwargs,
    ):
        # route this to the image grounding forward method
        out = self.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=find_target,
            geometric_prompt=geometric_prompt,
        )
        # trim the output to only include the necessary keys
        out = {
            "pred_logits": out["pred_logits"],
            "pred_boxes": out["pred_boxes"],
            "pred_boxes_xyxy": out["pred_boxes_xyxy"],
            "pred_masks": out["pred_masks"],
            "pred_object_ids": self._get_dummy_object_ids(out["pred_logits"]),
        }
        return out, backbone_out

    def _get_dummy_object_ids(self, pred_logits):
        """Generate dummy object IDs for the detected objects, based on their detection query indices."""
        # Assuming pred_logits has shape [batch_size, num_queries, num_classes]
        B, Q, _ = pred_logits.shape
        is_above_thresh = pred_logits.squeeze(2) > self.tracking_score_thresh
        dummy_obj_ids = torch.arange(Q, device=self.device).expand(B, -1)
        dummy_obj_ids = torch.where(is_above_thresh, dummy_obj_ids, -1)
        return dummy_obj_ids

    def _trim_outputs(self, *args, **kwargs):
        pass  # not needed for image-on-video

    def _batch_find_inputs(
        self,
        find_inputs: List[FindStage],
        chunk_start: int,
        chunk_end: int,
    ) -> FindStage:
        """
        Batch multiple FindStage objects into a single batched FindStage.

        For each frame in the chunk, creates img_ids that point to the correct
        frame index. When processing streaming video, the img_ids are the actual
        frame indices (e.g., [0, 1, 2, ..., 15] for chunk 0-16), and the modulo
        for circular buffer access is applied later in _get_img_feats.

        Args:
            find_inputs: List of FindStage objects for all frames.
            chunk_start: Start index of the chunk.
            chunk_end: End index of the chunk (exclusive).

        Returns:
            A single FindStage with batched tensors.
        """
        chunk_find_inputs = [
            find_inputs[i % len(find_inputs)] for i in range(chunk_start, chunk_end)
        ]

        # Generate img_ids based on chunk frame indices
        # Each frame in the chunk gets its corresponding frame index
        # The modulo for circular buffer access is handled in _get_img_feats
        device = chunk_find_inputs[0].img_ids.device
        dtype = chunk_find_inputs[0].img_ids.dtype
        img_ids_list = [
            torch.tensor([i], device=device, dtype=dtype)
            for i in range(chunk_start, chunk_end)
        ]
        batched_img_ids = torch.cat(img_ids_list, dim=0)

        # Generate img_ids_np to match
        img_ids_np_list = [np.array([i]) for i in range(chunk_start, chunk_end)]
        batched_img_ids_np = np.concatenate(img_ids_np_list, axis=0)

        # Concatenate text_ids
        text_ids_list = [fi.text_ids for fi in chunk_find_inputs]
        batched_text_ids = torch.cat(text_ids_list, dim=0)

        # Concatenate input_boxes
        input_boxes_list = [fi.input_boxes for fi in chunk_find_inputs]
        batched_input_boxes = (
            torch.cat(input_boxes_list, dim=0)
            if input_boxes_list[0] is not None
            else None
        )

        # Concatenate input_boxes_mask
        input_boxes_mask_list = [fi.input_boxes_mask for fi in chunk_find_inputs]
        batched_input_boxes_mask = (
            torch.cat(input_boxes_mask_list, dim=0)
            if input_boxes_mask_list[0] is not None
            else None
        )

        # Concatenate input_boxes_label
        input_boxes_label_list = [fi.input_boxes_label for fi in chunk_find_inputs]
        batched_input_boxes_label = (
            torch.cat(input_boxes_label_list, dim=0)
            if input_boxes_label_list[0] is not None
            else None
        )

        # Concatenate input_points
        input_points_list = [fi.input_points for fi in chunk_find_inputs]
        batched_input_points = (
            torch.cat(input_points_list, dim=0)
            if input_points_list[0] is not None
            else None
        )

        # Concatenate input_points_mask
        input_points_mask_list = [fi.input_points_mask for fi in chunk_find_inputs]
        batched_input_points_mask = (
            torch.cat(input_points_mask_list, dim=0)
            if input_points_mask_list[0] is not None
            else None
        )

        # Handle optional fields
        input_boxes_before_embed_list = [
            fi.input_boxes_before_embed for fi in chunk_find_inputs
        ]
        batched_input_boxes_before_embed = (
            torch.cat(input_boxes_before_embed_list, dim=0)
            if input_boxes_before_embed_list[0] is not None
            else None
        )

        input_points_before_embed_list = [
            fi.input_points_before_embed for fi in chunk_find_inputs
        ]
        batched_input_points_before_embed = (
            torch.cat(input_points_before_embed_list, dim=0)
            if input_points_before_embed_list[0] is not None
            else None
        )

        # Create batched FindStage
        batched_find_input = FindStage(
            img_ids=batched_img_ids,
            img_ids_np=batched_img_ids_np,
            text_ids=batched_text_ids,
            input_boxes=batched_input_boxes,
            input_boxes_mask=batched_input_boxes_mask,
            input_boxes_label=batched_input_boxes_label,
            input_points=batched_input_points,
            input_points_mask=batched_input_points_mask,
            ptrs=None,  # Not batching pointers for now
            ptrs_seg=None,
            object_ids=None,
            input_boxes_before_embed=batched_input_boxes_before_embed,
            input_points_before_embed=batched_input_points_before_embed,
        )

        return batched_find_input

    def _batch_geometric_prompts(
        self,
        geometric_prompts: List[Prompt],
        chunk_start: int,
        chunk_end: int,
    ) -> Prompt:
        """
        Batch multiple Prompt objects into a single batched Prompt.

        Args:
            geometric_prompts: List of Prompt objects for all frames.
            chunk_start: Start index of the chunk.
            chunk_end: End index of the chunk (exclusive).

        Returns:
            A single Prompt with batched tensors.
        """
        chunk_prompts = [geometric_prompts[i] for i in range(chunk_start, chunk_end)]
        return self._batch_geometric_prompts_from_list(chunk_prompts)

    def _batch_geometric_prompts_from_list(
        self,
        chunk_prompts: List[Prompt],
    ) -> Prompt:
        """
        Batch a list of Prompt objects into a single batched Prompt.

        Prompt uses seq-first, batch-second convention:
        - box_embeddings: N_boxes x B x C_box - batch along dim 1
        - box_mask: B x N_boxes - batch along dim 0
        - box_labels: N_boxes x B - batch along dim 1
        - point_embeddings: N_points x B x C_point - batch along dim 1
        - point_mask: B x N_points - batch along dim 0
        - point_labels: N_points x B - batch along dim 1

        Args:
            chunk_prompts: List of Prompt objects to batch.

        Returns:
            A single Prompt with batched tensors.
        """

        # Helper function to batch tensors along specified dimension
        def batch_tensors(tensors, dim):
            if tensors[0] is None:
                return None
            return torch.cat(tensors, dim=dim)

        # Batch box embeddings (N_boxes x B x C_box - batch along dim 1)
        box_embeddings_list = [p.box_embeddings for p in chunk_prompts]
        batched_box_embeddings = batch_tensors(box_embeddings_list, dim=1)

        # Batch box mask (B x N_boxes - batch along dim 0)
        box_mask_list = [p.box_mask for p in chunk_prompts]
        batched_box_mask = batch_tensors(box_mask_list, dim=0)

        # Batch box labels (N_boxes x B - batch along dim 1)
        box_labels_list = [p.box_labels for p in chunk_prompts]
        batched_box_labels = batch_tensors(box_labels_list, dim=1)

        # Batch point embeddings (N_points x B x C_point - batch along dim 1)
        point_embeddings_list = [p.point_embeddings for p in chunk_prompts]
        batched_point_embeddings = batch_tensors(point_embeddings_list, dim=1)

        # Batch point mask (B x N_points - batch along dim 0)
        point_mask_list = [p.point_mask for p in chunk_prompts]
        batched_point_mask = batch_tensors(point_mask_list, dim=0)

        # Batch point labels (N_points x B - batch along dim 1)
        point_labels_list = [p.point_labels for p in chunk_prompts]
        batched_point_labels = batch_tensors(point_labels_list, dim=1)

        # Create batched Prompt
        batched_prompt = Prompt(
            box_embeddings=batched_box_embeddings,
            box_mask=batched_box_mask,
            box_labels=batched_box_labels,
            point_embeddings=batched_point_embeddings,
            point_mask=batched_point_mask,
            point_labels=batched_point_labels,
        )

        return batched_prompt


class Sam3MultiplexDetector(Sam3MultiplexImageBase):
    def __init__(
        self,
        *args,
        async_all_gather=True,
        gather_backbone_out=None,
        is_multiplex=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.async_all_gather = async_all_gather

        # if gather_backbone is not set, default to gathering only for `SAM3VLBackbone`
        if gather_backbone_out is None:
            gather_backbone_out = isinstance(self.backbone, SAM3VLBackbone) or (
                SAM3VLBackboneTri is not None
                and isinstance(self.backbone, SAM3VLBackboneTri)
            )
        self.gather_backbone_out = gather_backbone_out
        self.is_multiplex = is_multiplex

    def forward_video_grounding_multigpu(
        self,
        backbone_out,
        find_inputs,
        geometric_prompt: Prompt,
        frame_idx,
        num_frames,
        # `multigpu_buffer` is a dict to cache FA outputs in a chunk between different calls
        multigpu_buffer,
        track_in_reverse=False,
        # whether to also return the SAM2 backbone features (in addition to FA results)
        return_sam2_backbone_feats=False,
        # whether to perform NMS and suppress the scores of those detections removed by NMS
        run_nms=False,
        nms_prob_thresh=None,
        nms_iou_thresh=None,
        nms_use_iom=False,
        # tracking bounds to respect max_frame_num_to_track
        max_frame_num_to_track=None,
        propagate_in_video_start_frame_idx=None,
        # feature_cache for buffered backbone computation
        feature_cache=None,
        **kwargs,
    ):
        """
        Compute the FA detection outputs in a distributed manner, where all GPUs process
        a chunk of frames (equal to the number of GPUs) at once and store them in cache.
        """
        # Calculate valid frame range based on max_frame_num_to_track
        # We prevent pre-fetching beyond the tracking window relative to current frame
        if max_frame_num_to_track is not None:
            if propagate_in_video_start_frame_idx is None:
                propagate_in_video_start_frame_idx = 0
            if track_in_reverse:
                # When going backwards, limit how far back we can go from current frame
                valid_frame_start = max(
                    0,
                    propagate_in_video_start_frame_idx - max_frame_num_to_track + 1,
                )
                valid_frame_end = num_frames
            else:
                # When going forwards, limit how far ahead we can go from current frame
                valid_frame_start = 0
                valid_frame_end = min(
                    num_frames,
                    propagate_in_video_start_frame_idx + max_frame_num_to_track,
                )
        else:
            # No tracking limit specified, use full video range
            valid_frame_start = 0
            valid_frame_end = num_frames

        # Step 1: fetch the FA outputs in the current chunk from buffer
        frame_idx_curr_b = frame_idx - frame_idx % self.world_size
        frame_idx_curr_e = min(frame_idx_curr_b + self.world_size, num_frames)

        # Clamp the current chunk to the valid tracking range
        frame_idx_curr_b = max(frame_idx_curr_b, valid_frame_start)
        frame_idx_curr_e = min(frame_idx_curr_e, valid_frame_end)
        # in case the current frame's FA results are not in the buffer yet, build the current chunk
        # (this should only happen on the first chunk, since we are also building the next chunk below)
        if frame_idx not in multigpu_buffer:
            with torch.profiler.record_function("build_multigpu_buffer_next_chunk1"):
                self._build_multigpu_buffer_next_chunk(
                    backbone_out=backbone_out,
                    find_inputs=find_inputs,
                    geometric_prompt=geometric_prompt,
                    frame_idx_begin=frame_idx_curr_b,
                    frame_idx_end=frame_idx_curr_e,
                    num_frames=num_frames,
                    multigpu_buffer=multigpu_buffer,
                    run_nms=run_nms,
                    nms_prob_thresh=nms_prob_thresh,
                    nms_iou_thresh=nms_iou_thresh,
                    nms_use_iom=nms_use_iom,
                    feature_cache=feature_cache,
                )

        # read out the current frame's results from `multigpu_buffer`
        out = {}
        for k, (v, handle) in multigpu_buffer[frame_idx].items():
            if self.is_multiplex:
                if (
                    k.startswith("interactive_backbone_")
                    or k.startswith("propagation_backbone_")
                ) and not return_sam2_backbone_feats:
                    continue
            else:
                if k.startswith("sam2_backbone_") and not return_sam2_backbone_feats:
                    continue
            if handle is not None:
                handle.wait()  # wait for async all-gather to finish
            out[k] = v

        # Step 2: remove FA outputs of the previous chunk from cache to save GPU memory
        if not track_in_reverse and frame_idx_curr_b - self.world_size >= 0:
            frame_idx_prev_e = frame_idx_curr_b
            frame_idx_prev_b = frame_idx_curr_b - self.world_size
        elif track_in_reverse and frame_idx_curr_e < num_frames:
            frame_idx_prev_b = frame_idx_curr_e
            frame_idx_prev_e = min(frame_idx_prev_b + self.world_size, num_frames)
        else:
            frame_idx_prev_b = frame_idx_prev_e = None
        if frame_idx_prev_b is not None:
            for frame_idx_rm in range(frame_idx_prev_b, frame_idx_prev_e):
                multigpu_buffer.pop(frame_idx_rm, None)

        # Step 3: compute and cache FA outputs of the next chunk ahead of time
        # (so that we can overlap computation with all-gather transfer)
        # Respect tracking bounds when calculating next chunk

        if not track_in_reverse and frame_idx_curr_e < valid_frame_end:
            frame_idx_next_b = frame_idx_curr_e
            frame_idx_next_e = min(frame_idx_next_b + self.world_size, valid_frame_end)
        elif (
            track_in_reverse and frame_idx_curr_b - self.world_size >= valid_frame_start
        ):
            frame_idx_next_e = frame_idx_curr_b
            frame_idx_next_b = max(
                frame_idx_curr_b - self.world_size, valid_frame_start
            )
        else:
            frame_idx_next_b = frame_idx_next_e = None
        if frame_idx_next_b is not None and frame_idx_next_b not in multigpu_buffer:
            with torch.profiler.record_function("build_multigpu_buffer_next_chunk2"):
                self._build_multigpu_buffer_next_chunk(
                    backbone_out=backbone_out,
                    find_inputs=find_inputs,
                    geometric_prompt=geometric_prompt,
                    frame_idx_begin=frame_idx_next_b,
                    frame_idx_end=frame_idx_next_e,
                    num_frames=num_frames,
                    multigpu_buffer=multigpu_buffer,
                    run_nms=run_nms,
                    nms_prob_thresh=nms_prob_thresh,
                    nms_iou_thresh=nms_iou_thresh,
                    feature_cache=feature_cache,
                )

        return out, backbone_out

    def _build_multigpu_buffer_next_chunk(
        self,
        backbone_out,
        find_inputs,
        geometric_prompt: Prompt,
        frame_idx_begin,
        frame_idx_end,
        num_frames,
        multigpu_buffer,
        run_nms=False,
        nms_prob_thresh=None,
        nms_iou_thresh=None,
        nms_use_iom=False,
        feature_cache=None,
    ):
        """Compute FA outputs on a chunk of frames and store their results in multigpu_buffer."""
        # each GPU computes FA on one frame in the chunk (in a round-robin manner)
        frame_idx_local_gpu = min(frame_idx_begin + self.rank, frame_idx_end - 1)
        # `forward_grounding` (from base class `Sam3MultiplexImageBase`) runs FA on a single frame
        with torch.profiler.record_function("forward_grounding"):
            out_local = self.forward_grounding(
                backbone_out=backbone_out,
                # HACK: Since find_inputs is on GPU having to realloc is expensive so changing the values in place for the prod usecase
                # i.e. when using the streaming frame loader resource instead of local file. For non-prod is always
                # frame_idx_local_gpu < len(find_inputs) so should be a no-op
                find_input=find_inputs[frame_idx_local_gpu % len(find_inputs)],
                find_target=None,
                geometric_prompt=geometric_prompt,
                feature_cache=feature_cache,
            )
        if run_nms:
            with torch.profiler.record_function("nms_masks"):
                # run NMS as a post-processing step on top of the detection outputs
                assert nms_prob_thresh is not None and nms_iou_thresh is not None
                pred_probs = out_local["pred_logits"].squeeze(-1).sigmoid()
                pred_masks = out_local["pred_masks"]
                # loop over text prompts (not an overhead for demo where there's only 1 prompt)
                for prompt_idx in range(pred_probs.size(0)):
                    keep = nms_masks(
                        pred_probs=pred_probs[prompt_idx],
                        pred_masks=pred_masks[prompt_idx],
                        prob_threshold=nms_prob_thresh,
                        iou_threshold=nms_iou_thresh,
                        nms_use_iom=nms_use_iom,
                        do_compile=getattr(self, "compile_model", False),
                        running_in_prod=getattr(self, "running_in_prod", False),
                    )
                    # set a very low threshold for those detections removed by NMS
                    out_local["pred_logits"][prompt_idx, :, 0] -= 1e4 * (~keep).float()

        if self.gather_backbone_out:
            # gather the SAM 2 backbone features across GPUs
            if self.is_multiplex:
                # Note that we should not need to compute the interaction features every frame
                # TODO: rooms for optimization

                # Interaction features
                inte_feats = out_local["prev_encoder_out"]["backbone_out"][
                    "interactive"
                ]
                assert inte_feats["vision_mask"] is None
                assert (
                    len(inte_feats["backbone_fpn"]) == 3
                )  # SAM2 backbone always have 3 levels
                assert all(x.mask is None for x in inte_feats["backbone_fpn"])
                # cast the SAM2 backbone features to bfloat16 for all-gather (this is usually
                # a no-op, SAM2 backbone features are likely already in bfloat16 due to AMP)
                inte_backbone_fpn_bf16 = [
                    x.to(torch.bfloat16) for x in inte_feats["backbone_fpn"]
                ]
                inte_fpn0, inte_fpn_handle0 = self._gather_tensor(
                    inte_backbone_fpn_bf16[0].tensors
                )
                inte_fpn1, inte_fpn_handle1 = self._gather_tensor(
                    inte_backbone_fpn_bf16[1].tensors
                )
                inte_fpn2, inte_fpn_handle2 = self._gather_tensor(
                    inte_backbone_fpn_bf16[2].tensors
                )
                # vision_pos_enc is the same on all frames, so no need to all-gather them
                inte_vision_pos_enc = inte_feats["vision_pos_enc"]

            feats = out_local["prev_encoder_out"]["backbone_out"]["sam2_backbone_out"]
            assert feats["vision_mask"] is None
            assert len(feats["backbone_fpn"]) == 3  # SAM2 backbone always have 3 levels
            assert all(x.mask is None for x in feats["backbone_fpn"])
            # cast the SAM2 backbone features to bfloat16 for all-gather (this is usually
            # a no-op, SAM2 backbone features are likely already in bfloat16 due to AMP)
            backbone_fpn_bf16 = [x.to(torch.bfloat16) for x in feats["backbone_fpn"]]
            fpn0, fpn_handle0 = self._gather_tensor(backbone_fpn_bf16[0].tensors)
            fpn1, fpn_handle1 = self._gather_tensor(backbone_fpn_bf16[1].tensors)
            fpn2, fpn_handle2 = self._gather_tensor(backbone_fpn_bf16[2].tensors)
            # vision_pos_enc is the same on all frames, so no need to all-gather them
            vision_pos_enc = feats["vision_pos_enc"]

        # trim the FA output to only include the necessary keys
        out_local = {
            "pred_logits": out_local["pred_logits"],
            "pred_boxes": out_local["pred_boxes"],
            "pred_boxes_xyxy": out_local["pred_boxes_xyxy"],
            "pred_masks": out_local["pred_masks"],
            "pred_object_ids": self._get_dummy_object_ids(out_local["pred_logits"]),
        }

        # gather the results: after this step, each GPU will receive FA outputs on
        # all frames in the chunk and store them in `multigpu_buffer`
        out_gathered = {k: self._gather_tensor(v) for k, v in out_local.items()}
        for rank in range(self.world_size):
            frame_idx_to_save = frame_idx_begin + rank
            if frame_idx_to_save >= num_frames:
                continue
            frame_buffer = {
                k: (v[rank], handle) for k, (v, handle) in out_gathered.items()
            }
            if self.gather_backbone_out:
                # also add gathered SAM 2 backbone features to frame_buffer
                if self.is_multiplex:
                    frame_buffer["interactive_backbone_fpn_0"] = (
                        inte_fpn0[rank],
                        inte_fpn_handle0,
                    )
                    frame_buffer["interactive_backbone_fpn_1"] = (
                        inte_fpn1[rank],
                        inte_fpn_handle1,
                    )
                    frame_buffer["interactive_backbone_fpn_2"] = (
                        inte_fpn2[rank],
                        inte_fpn_handle2,
                    )
                    frame_buffer["interactive_backbone_pos_enc"] = (
                        inte_vision_pos_enc,
                        None,
                    )
                frame_buffer["sam2_backbone_fpn_0"] = (fpn0[rank], fpn_handle0)
                frame_buffer["sam2_backbone_fpn_1"] = (fpn1[rank], fpn_handle1)
                frame_buffer["sam2_backbone_fpn_2"] = (fpn2[rank], fpn_handle2)
                frame_buffer["sam2_backbone_pos_enc"] = (vision_pos_enc, None)

            multigpu_buffer[frame_idx_to_save] = frame_buffer

    def _gather_tensor(self, x):
        if self.world_size == 1:
            return [x], None

        async_op = self.async_all_gather
        # here `.contiguous()` is required -- otherwise NCCL all_gather
        # sometimes gives wrong results (based on Ronghang's observations)
        x = x.contiguous()  # ensure contiguous memory for NCCL
        output_list = [torch.empty_like(x) for _ in range(self.world_size)]
        handle = torch.distributed.all_gather(output_list, x, async_op=async_op)
        return output_list, handle

    def forward_video_grounding_batched_multigpu(
        self,
        backbone_out,
        find_inputs,
        geometric_prompt: Prompt,
        frame_idx,
        num_frames,
        # `grounding_cache` is a dict to cache FA outputs in a chunk between different calls
        grounding_cache,
        track_in_reverse=False,
        # whether to also return the SAM2 backbone features (in addition to FA results)
        return_sam2_backbone_feats=False,
        # whether to perform NMS and suppress the scores of those detections removed by NMS
        run_nms=False,
        nms_prob_thresh=None,
        nms_iou_thresh=None,
        nms_use_iom=False,
        # tracking bounds to respect max_frame_num_to_track
        max_frame_num_to_track=None,
        propagate_in_video_start_frame_idx=None,
        # feature_cache for buffered backbone computation
        feature_cache=None,
        # batch_size for batched forward_grounding (default: 16)
        batch_size=16,
    ):
        """
        Fully batched forward_grounding that processes chunks of frames together on each GPU.

        Unlike forward_video_grounding_multigpu which processes 1 frame per GPU per chunk,
        this method processes `batch_size` frames at once using the batched forward_grounding
        approach from Sam3MultiplexImageBase.

        For single-GPU (world_size=1), this is equivalent to forward_grounding_batched.
        For multi-GPU, each GPU processes batch_size frames in parallel.

        Args:
            backbone_out: Dictionary containing backbone outputs and image batch.
            find_inputs: List of FindStage objects for all frames.
            geometric_prompt: Prompt object (used as template, individual prompts are
                constructed from find_inputs for batching).
            frame_idx: Current frame index to process.
            num_frames: Total number of frames in the video.
            grounding_cache: Dictionary to cache grounding outputs.
            track_in_reverse: If True, processing in reverse frame order.
            return_sam2_backbone_feats: Whether to also return SAM2 backbone features.
            run_nms: Whether to perform NMS on detection outputs.
            nms_prob_thresh: Probability threshold for NMS.
            nms_iou_thresh: IoU threshold for NMS.
            nms_use_iom: Whether to use IoM for NMS.
            max_frame_num_to_track: Maximum number of frames to track.
            propagate_in_video_start_frame_idx: Start frame index for propagation.
            feature_cache: Optional dictionary for backbone feature caching.
            batch_size: Number of frames to batch together per GPU (default: 16).

        Returns:
            Tuple of (out, backbone_out) where out contains detection results for frame_idx.
        """
        # Calculate valid frame range based on max_frame_num_to_track
        if max_frame_num_to_track is not None:
            if propagate_in_video_start_frame_idx is None:
                propagate_in_video_start_frame_idx = 0
            if track_in_reverse:
                valid_frame_start = (
                    propagate_in_video_start_frame_idx - max_frame_num_to_track + 1
                )
                valid_frame_end = propagate_in_video_start_frame_idx
            else:
                valid_frame_start = propagate_in_video_start_frame_idx
                valid_frame_end = (
                    propagate_in_video_start_frame_idx + max_frame_num_to_track
                )
        else:
            valid_frame_start = 0
            valid_frame_end = num_frames

        # Initialize grounding_buffer if not present
        if "grounding_buffer" not in grounding_cache:
            grounding_cache["grounding_buffer"] = {}

        # Calculate chunk boundaries - use batch_size instead of world_size
        chunk_start = (frame_idx // batch_size) * batch_size
        chunk_end = min(chunk_start + batch_size, valid_frame_end)
        chunk_key = (chunk_start, chunk_end)

        # Process chunk if not already cached
        if chunk_key not in grounding_cache["grounding_buffer"]:
            with torch.profiler.record_function(
                "forward_grounding_batched.process_chunk"
            ):
                chunk_outputs = self._process_grounding_chunk_batched(
                    backbone_out=backbone_out,
                    find_inputs=find_inputs,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    run_nms=run_nms,
                    nms_prob_thresh=nms_prob_thresh,
                    nms_iou_thresh=nms_iou_thresh,
                    nms_use_iom=nms_use_iom,
                    feature_cache=feature_cache,
                    return_sam2_backbone_feats=return_sam2_backbone_feats,
                )
                grounding_cache["grounding_buffer"][chunk_key] = chunk_outputs

            # Auto-cleanup previous chunks
            self._cleanup_previous_chunks_multigpu(
                grounding_cache=grounding_cache,
                current_chunk_key=chunk_key,
                batch_size=batch_size,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )

        # Retrieve the cached output for this frame
        chunk_outputs = grounding_cache["grounding_buffer"][chunk_key]
        local_idx = frame_idx - chunk_start

        # Slice out the output for this specific frame
        out = self._slice_batched_output(
            chunk_outputs, local_idx, return_sam2_backbone_feats
        )

        return out, backbone_out

    def _process_grounding_chunk_batched(
        self,
        backbone_out,
        find_inputs,
        chunk_start: int,
        chunk_end: int,
        run_nms: bool,
        nms_prob_thresh,
        nms_iou_thresh,
        nms_use_iom: bool,
        feature_cache,
        return_sam2_backbone_feats: bool,
    ):
        """
        Process a chunk of frames through the full forward_grounding pipeline in batch.
        """
        chunk_size = chunk_end - chunk_start

        # Build geometric prompts for the chunk
        chunk_geo_prompts = [
            self._get_geo_prompt_from_find_input(find_inputs[i % len(find_inputs)])
            for i in range(chunk_start, chunk_end)
        ]

        # Batch the find_inputs for this chunk
        batched_find_input = self._batch_find_inputs(
            find_inputs, chunk_start, chunk_end
        )

        # Batch the geometric prompts
        batched_geometric_prompt = self._batch_geometric_prompts_from_list(
            chunk_geo_prompts
        )

        # Run forward_grounding on the batched input
        with torch.profiler.record_function("forward_grounding_batched.forward"):
            out = self.forward_grounding(
                backbone_out=backbone_out,
                find_input=batched_find_input,
                find_target=None,
                geometric_prompt=batched_geometric_prompt,
                feature_cache=feature_cache,
            )

        # Apply NMS per frame in the batch
        if run_nms:
            with torch.profiler.record_function("forward_grounding_batched.nms"):
                assert nms_prob_thresh is not None and nms_iou_thresh is not None
                pred_probs = out["pred_logits"].squeeze(-1).sigmoid()
                pred_masks = out["pred_masks"]
                # pred_probs shape: [batch_size, num_queries]
                # pred_masks shape: [batch_size, num_queries, H, W]
                # Use batched NMS to process all frames at once
                keep = nms_masks(
                    pred_probs=pred_probs,
                    pred_masks=pred_masks,
                    prob_threshold=nms_prob_thresh,
                    iou_threshold=nms_iou_thresh,
                    nms_use_iom=nms_use_iom,
                    do_compile=getattr(self, "compile_model", False),
                    running_in_prod=getattr(self, "running_in_prod", False),
                )
                # Set a very low threshold for detections removed by NMS
                # keep shape: [batch_size, num_queries]
                out["pred_logits"][:, :, 0] -= 1e4 * (~keep).float()

        # Extract SAM2 backbone features if requested
        if return_sam2_backbone_feats and "prev_encoder_out" in out:
            backbone_data = out["prev_encoder_out"]["backbone_out"]
            if self.is_multiplex and "interactive" in backbone_data:
                out["_interactive_backbone"] = backbone_data["interactive"]
            if "sam2_backbone_out" in backbone_data:
                out["_sam2_backbone"] = backbone_data["sam2_backbone_out"]

        out["_chunk_size"] = chunk_size
        return out

    def _slice_batched_output(
        self,
        chunk_outputs,
        local_idx: int,
        return_sam2_backbone_feats: bool,
    ):
        """
        Slice a single frame's output from the batched chunk outputs.
        """
        out = {}

        # Keys to slice at batch dimension
        batch_dim_keys = {
            "pred_logits",
            "pred_boxes",
            "pred_boxes_xyxy",
            "pred_masks",
            "pred_logits_o2m",
            "pred_boxes_o2m",
            "pred_boxes_xyxy_o2m",
            "pred_masks_o2m",
            "queries",
            "presence_logit_dec",
        }

        # Keys to skip
        skip_keys = {
            "_chunk_size",
            "_interactive_backbone",
            "_sam2_backbone",
            "prev_encoder_out",
            "encoder_hidden_states",
            "aux_outputs",
        }

        for key, value in chunk_outputs.items():
            if key in skip_keys:
                continue
            if key in batch_dim_keys and isinstance(value, torch.Tensor):
                out[key] = value[local_idx : local_idx + 1]
            elif isinstance(value, torch.Tensor):
                try:
                    out[key] = value[local_idx : local_idx + 1]
                except (IndexError, RuntimeError):
                    out[key] = value

        # Add object IDs
        if "pred_logits" in out:
            out["pred_object_ids"] = self._get_dummy_object_ids(out["pred_logits"])

        # Add SAM2 backbone features if requested
        if return_sam2_backbone_feats:
            if "_sam2_backbone" in chunk_outputs:
                sam2_bb = chunk_outputs["_sam2_backbone"]
                out["sam2_backbone_fpn_0"] = sam2_bb["backbone_fpn"][0].tensors[
                    local_idx : local_idx + 1
                ]
                out["sam2_backbone_fpn_1"] = sam2_bb["backbone_fpn"][1].tensors[
                    local_idx : local_idx + 1
                ]
                out["sam2_backbone_fpn_2"] = sam2_bb["backbone_fpn"][2].tensors[
                    local_idx : local_idx + 1
                ]
                out["sam2_backbone_pos_enc"] = [
                    x[local_idx : local_idx + 1] for x in sam2_bb["vision_pos_enc"]
                ]

            if self.is_multiplex and "_interactive_backbone" in chunk_outputs:
                inte_bb = chunk_outputs["_interactive_backbone"]
                out["interactive_backbone_fpn_0"] = inte_bb["backbone_fpn"][0].tensors[
                    local_idx : local_idx + 1
                ]
                out["interactive_backbone_fpn_1"] = inte_bb["backbone_fpn"][1].tensors[
                    local_idx : local_idx + 1
                ]
                out["interactive_backbone_fpn_2"] = inte_bb["backbone_fpn"][2].tensors[
                    local_idx : local_idx + 1
                ]
                out["interactive_backbone_pos_enc"] = [
                    x[local_idx : local_idx + 1] for x in inte_bb["vision_pos_enc"]
                ]

        return out

    def _cleanup_previous_chunks_multigpu(
        self,
        grounding_cache,
        current_chunk_key,
        batch_size: int,
        num_frames: int,
        track_in_reverse: bool,
    ):
        """Remove previous chunks from cache to save GPU memory."""
        chunk_start, chunk_end = current_chunk_key

        if not track_in_reverse:
            prev_chunk_start = chunk_start - batch_size
            if prev_chunk_start >= 0:
                prev_chunk_end = chunk_start
                prev_chunk_key = (prev_chunk_start, prev_chunk_end)

                # Cleanup grounding_buffer entry
                chunk = grounding_cache["grounding_buffer"].pop(prev_chunk_key, None)
                if chunk is not None:
                    del chunk
        else:
            next_chunk_start = chunk_end
            if next_chunk_start < num_frames:
                next_chunk_end = min(next_chunk_start + batch_size, num_frames)
                next_chunk_key = (next_chunk_start, next_chunk_end)
                grounding_cache["grounding_buffer"].pop(next_chunk_key, None)
