# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

import logging
import math
import os
import sys
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Set

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, Tensor

from sam3 import perflib
from sam3.model.data_misc import BatchedDatapoint
from sam3.model.model_misc import NestedTensor
from sam3.model.sam3_image import Sam3ImageOnVideoMultiGPU
from sam3.model.sam3_image_on_video_multigpu_utils import mask_iou

from sam3.model.video_tracking_with_prompt_utils import (
    fill_holes_in_mask_scores,
    mask_to_box,
)
from sam3.train.masks_ops import rle_encode

logger = logging.getLogger(__name__)

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class Sam2Predictor(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.model = model
        self.per_obj_inference = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Use the sam2 predictor APIs instead. Check VideoTrackingWithPromptDemo class for details."
        )

    def __getattr__(self, name):
        # Expose all attributes of the underlying model
        model = super().__getattr__("model")
        if name == "model":
            return model
        return getattr(model, name)


# class Sam2Predictor(nn.Module):
#     def __init__(
#         self,
#         config_file,
#         checkpoint_file=None,
#         hydra_overrides=None,
#         per_obj_inference=False,
#         fill_hole_area=0,
#         use_fa3=False,
#         use_rope_real=False,
#     ):
#         """
#         Initialize the SAM2 predictor with the given configuration and checkpoint.
#         Args:
#             config_file (str): Path to the configuration file.
#             checkpoint_file (str, optional): Path to the checkpoint file. If None, the model will be initialized without loading weights.
#             hydra_overrides (list, optional): List of Hydra overrides to apply to the configuration.
#             per_obj_inference (bool): If True, the model will perform per-object inference instead of bucketized batching.
#         """

#         super().__init__()
#         #######################################
#         # Load model from config and checkpoint
#         #######################################

#         from hydra import compose, initialize_config_module
#         from hydra.core.global_hydra import GlobalHydra
#         from hydra.utils import instantiate
#         from omnivore.train_utils import (
#             handle_custom_hydra_commands,
#             register_omegaconf_resolvers,
#         )

#         # Ensure proper Hydra initialization
#         if not GlobalHydra().is_initialized():
#             logger.info("Sam2Predictor: GlobalHydra not initialized")
#             GlobalHydra.instance().clear()
#             # register_omegaconf_resolvers()
#             initialize_config_module("sam3.train.configs", version_base="1.2")

#         if hydra_overrides is None:
#             hydra_overrides = []
#         self.per_obj_inference = per_obj_inference
#         inference_model_class = (
#             "sam3.model.video_tracking_with_prompt_demo_per_obj_inference.Sam3VideoTrackingWithPromptDemoPerObjInference"  # Note: This class may not be available in sam3
#             if per_obj_inference
#             else "sam3.model.video_tracking_with_prompt_demo.Sam3VideoTrackingWithPromptDemo"
#         )
#         hydra_overrides = list(hydra_overrides)
#         hydra_overrides.extend(
#             [
#                 "launcher.experiment_log_dir=''",
#                 f"++trainer.model._target_={inference_model_class}",
#                 # Shared backbone cfg
#                 "++trainer.model.image_size=1008",
#                 "++trainer.model.backbone_stride=14",
#                 "++trainer.model.maskmem_backbone.mask_downsampler.interpol_size=[1152,1152]",
#                 "++trainer.model.backbone.forward_in_chunk_for_eval=false",
#                 # always start tracking from the frame where we receive the first annotation
#                 # (clicks or mask) and ignore the `start_frame_idx` passed to `propagate_in_video`
#                 "++trainer.model.always_start_from_first_ann_frame=false",
#                 # apply non-overlapping constraints on the object masks in the
#                 # memory encoder to avoid/alleviate superposing mask predictions
#                 "++trainer.model.non_overlap_masks_for_mem_enc=false",
#                 # Do not apply non-overlapping constraints on the output
#                 "++trainer.model.non_overlap_masks_for_output=false",
#                 # attend to at most 4 temporally closest conditioning frames in the encoder for
#                 # better temporal locality and a better handling to a large number of annotated frames
#                 "++trainer.model.max_cond_frames_in_attn=4",
#                 # turn off all offloading options in the demo (we handle them separately in the demo class)
#                 "++trainer.model.offload_output_to_cpu_for_eval=false",
#                 "++trainer.model.trim_past_non_cond_mem_for_eval=false",
#                 # torch.compile on the image backbone (w/ `dynamic=false` and `fullgraph=true` to capture a full graph)
#                 # "++trainer.model.backbone.compile_mode=max-autotune",
#                 # "++trainer.model.backbone.compile_extra_args.fullgraph=true",
#                 # "++trainer.model.backbone.compile_extra_args.dynamic=false",
#                 "++trainer.model.backbone.visual.trunk.weights_path=null",
#                 # Postprocessing/demo options
#                 # dynamically fall back to multi-mask if the single mask is not stable
#                 "++trainer.model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
#                 "++trainer.model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
#                 "++trainer.model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
#                 # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
#                 "++trainer.model.binarize_mask_from_pts_for_mem_enc=true",
#                 # only attend to object pointers in the past (before the current frame) in the encoder during evaluation
#                 "++trainer.model.only_obj_ptrs_in_the_past_for_eval=true",
#                 # clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks
#                 "++trainer.model.clear_non_cond_mem_around_input=true",
#                 "++trainer.model.transformer.encoder.layer.self_attention.feat_sizes=[72,72]",
#                 "++trainer.model.transformer.encoder.layer.cross_attention.feat_sizes=[72,72]",
#                 # fill small holes in the final masks up to `fill_hole_area` (after resizing them to the original video resolution)
#                 f"++trainer.model.fill_hole_area={fill_hole_area}",
#                 f"++trainer.model.transformer.encoder.layer.self_attention.use_fa3={use_fa3}",
#                 f"++trainer.model.transformer.encoder.layer.cross_attention.use_fa3={use_fa3}",
#                 f"++trainer.model.transformer.encoder.layer.self_attention.use_rope_real={use_rope_real}",
#                 f"++trainer.model.transformer.encoder.layer.cross_attention.use_rope_real={use_rope_real}",
#             ]
#         )

#         cfg = compose(config_name=config_file, overrides=hydra_overrides)
#         handle_custom_hydra_commands(cfg)
#         model = instantiate(cfg.trainer.model, _recursive_=True)
#         self.cfg = cfg
#         del model.backbone  # Remove backbone since it is shared with the sam3 model
#         if checkpoint_file is not None:
#             ckpt = torch.load(checkpoint_file, map_location="cpu")
#             model.load_state_dict(ckpt["model"], strict=False)
#         self.model = model
#         self.per_obj_inference = per_obj_inference
#         self.fill_hole_area = fill_hole_area
#         # use bfloat16 inference for Flash Attention kernel
#         self.bf16_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
#         self.bf16_context.__enter__()  # keep using for the entire model process

#     def __getattr__(self, name):
#         # Expose all attributes of the underlying model
#         model = super().__getattr__("model")
#         if name == "model":
#             return model
#         return getattr(model, name)

#     def forward(self, *args, **kwargs):
#         raise NotImplementedError(
#             "Use the sam2 predictor APIs instead. Check VideoTrackingWithPromptDemo class for details."
#         )

#     def add_output_per_object(self, *args, **kwargs):
#         if self.per_obj_inference:
#             # nothing needs to be done as each object is already stored separately
#             return

#         # for batched inference state, we also need to add per-object
#         # memory slides to support instance interactivity
#         self._add_output_per_object(*args, **kwargs)


class Sam3DenseTrackingMultiGPU(nn.Module):
    def __init__(
        self,
        sam2_model,
        sam3_model,
        ckpt_path=None,
        sam3_ckpt_path=None,
        # prob threshold for detection outputs -- only keep detections above this threshold
        # enters NMS and det-to-track matching
        score_threshold_detection=0.5,
        # IoU threshold for detection NMS
        det_nms_thresh=0.0,
        # IoU threshold for det-to-track matching -- a detection is considered "matched" to a tracklet it
        # overlaps with a tracklet above this threshold -- it is often a loose threshold like 0.1
        assoc_iou_thresh=0.5,
        # IoU threshold for det-to-track matching, which is used to determine whether a masklet is "unmatched"
        # by any detections -- it is often a stricter threshold like 0.5
        trk_assoc_iou_thresh=0.5,
        # prob threshold for a detection to be added as a new object
        new_det_thresh=0.0,
        # hotstart parameters: we hold off the outputs for `hotstart_delay` frames and
        # 1) remove those tracklets unmatched by any detections based on `hotstart_unmatch_thresh`
        # 2) remove those tracklets overlapping with one another based on `hotstart_dup_thresh`
        hotstart_delay=0,
        hotstart_unmatch_thresh=3,
        hotstart_dup_thresh=3,
        # Whether to suppress masks only within hotstart. If False, we can suppress masks even if they start before hotstart period.
        suppress_unmatched_only_within_hotstart=True,
        # Threshold for suppressing overlapping objects based on recent occlusion
        suppress_overlapping_based_on_recent_occlusion_threshold=0.0,
        o2o_matching_masklets_enable=False,  # Enable hungarian matching to match existing masklets
        suppress_det_close_to_boundary=False,
        fill_hole_area=64,
        # The maximum number of objects (masklets) to track across all GPUs (for no limit, set it to -1)
        max_num_objects=128,  # 128 objects (total across all GPUs) should be able to cover nearly all cases
        **kwargs,
    ):
        super().__init__()
        # assert isinstance(sam2_model, Sam2Predictor,)
        self.sam2_predictor = sam2_model
        # assert isinstance(sam3_model, Sam3ImageOnVideoMultiGPU)
        self.sam3_model = sam3_model
        if sam3_ckpt_path:
            ckpt = torch.load(sam3_ckpt_path, map_location="cpu", weights_only=True)
            self.sam3_model.load_state_dict(ckpt["model"], strict=False)
        elif ckpt_path:
            self._load_checkpoint(ckpt_path, strict=False)
        self.score_threshold_detection = score_threshold_detection
        self.det_nms_thresh = det_nms_thresh
        self.assoc_iou_thresh = assoc_iou_thresh
        self.trk_assoc_iou_thresh = trk_assoc_iou_thresh
        self.new_det_thresh = new_det_thresh
        # hotstart parameters
        if hotstart_delay > 0:
            assert hotstart_unmatch_thresh <= hotstart_delay
            assert hotstart_dup_thresh <= hotstart_delay
        self.hotstart_delay = hotstart_delay
        self.hotstart_unmatch_thresh = hotstart_unmatch_thresh
        self.hotstart_dup_thresh = hotstart_dup_thresh
        self.suppress_unmatched_only_within_hotstart = (
            suppress_unmatched_only_within_hotstart
        )
        self.suppress_overlapping_based_on_recent_occlusion_threshold = (
            suppress_overlapping_based_on_recent_occlusion_threshold
        )
        self.suppress_det_close_to_boundary = suppress_det_close_to_boundary
        self.o2o_matching_masklets_enable = o2o_matching_masklets_enable
        self.fill_hole_area = fill_hole_area
        self.eval()
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self._dist_pg_cpu = None  # CPU process group (lazy-initialized on first use)

        # Initialize profiling variables
        self._profiler = None
        self._frame_count = 0
        self._profile_save_dir = os.getenv("PROFILE_SAVE_DIR", "/tmp/profiling")
        self._profiling_enabled = os.getenv("ENABLE_PROFILING", "0").lower() == "1"

        # the maximum object number
        if max_num_objects > 0:
            num_obj_for_compile = math.ceil(max_num_objects / self.world_size)
        else:
            max_num_objects = 10000  # no limit
            num_obj_for_compile = 16
        logger.info(
            f"`setting max_num_objects` to {max_num_objects} -- creating {num_obj_for_compile=} objects for torch.compile cache"
        )
        self.max_num_objects = max_num_objects
        self.num_obj_for_compile = num_obj_for_compile

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def all_gather_cpu(self, tensor_list, tensor):
        if self._dist_pg_cpu is None:
            self._dist_pg_cpu = dist.new_group(backend="gloo")
        dist.broadcast(tensor_list, tensor, group=self._dist_pg_cpu)

    def all_gather_python_obj_cpu(self, object_list, python_obj):
        if self._dist_pg_cpu is None:
            self._dist_pg_cpu = dist.new_group(backend="gloo")
        dist.all_gather_object(object_list, python_obj, group=self._dist_pg_cpu)

    def broadcast_cpu(self, x, src):
        if self._dist_pg_cpu is None:
            self._dist_pg_cpu = dist.new_group(backend="gloo")
        dist.broadcast(x, src=src, group=self._dist_pg_cpu)

    def broadcast_python_obj_cpu(self, python_obj_list, src):
        if self._dist_pg_cpu is None:
            self._dist_pg_cpu = dist.new_group(backend="gloo")
        dist.broadcast_object_list(python_obj_list, src=src, group=self._dist_pg_cpu)

    def _start_profiling(self, frame_idx):
        self._profiling_enabled = os.getenv("ENABLE_PROFILING", "0").lower() == "1"
        self._profile_end_frame = int(os.getenv("PROFILE_END_FRAME", "-1"))
        """Start profiling for _det_track_one_frame if conditions are met."""
        if not self._profiling_enabled:
            return False

        if not self.is_warmup_complete:
            return False

        if self._profiler is not None:
            return True

        # Start profiling
        os.makedirs(self._profile_save_dir, exist_ok=True)
        profile_path = os.path.join(
            self._profile_save_dir, f"det_track_frame_rank_{self.rank}.json"
        )

        self._profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        )
        self._profiler.start()
        self._current_profile_path = profile_path
        print(f"Started profiling frame on {frame_idx} on rank {self.rank}")
        return True

    def _stop_profiling(self):
        """Stop profiling and save trace."""
        if self._profiler is not None:
            self._profiler.stop()
            self._profiler.export_chrome_trace(self._current_profile_path)
            print(f"Profiling trace saved to: {self._current_profile_path}")
            print(
                f"You can open this file in Perfetto (https://ui.perfetto.dev/) to visualize the trace"
            )
            self._profiler = None
            self._profiling_enabled = False
            os.environ["ENABLE_PROFILING"] = "0"

    def _det_track_one_frame(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        input_batch: BatchedDatapoint,
        geometric_prompt: Any,
        sam2_states_local: List[Any],
        sam2_metadata_prev: Dict[str, Any],
        feature_cache: Dict,
        orig_vid_height: int,
        orig_vid_width: int,
    ):
        profiling_enabled = self._start_profiling(frame_idx)

        try:
            return self._det_track_one_frame_impl(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                input_batch=input_batch,
                geometric_prompt=geometric_prompt,
                sam2_states_local=sam2_states_local,
                sam2_metadata_prev=sam2_metadata_prev,
                feature_cache=feature_cache,
                orig_vid_height=orig_vid_height,
                orig_vid_width=orig_vid_width,
            )
        finally:
            if profiling_enabled:
                if sys.exc_info()[0] is not None:
                    # If there is an exception, stop profiling
                    self._stop_profiling()
                else:
                    if (
                        (not reverse and frame_idx == num_frames - 1)
                        or (reverse and frame_idx == 0)
                        or self._profile_end_frame == frame_idx
                    ):
                        # Stop profiling if reached the last frame
                        self._stop_profiling()

    def _det_track_one_frame_impl(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        input_batch: BatchedDatapoint,
        geometric_prompt: Any,
        sam2_states_local: List[Any],
        sam2_metadata_prev: Dict[str, Any],
        feature_cache: Dict,
        orig_vid_height: int,
        orig_vid_width: int,
    ):
        """
        This function handles one-step inference for the DenseTracking model in an SPMD manner.
        At a high-level, all GPUs execute the same function calls as if it's done on a single GPU,
        while under the hood, some function calls involve distributed computation based on sharded
        SAM2 states.

        - `input_batch` contains image and other inputs on the entire video; it should be identical across GPUs
        - `sam2_states_local` holds the local masklet information in this GPU shard
        - `sam2_metadata_prev` manages the metadata for SAM2 objects, such as which masklet is hold on which GPUs
          it contains both global and local masklet information
        """

        # Step 1: run backbone and FA in a distributed manner -- this is done via Sam3ImageOnVideoMultiGPU,
        # a MultiGPU FA model (assigned to `self.sam3_model`) that shards frames in a round-robin manner.
        # It returns a "det_out" dict for `frame_idx` and fills SAM2 backbone features for `frame_idx`
        # into `feature_cache`. Despite its distributed inference under the hood, the results would be
        # the same as if it is running backbone and FA for every frame on a single GPU.
        with torch.profiler.record_function("run_backbone_and_detection"):
            det_out = self.run_backbone_and_detection(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                input_batch=input_batch,
                geometric_prompt=geometric_prompt,
                feature_cache=feature_cache,
            )

        # Step 2: each GPU propagates its local SAM2 states to get the SAM2 prediction masks.
        # the returned `sam2_low_res_masks_global` contains the concatenated masklet predictions
        # gathered from all GPUs (as if they are propagated on a single GPU). Note that this step only
        # runs the SAM2 propagation step, but doesn't encode new memory for the predicted masks;
        # we defer memory encoding to `run_sam2_update_execution_phase` after resolving all heuristics.
        with torch.profiler.record_function("run_sam2_propagation"):
            if sam2_metadata_prev == {}:
                # initialize masklet metadata if it's uninitialized (empty dict)
                sam2_metadata_prev.update(self._initialize_metadata())
            sam2_low_res_masks_global = self.run_sam2_propagation(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                sam2_states_local=sam2_states_local,
                sam2_metadata_prev=sam2_metadata_prev,
            )

        # Step 3: based on detection outputs and the propagated SAM2 prediction masks, we make plans
        # for SAM2 masklet updates (i.e. which objects to add and remove, how to load-balance them, etc).
        # We also run SAM2 memory encoder globally in this step to resolve non-overlapping constraints.
        # **This step should involve all the heuristics needed for any updates.** Most of the update
        # planning will be done on the master rank (GPU 0) and the resulting plan `sam2_update_plan` is
        # broadcasted to other GPUs (to be executed in a distributed manner). This step also generates the
        # new masklet metadata `sam2_metadata_new` (based on its previous version `sam2_metadata_prev`).
        with torch.profiler.record_function("run_sam2_update_planning_phase"):
            sam2_update_plan, sam2_metadata_new = self.run_sam2_update_planning_phase(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                det_out=det_out,
                sam2_low_res_masks_global=sam2_low_res_masks_global,
                sam2_metadata_prev=sam2_metadata_prev,
                sam2_states_local=sam2_states_local,
            )

        # Step 4: based on `sam2_update_plan`, each GPU executes the update w.r.t. its local SAM2 inference states
        with torch.profiler.record_function("run_sam2_update_execution_phase"):
            sam2_states_local_new = self.run_sam2_update_execution_phase(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                det_out=det_out,
                sam2_states_local=sam2_states_local,
                sam2_update_plan=sam2_update_plan,
                orig_vid_height=orig_vid_height,
                orig_vid_width=orig_vid_width,
                feature_cache=feature_cache,
            )

        # Step 5: finally, build the outputs for this frame (it only needs to be done on GPU 0 since
        # only GPU 0 will send outputs to the server).
        with torch.profiler.record_function("build_outputs"):
            if self.rank == 0:
                obj_id_to_mask = self.build_outputs(
                    frame_idx=frame_idx,
                    num_frames=num_frames,
                    reverse=reverse,
                    det_out=det_out,
                    sam2_low_res_masks_global=sam2_low_res_masks_global,
                    sam2_metadata_prev=sam2_metadata_prev,
                    sam2_update_plan=sam2_update_plan,
                    orig_vid_height=orig_vid_height,
                    orig_vid_width=orig_vid_width,
                )
                obj_id_to_score = sam2_metadata_new["obj_id_to_score"]
            else:
                obj_id_to_mask, obj_id_to_score = {}, {}  # dummy outputs on other GPUs
        # a few statistics for the current frame as a part of the output
        frame_stats = {
            "num_obj_tracked": np.sum(sam2_metadata_prev["num_obj_per_gpu"]),
            "num_obj_dropped": sam2_update_plan["num_obj_dropped_due_to_limit"],
        }

        return (
            obj_id_to_mask,  # a dict: obj_id --> output mask
            obj_id_to_score,  # a dict: obj_id --> output score (prob)
            sam2_states_local_new,
            sam2_metadata_new,
            frame_stats,
        )

    def _suppress_detections_close_to_boundary(self, boxes, margin=0.025):
        """
        Suppress detections too close to image edges (for normalized boxes).

        boxes: (N, 4) in xyxy format, normalized [0,1]
        margin: fraction of image
        """
        x_min, y_min, x_max, y_max = boxes.unbind(-1)
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        keep = (
            (x_c > margin)
            & (x_c < 1.0 - margin)
            & (y_c > margin)
            & (y_c < 1.0 - margin)
        )

        return keep

    def run_backbone_and_detection(
        self,
        frame_idx: int,
        num_frames: int,
        input_batch: BatchedDatapoint,
        geometric_prompt: Any,
        feature_cache: Dict,
        reverse: bool,
    ):
        # Step 1: if text feature is not cached in `feature_cache`, compute and cache it
        text_batch_key = tuple(input_batch.find_text_batch)
        if "text" not in feature_cache or text_batch_key not in feature_cache["text"]:
            text_outputs = self.sam3_model.backbone.forward_text(
                input_batch.find_text_batch, device=self.device
            )
            # note: we only cache the text feature of the most recent prompt
            feature_cache["text"] = {text_batch_key: text_outputs}
        else:
            text_outputs = feature_cache["text"][text_batch_key]

        # Step 2: run backbone, FA detection, and post-processing with NMS
        if "multigpu_buffer" not in feature_cache:
            # "multigpu_buffer" is a buffer cache used by `self.sam3_model` and it needs
            # to be passed to `forward_video_grounding_multigpu` for every call
            feature_cache["multigpu_buffer"] = {}
        with torch.profiler.record_function("forward_video_grounding_multigpu"):
            sam3_image_out, _ = self.sam3_model.forward_video_grounding_multigpu(
                backbone_out={
                    "img_batch_all_stages": input_batch.img_batch,
                    **text_outputs,
                },
                find_inputs=input_batch.find_inputs,
                geometric_prompt=geometric_prompt,
                frame_idx=frame_idx,
                num_frames=num_frames,
                multigpu_buffer=feature_cache["multigpu_buffer"],
                track_in_reverse=reverse,
                # also get the SAM2 backbone features
                return_sam2_backbone_feats=True,
                # run NMS as a part of distributed FA computation
                run_nms=self.det_nms_thresh > 0.0,
                nms_prob_thresh=self.score_threshold_detection,
                nms_iou_thresh=self.det_nms_thresh,
            )
        # note: detections in `sam3_image_out` has already gone through NMS
        pred_probs = sam3_image_out["pred_logits"].squeeze(-1).sigmoid()
        pred_boxes_xyxy = sam3_image_out["pred_boxes_xyxy"]
        pred_masks = sam3_image_out["pred_masks"]
        # get the positive detection outputs above threshold
        pos_pred_idx = torch.where(pred_probs > self.score_threshold_detection)
        det_out = {
            "bbox": pred_boxes_xyxy[pos_pred_idx[0], pos_pred_idx[1]],
            "mask": pred_masks[pos_pred_idx[0], pos_pred_idx[1]],
            "scores": pred_probs[pos_pred_idx[0], pos_pred_idx[1]],
        }

        # Step 3: build SAM2 backbone features and store them in `feature_cache`
        sam_mask_decoder = self.sam2_predictor.sam_mask_decoder
        sam2_backbone_fpn = [
            sam_mask_decoder.conv_s0(sam3_image_out["sam2_backbone_fpn_0"]),
            sam_mask_decoder.conv_s1(sam3_image_out["sam2_backbone_fpn_1"]),
            sam3_image_out["sam2_backbone_fpn_2"],  # fpn_2 doesn't need additional conv
        ]
        sam2_backbone_out = {
            "vision_features": sam2_backbone_fpn[-1],  # top-level feature
            "vision_mask": None,
            "vision_pos_enc": sam3_image_out["sam2_backbone_pos_enc"],
            "backbone_fpn": [NestedTensor(x, None) for x in sam2_backbone_fpn],
        }
        feature_cache[frame_idx] = (
            input_batch.img_batch.tensors[frame_idx],
            {"sam2_backbone_out": sam2_backbone_out},
        )
        # remove from `feature_cache` old features to save GPU memory
        feature_cache.pop(frame_idx - 1 if not reverse else frame_idx + 1, None)
        return det_out

    def run_sam2_propagation(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        sam2_states_local: List[Any],
        sam2_metadata_prev: Dict[str, np.ndarray],
    ):
        # Step 1: propagate the local SAM2 states to get the current frame's prediction
        # `low_res_masks_local` of the existing masklets on this GPU
        # - obj_ids_local: List[int] -- list of object IDs
        # - low_res_masks_local: Tensor -- (num_local_obj, H_mask, W_mask)
        with torch.profiler.record_function("propagate_sam2_one_frame_local_gpu"):
            obj_ids_local, low_res_masks_local = (
                self._propogate_sam2_one_frame_local_gpu(
                    sam2_states_local, frame_idx=frame_idx, reverse=reverse
                )
            )

        low_res_masks_local = fill_holes_in_mask_scores(
            low_res_masks_local.unsqueeze(1),
            max_area=self.fill_hole_area,
        )
        low_res_masks_local = low_res_masks_local.squeeze(1)

        assert np.all(
            obj_ids_local == sam2_metadata_prev["obj_ids_per_gpu"][self.rank]
        ), "{} != {}".format(
            obj_ids_local, sam2_metadata_prev["obj_ids_per_gpu"][self.rank]
        )

        # Step 2: all-gather `low_res_masks_local` into `low_res_masks_global`
        # - low_res_masks_global: Tensor -- (num_global_obj, H_mask, W_mask)
        with torch.profiler.record_function("all_gather_low_res_masks_local"):
            _, H_mask, W_mask = low_res_masks_local.shape
            if self.world_size > 1:
                low_res_masks_local = low_res_masks_local.contiguous()
                low_res_masks_peers = [
                    low_res_masks_local.new_empty(num_obj, H_mask, W_mask)
                    for num_obj in sam2_metadata_prev["num_obj_per_gpu"]
                ]
                dist.all_gather(low_res_masks_peers, low_res_masks_local)
                low_res_masks_global = torch.cat(low_res_masks_peers, dim=0)
            else:
                low_res_masks_global = low_res_masks_local
        return low_res_masks_global

    def run_sam2_update_planning_phase(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_out: Dict[str, Tensor],
        sam2_low_res_masks_global: Tensor,
        sam2_metadata_prev: Dict[str, np.ndarray],
        sam2_states_local: List[Any],
    ):
        # initialize new metadata from previous metadata (its values will be updated later)
        sam2_metadata_new = {
            "obj_ids_per_gpu": deepcopy(sam2_metadata_prev["obj_ids_per_gpu"]),
            "obj_ids_all_gpu": None,  # will be filled later
            "num_obj_per_gpu": deepcopy(sam2_metadata_prev["num_obj_per_gpu"]),
            "obj_id_to_score": deepcopy(sam2_metadata_prev["obj_id_to_score"]),
            "obj_id_to_last_occluded": {},  # will be filled later
            "max_obj_id": deepcopy(sam2_metadata_prev["max_obj_id"]),
        }

        # Step 1: make the update plan and resolve heuristics on GPU 0
        det_mask_preds: Tensor = det_out["mask"]  # low-res mask logits
        det_scores_np: np.ndarray = det_out["scores"].float().cpu().numpy()
        det_bbox_xyxy: Tensor = det_out["bbox"]
        if self.rank == 0:
            # a) match FA and SAM2 masks and find new objects
            with torch.profiler.record_function("associate_det_trk"):
                new_det_fa_inds, unmatched_trk_obj_ids, det_to_matched_trk_obj_ids = (
                    self._associate_det_trk(
                        det_masks=det_mask_preds,
                        det_scores_np=det_scores_np,
                        trk_masks=sam2_low_res_masks_global,
                        trk_obj_ids=sam2_metadata_prev["obj_ids_all_gpu"],
                    )
                )
            if self.suppress_det_close_to_boundary:
                # TODO: move to `run_backbone_and_detection`. Note that this runs on higher detection threshold (self.new_det_thresh)
                keep = self._suppress_detections_close_to_boundary(
                    det_bbox_xyxy[new_det_fa_inds]
                )
                new_det_fa_inds = new_det_fa_inds[keep.cpu().numpy()]

            # check whether we've hit the maximum number of objects we can track (and if so, drop some detections)
            prev_obj_num = np.sum(sam2_metadata_prev["num_obj_per_gpu"])
            new_det_num = len(new_det_fa_inds)
            num_obj_dropped_due_to_limit = 0
            if prev_obj_num + new_det_num > self.max_num_objects:
                logger.warning(
                    f"hitting {self.max_num_objects=} with {new_det_num=} and {prev_obj_num=}"
                )
                new_det_num_to_keep = self.max_num_objects - prev_obj_num
                num_obj_dropped_due_to_limit = new_det_num - new_det_num_to_keep
                new_det_fa_inds = self._drop_new_det_with_obj_limit(
                    new_det_fa_inds, det_scores_np, new_det_num_to_keep
                )
                assert len(new_det_fa_inds) == new_det_num_to_keep
                new_det_num = len(new_det_fa_inds)

            # assign object IDs to new detections and decide which GPU to place them
            new_det_start_obj_id = sam2_metadata_prev["max_obj_id"] + 1
            new_det_obj_ids = new_det_start_obj_id + np.arange(new_det_num)
            new_det_gpu_ids = self._assign_new_det_to_gpus(
                new_det_num=new_det_num,
                num_prev_obj_per_gpu=sam2_metadata_prev["num_obj_per_gpu"],
            )

            # b) handle hotstart heuristics to remove objects
            # here `rank0_metadata` contains metadata stored on (and only accessible to) GPU 0;
            # we avoid broadcasting them to other GPUs to save communication cost, assuming
            # that `rank0_metadata` is not needed by other GPUs
            rank0_metadata_new = deepcopy(sam2_metadata_prev["rank0_metadata"])
            if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
                obj_ids_newly_removed, rank0_metadata_new = self._process_hotstart(
                    frame_idx=frame_idx,
                    num_frames=num_frames,
                    reverse=reverse,
                    det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                    new_det_obj_ids=new_det_obj_ids,
                    unmatched_trk_obj_ids=unmatched_trk_obj_ids,
                    rank0_metadata=rank0_metadata_new,
                    sam2_metadata=sam2_metadata_prev,
                )
            else:
                # if warm-up is not complete, we don't remove any objects
                obj_ids_newly_removed = set()
            sam2_metadata_new["rank0_metadata"] = rank0_metadata_new

        # Step 2: broadcast the update plan to other GPUs
        if self.rank == 0 and self.world_size > 1:
            update_plan = [
                new_det_fa_inds,
                new_det_obj_ids,
                new_det_gpu_ids,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                obj_ids_newly_removed,
                num_obj_dropped_due_to_limit,
            ]
            self.broadcast_python_obj_cpu(update_plan, src=0)
        elif self.rank > 0 and self.world_size > 1:
            update_plan = [None] * 7  # other ranks receive the plan from rank 0
            self.broadcast_python_obj_cpu(update_plan, src=0)
            (
                new_det_fa_inds,
                new_det_obj_ids,
                new_det_gpu_ids,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                obj_ids_newly_removed,
                num_obj_dropped_due_to_limit,
            ) = update_plan
        # `sam2_update_plan` should be identical on all GPUs after broadcasting
        sam2_update_plan = {
            "new_det_fa_inds": new_det_fa_inds,  # np.ndarray
            "new_det_obj_ids": new_det_obj_ids,  # np.ndarray
            "new_det_gpu_ids": new_det_gpu_ids,  # np.ndarray
            "unmatched_trk_obj_ids": unmatched_trk_obj_ids,  # np.ndarray
            "det_to_matched_trk_obj_ids": det_to_matched_trk_obj_ids,  # dict
            "obj_ids_newly_removed": obj_ids_newly_removed,  # set
            "num_obj_dropped_due_to_limit": num_obj_dropped_due_to_limit,  # int
        }

        # Step 3: Run SAM2 memory encoder on the current frame's prediction masks
        # This is done on all GPUs
        batch_size = sam2_low_res_masks_global.size(0)
        if batch_size > 0:
            if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
                if self.suppress_overlapping_based_on_recent_occlusion_threshold > 0.0:
                    # NOTE: sam2_low_res_masks_global is updated in-place then returned
                    sam2_low_res_masks_global = (
                        self._suppress_overlapping_based_on_recent_occlusion(
                            frame_idx,
                            sam2_low_res_masks_global,
                            sam2_metadata_prev,
                            sam2_metadata_new,
                            obj_ids_newly_removed,
                            reverse,
                        )
                    )

            self._sam2_update_memories(
                sam2_states_local,
                frame_idx,
                sam2_metadata=sam2_metadata_prev,
                low_res_masks=sam2_low_res_masks_global,
            )

        # Step 4: update the SAM2 metadata based on the update plan
        # note: except for "rank0_metadata" (that is only available on GPU 0),
        # the updated `sam2_metadata_new` should be identical on all GPUs
        for rank in range(self.world_size):
            new_det_obj_ids_this_gpu = new_det_obj_ids[new_det_gpu_ids == rank]
            updated_obj_ids_this_gpu = sam2_metadata_new["obj_ids_per_gpu"][rank]
            if len(new_det_obj_ids_this_gpu) > 0:
                updated_obj_ids_this_gpu = np.concatenate(
                    [updated_obj_ids_this_gpu, new_det_obj_ids_this_gpu]
                )
            if len(obj_ids_newly_removed) > 0:
                is_removed = np.isin(
                    updated_obj_ids_this_gpu, list(obj_ids_newly_removed)
                )
                updated_obj_ids_this_gpu = updated_obj_ids_this_gpu[~is_removed]
            sam2_metadata_new["obj_ids_per_gpu"][rank] = updated_obj_ids_this_gpu
            sam2_metadata_new["num_obj_per_gpu"][rank] = len(updated_obj_ids_this_gpu)
        sam2_metadata_new["obj_ids_all_gpu"] = np.concatenate(
            sam2_metadata_new["obj_ids_per_gpu"]
        )
        # update object scores and the maximum object ID assigned so far
        if len(new_det_obj_ids) > 0:
            sam2_metadata_new["obj_id_to_score"].update(
                zip(new_det_obj_ids, det_scores_np[new_det_fa_inds])
            )
            sam2_metadata_new["max_obj_id"] = max(
                sam2_metadata_new["max_obj_id"],
                np.max(new_det_obj_ids),
            )
        # for removed objects, we set their scores to a very low value (-1e4) but still
        # keep them in "obj_id_to_score" (it's easier to handle outputs this way)
        for obj_id in obj_ids_newly_removed:
            sam2_metadata_new["obj_id_to_score"][obj_id] = -1e4
            sam2_metadata_new["obj_id_to_last_occluded"].pop(obj_id, None)
        # check that "rank0_metadata" is in sam2_metadata_new if and only if it's GPU 0
        assert ("rank0_metadata" in sam2_metadata_new) == (self.rank == 0)

        return sam2_update_plan, sam2_metadata_new

    def _suppress_overlapping_based_on_recent_occlusion(
        self,
        frame_idx: int,
        sam2_low_res_masks_global: Tensor,
        sam2_metadata_prev: Dict[str, Any],
        sam2_metadata_new: Dict[str, Any],
        obj_ids_newly_removed: Set[int],
        reverse: bool = False,
    ):
        """
        Suppress overlapping masks based on the most recent occlusion information. If an object is removed by hotstart, we always suppress it if it overlaps with any other object.
        Args:
            frame_idx (int): The current frame index.
            sam2_low_res_masks_global (Tensor): The low-resolution masks for the current frame.
            sam2_metadata_prev (Dict[str, Any]): The metadata from the previous frame.
            sam2_metadata_new (Dict[str, Any]): The metadata for the current frame.
            obj_ids_newly_removed (Set[int]): The object IDs that have been removed.
        Return:
            Tensor: The updated low-resolution masks with some objects suppressed.
        """
        obj_ids_global = sam2_metadata_prev["obj_ids_all_gpu"]
        binary_sam2_low_res_masks_global = sam2_low_res_masks_global > 0
        batch_size = sam2_low_res_masks_global.size(0)
        if batch_size > 0:
            assert (
                len(obj_ids_global) == batch_size
            ), f"Mismatch in number of objects: {len(obj_ids_global)} vs {batch_size}"
            NEVER_OCCLUDED = -1
            ALWAYS_OCCLUDED = 100000  # This value should be larger than any possible frame index, indicates that the object was removed by hotstart logic
            last_occluded_prev = torch.cat(
                [
                    sam2_metadata_prev["obj_id_to_last_occluded"].get(
                        obj_id,
                        torch.full(
                            (1,),
                            fill_value=(
                                NEVER_OCCLUDED
                                if obj_id not in obj_ids_newly_removed
                                else ALWAYS_OCCLUDED
                            ),
                            device=binary_sam2_low_res_masks_global.device,
                            dtype=torch.long,
                        ),
                    )
                    for obj_id in obj_ids_global
                ],
                dim=0,
            )
            to_suppress = self._get_objects_to_suppress_based_on_most_recently_occluded(
                binary_sam2_low_res_masks_global,
                last_occluded_prev,
                obj_ids_global,
                frame_idx,
                reverse,
            )

            # Update metadata with occlusion information
            is_obj_occluded = ~(binary_sam2_low_res_masks_global.any(dim=(-1, -2)))
            is_obj_occluded_or_suppressed = is_obj_occluded | to_suppress
            last_occluded_new = last_occluded_prev.clone()
            last_occluded_new[is_obj_occluded_or_suppressed] = frame_idx
            # Slice out the last occluded frame for each object
            sam2_metadata_new["obj_id_to_last_occluded"] = {
                obj_id: last_occluded_new[obj_idx : obj_idx + 1]
                for obj_idx, obj_id in enumerate(obj_ids_global)
            }

            # Zero out suppressed masks before memory encoding
            NO_OBJ_LOGIT = -10
            sam2_low_res_masks_global[to_suppress] = NO_OBJ_LOGIT

        return sam2_low_res_masks_global

    def run_sam2_update_execution_phase(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_out: Dict[str, Tensor],
        sam2_states_local: List[Any],
        sam2_update_plan: Dict[str, np.ndarray],
        orig_vid_height: int,
        orig_vid_width: int,
        feature_cache: Dict,
    ):
        # initialize tracking scores with detection scores
        new_det_fa_inds: np.ndarray = sam2_update_plan["new_det_fa_inds"]
        new_det_obj_ids: np.ndarray = sam2_update_plan["new_det_obj_ids"]
        new_det_gpu_ids: np.ndarray = sam2_update_plan["new_det_gpu_ids"]
        is_on_this_gpu: np.ndarray = new_det_gpu_ids == self.rank
        new_det_obj_ids_local: np.ndarray = new_det_obj_ids[is_on_this_gpu]
        new_det_fa_inds_local: np.ndarray = new_det_fa_inds[is_on_this_gpu]
        obj_ids_newly_removed: Set[int] = sam2_update_plan["obj_ids_newly_removed"]

        # Step 1: add new objects from FA detection to SAM2 inference states
        if len(new_det_fa_inds_local) > 0:
            new_det_fa_inds_local_t = torch.from_numpy(new_det_fa_inds_local)
            new_det_masks: Tensor = det_out["mask"][new_det_fa_inds_local_t]
            # initialize SAM2 with new object masks
            sam2_states_local = self._sam2_add_new_objects(
                frame_idx=frame_idx,
                num_frames=num_frames,
                new_obj_ids=new_det_obj_ids_local,
                new_obj_masks=new_det_masks,
                sam2_states_local=sam2_states_local,
                orig_vid_height=orig_vid_height,
                orig_vid_width=orig_vid_width,
                feature_cache=feature_cache,
            )

        # Step 2: remove from SAM2 inference states those objects removed by heuristics
        if len(obj_ids_newly_removed) > 0:
            for obj_id in obj_ids_newly_removed:
                self._sam2_remove_object(sam2_states_local, obj_id)

        return sam2_states_local

    def build_outputs(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_out: Dict[str, Tensor],
        sam2_low_res_masks_global: Tensor,
        sam2_metadata_prev: Dict[str, np.ndarray],
        sam2_update_plan: Dict[str, np.ndarray],
        orig_vid_height: int,
        orig_vid_width: int,
    ):
        new_det_fa_inds: np.ndarray = sam2_update_plan["new_det_fa_inds"]
        new_det_obj_ids: np.ndarray = sam2_update_plan["new_det_obj_ids"]
        obj_id_to_mask = {}  # obj_id --> output mask tensor

        # Part 1: masks from previous SAM2 propagation
        existing_masklet_obj_ids = sam2_metadata_prev["obj_ids_all_gpu"]
        existing_masklet_video_res_masks = F.interpolate(
            sam2_low_res_masks_global.unsqueeze(1),
            size=(orig_vid_height, orig_vid_width),
            mode="bilinear",
            align_corners=False,
        )  # (num_obj, 1, H_video, W_video)
        # apply non-overlapping constraints on the existing masklets
        if existing_masklet_video_res_masks.shape[0] > 0:
            existing_masklet_video_res_masks = (
                self.sam2_predictor._apply_non_overlapping_constraints(
                    existing_masklet_video_res_masks
                )
            )
        existing_masklet_binary = existing_masklet_video_res_masks > 0
        assert len(existing_masklet_obj_ids) == len(existing_masklet_binary)
        for obj_id, mask in zip(existing_masklet_obj_ids, existing_masklet_binary):
            obj_id_to_mask[obj_id] = mask  # (1, H_video, W_video)

        # Part 2: masks from new detections
        new_det_fa_inds_t = torch.from_numpy(new_det_fa_inds)
        new_det_low_res_masks = det_out["mask"][new_det_fa_inds_t].unsqueeze(1)
        new_det_low_res_masks = fill_holes_in_mask_scores(
            new_det_low_res_masks, max_area=self.fill_hole_area
        )
        new_masklet_video_res_masks = F.interpolate(
            new_det_low_res_masks,
            size=(orig_vid_height, orig_vid_width),
            mode="bilinear",
            align_corners=False,
        )  # (num_obj, 1, H_video, W_video)

        new_masklet_binary = new_masklet_video_res_masks > 0
        assert len(new_det_obj_ids) == len(new_masklet_video_res_masks)
        for obj_id, mask in zip(new_det_obj_ids, new_masklet_binary):
            obj_id_to_mask[obj_id] = mask  # (1, H_video, W_video)

        return obj_id_to_mask

    def _get_objects_to_suppress_based_on_most_recently_occluded(
        self,
        binary_low_res_masks: Tensor,
        last_occluded: List[int],
        obj_ids: List[int],
        frame_idx: int = None,
        reverse: bool = False,
    ):
        # Suppress overlapping masks for objects that were most recently occluded
        assert (
            binary_low_res_masks.dtype == torch.bool
        ), f"Expected boolean tensor, got {binary_low_res_masks.dtype}"
        to_suppress = torch.zeros(
            binary_low_res_masks.size(0),
            device=binary_low_res_masks.device,
            dtype=torch.bool,
        )
        if len(obj_ids) <= 1:
            return to_suppress

        iou = mask_iou(binary_low_res_masks, binary_low_res_masks)  # [N,N]

        # Create masks for upper triangular matrix (i < j) and IoU threshold
        mask_iou_thresh = (
            iou >= self.suppress_overlapping_based_on_recent_occlusion_threshold
        )
        overlapping_pairs = torch.triu(mask_iou_thresh, diagonal=1)  # [N,N]

        last_occ_expanded_i = last_occluded.unsqueeze(1)  # (N, 1)
        last_occ_expanded_j = last_occluded.unsqueeze(0)  # (1, N)
        # Suppress most recently occluded
        cmp_op = torch.gt if not reverse else torch.lt
        suppress_i_mask = (
            overlapping_pairs
            & cmp_op(
                last_occ_expanded_i, last_occ_expanded_j
            )  # (last_occ_expanded_i > last_occ_expanded_j)
            & (
                last_occ_expanded_j > -1
            )  # j can suppress i only if i was previously occluded
        )
        suppress_j_mask = (
            overlapping_pairs
            & cmp_op(last_occ_expanded_j, last_occ_expanded_i)
            & (
                last_occ_expanded_i > -1
            )  # i can suppress j only if j was previously occluded
        )
        # Apply suppression
        to_suppress = suppress_i_mask.any(dim=1) | suppress_j_mask.any(dim=0)

        # Log for debugging
        if (
            self.rank == 0
            and logger.isEnabledFor(logging.DEBUG)
            and frame_idx is not None
        ):
            suppress_i_mask = suppress_i_mask.cpu().numpy()
            suppress_j_mask = suppress_j_mask.cpu().numpy()
            last_occluded = last_occluded.cpu().numpy()

            # Find all suppression pairs without using torch.where
            batch_size = suppress_i_mask.shape[0]

            # Log i-suppression cases (where i gets suppressed in favor of j)
            for i in range(batch_size):
                for j in range(batch_size):
                    if suppress_i_mask[i, j]:
                        logger.debug(
                            f"{frame_idx=}: Suppressing obj {obj_ids[i]} last occluded {last_occluded[i]} in favor of {obj_ids[j]} last occluded {last_occluded[j]}"
                        )

            # Log j-suppression cases (where j gets suppressed in favor of i)
            for i in range(batch_size):
                for j in range(batch_size):
                    if suppress_j_mask[i, j]:
                        logger.debug(
                            f"{frame_idx=}: Suppressing obj {obj_ids[j]} last occluded {last_occluded[j]} in favor of {obj_ids[i]} last occluded {last_occluded[i]}"
                        )

        return to_suppress

    def _propogate_sam2_one_frame_local_gpu(
        self,
        inference_states: List[Any],
        frame_idx: int,
        reverse: bool,
        # by default, we disable memory encoding until we gather all outputs
        run_mem_encoder: bool = False,
    ):
        """
        inference_states: List of inference states, each state corresponds to a different set of objects.
        """
        obj_ids_local = []
        low_res_masks_list = []
        for inference_state in inference_states:
            if len(inference_state["obj_ids"]) == 0:
                continue  # skip propagation on empty inference states

            # propagate one frame
            num_frames_propagated = 0
            with torch.profiler.record_function("sam2_predictor.propagate_in_video"):
                for out in self.sam2_predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=frame_idx,
                    # end_frame_idx = start_frame_idx + max_frame_num_to_track
                    # (i.e. propagating 1 frame since end_frame_idx is inclusive)
                    max_frame_num_to_track=0,
                    reverse=reverse,
                    tqdm_disable=True,
                    run_mem_encoder=run_mem_encoder,
                ):
                    # TODO we only need low-res outputs here for all-gather across GPUs,
                    # so we can remove the high-res interpolation in `propagate_in_video`
                    out_frame_idx, out_obj_ids, out_low_res_masks, _ = out
                    num_frames_propagated += 1

            # only 1 frames should be propagated
            assert (
                num_frames_propagated == 1 and out_frame_idx == frame_idx
            ), f"num_frames_propagated: {num_frames_propagated}, out_frame_idx: {out_frame_idx}, frame_idx: {frame_idx}"
            assert isinstance(out_obj_ids, list)
            obj_ids_local.extend(out_obj_ids)
            low_res_masks_list.append(out_low_res_masks.squeeze(1))

        # concatenate the output masklets from all local inference states
        H_mask = W_mask = self.sam2_predictor.low_res_mask_size
        if len(low_res_masks_list) > 0:
            low_res_masks_local = torch.cat(low_res_masks_list, dim=0)
            assert low_res_masks_local.shape[1:] == (H_mask, W_mask)
        else:
            low_res_masks_local = torch.zeros(0, H_mask, W_mask, device=self.device)

        return obj_ids_local, low_res_masks_local

    def _associate_det_trk(
        self,
        det_masks: Tensor,
        det_scores_np: np.ndarray,
        trk_masks: Tensor,
        trk_obj_ids: np.ndarray,
    ):
        """
        Match detections on the current frame with the existing masklets.

        Args:
          - det_masks: (N, H, W) tensor of predicted masks
          - det_scores_np: (N,) array of detection scores
          - trk_masks: (M, H, W) tensor of track masks
          - trk_obj_ids: (M,) array of object IDs corresponding to trk_masks

        Returns:
          - new_det_fa_inds: array of new object indices among in FA detection outputs
          - unmatched_trk_obj_ids: array of existing masklet object IDs that are not matched
            to any detections on this frame (for unmatched, we only count masklets with >0 area)
          - det_to_matched_trk_obj_ids: dict[int, np.ndarray]: mapping from FA detection indices
            to the list of matched tracklet object IDs
        """
        iou_threshold = self.assoc_iou_thresh
        iou_threshold_trk = self.trk_assoc_iou_thresh
        new_det_thresh = self.new_det_thresh

        assert det_masks.is_floating_point(), "float tensor expected (do not binarize)"
        assert trk_masks.is_floating_point(), "float tensor expected (do not binarize)"
        assert trk_masks.size(0) == len(
            trk_obj_ids
        ), f"trk_masks and trk_obj_ids should have the same length, {trk_masks.size(0)} vs {len(trk_obj_ids)}"
        if trk_masks.size(0) == 0:
            # all detections are new
            new_det_fa_inds = np.arange(det_masks.size(0))
            unmatched_trk_obj_ids = np.array([], np.int64)
            det_to_matched_trk_obj_ids = {}
            return new_det_fa_inds, unmatched_trk_obj_ids, det_to_matched_trk_obj_ids
        elif det_masks.size(0) == 0:
            # all previous tracklets are unmatched if they have a non-zero area
            new_det_fa_inds = np.array([], np.int64)
            trk_is_nonempty = (trk_masks > 0).any(dim=(1, 2)).cpu().numpy()
            unmatched_trk_obj_ids = trk_obj_ids[trk_is_nonempty]
            det_to_matched_trk_obj_ids = {}
            return new_det_fa_inds, unmatched_trk_obj_ids, det_to_matched_trk_obj_ids

        if det_masks.shape[-2:] != trk_masks.shape[-2:]:
            # resize to the smaller size to save GPU memory
            if np.prod(det_masks.shape[-2:]) < np.prod(trk_masks.shape[-2:]):
                trk_masks = F.interpolate(
                    trk_masks.unsqueeze(1),
                    size=det_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            else:
                # resize detections to track size
                det_masks = F.interpolate(
                    det_masks.unsqueeze(1),
                    size=trk_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

        det_masks_binary = det_masks > 0
        trk_masks_binary = trk_masks > 0
        ious = mask_iou(det_masks_binary, trk_masks_binary)  # (N, M)

        # Hungarian matching for tracks (one-to-one: each track matches at most one detection)
        # For detections: allow many tracks to match to the same detection (many-to-one)
        # TODO now that we're already doing one-to-many matching, we can remove Hungarian matching
        # (calling "linear_sum_assignment" creates an unnecessary GPU-to-CPU transfer)

        # TODO: remove the GPU->CPU copy if hungarian matching disabled
        ious_np = ious.cpu().numpy()
        if self.o2o_matching_masklets_enable:
            from scipy.optimize import linear_sum_assignment

            cost_matrix = 1 - ious_np  # Hungarian solves for minimum cost
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            trk_is_matched = np.zeros(trk_masks.size(0), dtype=bool)
            for d, t in zip(row_ind, col_ind):
                if ious_np[d, t] >= iou_threshold_trk:
                    trk_is_matched[t] = True
        else:
            trk_is_matched = (ious_np >= iou_threshold_trk).any(axis=0)
        # Non-empty tracks not matched by Hungarian assignment above threshold are unmatched
        trk_is_nonempty = trk_masks_binary.any(dim=(1, 2)).cpu().numpy()
        trk_is_unmatched = np.logical_and(trk_is_nonempty, ~trk_is_matched)
        unmatched_trk_obj_ids = trk_obj_ids[trk_is_unmatched]

        # For detections: allow many tracks to match to the same detection (many-to-one)
        # So, a detection is 'new' if it does not match any track above threshold
        is_new_det = np.logical_and(
            det_scores_np >= new_det_thresh,
            np.logical_not(np.any(ious_np >= iou_threshold, axis=1)),
        )
        new_det_fa_inds = np.nonzero(is_new_det)[0]

        # for each detection, which tracks it matched to (above threshold)
        det_to_matched_trk_obj_ids = {}
        for d in range(det_masks.size(0)):
            det_to_matched_trk_obj_ids[d] = trk_obj_ids[ious_np[d, :] >= iou_threshold]

        return new_det_fa_inds, unmatched_trk_obj_ids, det_to_matched_trk_obj_ids

    def _assign_new_det_to_gpus(self, new_det_num, num_prev_obj_per_gpu):
        """Distribute the new objects to the GPUs with the least workload."""
        workload_per_gpu: np.ndarray = num_prev_obj_per_gpu.copy()
        new_det_gpu_ids = np.zeros(new_det_num, np.int64)
        for i in range(len(new_det_gpu_ids)):
            # find the GPU with the least workload
            min_gpu = np.argmin(workload_per_gpu)
            new_det_gpu_ids[i] = min_gpu
            workload_per_gpu[min_gpu] += 1
        return new_det_gpu_ids

    def _process_hotstart(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_to_matched_trk_obj_ids: Dict[int, np.ndarray],
        new_det_obj_ids: np.ndarray,
        unmatched_trk_obj_ids: np.ndarray,
        rank0_metadata: Dict[str, Any],
        sam2_metadata: Dict[str, Any],
    ):
        """Handle hotstart heuristics to remove unmatched or duplicated objects."""
        # obj_id --> first frame index where the object was detected
        obj_first_frame_idx = rank0_metadata["obj_first_frame_idx"]
        # obj_id --> [mismatched frame indices]
        unmatched_frame_inds = rank0_metadata["unmatched_frame_inds"]
        trk_keep_alive = rank0_metadata["trk_keep_alive"]
        # (first_appear_obj_id, obj_id) --> [overlap frame indices]
        overlap_pair_to_frame_inds = rank0_metadata["overlap_pair_to_frame_inds"]
        # removed_obj_ids: object IDs that are suppressed via hot-start
        removed_obj_ids = rank0_metadata["removed_obj_ids"]
        suppressed_obj_ids = rank0_metadata["suppressed_obj_ids"][frame_idx]

        obj_ids_newly_removed = set()  # object IDs to be newly removed on this frame
        hotstart_diff = (
            frame_idx - self.hotstart_delay
            if not reverse
            else frame_idx + self.hotstart_delay
        )

        # Step 1: log the frame index where each object ID first appears
        for obj_id in new_det_obj_ids:
            if obj_id not in obj_first_frame_idx:
                obj_first_frame_idx[obj_id] = frame_idx
            assert obj_id not in trk_keep_alive
            trk_keep_alive[obj_id] = 0

        matched_trks = set()
        # We use the det-->tracks list to check for matched objects. Otherwise, we need to compute areas to decide whether they're occluded
        for matched_trks_per_det in det_to_matched_trk_obj_ids.values():
            matched_trks.update(matched_trks_per_det)
        for obj_id in matched_trks:
            # NOTE: To minimize number of configurable params, we use the hotstart_unmatch_thresh to set the max value of trk_keep_alive
            trk_keep_alive[obj_id] = min(
                self.hotstart_unmatch_thresh, trk_keep_alive[obj_id] + 1
            )
        for obj_id in unmatched_trk_obj_ids:
            unmatched_frame_inds[obj_id].append(frame_idx)
            # NOTE: To minimize number of configurable params, we use the hotstart_unmatch_thresh to set the min value of trk_keep_alive
            # The max keep alive is 2x the min, means the model prefers to keep the prediction rather than suppress it if it was matched long enough.
            trk_keep_alive[obj_id] = max(
                -self.hotstart_unmatch_thresh // 2, trk_keep_alive[obj_id] - 1
            )
        # Step 2: removed tracks that has not matched with detections for `hotstart_unmatch_thresh` frames with hotstart period
        # a) add unmatched frame indices for each existing object ID
        # note that `unmatched_trk_obj_ids` contains those frames where the SAM2 output mask
        # doesn't match any FA detection; it excludes those frames where SAM2 gives an empty mask
        # b) remove a masklet if it first appears after `hotstart_diff` and is unmatched for more
        # than `self.hotstart_unmatch_thresh` frames
        for obj_id, frame_indices in unmatched_frame_inds.items():
            if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue  # skip if the object is already removed
            if len(frame_indices) >= self.hotstart_unmatch_thresh:
                is_within_hotstart = (
                    obj_first_frame_idx[obj_id] > hotstart_diff and not reverse
                ) or (obj_first_frame_idx[obj_id] < hotstart_diff and reverse)
                if is_within_hotstart:
                    obj_ids_newly_removed.add(obj_id)
                    logger.info(
                        f"Removing object {obj_id} at frame {frame_idx} "
                        f"since it is unmatched for frames: {frame_indices}"
                    )
            if (
                trk_keep_alive[obj_id] <= 0  # Object has not been matched for too long
                and not self.suppress_unmatched_only_within_hotstart
                and obj_id not in removed_obj_ids
                and obj_id not in obj_ids_newly_removed
            ):
                logger.debug(
                    f"Suppressing object {obj_id} at frame {frame_idx}, due to being unmatched"
                )
                suppressed_obj_ids.add(obj_id)

        # Step 3: removed tracks that overlaps with another track for `hotstart_dup_thresh` frames
        # a) find overlaps tracks -- we consider overlap if they match to the same detection
        for _, matched_trk_obj_ids in det_to_matched_trk_obj_ids.items():
            if len(matched_trk_obj_ids) < 2:
                continue  # only count detections that are matched to multiple (>=2) masklets
            # if there are multiple matched track ids, we need to find the one that appeared first;
            # these later appearing ids may be removed since they may be considered as duplicates
            first_appear_obj_id = (
                min(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
                if not reverse
                else max(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
            )
            for obj_id in matched_trk_obj_ids:
                if obj_id != first_appear_obj_id:
                    key = (first_appear_obj_id, obj_id)
                    overlap_pair_to_frame_inds[key].append(frame_idx)

        # b) remove a masklet if it first appears after `hotstart_diff` and it overlaps with another
        # masklet (that appears earlier) for more than `self.hotstart_dup_thresh` frames
        for (first_obj_id, obj_id), frame_indices in overlap_pair_to_frame_inds.items():
            if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue  # skip if the object is already removed
            if (obj_first_frame_idx[obj_id] > hotstart_diff and not reverse) or (
                obj_first_frame_idx[obj_id] < hotstart_diff and reverse
            ):
                if len(frame_indices) >= self.hotstart_dup_thresh:
                    obj_ids_newly_removed.add(obj_id)
                    logger.info(
                        f"Removing object {obj_id} at frame {frame_idx} "
                        f"since it overlaps with another track {first_obj_id} at frames: {frame_indices}"
                    )

        removed_obj_ids.update(obj_ids_newly_removed)
        return obj_ids_newly_removed, rank0_metadata

    def _sam2_update_memories(
        self,
        sam2_inference_states: List[Any],
        frame_idx: int,
        sam2_metadata: Dict[str, Any],
        low_res_masks: Tensor,
    ):
        """
        Run Sam2 memory encoder, enforcing non-overlapping constraints globally.
        """
        # TODO: Add most recently occluded heuristic for suppression of overlapping masks
        if len(sam2_inference_states) == 0:
            return
        # Avoid an extra interpolation step by directly interpolating to `interpol_size`
        high_res_H, high_res_W = (
            self.sam2_predictor.maskmem_backbone.mask_downsampler.interpol_size
        )
        # NOTE: inspect this part if we observe OOMs in the demo
        high_res_masks = F.interpolate(
            low_res_masks.unsqueeze(1),
            size=(high_res_H, high_res_W),
            mode="bilinear",
            align_corners=False,
        )
        # We first apply non-overlapping constraints before memory encoding. This may include some suppression heuristics.
        if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
            high_res_masks = self.sam2_predictor._apply_non_overlapping_constraints(
                high_res_masks
            )
        # Instead of gathering the predicted object scores, we use mask areas as a proxy.
        object_score_logits = torch.where(
            (high_res_masks > 0).any(dim=(-1, -2)), 10.0, -10.0
        )

        # Run the memory encoder on local slices for each GPU
        start_idx_gpu = sum(sam2_metadata["num_obj_per_gpu"][: self.rank])
        start_idx_state = start_idx_gpu
        for sam2_state in sam2_inference_states:
            num_obj_per_state = len(sam2_state["obj_ids"])
            if num_obj_per_state == 0:
                continue
            # Get the local high-res masks and object score logits for this inference state
            end_idx_state = start_idx_state + num_obj_per_state
            local_high_res_masks = high_res_masks[start_idx_state:end_idx_state]
            local_object_score_logits = object_score_logits[
                start_idx_state:end_idx_state
            ]
            local_batch_size = local_high_res_masks.size(0)
            # Run Sam2 memory encoder. Note that we do not re-enforce the non-overlapping constraint as it is turned off by default
            local_maskmem_features, local_maskmem_pos_enc = (
                self.sam2_predictor._run_memory_encoder(
                    sam2_state,
                    frame_idx,
                    local_batch_size,
                    local_high_res_masks,
                    local_object_score_logits,
                    is_mask_from_pts=False,
                )
            )
            # Store encoded memories in the local inference state
            output_dict = sam2_state["output_dict"]
            for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                if frame_idx not in output_dict[storage_key]:
                    continue
                output_dict[storage_key][frame_idx][
                    "maskmem_features"
                ] = local_maskmem_features
                output_dict[storage_key][frame_idx]["maskmem_pos_enc"] = [
                    pos for pos in local_maskmem_pos_enc
                ]
                # for batched inference state, we also need to add per-object
                # memory slides to support instance interactivity
                self.sam2_predictor.add_output_per_object(
                    inference_state=sam2_state,
                    frame_idx=frame_idx,
                    current_out=output_dict[storage_key][frame_idx],
                    storage_key=storage_key,
                )
            start_idx_state += num_obj_per_state

    def _sam2_add_new_objects(
        self,
        frame_idx: int,
        num_frames: int,
        new_obj_ids: List[int],
        new_obj_masks: Tensor,
        sam2_states_local: List[Any],
        orig_vid_height: int,
        orig_vid_width: int,
        feature_cache: Dict,
    ):
        """Add a new object to SAM2 inference states."""
        prev_sam2_state = sam2_states_local[0] if len(sam2_states_local) > 0 else None

        # prepare inference_state
        if self.sam2_predictor.per_obj_inference:
            # in per_obj_inference mode, init_state happens only once,
            # new obj_ids will be added to the existing inference state
            if prev_sam2_state is not None:
                new_sam2_state = prev_sam2_state
            else:
                new_sam2_state = self.sam2_predictor.init_state(
                    cached_features=feature_cache,
                    video_height=orig_vid_height,
                    video_width=orig_vid_width,
                    num_frames=num_frames,
                )
                new_sam2_state["backbone_out"] = None
        else:
            # batch objects that first appear on the same frame together
            # Clear inference state. Keep the cached image features if available.
            new_sam2_state = self.sam2_predictor.init_state(
                cached_features=feature_cache,
                video_height=orig_vid_height,
                video_width=orig_vid_width,
                num_frames=num_frames,
            )
            new_sam2_state["backbone_out"] = (
                prev_sam2_state.get("backbone_out", None)
                if prev_sam2_state is not None
                else None
            )

        assert len(new_obj_ids) == new_obj_masks.size(0)
        assert new_obj_masks.is_floating_point()
        # TODO consider removing this interpolation -- it's probably no longer needed
        # we should edit `self.sam2_predictor.add_new_mask` to directly take low-res input masks
        input_mask_res = self.sam2_predictor.input_mask_size
        new_obj_masks = F.interpolate(
            new_obj_masks.unsqueeze(1),
            size=(input_mask_res, input_mask_res),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        new_obj_masks = new_obj_masks > 0
        for new_obj_id, new_mask in zip(new_obj_ids, new_obj_masks):
            self.sam2_predictor.add_new_mask(
                inference_state=new_sam2_state,
                frame_idx=frame_idx,
                obj_id=new_obj_id,
                mask=new_mask,
                add_mask_to_memory=True,
            )
        # NOTE: we skip enforcing the non-overlapping constraint **globally** when adding new objects.
        self.sam2_predictor.propagate_in_video_preflight(
            new_sam2_state, run_mem_encoder=True
        )
        if self.sam2_predictor.per_obj_inference:
            sam2_states_local = [new_sam2_state]
        else:
            sam2_states_local.append(new_sam2_state)
        return sam2_states_local

    def _sam2_remove_object(self, sam2_states_local: List[Any], obj_id: int):
        """
        Remove an object from SAM2 inference states. This would remove the object from
        all frames in the video.
        """
        sam2_states_local_before_removal = sam2_states_local.copy()
        sam2_states_local.clear()
        for sam2_inference_state in sam2_states_local_before_removal:
            # we try to remove `obj_id` on every inference state with `strict=False`
            # it will not do anything if an inference state doesn't contain `obj_id`
            new_obj_ids, _ = self.sam2_predictor.remove_object(
                sam2_inference_state, obj_id, strict=False, need_output=False
            )
            # only keep an inference state if it's non-empty after object removal
            if len(new_obj_ids) > 0:
                sam2_states_local.append(sam2_inference_state)

    def _initialize_metadata(self):
        """Initialize metadata for the masklets."""
        sam2_metadata = {
            "obj_ids_per_gpu": [np.array([], np.int64) for _ in range(self.world_size)],
            "obj_ids_all_gpu": np.array([], np.int64),
            "num_obj_per_gpu": np.zeros(self.world_size, np.int64),
            "max_obj_id": -1,
            "obj_id_to_score": {},
            "obj_id_to_last_occluded": {},
        }
        if self.rank == 0:
            # "rank0_metadata" contains metadata that is only stored on (and accessible to) GPU 0
            # - obj_first_frame_idx: obj_id --> first frame index where the object was detected
            # - unmatched_frame_inds: obj_id --> [mismatched frame indices]
            # - overlap_pair_to_frame_inds: (first_appear_obj_id, obj_id) --> [overlap frame indices]
            # - removed_obj_ids: object IDs that are suppressed via hot-start
            rank0_metadata = {
                "obj_first_frame_idx": {},
                "unmatched_frame_inds": defaultdict(list),
                "trk_keep_alive": defaultdict(
                    int
                ),  # This is used only for object suppression not for removal
                "overlap_pair_to_frame_inds": defaultdict(list),
                "removed_obj_ids": set(),
                "suppressed_obj_ids": defaultdict(
                    set
                ),  # frame_idx --> set of objects with suppressed outputs, but still continue to be tracked
            }
            sam2_metadata["rank0_metadata"] = rank0_metadata

        return sam2_metadata

    def forward(self, input: BatchedDatapoint, is_inference: bool = False):
        raise NotImplementedError("Evaluation outside demo is not implemented yet")

    def _load_checkpoint(self, ckpt_path: str, strict: bool = True):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=strict)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            logger.warning(f"Loaded ckpt with {missing_keys=}, {unexpected_keys=}")
        else:
            logger.info("Loaded ckpt successfully without missing or unexpected keys")

    def prep_for_evaluator(self, video_frames, tracking_res, scores_labels):
        """This method is only used for benchmark eval (not used in the demo)."""
        num_frames = len(video_frames)
        w, h = video_frames[0].size
        zero_mask = torch.zeros((1, h, w), dtype=torch.bool)
        object_ids = list(scores_labels.keys())
        preds = {"scores": [], "labels": [], "boxes": [], "masks_rle": []}
        for oid in object_ids:
            o_masks = []
            o_score = scores_labels[oid][0].item()
            o_label = scores_labels[oid][1]
            for frame_idx in range(num_frames):
                if frame_idx not in tracking_res:
                    o_masks.append(zero_mask)
                else:
                    o_masks.append(tracking_res[frame_idx].get(oid, zero_mask))

            o_masks = torch.cat(o_masks, dim=0)  # (n_frames, H, W)
            preds["scores"].append(o_score)
            preds["labels"].append(o_label)
            preds["boxes"].append(mask_to_box(o_masks.unsqueeze(1)).squeeze())
            preds["masks_rle"].append(rle_encode(o_masks, return_areas=True))

        preds["boxes"] = (
            torch.stack(preds["boxes"], dim=0)
            if len(preds["boxes"]) > 0
            else torch.empty(
                (0, num_frames, 4), dtype=torch.float32, device=self.device
            )
        )
        preds["scores"] = (
            torch.tensor(preds["scores"], device=self.device)
            if len(preds["scores"]) > 0
            else torch.empty((0,), device=self.device)
        )
        preds["per_frame_scores"] = preds["scores"]
        preds["labels"] = (
            torch.tensor(preds["labels"], device=self.device)
            if len(preds["labels"]) > 0
            else torch.empty((0,), device=self.device)
        )
        return preds

    def _encode_prompt(self, **kwargs):
        return self.sam3_model._encode_prompt(**kwargs)

    def _drop_new_det_with_obj_limit(self, new_det_fa_inds, det_scores_np, num_to_keep):
        """
        Drop a few new detections based on the maximum number of objects. We drop new objects based
        on their detection scores, keeping the high-scoring ones and dropping the low-scoring ones.
        """
        assert 0 <= num_to_keep <= len(new_det_fa_inds)
        if num_to_keep == 0:
            return np.array([], np.int64)  # keep none
        if num_to_keep == len(new_det_fa_inds):
            return new_det_fa_inds  # keep all

        # keep the top-scoring detections
        score_order = np.argsort(det_scores_np[new_det_fa_inds])[::-1]
        new_det_fa_inds = new_det_fa_inds[score_order[:num_to_keep]]
        return new_det_fa_inds
