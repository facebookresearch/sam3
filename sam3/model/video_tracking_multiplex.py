from collections import defaultdict

"""
Video tracking model with multiplexing support.

This file extends the base video tracking with prompt functionality to add:
    - Multiplexing: Support for processing multiple objects simultaneously
    - Recording image features in memory to support the decoupled transformer for memory reading
"""

import logging
from copy import deepcopy

try:
    from typing import Iterable, Literal, NotRequired, Optional, Required, TypedDict
except ImportError:
    from typing_extensions import (
        Iterable,
        Literal,
        NotRequired,  # not available in Python 3.10
        Optional,
        Required,  # not available in Python 3.10
        TypedDict,
    )

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from sam3.model.data_misc import BatchedDatapoint, NestedTensor
from sam3.model.memory import SimpleMaskEncoder
from sam3.model.multiplex_mask_decoder import MLP, MultiplexMaskDecoder
from sam3.model.multiplex_utils import MultiplexController, MultiplexState
from sam3.model.sam3_tracker_utils import (
    get_1d_sine_pe,
    get_next_point,
    sample_box_points,
    select_closest_cond_frames,
)
from sam3.sam.mask_decoder import MaskDecoder
from sam3.sam.prompt_encoder import PositionEmbeddingRandom, PromptEncoder
from sam3.sam.transformer import TwoWayTransformer
from timm.models.layers import trunc_normal_


# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0

neck_outs = ["interactive", "sam2_backbone_out"]


class SAMOutput(TypedDict, total=True):
    # Outputs from a single SAM head forward
    low_res_multimasks: torch.Tensor
    high_res_multimasks: torch.Tensor
    ious: torch.Tensor
    low_res_masks: torch.Tensor
    high_res_masks: torch.Tensor
    object_score_logits: torch.Tensor
    obj_ptr: NotRequired[torch.Tensor]  # [num_objects, C], in data space


class StageOutput(TypedDict, total=False):
    # metadata
    conditioning_objects: Required[set[int]]

    # The outputs from a single stage; could be used as memory
    pred_masks: torch.Tensor
    pred_masks_high_res: torch.Tensor
    point_inputs: dict[str, torch.Tensor]
    mask_inputs: torch.Tensor
    object_score_logits: torch.Tensor
    obj_ptr: torch.Tensor  # [num_buckets, multiplex_count, C], in mux space
    maskmem_features: torch.Tensor
    maskmem_pos_enc: list[torch.Tensor]
    image_features: torch.Tensor
    image_pos_enc: torch.Tensor

    # for memory filtering
    iou_score: torch.Tensor
    eff_iou_score: torch.Tensor

    # Multi-step prediction fields for state tracking or training
    multistep_pred_masks: torch.Tensor
    multistep_pred_masks_high_res: torch.Tensor
    multistep_pred_multimasks: list[torch.Tensor]
    multistep_pred_multimasks_high_res: list[torch.Tensor]
    multistep_pred_ious: list[torch.Tensor]
    multistep_point_inputs: list[dict]
    multistep_object_score_logits: list[torch.Tensor]


class VideoTrackingMultiplex(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        maskmem_backbone: nn.Module,
        multiplex_controller: MultiplexController,
        num_maskmem: int = 7,  # default 1 input frame + 6 previous frames as in CAE
        image_size: int = 512,
        backbone_stride: int = 16,  # default to 16 as in CAE (truncated Hiera backbone)
        prob_to_use_pt_input_for_train: float = 0.0,
        prob_to_use_pt_input_for_eval: float = 0.0,
        prob_to_use_box_input_for_train: float = 0.0,
        prob_to_use_box_input_for_eval: float = 0.0,
        # always_keep_first_frame_mem=True,  # this option is removed (we've always set it to True)
        apply_sigmoid_to_mask_logits_for_mem_enc: bool = False,
        sigmoid_scale_for_mem_enc: float = 1.0,  # scale factor for mask sigmoid prob, only effective when `apply_sigmoid_to_mask_logits_for_mem_enc` is True
        sigmoid_bias_for_mem_enc: float = 0.0,  # bias factor for mask sigmoid prob, only effective when `apply_sigmoid_to_mask_logits_for_mem_enc` is True
        # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks, only effective when `apply_sigmoid_to_mask_logits_for_mem_enc` is True
        binarize_mask_from_pts_for_mem_enc: bool = False,
        use_mask_input_as_output_without_sam: bool = False,  # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        # how many frames for interactive point sampling (only effective when using point inputs per video; the first frame is always used)
        # - if `num_frames_to_correct` below is True, we randomly sample 1~num_frames_to_correct frames for interactive point sampling
        # - otherwise we used a fixed number of num_frames_to_correct frames for interactive point sampling
        # if it is 1, we do interactive point sampling only on the 1st frame
        # if it is greater than 1, we interactive point sampling in the 1st frame and other randomly selected frames
        num_frames_to_correct_for_train: int = 1,  # default: only iteratively sample on first frame
        num_frames_to_correct_for_eval: int = 1,  # default: only iteratively sample on first frame
        rand_frames_to_correct_for_train: bool = False,
        rand_frames_to_correct_for_eval: bool = False,
        prob_correct_all_objects_for_train: float = 0.0,
        ratio_of_objects_to_correct_for_train: float = 1.0,
        force_correct_all_for_conditional_inputs: bool = False,
        rand_objects_to_correct_for_train: bool = True,
        # how many frames to use as initial conditioning frames (for both point input and mask input; the first frame is always used as an initial conditioning frame)
        # - if `rand_init_cond_frames` below is True, we randomly sample 1~num_init_cond_frames initial conditioning frames
        # - otherwise we sample a fixed number of num_init_cond_frames initial conditioning frames
        # note: for point input, we sample correction points on all such initial conditioning frames, and we require that `num_frames_to_correct` >= `num_init_cond_frames`;
        # these are initial conditioning frames because as we track the video, more conditioning frames might be added
        # when a frame receives correction clicks under point input if `add_all_frames_to_correct_as_cond=True`
        num_init_cond_frames_for_train: int = 1,  # default: only use the first frame as initial conditioning frame
        num_init_cond_frames_for_eval: int = 1,  # default: only use the first frame as initial conditioning frame
        rand_init_cond_frames_for_train: bool = True,  # default: random 1~num_init_cond_frames_for_train cond frames (to be constent w/ previous TA data loader)
        rand_init_cond_frames_for_eval: bool = False,
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        max_cond_frames_in_attn: int = -1,
        # Whether to always keep the first conditioning frame in case we exceed the maximum number of conditioning frames allowed
        keep_first_cond_frame=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond: bool = False,
        # how many additional correction points to sample (on each frame selected to be corrected)
        # note that the first frame receives an initial input click (in addition to any correction clicks)
        num_correction_pt_per_frame: int = 7,
        # method for point sampling during evaluation
        # "uniform" (sample uniformly from error region) or "center" (use the point with the largest distance to error region boundary)
        # default to "center" to be consistent with evaluation in the SAM paper
        pt_sampling_for_eval: Literal["uniform", "center"] = "center",
        # During training, we optionally allow sampling the correction points from GT regions
        # instead of the prediction error regions with a small probability. This might allow the
        # model to overfit less to the error regions in training datasets
        prob_to_sample_from_gt_for_train: float = 0.0,
        # on the first frame, whether to directly add the no-memory embedding to the image feature
        # (instead of using the transformer encoder)
        directly_add_no_mem_embed: bool = False,
        # whether to use high-resolution feature maps in the SAM mask decoder
        use_high_res_features_in_sam: bool = False,
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        multimask_output_in_sam: bool = False,
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
        multimask_min_pt_num: int = 1,
        multimask_max_pt_num: int = 1,
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        multimask_output_for_tracking: bool = False,
        # Whether to use multimask tokens for obj ptr; Only relevant when both
        # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
        use_multimask_token_for_obj_ptr: bool = False,
        # if the last output is multimask during training, whether to select the mask w/ highest IoU to the ground-truth for memory encoder
        # (instead of the mask with the highest prediction score; this resembles teacher-forcing for multi-mask prediction in tracking)
        use_best_iou_mask_for_mem_enc: bool = False,
        # whether to use sigmoid to restrict ious prediction to [0-1]
        iou_prediction_use_sigmoid: bool = False,
        # whether to feed the previously predicted low-res mask logits as a mask prompt into the SAM mask decoder during iterative point sampling
        iter_use_prev_mask_pred: bool = False,
        # whether to forward image features per frame (as it's being tracked) during evaluation, instead of forwarding image features
        # of all frames at once. This avoids backbone OOM errors on very long videos in evaluation, but could be slightly slower.
        forward_backbone_per_frame_for_eval: bool = False,
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
        memory_temporal_stride_for_eval: int = 1,
        # whether to offload outputs to CPU memory during evaluation, to avoid GPU OOM on very long videos or very large resolutions or too many objects
        # (it's recommended to use `forward_backbone_per_frame_for_eval=True` first before setting this option to True)
        offload_output_to_cpu_for_eval: bool = False,
        # whether to trim the output of past non-conditioning frames (num_maskmem frames before the current frame) during evaluation
        # (this helps save GPU or CPU memory on very long videos for semi-supervised VOS eval, where only the first frame receives prompts)
        trim_past_non_cond_mem_for_eval: bool = False,
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        non_overlap_masks_for_mem_enc: bool = False,
        # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        use_obj_ptrs_in_encoder: bool = False,
        # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
        max_obj_ptrs_in_encoder: int = 16,
        # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
        add_tpos_enc_to_obj_ptrs: bool = True,
        # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
        # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        proj_tpos_enc_in_obj_ptrs: bool = False,
        # whether to use signed distance (instead of unsigned absolute distance) in the temporal positional encoding in the object pointers
        # (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        use_signed_tpos_enc_to_obj_ptrs: bool = False,
        # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
        # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
        only_obj_ptrs_in_the_past_for_eval: bool = False,
        # Whether to predict if there is an object in the frame
        pred_obj_scores: bool = False,
        # Whether to use an MLP to predict object scores
        pred_obj_scores_mlp: bool = False,
        # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
        # Whether to have a fixed no obj pointer when there is no object present
        # or to use it as an additive embedding with obj_ptr produced by decoder
        fixed_no_obj_ptr: bool = False,
        use_no_obj_ptr: bool = True,
        use_mlp_for_obj_ptr_proj: bool = False,
        # replace per-slot static no-obj embeddings with linear projections of object embeddings
        use_linear_no_obj_ptr: bool = False,
        # add no obj embedding to spatial frames
        no_obj_embed_spatial: bool = False,
        # does not apply to spatial memories (only to obj ptrs), unless unified_tpos_enc=True
        sincos_tpos_enc: bool = True,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args: Optional[dict] = None,
        # whether to compile all the model compoents
        compile_all_components: bool = False,
        # save and use image features in the memory
        save_image_features: bool = False,
        # number of multimask outputs in the SAM mask decoder
        num_multimask_outputs: int = 3,
        # use a single mask token to predict all masks
        decode_mask_with_shared_tokens: bool = False,
        # use the mask token for predicting ious and object scores
        decode_mask_attribute_with_shared_tokens: bool = False,
        share_necks: bool = False,  # share the interactive and sam2_backbone necks
        # if enabled, use a different rng generator for operations that differ between GPUs,
        # such that the base rng that controls flow does not go out-of-sync among GPUs
        # There will be a slight performance penalty when turned off due to uneven workload but it's minor
        randomness_fix: bool = False,
        # add a learnable embeddings to the object queries that corresponding to paddings/removed objects
        add_output_suppression_embeddings: bool = False,
        # add a per-object embedding to the spatial memory features if that object is a conditioning input
        add_object_conditional_embeddings: bool = False,
        # if None, follow add_object_conditional_embeddings
        add_object_unconditional_embeddings: Optional[bool] = None,
        # for each object, add an additional channel in the mask encoder to indicate conditional/unconditional objects
        condition_as_mask_input: bool = False,
        condition_as_mask_input_fg: float = 1.0,
        condition_as_mask_input_bg: float = 0.0,
        # use v2 memory positional encodings
        # in v2, the last slot in the positional encoding no longer refers to the conditional frame
        # it now refers to "out-of-bound" frames.
        # The motivation is to shift all encodings of "conditioning" to the object_conditional embeddings
        use_maskmem_tpos_v2: bool = False,
        # select the frame with object existence
        use_memory_selection: bool = False,
        # when using memory selection, the threshold to determine if the frame is good
        mf_threshold: float = 0.01,
        # this is a flag for demo purposes; it does not need to be explicitly set
        is_dynamic_model: bool = False,
        object_score_logit_threshold: float = 0.0,
        stability_score_attentuation: bool = False,  # select from multimask based on iou*stability_score
    ):
        super().__init__()

        # the interactive sam mask deocder can use dynamic_multimask_via_stability
        interactive_sam_mask_decoder_extra_args = deepcopy(sam_mask_decoder_extra_args)
        if sam_mask_decoder_extra_args is not None:
            dynamic_multimask_via_stability = sam_mask_decoder_extra_args.get(
                "dynamic_multimask_via_stability", False
            )
            if dynamic_multimask_via_stability:
                sam_mask_decoder_extra_args["dynamic_multimask_via_stability"] = False
                print(
                    "dynamic_multimask_via_stability is reset to False in the multiplex model"
                )

        # Part 1: the image backbone
        self.backbone = backbone
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # A conv layer to downsample the GT mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            self.interactive_mask_downsample = torch.nn.Conv2d(
                1, 1, kernel_size=4, stride=4
            )

        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval
        self.multiplex_controller = multiplex_controller
        self.save_image_features = save_image_features
        self.multiplex_count = self.multiplex_controller.multiplex_count

        # Part 2: encoder-only transformer to fuse current frame's visual features
        # with memories from past frames
        assert transformer.decoder is None, "transformer should be encoder-only"
        self.transformer = transformer
        self.hidden_dim: int = transformer.d_model

        # Part 3: memory encoder for the previous frame's outputs
        self.maskmem_backbone = maskmem_backbone
        self.mem_dim = self.hidden_dim
        if hasattr(self.maskmem_backbone, "out_proj") and hasattr(
            self.maskmem_backbone.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            mem_dim = self.maskmem_backbone.out_proj.weight.shape[0]
            assert mem_dim == self.hidden_dim, (
                "there should be no compression of memory embeddings"
            )
        self.num_maskmem = num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.sincos_tpos_enc = sincos_tpos_enc
        self.use_maskmem_tpos_v2 = use_maskmem_tpos_v2
        # tpos specific to spatial memories only
        # last token actually corresponds to conditioning
        # frame embedding, indep of temporal position
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)

        # a single token to indicate no memory embedding from previous frames
        self.interactivity_no_mem_embed = torch.nn.Parameter(
            torch.zeros(1, 1, self.hidden_dim)
        )
        trunc_normal_(self.interactivity_no_mem_embed, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed

        # Whether to apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.apply_sigmoid_to_mask_logits_for_mem_enc = (
            apply_sigmoid_to_mask_logits_for_mem_enc
        )
        if apply_sigmoid_to_mask_logits_for_mem_enc:
            self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
            self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc

            if binarize_mask_from_pts_for_mem_enc:
                logging.warning(
                    """
                The current model is not trained with binarize_mask_from_pts_for_mem_enc;
                We force it to False here because external callers often hardcoded this
                to True, ignoring the config.
                Re-training should be possible.
                """
                )
                binarize_mask_from_pts_for_mem_enc = False

            self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.use_best_iou_mask_for_mem_enc = use_best_iou_mask_for_mem_enc
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid
        self.object_score_logit_threshold = object_score_logit_threshold
        self.stability_score_attentuation = stability_score_attentuation
        if iter_use_prev_mask_pred:
            # In this case, we are feeding the previously predicted SAM mask logits
            # as mask prompt into the SAM mask decoder, which has a different format
            # and magnitude from GT mask input in VOS. Therefore in this case, the GT
            # mask input must be encoded directly (not through the SAM mask decoder).
            if min(prob_to_use_pt_input_for_train, prob_to_use_pt_input_for_eval) < 1:
                assert use_mask_input_as_output_without_sam
        self.iter_use_prev_mask_pred = iter_use_prev_mask_pred

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.low_res_mask_size = self.image_size // self.backbone_stride * 4
        # we resize the mask if it doesn't match `self.input_mask_size` (which is always 4x
        # the low-res mask size, regardless of the actual input image size); this is because
        # `_use_mask_as_output` always downsamples the input masks by 4x
        self.input_mask_size = self.low_res_mask_size * 4
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval
        self.offload_output_to_cpu_for_eval = offload_output_to_cpu_for_eval
        if trim_past_non_cond_mem_for_eval:
            assert num_frames_to_correct_for_eval <= 1, (
                "trim_past_non_cond_mem_for_eval=True requires that only the first frame receives prompts"
            )
        self.trim_past_non_cond_mem_for_eval = trim_past_non_cond_mem_for_eval
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.interactive_sam_mask_decoder_extra_args = (
            interactive_sam_mask_decoder_extra_args
        )
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.use_no_obj_ptr = use_no_obj_ptr
        self.use_linear_no_obj_ptr = use_linear_no_obj_ptr

        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if (
            self.pred_obj_scores
            and self.use_obj_ptrs_in_encoder
            and self.use_no_obj_ptr
        ):
            if self.use_linear_no_obj_ptr:
                self.no_obj_ptr_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
            else:
                self.no_obj_ptr = torch.nn.Parameter(
                    torch.zeros(self.multiplex_count, self.hidden_dim)
                )
                trunc_normal_(self.no_obj_ptr, std=0.02)

        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(
                torch.zeros(self.multiplex_count, self.hidden_dim)
            )
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)
        self.num_multimask_outputs = num_multimask_outputs
        self.decode_mask_with_shared_tokens = decode_mask_with_shared_tokens
        self.decode_mask_attribute_with_shared_tokens = (
            decode_mask_attribute_with_shared_tokens
        )
        self.share_necks = share_necks

        self.add_output_suppression_embeddings = add_output_suppression_embeddings
        if self.add_output_suppression_embeddings:
            self.output_valid_embed = torch.nn.Parameter(
                torch.zeros(self.multiplex_count, self.hidden_dim)
            )
            self.output_invalid_embed = torch.nn.Parameter(
                torch.zeros(self.multiplex_count, self.hidden_dim)
            )
            trunc_normal_(self.output_valid_embed, std=0.02)
            trunc_normal_(self.output_invalid_embed, std=0.02)
        self.add_object_conditional_embeddings = add_object_conditional_embeddings
        if add_object_unconditional_embeddings is None:
            add_object_unconditional_embeddings = add_object_conditional_embeddings
        self.add_object_unconditional_embeddings = add_object_unconditional_embeddings
        if add_object_unconditional_embeddings:
            assert add_object_conditional_embeddings
        if self.add_object_conditional_embeddings:
            # have embeddings for both conditional and non-conditional objects
            # such that the features are more "balanced"
            # these three sets should be disjoint and their union should cover all objects
            # for conditioning objects
            self.obj_cond_embed = torch.nn.Parameter(
                torch.zeros(self.multiplex_count, self.hidden_dim)
            )
            trunc_normal_(self.obj_cond_embed, std=0.02)
            if self.add_object_unconditional_embeddings:
                # for non-conditioning objects
                self.obj_non_cond_embed = torch.nn.Parameter(
                    torch.zeros(self.multiplex_count, self.hidden_dim)
                )
                trunc_normal_(self.obj_non_cond_embed, std=0.02)

        self.condition_as_mask_input = condition_as_mask_input
        self.condition_as_mask_input_fg = condition_as_mask_input_fg
        self.condition_as_mask_input_bg = condition_as_mask_input_bg

        self.is_dynamic_model = is_dynamic_model

        self._build_sam_heads()

        # Point sampler and conditioning frames
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        if prob_to_use_pt_input_for_train > 0 or prob_to_use_pt_input_for_eval > 0:
            logging.info("Using points (sampled from masks) as inputs")
            assert num_frames_to_correct_for_train >= num_init_cond_frames_for_train
            assert num_frames_to_correct_for_eval >= num_init_cond_frames_for_eval
        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        self.prob_correct_all_objects_for_train = prob_correct_all_objects_for_train
        self.ratio_of_objects_to_correct_for_train = (
            ratio_of_objects_to_correct_for_train
        )
        self.rand_objects_to_correct_for_train = rand_objects_to_correct_for_train
        self.force_correct_all_for_conditional_inputs = (
            force_correct_all_for_conditional_inputs
        )
        # Initial multi-conditioning frames
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.max_cond_frames_in_attn = max_cond_frames_in_attn
        self.keep_first_cond_frame = keep_first_cond_frame
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        # A random number generator with a fixed initial seed across GPUs
        self.rng = np.random.default_rng(seed=42)
        if randomness_fix:
            self.rng2 = np.random.default_rng(seed=42)
        else:
            self.rng2 = self.rng

        # Use frame filtering according to SAM2Long
        self.use_memory_selection = use_memory_selection
        self.mf_threshold = mf_threshold

        # Compile all components of the model
        self.compile_all_components = compile_all_components
        if self.compile_all_components:
            self._compile_all_components()

    def _get_tpos_enc(self, rel_pos_list, device, max_abs_pos=None, dummy=False):
        if dummy:
            return torch.zeros(len(rel_pos_list), self.mem_dim, device=device)

        t_diff_max = max_abs_pos - 1 if max_abs_pos is not None else 1
        pos_enc = (
            torch.tensor(rel_pos_list).pin_memory().to(device=device, non_blocking=True)
            / t_diff_max
        )
        if self.sincos_tpos_enc:
            tpos_dim = (
                self.hidden_dim if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
            )
            pos_enc = get_1d_sine_pe(pos_enc, dim=tpos_dim)
        else:
            raise NotImplementedError
        pos_enc = self.obj_ptr_tpos_proj(pos_enc)

        return pos_enc

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        self.image_pe_layer = PositionEmbeddingRandom(self.hidden_dim // 2)

        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.interactive_sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )

        self.interactive_sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.interactive_sam_mask_decoder_extra_args or {}),
        )
        if self.share_necks:
            # we will use self.sam_mask_decoder's convs
            del self.interactive_sam_mask_decoder.conv_s0
            del self.interactive_sam_mask_decoder.conv_s1

        self.sam_mask_decoder = MultiplexMaskDecoder(
            multiplex_count=self.multiplex_count,
            num_multimask_outputs=self.num_multimask_outputs,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.hidden_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.hidden_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            decode_mask_with_shared_tokens=self.decode_mask_with_shared_tokens,
            decode_mask_attribute_with_shared_tokens=self.decode_mask_attribute_with_shared_tokens,
            multimask_outputs_only=self.num_multimask_outputs > 0
            and self.multimask_output_in_sam,
            **(self.sam_mask_decoder_extra_args or {}),
        )

        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            self.interactive_obj_ptr_proj = torch.nn.Linear(
                self.hidden_dim, self.hidden_dim
            )
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
                self.interactive_obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
            self.interactive_obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def _get_interactive_pix_mem(
        self, features: torch.Tensor, feat_sizes: list[tuple]
    ) -> torch.Tensor:
        assert self.directly_add_no_mem_embed
        pix_feat_with_mem = features[-1] + self.interactivity_no_mem_embed
        B = features[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _forward_sam_heads(
        self,
        backbone_features: torch.Tensor,
        *,
        point_inputs: Optional[dict[str, torch.Tensor]] = None,
        mask_inputs: Optional[torch.Tensor] = None,
        interactive_high_res_features: Optional[list[torch.Tensor]] = None,
        propagation_high_res_features: Optional[list[torch.Tensor]] = None,
        multimask_output: bool = False,
        gt_masks=None,
        multiplex_state: MultiplexState,
        objects_to_interact: Optional[list[int]] = None,
    ) -> SAMOutput:
        """
        Forward SAM prompt encoders and mask heads.
        We run the propagation head, the interactive head, or both, based on the inputs.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious: [B, M] shape (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [num_buckets, multiplex_count, C] shape, the object pointer vector for
          the output mask, extracted based on the output token from the SAM mask decoder.
        """

        device = backbone_features.device
        assert backbone_features.size(1) == self.hidden_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        is_interactive = point_inputs is not None or mask_inputs is not None

        if is_interactive:
            """
            Image-level, per-object interactive path
            """
            assert interactive_high_res_features is not None
            assert objects_to_interact is not None

            # a) Handle point prompts
            if point_inputs is not None:
                sam_point_coords = point_inputs["point_coords"]
                sam_point_labels = point_inputs["point_labels"]
            else:
                assert mask_inputs is not None
                # If no points are provided, pad with an empty point (with label -1)
                sam_point_coords = torch.zeros(
                    mask_inputs.shape[0], 1, 2, device=device
                )
                sam_point_labels = -torch.ones(
                    mask_inputs.shape[0], 1, dtype=torch.int32, device=device
                )

            # b) Handle mask prompts
            if mask_inputs is not None:
                # If mask_inputs is provided, downsize it into low-res mask input if needed
                # and feed it as a dense mask prompt into the SAM mask encoder
                assert len(mask_inputs.shape) == 4
                if (
                    mask_inputs.shape[-2:]
                    != self.interactive_sam_prompt_encoder.mask_input_size
                ):
                    sam_mask_prompt = F.interpolate(
                        mask_inputs.float(),
                        size=self.interactive_sam_prompt_encoder.mask_input_size,
                        align_corners=False,
                        mode="bilinear",
                        antialias=True,  # use antialias for downsampling
                    )
                else:
                    sam_mask_prompt = mask_inputs
            else:
                # Otherwise, simply feed None (and SAM's prompt encoder will add
                # a learned `no_mask_embed` to indicate no mask input in this case).
                sam_mask_prompt = None

            sparse_embeddings, dense_embeddings = self.interactive_sam_prompt_encoder(
                points=(sam_point_coords, sam_point_labels),
                boxes=None,
                masks=sam_mask_prompt,
            )

            # Clone image_pe and the outputs of sam_prompt_encoder
            # to enable compilation
            sparse_embeddings = self._maybe_clone(sparse_embeddings)
            dense_embeddings = self._maybe_clone(dense_embeddings)
            image_pe = self._maybe_clone(
                self.interactive_sam_prompt_encoder.get_dense_pe()
            )
            (
                low_res_multimasks,
                ious,
                sam_output_tokens,
                object_score_logits,
            ) = self.interactive_sam_mask_decoder(
                image_embeddings=backbone_features,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=True,
                high_res_features=interactive_high_res_features,
            )

        else:
            """
            Multiplexed propagation path
            """
            assert propagation_high_res_features is not None
            assert multiplex_state is not None

            if self.add_output_suppression_embeddings:
                # the suppression embeddings inform the mask decoder the objects that should be decoded
                output_valid_embed = self.output_valid_embed.unsqueeze(0)
                output_invalid_embed = self.output_invalid_embed.unsqueeze(0)
                valid_object_mask = (
                    multiplex_state.get_valid_object_mask().unsqueeze(-1).float()
                )
                output_merged_embed = (
                    valid_object_mask * output_valid_embed
                    + (1 - valid_object_mask) * output_invalid_embed
                )
            else:
                output_merged_embed = None

            # Clone image_pe to enable compilation
            image_pe = self._maybe_clone(self.get_propagation_dense_pe())
            out = self.sam_mask_decoder(
                image_embeddings=backbone_features,
                image_pe=image_pe,
                high_res_features=propagation_high_res_features,
                multimask_output=multimask_output,
                extra_per_object_embeddings=output_merged_embed,
            )
            low_res_multimasks = out["masks"]  # [B, M, 3/1, H*4, W*4]
            ious = out["iou_pred"]  # [B, M, 3/1]
            sam_output_tokens = out["sam_tokens_out"]  # [B, M, 3/1, C]
            object_score_logits = out["object_score_logits"]

            low_res_multimasks = multiplex_state.demux(low_res_multimasks)
            ious = multiplex_state.demux(ious)
            object_score_logits = multiplex_state.demux(object_score_logits)
            sam_output_tokens = multiplex_state.demux(sam_output_tokens)

        """
        The interactive and the propagation paths converge here
        """
        # Clone the output of sam_mask_decoder
        # to enable compilation
        low_res_multimasks = self._maybe_clone(low_res_multimasks)
        ious = self._maybe_clone(ious)
        object_score_logits = self._maybe_clone(object_score_logits)
        sam_output_tokens = self._maybe_clone(sam_output_tokens)

        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > self.object_score_logit_threshold

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output and (
            not self.decode_mask_with_shared_tokens or is_interactive
        ):
            # take the best mask prediction (with the highest IoU estimation)
            if self.stability_score_attentuation:
                # prefer selecting masks with high stability score
                stability_score = self.sam_mask_decoder._get_stability_scores(
                    low_res_multimasks
                )
                ious = ious * stability_score

            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(ious.shape[0], device=device)

            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            if multimask_output and not is_interactive:
                assert self.decode_mask_with_shared_tokens
                low_res_masks = low_res_multimasks[:, 0:1]
                high_res_masks = high_res_multimasks[:, 0:1]
            else:
                low_res_masks = low_res_multimasks
                high_res_masks = high_res_multimasks

        # Extract object pointer from the SAM output token
        if self.use_obj_ptrs_in_encoder:
            if is_interactive:
                obj_ptr = self.interactive_obj_ptr_proj(sam_output_token)
            else:
                obj_ptr = self.obj_ptr_proj(sam_output_token)

            if self.pred_obj_scores and self.use_no_obj_ptr:
                lambda_is_obj_appearing = is_obj_appearing.float()
                if self.use_linear_no_obj_ptr:
                    obj_ptr = lambda_is_obj_appearing * obj_ptr + (
                        1 - lambda_is_obj_appearing
                    ) * self.no_obj_ptr_linear(obj_ptr)
                else:
                    if self.fixed_no_obj_ptr:
                        obj_ptr = lambda_is_obj_appearing * obj_ptr

                    # use demux to locate the corresponding no_obj_ptr entries
                    selected_no_obj_ptr = self.no_obj_ptr.unsqueeze(0).repeat(
                        multiplex_state.num_buckets, 1, 1
                    )
                    selected_no_obj_ptr = multiplex_state.demux(selected_no_obj_ptr)
                    if is_interactive:
                        # if is_interactive, the object pointers are in the data space
                        selected_no_obj_ptr = selected_no_obj_ptr[objects_to_interact]

                    obj_ptr = (
                        obj_ptr + (1 - lambda_is_obj_appearing) * selected_no_obj_ptr
                    )

        outputs: SAMOutput = {
            "low_res_multimasks": low_res_multimasks,
            "high_res_multimasks": high_res_multimasks,
            "ious": ious,
            "low_res_masks": low_res_masks,
            "high_res_masks": high_res_masks,
            "object_score_logits": object_score_logits,
        }
        if self.use_obj_ptrs_in_encoder:
            outputs["obj_ptr"] = obj_ptr  # [num_objects, C], in data space
        return outputs

    def _use_mask_as_output(
        self,
        backbone_features: torch.Tensor,
        high_res_features: list[torch.Tensor],
        mask_inputs: torch.Tensor,
        multiplex_state: MultiplexState,
        objects_in_mask: Optional[list[int]] = None,
    ) -> SAMOutput:
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        if objects_in_mask is None:
            objects_in_mask = list(range(multiplex_state.total_valid_entries))

        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.to(backbone_features.dtype)
        assert mask_inputs.shape[0] == len(objects_in_mask), (
            f"{mask_inputs.shape[0]} != {len(objects_in_mask)}"
        )
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(
            mask_inputs.size(0), 1, dtype=backbone_features.dtype
        )

        if self.use_obj_ptrs_in_encoder:
            # produce an object pointer using the SAM decoder from the mask input
            sam_outputs = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.interactive_mask_downsample(mask_inputs_float),
                interactive_high_res_features=high_res_features,
                gt_masks=mask_inputs,
                objects_to_interact=objects_in_mask,
                multiplex_state=multiplex_state,
            )
            obj_ptr = sam_outputs["obj_ptr"]

            # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
            # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
            # on the object_scores from the SAM decoder.
            is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
            is_obj_appearing = is_obj_appearing[..., None]
            lambda_is_obj_appearing = is_obj_appearing.float()
            object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
            # Note that although this logic has already been applied in _forward_sam_heads
            # it is ok because lambda_is_obj_appearing is binary
            # when it is zero it forces no_obj_ptr
            # when it is one it keeps the output from _forward_sam_heads
            if self.pred_obj_scores and self.use_no_obj_ptr:
                if self.use_linear_no_obj_ptr:
                    obj_ptr = lambda_is_obj_appearing * obj_ptr + (
                        1 - lambda_is_obj_appearing
                    ) * self.no_obj_ptr_linear(obj_ptr)
                else:
                    if self.fixed_no_obj_ptr:
                        obj_ptr = lambda_is_obj_appearing * obj_ptr
                    # use demux to locate the corresponding no_obj_ptr entries
                    selected_no_obj_ptr = self.no_obj_ptr.unsqueeze(0).repeat(
                        multiplex_state.num_buckets, 1, 1
                    )
                    selected_no_obj_ptr = multiplex_state.demux(selected_no_obj_ptr)
                    selected_no_obj_ptr = selected_no_obj_ptr[objects_in_mask]
                    obj_ptr = (
                        obj_ptr + (1 - lambda_is_obj_appearing) * selected_no_obj_ptr
                    )

        outputs: SAMOutput = {
            "low_res_multimasks": low_res_masks,
            "high_res_multimasks": high_res_masks,
            "ious": ious,
            "low_res_masks": low_res_masks,
            "high_res_masks": high_res_masks,
            "object_score_logits": object_score_logits,
        }
        if self.use_obj_ptrs_in_encoder:
            outputs["obj_ptr"] = obj_ptr  # [num_objects, C], in data space
        return outputs

    def forward(self, input: BatchedDatapoint, is_inference=False):
        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all frames before tracking
            backbone_out = self.forward_image(
                input.img_batch, need_interactive_out=True, need_propagation_out=True
            )
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {}
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        previous_stages_out = self.forward_tracking(backbone_out, input)

        # "None" for get_queries to be compatible with the trainer
        return previous_stages_out, None

    def forward_image(
        self,
        img_batch,
        *,
        need_sam3_out: bool = False,
        need_interactive_out: bool = False,
        need_propagation_out: bool = False,
    ):
        """Get the image feature on the input batch."""
        if self.share_necks:
            need_propagation_out = need_interactive_out or need_propagation_out
            need_interactive_out = False
            # this also means that convs for backbone_fpn are shared
            backbone_out = self.backbone.forward_image(
                img_batch,
                need_sam3_out=need_sam3_out,
                need_sam2_out=need_propagation_out,
            )
            backbone_out["interactive"] = backbone_out["sam2_backbone_out"]
        else:
            backbone_out = self.backbone.forward_image(
                img_batch,
                need_sam3_out=need_sam3_out,
                need_interactive_out=need_interactive_out,
                need_propagation_out=need_propagation_out,
            )
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            if need_interactive_out:
                backbone_out["interactive"]["backbone_fpn"][
                    0
                ].tensors = self.interactive_sam_mask_decoder.conv_s0(
                    backbone_out["interactive"]["backbone_fpn"][0].tensors
                )
                backbone_out["interactive"]["backbone_fpn"][
                    1
                ].tensors = self.interactive_sam_mask_decoder.conv_s1(
                    backbone_out["interactive"]["backbone_fpn"][1].tensors
                )
            if need_propagation_out:
                backbone_out["sam2_backbone_out"]["backbone_fpn"][
                    0
                ].tensors = self.sam_mask_decoder.conv_s0(
                    backbone_out["sam2_backbone_out"]["backbone_fpn"][0].tensors
                )
                backbone_out["sam2_backbone_out"]["backbone_fpn"][
                    1
                ].tensors = self.sam_mask_decoder.conv_s1(
                    backbone_out["sam2_backbone_out"]["backbone_fpn"][1].tensors
                )
        # Clone to help torch.compile
        for out_type in backbone_out.keys():
            for i in range(len(backbone_out[out_type]["backbone_fpn"])):
                backbone_out[out_type]["backbone_fpn"][i].tensors = self._maybe_clone(
                    backbone_out[out_type]["backbone_fpn"][i].tensors
                )
                backbone_out[out_type]["vision_pos_enc"][i] = self._maybe_clone(
                    backbone_out[out_type]["vision_pos_enc"][i]
                )
        return backbone_out

    def _prepare_prompt_inputs_meta(self, backbone_out, input, start_frame_idx=0):
        # Load the ground-truth masks on all frames (so that we can later
        # sample correction points from them)
        gt_masks_per_frame = {
            stage_id: targets.segments.unsqueeze(1)  # [B, 1, H_im, W_im]
            for stage_id, targets in enumerate(input.find_targets)
        }
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = len(input.find_targets)
        backbone_out["num_frames"] = num_frames

        # Randomly decide whether to use point inputs or mask inputs
        if self.training:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval
        if num_frames == 1:
            # here we handle a special case for mixing video + SAM on image training,
            # where we force using point input for the SAM task on static images
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1
        assert num_init_cond_frames >= 1
        # (here `self.rng.random()` returns value in range 0.0 <= X < 1.0)
        use_pt_input = self.rng.random() < prob_to_use_pt_input
        if rand_init_cond_frames and num_init_cond_frames > 1:
            # randomly select 1 to `num_init_cond_frames` frames as initial conditioning frames
            num_init_cond_frames = self.rng.integers(
                1, num_init_cond_frames, endpoint=True
            )
        if (
            use_pt_input
            and rand_frames_to_correct
            and num_frames_to_correct > num_init_cond_frames
        ):
            # randomly select `num_init_cond_frames` to `num_frames_to_correct` frames to sample
            # correction clicks (only for the case of point input)
            num_frames_to_correct = self.rng.integers(
                num_init_cond_frames, num_frames_to_correct, endpoint=True
            )
        backbone_out["use_pt_input"] = use_pt_input

        # Sample initial conditioning frames
        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]  # starting frame
        else:
            # starting frame + randomly selected remaining frames (without replacement)
            init_cond_frames = [start_frame_idx] + self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]

        # Sample frames where we will add correction clicks on the fly
        # based on the error between prediction and ground-truth masks
        if not use_pt_input:
            # no correction points will be sampled when using mask inputs
            frames_to_add_correction_pt = []
        elif num_frames_to_correct == num_init_cond_frames:
            frames_to_add_correction_pt = init_cond_frames
        else:
            assert num_frames_to_correct > num_init_cond_frames
            # initial cond frame + randomly selected remaining frames (without replacement)
            extra_num = num_frames_to_correct - num_init_cond_frames
            frames_to_add_correction_pt = (
                init_cond_frames
                + self.rng.choice(
                    backbone_out["frames_not_in_init_cond"], extra_num, replace=False
                ).tolist()
            )
        backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt

        return backbone_out

    def _prepare_conditional_frames(self, backbone_out):
        init_cond_frames = backbone_out["init_cond_frames"]
        gt_masks_per_frame = backbone_out["gt_masks_per_frame"]
        use_pt_input = backbone_out["use_pt_input"]

        if self.training:
            prob_to_use_box_input = self.prob_to_use_box_input_for_train
        else:
            prob_to_use_box_input = self.prob_to_use_box_input_for_eval

        # Prepare mask or point inputs on initial conditioning frames
        backbone_out["mask_inputs_per_frame"] = {}  # {frame_idx: <input_masks>}
        backbone_out["point_inputs_per_frame"] = {}  # {frame_idx: <input_points>}
        for t in init_cond_frames:
            if not use_pt_input:
                backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]
            else:
                # During training # P(box) = prob_to_use_pt_input * prob_to_use_box_input
                use_box_input = self.rng.random() < prob_to_use_box_input
                if use_box_input:
                    points, labels = sample_box_points(
                        gt_masks_per_frame[t],
                    )
                else:
                    # (here we only sample **one initial point** on initial conditioning frames from the
                    # ground-truth mask; we may sample more correction points on the fly)
                    points, labels = get_next_point(
                        gt_masks=gt_masks_per_frame[t],
                        pred_masks=None,
                        method=(
                            "uniform" if self.training else self.pt_sampling_for_eval
                        ),
                    )

                point_inputs = {"point_coords": points, "point_labels": labels}
                backbone_out["point_inputs_per_frame"][t] = point_inputs

        return backbone_out

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        Prepare input mask, point or box prompts. Optionally, we allow tracking from
        a custom `start_frame_idx` to the end of the video (for evaluation purposes).
        """
        backbone_out = self._prepare_prompt_inputs_meta(
            backbone_out, input, start_frame_idx
        )
        backbone_out = self._prepare_conditional_frames(backbone_out)
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features (same as in MDETR_API model)."""

        backbone_features = {}

        for neck_k in neck_outs:
            if neck_k not in backbone_out:
                continue
            neck_out = backbone_out[neck_k]
            assert len(neck_out["backbone_fpn"]) == len(neck_out["vision_pos_enc"])
            assert len(neck_out["backbone_fpn"]) >= self.num_feature_levels

            feature_maps = neck_out["backbone_fpn"][-self.num_feature_levels :]
            vision_pos_embeds = neck_out["vision_pos_enc"][-self.num_feature_levels :]

            feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
            # flatten NxCxHxW to HWxNxC
            vision_feats = [x.tensors.flatten(2).permute(2, 0, 1) for x in feature_maps]
            vision_pos_embeds = [
                x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds
            ]
            vision_masks = [x.mask for x in feature_maps]

            for i, vision_mask in enumerate(vision_masks):
                if vision_mask is not None:
                    vision_masks[i] = vision_mask.flatten(1)

            backbone_features[neck_k] = {
                "vision_feats": vision_feats,
                "vision_pos_embeds": vision_pos_embeds,
                "vision_masks": vision_masks,
                "feat_sizes": feat_sizes,
            }

        return backbone_features

    def _prepare_backbone_features_per_frame(
        self,
        img_batch,
        img_ids,
        *,
        need_interactive_out: bool = False,
        need_propagation_out: bool = False,
    ):
        """Compute the image backbone features on the fly for the given img_ids."""
        # all image ids should be the same
        assert img_ids.numel() == 1
        unique_img_ids = img_ids

        # Compute the image features on those unique image ids
        image = img_batch.tensors[unique_img_ids]
        image_mask = (
            img_batch.mask[unique_img_ids] if img_batch.mask is not None else None
        )

        backbone_out = self.forward_image(
            NestedTensor(tensors=image, mask=image_mask),
            need_interactive_out=need_interactive_out,
            need_propagation_out=need_propagation_out,
        )

        backbone_features = self._prepare_backbone_features(backbone_out)
        return image, backbone_features

    def _prepare_memory_conditioned_features(
        self,
        *,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_masks,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        use_prev_mem_frame=True,  # whether to condition on previous memory frames
        multiplex_state: MultiplexState,
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = multiplex_state.num_buckets
        # B = current_vision_feats[-1].size(1)  # batch size on this frame
        vision_feat = current_vision_feats[-1].expand(-1, B, -1)
        vision_mask = (
            current_vision_masks[-1].expand(-1, B, -1)
            if current_vision_masks[-1] is not None
            else None
        )
        vision_pos_embed = current_vision_pos_embeds[-1].expand(-1, B, -1)

        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = vision_feat.permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame and use_prev_mem_frame:
            # Retrieve the memories encoded with the maskmem backbone
            # to_cat_prompt, to_cat_prompt_mask, to_cat_prompt_pos_embed = [], [], []
            to_cat_prompt, to_cat_prompt_pos_embed = [], []
            if self.save_image_features:
                to_cat_image_feat, to_cat_image_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx,
                cond_outputs,
                self.max_cond_frames_in_attn,
                keep_first_cond_frame=self.keep_first_cond_frame,
            )

            t_pos_and_prevs = [
                ((frame_idx - t) * tpos_sign_mul, out, True)
                for t, out in selected_cond_outputs.items()
            ]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with r>1), in which case
            # we take (self.num_maskmem - 2) frames among every r-th frames plus the last frame.
            r = 1 if self.training else self.memory_temporal_stride_for_eval

            if self.use_memory_selection:
                valid_indices = self.frame_filter(
                    output_dict, track_in_reverse, frame_idx, num_frames, r
                )

            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if self.use_memory_selection:
                    if t_rel > len(valid_indices):
                        continue
                    prev_frame_idx = valid_indices[-t_rel]
                else:
                    if t_rel == 1:
                        # for t_rel == 1, we take the last frame (regardless of r)
                        if not track_in_reverse:
                            # the frame immediately before this frame (i.e. frame_idx - 1)
                            prev_frame_idx = frame_idx - t_rel
                        else:
                            # the frame immediately after this frame (i.e. frame_idx + 1)
                            prev_frame_idx = frame_idx + t_rel
                    else:
                        # for t_rel >= 2, we take the memory frame from every r-th frames
                        if not track_in_reverse:
                            # first find the nearest frame among every r-th frames before this frame
                            # for r=1, this would be (frame_idx - 2)
                            prev_frame_idx = ((frame_idx - 2) // r) * r
                            # then seek further among every r-th frames
                            prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                        else:
                            # first find the nearest frame among every r-th frames after this frame
                            # for r=1, this would be (frame_idx + 2)
                            prev_frame_idx = -(-(frame_idx + 2) // r) * r
                            # then seek further among every r-th frames
                            prev_frame_idx = prev_frame_idx + (t_rel - 2) * r
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out, False))

            for t_pos, prev, is_selected_cond_frame in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames

                feats = prev.get("maskmem_features")
                if feats is None:
                    continue
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = feats.cuda(non_blocking=True)
                if feats.dim() == 5:
                    feats = multiplex_state.demux(feats).contiguous()
                    prev["maskmem_features"] = (
                        feats.cpu() if not feats.is_cuda else feats
                    )

                if feats.shape[0] == 0:
                    continue

                to_cat_prompt.append(feats.flatten(2).permute(2, 0, 1))
                # to_cat_prompt_mask.append(None)
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_pos_list = prev.get("maskmem_pos_enc")
                if not maskmem_pos_list:
                    continue
                maskmem_enc = maskmem_pos_list[-1]
                if maskmem_enc is None:
                    continue
                maskmem_enc = maskmem_enc.cuda(non_blocking=True)
                if maskmem_enc.dim() == 5:
                    maskmem_enc = multiplex_state.demux(maskmem_enc).contiguous()
                    prev["maskmem_pos_enc"][-1] = (
                        maskmem_enc.cpu() if not maskmem_enc.is_cuda else maskmem_enc
                    )
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)

                if self.use_maskmem_tpos_v2:
                    # the last of maskmem_tpos_enc is an "out-of-range" embedding
                    if t_pos <= 0 or t_pos >= self.num_maskmem:
                        tpos_enc = self.maskmem_tpos_enc[self.num_maskmem - 1]
                    else:
                        tpos_enc = self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                else:
                    # cond_frame NOT temporally encoded in this setting
                    # and last of the maskmem_tpos_enc is actually an
                    # indicator for being a cond_frame
                    t = t_pos if not is_selected_cond_frame else 0
                    tpos_enc = self.maskmem_tpos_enc[self.num_maskmem - t - 1]

                maskmem_enc = maskmem_enc + tpos_enc

                if self.save_image_features:
                    # image features are in (HW)BC
                    image_feat = prev["image_features"].cuda()
                    image_pos_embed = prev["image_pos_enc"].cuda() + tpos_enc
                    to_cat_image_feat.append(image_feat)
                    to_cat_image_pos_embed.append(image_pos_embed)

                to_cat_prompt_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_outs_for_ptr = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out,
                        True,  # is_selected_cond_frame
                    )
                    for t, out in ptr_cond_outputs.items()
                ]

                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    if not self.use_memory_selection:
                        t = (
                            frame_idx + t_diff
                            if track_in_reverse
                            else frame_idx - t_diff
                        )
                        if t < 0 or (num_frames is not None and t >= num_frames):
                            break
                    else:
                        if -t_diff <= -len(valid_indices):
                            break
                        t = valid_indices[-t_diff]

                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_outs_for_ptr.append((t_diff, out, False))

                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_outs_for_ptr) > 0:
                    pos_list, out_list, is_selected_cond_frame_list = zip(
                        *pos_and_outs_for_ptr
                    )
                    # Filter out outputs that don't have obj_ptr (e.g., when object has empty mask)
                    filtered_data = [
                        (pos, out, is_cond)
                        for pos, out, is_cond in zip(
                            pos_list, out_list, is_selected_cond_frame_list
                        )
                        if "obj_ptr" in out
                    ]

                    # Only proceed if we have at least one valid obj_ptr
                    if len(filtered_data) > 0:
                        pos_list, out_list, is_selected_cond_frame_list = zip(
                            *filtered_data
                        )
                        # each out["obj_ptr"] is a tensor of shape (num_buckets, seq_len, C)
                        # cat object pointers along dim=0 into [ptr_seq_len, B, C] shape
                        obj_ptrs = torch.cat(
                            [out["obj_ptr"] for out in out_list], dim=1
                        ).transpose(0, 1)

                        # a temporal positional embedding based on how far each object pointer is from
                        # the current frame (sine embedding normalized by the max pointer num).
                        if self.add_tpos_enc_to_obj_ptrs:
                            obj_pos = self._get_tpos_enc(
                                pos_list,
                                max_abs_pos=max_obj_ptrs_in_encoder,
                                device=device,
                            )
                        else:
                            obj_pos = self._get_tpos_enc(
                                pos_list, device=device, dummy=True
                            )
                        # expand to batch size
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, -1)

                        assert self.mem_dim == C, (
                            f"obj_ptrs.shape = {obj_ptrs.shape}, C = {C}"
                        )

                        # each frame has [bucket_size] pointers, except the first frame
                        obj_pos = obj_pos.repeat_interleave(
                            multiplex_state.multiplex_count, dim=0
                        )

                        to_cat_prompt.append(obj_ptrs)
                        to_cat_prompt_pos_embed.append(obj_pos)
                        # number of object pointer tokens for the encoder
                        num_obj_ptr_tokens = obj_ptrs.shape[0]
                    else:
                        # All outputs were filtered out (empty masks), no obj_ptrs available
                        num_obj_ptr_tokens = 0
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            raise NotImplementedError(
                "Any init cond frame should have gone to _use_mask_as_output instead"
            )

        # Step 2: Concatenate the memories and forward through the transformer encoder
        if len(to_cat_prompt) == 0:
            # No available memory features (e.g. mask was cleared). Skip fusion and
            # fall back to the current frame features so the object can continue to
            # propagate as empty without raising errors.
            pix_feat = vision_feat.permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        prompt = torch.cat(to_cat_prompt, dim=0)
        prompt_mask = None  # For now, we always masks are zeros anyways
        prompt_pos_embed = torch.cat(to_cat_prompt_pos_embed, dim=0)

        if self.save_image_features:
            assert prompt_mask is None
            assert vision_mask is None
            if len(to_cat_image_feat) == 0 or len(to_cat_image_pos_embed) == 0:
                # Memory image features were cleared; fall back to current-frame features.
                pix_feat = vision_feat.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat
            image_feat = torch.cat(to_cat_image_feat, dim=0)
            image_pos_embed = torch.cat(to_cat_image_pos_embed, dim=0)

            encoder_out = self.transformer.encoder(
                image=current_vision_feats[-1],
                src=vision_feat,
                memory_image=image_feat,
                memory=prompt,
                image_pos=current_vision_pos_embeds[-1],
                src_pos=vision_pos_embed,
                memory_image_pos=image_pos_embed,
                memory_pos=prompt_pos_embed,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
            )
        else:
            encoder_out = self.transformer.encoder(
                src=vision_feat,
                src_key_padding_mask=vision_mask,
                src_pos=vision_pos_embed,
                prompt=prompt,
                prompt_pos=prompt_pos_embed,
                prompt_key_padding_mask=prompt_mask,
                feat_sizes=feat_sizes,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
            )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = encoder_out["memory"].permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        image,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
        *,
        conditioning_objects: Optional[Iterable[int]] = None,
        multiplex_state: MultiplexState,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        if self.apply_sigmoid_to_mask_logits_for_mem_enc:
            # scale the raw mask logits with a temperature before applying sigmoid
            assert not self.binarize_mask_from_pts_for_mem_enc, (
                "haven't been trained this way; beware of hardcoded config override"
            )
            binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
            if binarize and not self.training:
                mask_for_mem = (pred_masks_high_res > 0).float()
            else:
                # apply sigmoid on the raw mask logits to turn them into range (0, 1)
                mask_for_mem = torch.sigmoid(pred_masks_high_res)
            # apply scale and bias terms to the sigmoid probabilities
            if self.sigmoid_scale_for_mem_enc != 1.0:
                mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
            if self.sigmoid_bias_for_mem_enc != 0.0:
                mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        else:
            mask_for_mem = pred_masks_high_res

        if self.add_object_conditional_embeddings or self.condition_as_mask_input:
            # figure out the set of objects that are "conditional" on this frame
            if conditioning_objects is None:
                conditioning_objects = []
                unconditioning_objects = sorted(
                    list(multiplex_state.get_all_valid_object_idx())
                )
            else:
                conditioning_objects = sorted(list(conditioning_objects))
                all_objects_idx = multiplex_state.get_all_valid_object_idx()
                unconditioning_objects = sorted(
                    [i for i in all_objects_idx if i not in conditioning_objects]
                )

        mux_mask_for_mem = multiplex_state.mux(mask_for_mem).squeeze(2)

        if self.condition_as_mask_input:
            # create num_objects channels spatial features that encode the
            # list of objects that are conditional with fg and bg values
            num_objects = mask_for_mem.shape[0]
            # Create a 1D conditioning mask on GPU and broadcast it
            cond_values = torch.full(
                (num_objects,),
                self.condition_as_mask_input_bg,
                device=mask_for_mem.device,
                dtype=mask_for_mem.dtype,
            )
            if len(conditioning_objects) > 0:
                cond_values[conditioning_objects] = self.condition_as_mask_input_fg
            # Broadcast to full spatial dimensions: [N] -> [N, 1, H, W]
            embedded_conditions = cond_values.view(-1, 1, 1, 1).expand_as(mask_for_mem)
            embedded_conditions = multiplex_state.mux(embedded_conditions).squeeze(2)

            mux_mask_for_mem = torch.cat([mux_mask_for_mem, embedded_conditions], dim=1)

        if isinstance(self.maskmem_backbone, SimpleMaskEncoder):
            maskmem_out = self.maskmem_backbone(
                pix_feat,
                mux_mask_for_mem,
                skip_mask_sigmoid=True,
            )
        else:
            maskmem_out = self.maskmem_backbone(image, pix_feat, mux_mask_for_mem)
        # Clone the feats and pos_enc to enable compilation
        maskmem_features = self._maybe_clone(maskmem_out["vision_features"])
        maskmem_pos_enc = [self._maybe_clone(m) for m in maskmem_out["vision_pos_enc"]]

        if self.no_obj_embed_spatial is not None:
            # since maskmem_features are deeply detangled between objects
            # we simply add a projected embedding for each empty object
            # num_buckets * multiplex_count * C
            no_obj_embed_spatial = self.no_obj_embed_spatial.unsqueeze(0).repeat(
                multiplex_state.num_buckets, 1, 1
            )
            # Align object_score_logits length to multiplex expectations before mux
            if object_score_logits is not None:
                obj_expected = multiplex_state.total_valid_entries
                obj_current = object_score_logits.shape[0]
                if obj_current != obj_expected:
                    if obj_current < obj_expected:
                        pad_shape = (obj_expected - obj_current,) + tuple(
                            object_score_logits.shape[1:]
                        )
                        obj_pad = object_score_logits.new_zeros(pad_shape)
                        object_score_logits = torch.cat(
                            [object_score_logits, obj_pad], dim=0
                        )
                    else:
                        object_score_logits = object_score_logits[:obj_expected]
            object_score_logits = multiplex_state.mux(object_score_logits)
            is_obj_appearing = (
                object_score_logits > self.object_score_logit_threshold
            ).float()

            no_obj_embed = ((1 - is_obj_appearing) * no_obj_embed_spatial).sum(dim=1)
            maskmem_features += no_obj_embed[..., None, None].expand_as(
                maskmem_features
            )

        if self.add_object_conditional_embeddings:
            # add object conditional embeddings to the maskmem_features
            # num_buckets * multiplex_count * C
            obj_cond_embed = self.obj_cond_embed.unsqueeze(0).repeat(
                multiplex_state.num_buckets, 1, 1
            )
            obj_cond_embed = multiplex_state.demux(obj_cond_embed)
            obj_merged_embed = obj_cond_embed

            if self.add_object_unconditional_embeddings:
                obj_non_cond_embed = self.obj_non_cond_embed.unsqueeze(0).repeat(
                    multiplex_state.num_buckets, 1, 1
                )
                obj_non_cond_embed = multiplex_state.demux(obj_non_cond_embed)
                if self.training:
                    obj_merged_embed = obj_merged_embed.clone()
                obj_merged_embed[unconditioning_objects] = obj_non_cond_embed[
                    unconditioning_objects
                ]

            obj_merged_embed = multiplex_state.mux(obj_merged_embed).sum(dim=1)
            maskmem_features = maskmem_features + obj_merged_embed[
                ..., None, None
            ].expand_as(maskmem_features)

        if maskmem_features.dim() == 5:
            maskmem_features = multiplex_state.demux(maskmem_features).contiguous()

        demuxed_pos_enc = []
        for pos_enc in maskmem_pos_enc:
            pos_enc_clone = pos_enc
            if pos_enc_clone is not None and pos_enc_clone.dim() == 5:
                pos_enc_clone = multiplex_state.demux(pos_enc_clone).contiguous()
            demuxed_pos_enc.append(pos_enc_clone)
        maskmem_pos_enc = demuxed_pos_enc

        return maskmem_features, maskmem_pos_enc

    def forward_tracking(
        self,
        backbone_out,
        input,
        return_dict=False,
        objects_to_interact: Optional[list[int]] = None,
    ):
        """Forward video tracking on each frame (and sample correction clicks)."""
        img_feats_already_computed = (
            "interactive" in backbone_out or "sam2_backbone_out" in backbone_out
        )
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            # - vision_masks are in B(HW) format, dtype=bool (False is valid, True is padding)
            backbone_features = self._prepare_backbone_features(backbone_out)

        # Starting the stage loop
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        # first process all the initial conditioning frames to encode them as memory,
        # and then conditioning on them to track the remaining frames
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]

        cond_frame_outputs: dict[int, StageOutput] = {}
        non_cond_frame_outputs: dict[int, StageOutput] = {}
        output_dict = {
            "cond_frame_outputs": cond_frame_outputs,
            "non_cond_frame_outputs": non_cond_frame_outputs,
        }

        multiplex_state = self.multiplex_controller.get_state(
            backbone_out["gt_masks_per_frame"][0].shape[0],
            device=backbone_out["gt_masks_per_frame"][0].device,
            dtype=torch.float,
            random=self.training,
        )

        for stage_id in processing_order:
            # Get the image features for the current frames
            img_ids = input.find_inputs[stage_id].img_ids
            # the image ids are for the entire batch
            assert all(
                [img_id == img_ids[0] for img_id in img_ids]
            )  # should be all the same
            # force this to have a batch size of 1
            img_ids = torch.tensor(
                [img_ids[0]], device=img_ids.device, dtype=img_ids.dtype
            )

            if img_feats_already_computed:
                # Retrieve image features according to img_ids (if they are already computed).
                current_image = input.img_batch.tensors[img_ids]
                current_backbone_features = {}
                for neck_k, neck_out in backbone_features.items():
                    current_backbone_features[neck_k] = {
                        "vision_feats": [
                            x[:, img_ids] for x in neck_out["vision_feats"]
                        ],
                        "vision_masks": [
                            x[img_ids] if x is not None else None
                            for x in neck_out["vision_masks"]
                        ],
                        "vision_pos_embeds": [
                            x[:, img_ids] for x in neck_out["vision_pos_embeds"]
                        ],
                        "feat_sizes": neck_out["feat_sizes"],
                    }
            else:
                # Otherwise, compute the image features on the fly for the given img_ids
                # (this might be used for evaluation on long videos to avoid backbone OOM).
                need_interactive_out = (stage_id in frames_to_add_correction_pt) or (
                    stage_id in init_cond_frames
                )
                (current_image, current_backbone_features) = (
                    self._prepare_backbone_features_per_frame(
                        input.img_batch,
                        img_ids,
                        need_interactive_out=need_interactive_out,
                        need_propagation_out=True,
                    )
                )

            # Get output masks based on this frame's prompts and previous memory
            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                backbone_features_interactive=current_backbone_features.get(
                    "interactive"
                ),
                backbone_features_propagation=current_backbone_features.get(
                    "sam2_backbone_out"
                ),
                image=current_image,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
                gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                output_dict=output_dict,
                num_frames=num_frames,
                multiplex_state=multiplex_state,
                objects_to_interact=objects_to_interact,
            )
            # Append the output, depending on whether it's a conditioning frame
            add_output_as_cond_frame = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond
                and stage_id in frames_to_add_correction_pt
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        output_dict["multiplex_state"] = multiplex_state

        if return_dict:
            return output_dict
        # turn `output_dict` into a list for loss function
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        # Make DDP happy with activation checkpointing by removing unused keys
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]

        return all_frame_outputs

    def _track_step_aux(
        self,
        *,
        frame_idx,
        is_init_cond_frame,
        backbone_features_interactive,
        backbone_features_propagation,
        image,
        point_inputs,
        mask_inputs,
        gt_masks,
        frames_to_add_correction_pt,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
        multiplex_state: MultiplexState,
        objects_to_interact: Optional[list[int]] = None,
        need_aux_output: bool = False,
    ) -> tuple[StageOutput, dict]:
        """
        There are four different modes that track_step might enter, based on the inputs
        1. Mask-as-output. This is when mask_inputs is not None.
           The input mask is returned directly. This case is for FA/VOS initialization.
        2. Propagation-only. This is when mask_inputs and point_inputs are empty.
           We propagate masks using the memory only. This case is for VOS propagation.
        3. Interaction-only. This is when mask_inputs is None, point_inputs is not None,
           and one of the followings is satisified:
           a) prev_sam_mask_logits is not None. In this case, we refine prev_sam_mask_logits
              with additional interactions, updating only the objects specified in objects_to_interact.
              objects_to_interact must not be None.
              This occurs when we refine the same frame with multiple point inputs iteratively.
           b) prev_sam_mask_logits is None, and is_init_cond_frame is True.
              This case is for initializing the first frame. All objects will have point inputs.
              This mostly happens during training/interactive eval.
        4. Propagation-and-interaction. This is when mask_inputs is None, point_inputs is not None,
           prev_sam_mask_logits is None, and objects_to_interact is not None.
           This is when we are propagating to a new frame that has point inputs (from previous interactions).
           This is more of an edge case that could happen in offline interactive eval.
           We first propagate the mask to the current frame, and then perform interaction on the selected
           objects. Finally, we replace the masks of the interacted objects in the propagated output
           with the masks from the interaction output.
        """
        current_out: StageOutput = {
            "conditioning_objects": set(),
            "point_inputs": point_inputs,
            "mask_inputs": mask_inputs,
        }

        mode = None
        if mask_inputs is not None:
            mode = "mask_as_output"
        elif point_inputs is None:
            mode = "propagation_only"
        elif point_inputs is not None:
            # Case 3a: Refining existing predictions
            if prev_sam_mask_logits is not None:
                assert objects_to_interact is not None, (
                    "objects_to_interact must be specified when refining with prev_sam_mask_logits"
                )
                mode = "interaction_only"
            # Case 3b: Initial conditioning frame
            elif is_init_cond_frame:
                mode = "interaction_only"
            # Case 4: Propagation then interaction
            elif objects_to_interact is not None and prev_sam_mask_logits is None:
                assert not self.training
                mode = "propagation_and_interaction"

        if mode is None:
            raise ValueError(
                f"Unable to determine tracking case. "
                f"mask_inputs={mask_inputs is not None}, "
                f"point_inputs={point_inputs is not None}, "
                f"prev_sam_mask_logits={prev_sam_mask_logits is not None}, "
                f"objects_to_interact={objects_to_interact}, "
                f"is_init_cond_frame={is_init_cond_frame}"
            )
        # partition the backbone features
        interactive_high_res_features = interactive_vision_feats = None
        interactive_feat_sizes = None
        if backbone_features_interactive is not None:
            interactive_vision_feats = backbone_features_interactive["vision_feats"]
            interactive_feat_sizes = backbone_features_interactive["feat_sizes"]

            # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
            if len(interactive_vision_feats) > 1:
                interactive_high_res_features = [
                    x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                    for x, s in zip(
                        interactive_vision_feats[:-1], interactive_feat_sizes[:-1]
                    )
                ]
        else:
            # cannot do point interaction without interactive features
            assert mode not in ["interaction_only", "propagation_and_interaction"]

        propagation_high_res_features = propagation_vision_feats = None
        propagation_vision_masks = None
        propagation_vision_pos_embeds = propagation_feat_sizes = None
        if backbone_features_propagation is not None:
            propagation_vision_feats = backbone_features_propagation["vision_feats"]
            propagation_vision_masks = backbone_features_propagation["vision_masks"]
            propagation_vision_pos_embeds = backbone_features_propagation[
                "vision_pos_embeds"
            ]
            propagation_feat_sizes = backbone_features_propagation["feat_sizes"]

            # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
            if len(propagation_vision_feats) > 1:
                propagation_high_res_features = [
                    x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                    for x, s in zip(
                        propagation_vision_feats[:-1], propagation_feat_sizes[:-1]
                    )
                ]
        else:
            # we can get away without propagation features if we are interacting and not encoding new memory
            assert mode not in ["propagation_only", "propagation_and_interaction"]
            assert not run_mem_encoder

        interactive_pix_feat = None
        if mode == "mask_as_output":
            # simple encoding
            assert self.use_mask_input_as_output_without_sam
            # pix_feat = interactive_vision_feats[-1].permute(1, 2, 0)
            # pix_feat = pix_feat.view(-1, self.hidden_dim, *interactive_feat_sizes[-1])
            # use no_mem_embed here as well to better align first-frame mask input vs point input
            interactive_pix_feat = self._get_interactive_pix_mem(
                interactive_vision_feats, interactive_feat_sizes
            )
            sam_outputs = self._use_mask_as_output(
                backbone_features=interactive_pix_feat,
                high_res_features=interactive_high_res_features,
                mask_inputs=mask_inputs,
                multiplex_state=multiplex_state,
            )
            # all the objects are conditional here
            current_out["conditioning_objects"].update(range(mask_inputs.shape[0]))
        else:
            # propagation, interaction, or both
            propagation_out = None
            if mode in ["propagation_only", "propagation_and_interaction"]:
                # gather the memory
                assert backbone_features_propagation is not None
                assert propagation_vision_feats is not None
                assert propagation_vision_masks is not None
                assert propagation_vision_pos_embeds is not None
                assert propagation_feat_sizes is not None
                pix_feat_with_mem = self._prepare_memory_conditioned_features(
                    frame_idx=frame_idx,
                    is_init_cond_frame=is_init_cond_frame,
                    current_vision_feats=propagation_vision_feats[-1:],
                    current_vision_masks=propagation_vision_masks[-1:],
                    current_vision_pos_embeds=propagation_vision_pos_embeds[-1:],
                    feat_sizes=propagation_feat_sizes[-1:],
                    output_dict=output_dict,
                    num_frames=num_frames,
                    track_in_reverse=track_in_reverse,
                    multiplex_state=multiplex_state,
                )

                # propagate the mask
                # this is the propagation step; do not consider point_inputs here
                multimask_output = self._use_multimask(
                    is_init_cond_frame, point_inputs=None
                )
                propagation_out = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    propagation_high_res_features=propagation_high_res_features,
                    multimask_output=multimask_output,
                    objects_to_interact=list(
                        range(multiplex_state.total_valid_entries)
                    ),
                    multiplex_state=multiplex_state,
                )

            interaction_out = None
            if mode in ["interaction_only", "propagation_and_interaction"]:
                assert backbone_features_interactive is not None
                assert interactive_vision_feats is not None
                assert interactive_feat_sizes is not None
                interactive_pix_feat = self._get_interactive_pix_mem(
                    interactive_vision_feats, interactive_feat_sizes
                )

                # apply SAM-style segmentation head
                # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
                # e.g. in demo where such logits come from earlier interaction instead of correction sampling
                # (in this case, the SAM mask decoder should have `self.iter_use_prev_mask_pred=True`, and
                # any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
                assert mask_inputs is None and point_inputs is not None
                if prev_sam_mask_logits is not None:
                    assert objects_to_interact is not None
                    assert self.iter_use_prev_mask_pred
                    assert mode != "propagation_and_interaction"
                    mask_inputs = prev_sam_mask_logits[objects_to_interact]
                elif mode == "propagation_and_interaction":
                    # use propagated masks as mask input
                    assert objects_to_interact is not None
                    assert propagation_out is not None
                    mask_inputs = propagation_out["low_res_masks"][objects_to_interact]

                if objects_to_interact is not None:
                    assert point_inputs["point_coords"].shape[0] == len(
                        objects_to_interact
                    )
                    assert point_inputs["point_labels"].shape[0] == len(
                        objects_to_interact
                    )

                multimask_output = self._use_multimask(
                    is_init_cond_frame, point_inputs=point_inputs
                )
                interaction_out = self._forward_sam_heads(
                    backbone_features=interactive_pix_feat,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    interactive_high_res_features=interactive_high_res_features,
                    multimask_output=multimask_output,
                    objects_to_interact=(
                        objects_to_interact
                        if objects_to_interact is not None
                        else list(range(multiplex_state.total_valid_entries))
                    ),
                    multiplex_state=multiplex_state,
                )
                if objects_to_interact is None:
                    current_out["conditioning_objects"].update(
                        multiplex_state.get_all_valid_object_idx()
                    )
                else:
                    current_out["conditioning_objects"].update(objects_to_interact)

            if propagation_out is None and interaction_out is not None:
                sam_outputs = interaction_out
            elif interaction_out is None and propagation_out is not None:
                sam_outputs = propagation_out
            else:
                # merge the output
                assert propagation_out is not None and interaction_out is not None
                keys_to_merge = [
                    "low_res_multimasks",
                    "high_res_multimasks",
                    "low_res_masks",
                    "high_res_masks",
                    "ious",
                    "object_score_logits",
                    "obj_ptr",
                ]
                for k in keys_to_merge:
                    src = interaction_out[k]
                    dst = propagation_out[k]
                    # Align dtype for floating tensors before indexed assignment
                    if torch.is_tensor(src) and torch.is_tensor(dst):
                        if torch.is_floating_point(src) and src.dtype != dst.dtype:
                            src = src.to(dtype=dst.dtype)
                    propagation_out[k][objects_to_interact] = src
                sam_outputs = propagation_out

        low_res_multimasks = sam_outputs["low_res_multimasks"]
        high_res_multimasks = sam_outputs["high_res_multimasks"]
        ious = sam_outputs["ious"]
        low_res_masks = sam_outputs["low_res_masks"]
        high_res_masks = sam_outputs["high_res_masks"]
        object_score_logits = sam_outputs["object_score_logits"]

        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]

        if self.use_obj_ptrs_in_encoder:
            obj_ptr = sam_outputs["obj_ptr"]

        # Optionally, sample correction points iteratively to correct the mask
        if frame_idx in frames_to_add_correction_pt:
            assert gt_masks is not None
            assert interactive_vision_feats is not None
            assert interactive_feat_sizes is not None
            all_pred_masks = [low_res_masks]
            all_pred_high_res_masks = [high_res_masks]
            all_pred_multimasks = [low_res_multimasks]
            all_pred_high_res_multimasks = [high_res_multimasks]
            all_pred_ious = [ious]
            all_point_inputs = [point_inputs]
            all_object_score_logits = [object_score_logits]

            # select a subset of objects to interact with
            if self.training:
                assert objects_to_interact is None

                interact_with_all_objects = (
                    self.rng.random() < self.prob_correct_all_objects_for_train
                ) or (
                    self.force_correct_all_for_conditional_inputs and is_init_cond_frame
                )

                if interact_with_all_objects:
                    num_objects_to_correct = gt_masks.shape[0]
                elif self.rand_objects_to_correct_for_train:
                    num_objects_to_correct = self.rng2.integers(
                        1,
                        int(
                            gt_masks.shape[0]
                            * self.ratio_of_objects_to_correct_for_train
                        )
                        + 1,
                    )
                else:
                    num_objects_to_correct = max(
                        1,
                        int(
                            gt_masks.shape[0]
                            * self.ratio_of_objects_to_correct_for_train
                        ),
                    )

                objects_to_interact = self.rng2.choice(
                    range(gt_masks.shape[0]),
                    size=num_objects_to_correct,
                    replace=False,
                ).tolist()

                if point_inputs is not None:
                    # don't modify the point inputs in-place
                    point_inputs = {
                        "point_coords": point_inputs["point_coords"][
                            objects_to_interact
                        ],
                        "point_labels": point_inputs["point_labels"][
                            objects_to_interact
                        ],
                    }
            else:
                assert objects_to_interact is not None
                # the point inputs should have been preselected, i.e., the following assertion should hold

            if point_inputs is not None:
                assert point_inputs["point_coords"].shape[0] == len(objects_to_interact)
                assert point_inputs["point_labels"].shape[0] == len(objects_to_interact)

            for _ in range(self.num_correction_pt_per_frame):
                # sample a new point from the error between prediction and ground-truth
                # (with a small probability, directly sample from GT masks instead of errors)
                if self.training and self.prob_to_sample_from_gt_for_train > 0:
                    sample_from_gt = (
                        self.rng.random() < self.prob_to_sample_from_gt_for_train
                    )
                else:
                    sample_from_gt = False
                # if `pred_for_new_pt` is None, only GT masks will be used for point sampling
                pred_for_new_pt = None if sample_from_gt else (high_res_masks > 0)
                new_points, new_labels = get_next_point(
                    gt_masks=gt_masks[objects_to_interact],
                    pred_masks=(
                        pred_for_new_pt[objects_to_interact]
                        if pred_for_new_pt is not None
                        else None
                    ),
                    method="uniform" if self.training else self.pt_sampling_for_eval,
                )
                point_inputs = concat_points(point_inputs, new_points, new_labels)
                assert low_res_masks.shape[0] > max(objects_to_interact), (
                    f"interacting {objects_to_interact} in {low_res_masks.shape}?"
                )
                if self.iter_use_prev_mask_pred:
                    # Feed the mask logits of the previous SAM outputs in the next SAM decoder step.
                    # For tracking, this means that when the user adds a correction click, we also feed
                    # the tracking output mask logits along with the click as input to the SAM decoder.
                    mask_inputs = low_res_masks[objects_to_interact]
                multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
                pix_feat_with_mem = self._get_interactive_pix_mem(
                    interactive_vision_feats, interactive_feat_sizes
                )
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    interactive_high_res_features=interactive_high_res_features,
                    propagation_high_res_features=propagation_high_res_features,
                    multimask_output=multimask_output,
                    gt_masks=gt_masks,
                    objects_to_interact=objects_to_interact,
                    multiplex_state=multiplex_state,
                )
                interact_low_res_multimasks = sam_outputs["low_res_multimasks"]
                interact_high_res_multimasks = sam_outputs["high_res_multimasks"]
                interact_ious = sam_outputs["ious"]
                interact_low_res_masks = sam_outputs["low_res_masks"]
                interact_high_res_masks = sam_outputs["high_res_masks"]
                interact_object_score_logits = sam_outputs["object_score_logits"]
                if self.use_obj_ptrs_in_encoder:
                    interact_obj_ptr = sam_outputs["obj_ptr"]

                if self.training:
                    # combine the masks from the interacted and non-interacted objects
                    low_res_masks = low_res_masks.clone()
                    high_res_masks = high_res_masks.clone()
                    low_res_multimasks = low_res_multimasks.clone()
                    high_res_multimasks = high_res_multimasks.clone()
                    ious = ious.clone()
                    object_score_logits = object_score_logits.clone()
                    obj_ptr = obj_ptr.clone() if self.use_obj_ptrs_in_encoder else None

                # Update masks for the interacted objects
                if (
                    torch.is_floating_point(interact_low_res_masks)
                    and interact_low_res_masks.dtype != low_res_masks.dtype
                ):
                    interact_low_res_masks = interact_low_res_masks.to(
                        dtype=low_res_masks.dtype
                    )
                low_res_masks[objects_to_interact] = interact_low_res_masks
                if (
                    torch.is_floating_point(interact_high_res_masks)
                    and interact_high_res_masks.dtype != high_res_masks.dtype
                ):
                    interact_high_res_masks = interact_high_res_masks.to(
                        dtype=high_res_masks.dtype
                    )
                high_res_masks[objects_to_interact] = interact_high_res_masks
                if (
                    torch.is_floating_point(interact_low_res_multimasks)
                    and interact_low_res_multimasks.dtype != low_res_multimasks.dtype
                ):
                    interact_low_res_multimasks = interact_low_res_multimasks.to(
                        dtype=low_res_multimasks.dtype
                    )
                low_res_multimasks[objects_to_interact] = interact_low_res_multimasks
                if (
                    torch.is_floating_point(interact_high_res_multimasks)
                    and interact_high_res_multimasks.dtype != high_res_multimasks.dtype
                ):
                    interact_high_res_multimasks = interact_high_res_multimasks.to(
                        dtype=high_res_multimasks.dtype
                    )
                high_res_multimasks[objects_to_interact] = interact_high_res_multimasks
                if (
                    torch.is_floating_point(interact_ious)
                    and interact_ious.dtype != ious.dtype
                ):
                    interact_ious = interact_ious.to(dtype=ious.dtype)
                ious[objects_to_interact] = interact_ious
                if (
                    torch.is_floating_point(interact_object_score_logits)
                    and interact_object_score_logits.dtype != object_score_logits.dtype
                ):
                    interact_object_score_logits = interact_object_score_logits.to(
                        dtype=object_score_logits.dtype
                    )
                object_score_logits[objects_to_interact] = interact_object_score_logits
                if self.use_obj_ptrs_in_encoder:
                    obj_ptr[objects_to_interact] = interact_obj_ptr

                all_pred_masks.append(low_res_masks)
                all_pred_high_res_masks.append(high_res_masks)
                all_pred_multimasks.append(low_res_multimasks)
                all_pred_high_res_multimasks.append(high_res_multimasks)
                all_pred_ious.append(ious)
                all_point_inputs.append(point_inputs)
                all_object_score_logits.append(object_score_logits)

            # Concatenate the masks along channel (to compute losses on all of them,
            # using `onevision.losses.loss_fns.MultiStepIteractiveMasks`)
            current_out["multistep_pred_masks"] = torch.cat(all_pred_masks, dim=1)
            current_out["multistep_pred_masks_high_res"] = torch.cat(
                all_pred_high_res_masks, dim=1
            )
            current_out["multistep_pred_multimasks"] = all_pred_multimasks
            current_out["multistep_pred_multimasks_high_res"] = (
                all_pred_high_res_multimasks
            )
            current_out["multistep_pred_ious"] = all_pred_ious
            current_out["multistep_point_inputs"] = all_point_inputs
            current_out["multistep_object_score_logits"] = all_object_score_logits

            if self.add_all_frames_to_correct_as_cond:
                if objects_to_interact is None:
                    current_out["conditioning_objects"].update(
                        multiplex_state.get_all_valid_object_idx()
                    )
                else:
                    current_out["conditioning_objects"].update(set(objects_to_interact))

        # Use the final prediction (after all correction steps for output and eval)
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        if self.use_obj_ptrs_in_encoder:
            # similar to spatial memory, the object pointers are stored with multiplex
            current_out["obj_ptr"] = multiplex_state.mux(obj_ptr)
        if self.use_memory_selection:
            current_out["object_score_logits"] = object_score_logits
            iou_score = current_out["multistep_pred_ious"][-1].max(-1)[0]
            current_out["iou_score"] = iou_score
            current_out["eff_iou_score"] = self.cal_mem_score(
                object_score_logits, iou_score
            )
        # we need to return this for encoding new masks in the dynamic mode
        current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        # (note that `self.num_maskmem == 0` is primarily used for reproducing SAM on
        # images, in which case we'll just skip memory encoder to save compute).
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                image=image,
                current_vision_feats=propagation_vision_feats,
                feat_sizes=propagation_feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
                conditioning_objects=current_out["conditioning_objects"],
                multiplex_state=multiplex_state,
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc

        if self.save_image_features:
            current_out["image_features"] = propagation_vision_feats[-1]
            current_out["image_pos_enc"] = propagation_vision_pos_embeds[-1]

        # this is to avoid recomputing some of these features for add_new_masks_to_existing_state
        aux_output = {}
        if need_aux_output:
            if interactive_pix_feat is None:
                interactive_pix_feat = self._get_interactive_pix_mem(
                    interactive_vision_feats, interactive_feat_sizes
                )
            aux_output["interactive_pix_feat"] = interactive_pix_feat
            aux_output["interactive_high_res_features"] = interactive_high_res_features
            aux_output["propagation_vision_feats"] = propagation_vision_feats
            aux_output["propagation_feat_sizes"] = propagation_feat_sizes

        return current_out, aux_output

    def _trim_output_and_memory(
        self,
        frame_idx: int,
        output_dict: dict[str, dict[int, StageOutput]],
        current_out: StageOutput,
        memory_encoder_was_used: bool,
    ) -> StageOutput:
        # Optionally, offload the outputs to CPU memory during evaluation to avoid
        # GPU OOM on very long videos or very large resolution or too many objects
        if self.offload_output_to_cpu_for_eval and not self.training:
            # Here we only keep those keys needed for evaluation to get a compact output
            trimmed_out: StageOutput = {
                "conditioning_objects": current_out["conditioning_objects"],
                "pred_masks": current_out["pred_masks"].cpu(),
                "pred_masks_high_res": current_out["pred_masks_high_res"].cpu(),
                # other items for evaluation (these are small tensors so we keep them on GPU)
                "object_score_logits": current_out["object_score_logits"],
                "multistep_point_inputs": current_out["multistep_point_inputs"],
            }
            if self.use_obj_ptrs_in_encoder:
                trimmed_out["obj_ptr"] = current_out["obj_ptr"]
            if memory_encoder_was_used and self.num_maskmem > 0:
                trimmed_out["maskmem_features"] = current_out["maskmem_features"].cpu()
                trimmed_out["maskmem_pos_enc"] = [
                    x.cpu() for x in current_out["maskmem_pos_enc"]
                ]
            if self.save_image_features:
                trimmed_out["image_features"] = current_out["image_features"].cpu()
                trimmed_out["image_pos_enc"] = current_out["image_pos_enc"].cpu()
            current_out = trimmed_out

        # Optionally, trim the output of past non-conditioning frame (r * num_maskmem frames
        # before the current frame) during evaluation. This is intended to save GPU or CPU
        # memory for semi-supervised VOS eval, where only the first frame receives prompts.
        def _trim_past_out(
            past_out: StageOutput, current_out: StageOutput
        ) -> Optional[StageOutput]:
            if past_out is None:
                return None
            trimmed_past_out: StageOutput = {
                "conditioning_objects": past_out["conditioning_objects"],
                "pred_masks": past_out["pred_masks"],
                "object_score_logits": past_out["object_score_logits"],
                # Why would this be current_out?
                # "multistep_point_inputs": current_out["multistep_point_inputs"],
                "multistep_point_inputs": past_out["multistep_point_inputs"],
            }
            if self.use_obj_ptrs_in_encoder:
                trimmed_past_out["obj_ptr"] = past_out["obj_ptr"]
            return trimmed_past_out

        if self.trim_past_non_cond_mem_for_eval and not self.training:
            r = self.memory_temporal_stride_for_eval
            past_frame_idx = frame_idx - r * self.num_maskmem
            past_out = output_dict["non_cond_frame_outputs"].get(past_frame_idx, None)

            if past_out is not None:
                if (
                    self.use_memory_selection
                    and past_out.get("eff_iou_score", 0) < self.mf_threshold
                ) or not self.use_memory_selection:
                    output_dict["non_cond_frame_outputs"][past_frame_idx] = (
                        _trim_past_out(past_out, current_out)
                    )

            if (
                self.use_memory_selection and not self.offload_output_to_cpu_for_eval
            ):  # design for memory selection, trim too old frames to save memory
                far_old_frame_idx = frame_idx - 20 * self.max_obj_ptrs_in_encoder
                past_out = output_dict["non_cond_frame_outputs"].get(
                    far_old_frame_idx, None
                )
                if past_out is not None:
                    output_dict["non_cond_frame_outputs"][far_old_frame_idx] = (
                        _trim_past_out(past_out, current_out)
                    )

        return current_out

    def track_step(
        self,
        *,
        frame_idx,
        is_init_cond_frame,
        backbone_features_interactive,
        backbone_features_propagation,
        image,
        point_inputs,
        mask_inputs,
        gt_masks,
        frames_to_add_correction_pt,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
        multiplex_state: MultiplexState,
        # The list of object idx that point_inputs correspond to; only this set of objects will
        # be interacted with in the correction stage
        objects_to_interact: Optional[list[int]] = None,
    ) -> StageOutput:
        current_out, _ = self._track_step_aux(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            backbone_features_interactive=backbone_features_interactive,
            backbone_features_propagation=backbone_features_propagation,
            image=image,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            gt_masks=gt_masks,
            frames_to_add_correction_pt=frames_to_add_correction_pt,
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
            multiplex_state=multiplex_state,
            objects_to_interact=objects_to_interact,
            need_aux_output=False,
        )
        current_out = self._trim_output_and_memory(
            frame_idx, output_dict, current_out, memory_encoder_was_used=run_mem_encoder
        )

        return current_out

    def back_convert(self, targets):
        """To be compatible with SetCriterionAPI losses (mask loss only)."""
        batched_targets = {}
        batched_targets["num_boxes"] = targets.num_boxes
        batched_targets["masks"] = targets.segments
        batched_targets["is_valid_mask"] = targets.is_valid_segment
        return batched_targets

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
            and self.num_multimask_outputs > 0
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks

    def _compile_all_components(self):
        """Compile all model components for faster inference."""
        # a larger cache size to hold varying number of shapes for torch.compile
        # see https://github.com/pytorch/pytorch/blob/v2.5.1/torch/_dynamo/config.py#L42-L49
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 2048

        logging.info("Compiling all components. First time may be very slow.")

        self.maskmem_backbone.forward = torch.compile(
            self.maskmem_backbone.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
        self.transformer.encoder.forward = torch.compile(
            self.transformer.encoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=True,  # Num. of memories varies
        )
        # We disable compilation of sam_prompt_encoder as it sometimes gives a large accuracy regression,
        # especially when sam_mask_prompt (previous mask logits) is not None
        # self.sam_prompt_encoder.forward = torch.compile(
        #     self.sam_prompt_encoder.forward,
        #     mode="max-autotune",
        #     fullgraph=True,
        #     dynamic=False,  # Accuracy regression on True
        # )
        self.sam_mask_decoder.forward = torch.compile(
            self.sam_mask_decoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,  # Accuracy regression on True
        )

    def _maybe_clone(self, x):
        """Clone a tensor if and only if `self.compile_all_components` is True."""
        return x.clone() if self.compile_all_components else x

    def get_propagation_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.image_pe_layer(
            (self.sam_image_embedding_size, self.sam_image_embedding_size)
        ).unsqueeze(0)

    def cal_mem_score(self, object_score_logits, iou_score):
        object_score_norm = torch.where(
            object_score_logits > 0,
            object_score_logits.sigmoid() * 2 - 1,  # rescale to [0, 1]
            torch.zeros_like(object_score_logits),
        )
        score_per_frame = (object_score_norm * iou_score).mean()
        return score_per_frame

    def frame_filter(self, output_dict, track_in_reverse, frame_idx, num_frames, r):
        if (frame_idx == 0 and not track_in_reverse) or (
            frame_idx == num_frames - 1 and track_in_reverse
        ):
            return []

        max_num = min(
            num_frames, self.max_obj_ptrs_in_encoder
        )  # maximum number of pointer memory frames to consider

        if not track_in_reverse:
            start = frame_idx - 1
            end = 0
            step = -r
            must_include = frame_idx - 1
        else:
            start = frame_idx + 1
            end = num_frames
            step = r
            must_include = frame_idx + 1

        valid_indices = []
        for i in range(start, end, step):
            if (
                i not in output_dict["non_cond_frame_outputs"]
                or "eff_iou_score" not in output_dict["non_cond_frame_outputs"][i]
            ):
                continue

            score_per_frame = output_dict["non_cond_frame_outputs"][i]["eff_iou_score"]

            if score_per_frame > self.mf_threshold:  # threshold
                valid_indices.insert(0, i)

            if len(valid_indices) >= max_num - 1:
                break

        if must_include not in valid_indices:
            valid_indices.append(must_include)

        return valid_indices


def concat_points(old_point_inputs, new_points, new_labels):
    """Add new points and labels to previous point inputs (add at the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)

    return {"point_coords": points, "point_labels": labels}


def _append(
    d1: StageOutput, d2: SAMOutput, k1: str, k2: str, dim: int = 0, strict: bool = True
):
    if strict:
        assert k1 in d1, f"{k1} not found"
    else:
        if k1 not in d1:
            return

    d1[k1] = torch.cat([d1[k1], d2[k2]], dim=dim)


def _merge(
    d1: StageOutput,
    d2: SAMOutput,
    k1: str,
    k2: str,
    d2_idx: list[int],
    strict: bool = True,
):
    if strict:
        assert k1 in d1, f"{k1} not found"
    else:
        if k1 not in d1:
            return
    d1[k1][d2_idx] = d2[k2].to(dtype=d1[k1].dtype)


class VideoTrackingDynamicMultiplex(VideoTrackingMultiplex):
    def __init__(
        self,
        enable_dynamic_training: bool = True,  # Allows the number of objects to increase across frames during training
        rand_num_transition_points: bool = True,  # Randomizes the number of transition points
        max_num_transition_points: int = 3,  # Maximum number of transition points
        add_all_transition_frames_as_cond: bool = True,
        max_trans_frames_in_attn: int = 4,
        is_dynamic_model: bool = True,  # Overrides the default
        is_dynamic_vos_evaluation: bool = False,  # For datasets like YouTubeVOS which have new objects
        **kwargs,
    ):
        super().__init__(is_dynamic_model=is_dynamic_model, **kwargs)

        self.enable_dynamic_training = enable_dynamic_training
        self.rand_num_transition_points = rand_num_transition_points
        self.max_num_transition_points = max_num_transition_points

        self.add_all_transition_frames_as_cond = add_all_transition_frames_as_cond
        self.max_trans_frames_in_attn = max_trans_frames_in_attn
        self.is_dynamic_vos_evaluation = is_dynamic_vos_evaluation

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        Prepare input mask, point or box prompts. Optionally, we allow tracking from
        a custom `start_frame_idx` to the end of the video (for evaluation purposes).
        """

        """
        This function, in addition to the prompt preparation done in the parent class, preprocesses the
        masks and pre-computes visibility/validity attributes necessary for training with dynamic bucketing.

        **Data**
        We use a modified dataset class and a modified collate_fn such that:
        1. The mask for an object is loaded if it is visible (area>0) on any of the loaded frames
        2. A "visible_objects_per_frame" attribute is computed, which contains the set of objects with area>0 on each frame

        Here, we use [] to denote a set of objects; i.e., object A and B are represented as [A, B].
        Consider the masks given by the dataloader in an arbitrary yet deterministic order.
        That is, [2, 3] can appear on the first frame, and [1, 2, 3, 17] can appear on the second frame.

        This is incompatible with the object addition implementation, since we assume new objects are appended, not inserted.
        Thus, we compute object_appearance_order which sorts the object idx using the frame at which they appear
        (conditional frames always appear first). For objects that appear on the same frame, we shuffle them as augmentation.
        We also reorder the ground-truth masks used for supervision.

        **Causal supervision**
        Since not all objects appear on the first frame, we should not supervise on the objects that the model has no knowledge of yet.
        Thus, we keep track of the set of objects that have been introduced, and the frame at which that happens.
        We compute valid_idx_per_frame (and correspondingly trim the ground-truth) to enforce reasonable supervisions.

        **Transition points**
        Transition points are non-initial-conditioning frames that introduce new objects. We uniformly sample some frames
        to be candidates for transition points, and use them if they actually introduce new objects compared to the last seen
        conditional frame/transition point.
        Transitions do not always happen when an object first becomes visible, because our (initial) sampling is agnostic to visibility.
        This is intended, as new objects do not always get detected immediately in the dense tracking setting.
        """

        # First, prepare the prompt inputs following the parent class
        backbone_out = super()._prepare_prompt_inputs_meta(
            backbone_out, input, start_frame_idx=start_frame_idx
        )

        num_frames = backbone_out["num_frames"]
        gt_masks_per_frame = backbone_out["gt_masks_per_frame"]

        if self.training or self.is_dynamic_vos_evaluation:
            visible_objects_per_frame: dict[int, set[int]] = (
                input.visible_objects_per_frame
            )
        else:
            visible_objects_per_frame: dict[int, set[int]] = {
                stage_id: set(range(gt_masks_per_frame[stage_id].shape[0]))
                for stage_id in range(num_frames)
            }

        # If we have more than one conditioning frame,
        # all visible objects on any of the conditioning frames become valid for all frames
        init_cond_frames: list[int] = backbone_out["init_cond_frames"]
        init_cond_frames = sorted(init_cond_frames)
        frames_not_in_init_cond: list[int] = backbone_out["frames_not_in_init_cond"]

        # Rare case: the data guard might fail and we could have an empty first frame.
        # In this case, we track an empty object.
        if len(visible_objects_per_frame[start_frame_idx]) == 0:
            if self.training:
                logging.warning("Empty first frame, tracking an empty object")
                visible_objects_per_frame[start_frame_idx] = {0}
                # set the GT mask for this object to be all zeros
                for stage_id in range(num_frames):
                    gt_masks_per_frame[stage_id][0] = torch.zeros_like(
                        gt_masks_per_frame[stage_id][0]
                    )
            else:
                # During evaluation, this should only happen for YouTubeVOS.
                # We will skip the frames before the first conditional frame.
                assert self.is_dynamic_vos_evaluation, (
                    f"{visible_objects_per_frame=} invalid"
                )
                assert len(init_cond_frames) == 1
                for stage_id in range(start_frame_idx, num_frames):
                    if len(visible_objects_per_frame[stage_id]) > 0:
                        init_cond_frames = [stage_id]
                        break
                for i in range(
                    init_cond_frames[0] + 1
                ):  # also remove init_cond_frames[0]
                    if i in frames_not_in_init_cond:
                        frames_not_in_init_cond.remove(i)

        backbone_out["init_cond_frames"] = init_cond_frames

        # The object idx in valid_idx_per_frame should be in sequential order.
        # We will first reshuffle the objects using object_appearance_order,
        # and then index via valid_idx_per_frame.
        valid_idx_per_frame: dict[int, list[int]] = {}
        # Importantly, we cannot simply use valid_idx_per_frame[stage_id-1] because it might be a conditional frame.
        valid_idx_prior_to_each_transition: dict[int, list[int]] = {}
        new_idx_per_transition: dict[int, list[int]] = {}

        if self.training and self.enable_dynamic_training:
            # Select the number of transition points
            if self.rand_num_transition_points:
                # Randomly select 1 to `max_num_transition_points` transition points
                num_transition_points = self.rng.integers(
                    1, self.max_num_transition_points, endpoint=True
                )
            else:
                num_transition_points = self.max_num_transition_points

            available_transition_points = frames_not_in_init_cond
            num_transition_points = min(
                num_transition_points, len(available_transition_points)
            )
            # num_transition_points can differ between GPUs so we use rng2
            transition_points = self.rng2.choice(
                available_transition_points, num_transition_points, replace=False
            ).tolist()
            transition_points = sorted(transition_points)

            # Filter for the transition points that do introduce new objects
            filtered_transition_points = []
            objects_seen = set()
            for stage_id in init_cond_frames:
                objects_seen.update(visible_objects_per_frame[stage_id])

            for stage_id in range(start_frame_idx, num_frames):
                if stage_id in transition_points:
                    new_objects_seen = (
                        visible_objects_per_frame[stage_id] - objects_seen
                    )
                    if len(new_objects_seen) > 0:
                        filtered_transition_points.append(stage_id)
                        objects_seen.update(new_objects_seen)
                        new_idx_per_transition[stage_id] = list(new_objects_seen)
            transition_points = filtered_transition_points

            # Create appearance-based object ordering with randomization
            init_objects = set()
            for stage_id in init_cond_frames:
                init_objects.update(visible_objects_per_frame[stage_id])
            init_objects = list(init_objects)
            self.rng2.shuffle(init_objects)

            object_appearance_order = init_objects.copy()
            valid_idx_per_frame[start_frame_idx] = list(range(len(init_objects)))
            for stage_id in range(start_frame_idx + 1, num_frames):
                if stage_id in transition_points:
                    # When objects appear at a transition point, we add them to the end of the list
                    stage_objects = new_idx_per_transition[stage_id].copy()
                    self.rng2.shuffle(stage_objects)
                    valid_idx_prior_to_each_transition[stage_id] = list(
                        range(len(object_appearance_order))
                    )
                    new_idx_per_transition[stage_id] = list(
                        range(
                            len(object_appearance_order),
                            len(object_appearance_order) + len(stage_objects),
                        )
                    )
                    object_appearance_order.extend(stage_objects)

                # Update the valid objects at this frame
                if stage_id in init_cond_frames:
                    # Note: on any non-first init cond frame, the number of valid objects
                    # might be fewer than the previous frame because we always process the init cond frames first.
                    # For example, if [1, 2, 4] are visible on the two init cond frames (e.g., frame 0 and frame 5),
                    # and object 3 appears on frame 4 (as a transition point), object 3 would not be considered valid on frame 5.
                    # This should not break any processing steps or affect correctness (since invalid objects are marked as floating).
                    valid_idx_per_frame[stage_id] = valid_idx_per_frame[
                        start_frame_idx
                    ].copy()
                elif stage_id in frames_not_in_init_cond:
                    valid_idx_per_frame[stage_id] = list(
                        range(len(object_appearance_order))
                    )
                else:
                    raise ValueError(
                        f"Unexpected {stage_id=}? {init_cond_frames=} {frames_not_in_init_cond=} {transition_points=}"
                    )
        elif self.is_dynamic_vos_evaluation and not self.training:
            # In dynamic VOS evaluation, we find the transition points manually.
            # Each object should appear on exactly one frame.
            # NOTE: The new release of YouTubeVOS apparently did not enforce this.
            # We are enforcing it here.

            # Find first appearance of each object
            object_appearance_order: list[int] = []
            object_appear_at_stage: dict[int, int] = {}
            transition_points: list[int] = []
            stage_to_new_objects: dict[int, list[int]] = defaultdict(list)
            for stage_id in range(start_frame_idx, num_frames):
                visible_objects = sorted(list(visible_objects_per_frame[stage_id]))
                for obj_id in visible_objects:
                    if obj_id in object_appear_at_stage:
                        continue  # skip seen objects

                    object_appear_at_stage[obj_id] = stage_id
                    object_appearance_order.append(obj_id)
                    stage_to_new_objects[stage_id].append(obj_id)
                    if stage_id not in init_cond_frames:
                        transition_points.append(stage_id)

            # Track cumulative object count
            objects_seen_so_far = []
            for stage_id in range(start_frame_idx, num_frames):
                if stage_id in transition_points:
                    # New objects appear at this frame
                    new_objects = stage_to_new_objects[stage_id]
                    num_objects_before = len(objects_seen_so_far)

                    # Record which objects were valid before this transition
                    valid_idx_prior_to_each_transition[stage_id] = list(
                        range(num_objects_before)
                    )
                    # Record the indices of new objects
                    new_idx_per_transition[stage_id] = list(
                        range(num_objects_before, num_objects_before + len(new_objects))
                    )

                    objects_seen_so_far.extend(new_objects)

                # Set valid objects for this frame
                if stage_id in init_cond_frames:
                    # For init cond frames, only the initial objects are valid
                    valid_idx_per_frame[stage_id] = list(
                        range(len(stage_to_new_objects[stage_id]))
                    )
                    objects_seen_so_far.extend(stage_to_new_objects[stage_id])
                else:
                    # For other frames, all objects seen so far are valid
                    valid_idx_per_frame[stage_id] = list(
                        range(len(objects_seen_so_far))
                    )

        else:
            # Use no transition points when dynamic training is disabled
            transition_points = []
            visible_objects_on_first_frame = sorted(
                list(visible_objects_per_frame[start_frame_idx])
            )
            # Since visible_objects_on_first_frame might not be consecutive
            object_orderings = list(range(len(visible_objects_on_first_frame)))
            # Use the original order for evaluation
            object_appearance_order = visible_objects_on_first_frame.copy()
            for stage_id in range(start_frame_idx, num_frames):
                valid_idx_per_frame[stage_id] = object_orderings.copy()

        # Apply the appearance-based mapping to ground-truth masks
        for stage_id in range(start_frame_idx, num_frames):
            gt_masks_per_frame[stage_id] = gt_masks_per_frame[stage_id][
                object_appearance_order
            ][valid_idx_per_frame[stage_id]]

        # We also want to apply this change in-place to the input, such that loss can be computed correctly.
        # For targets.segments, we need to delay the object introduction by 1 frame.
        # At transition points, use current frame's masks but only for objects that existed in the previous frame.
        # This allows us to compute the loss on the existing objects and not on the newly added objects.
        for stage_id, targets in enumerate(input.find_targets):
            if stage_id in transition_points:
                # At transition points, use current frame's masks but only keep objects from the previous frame
                prev_objects = valid_idx_prior_to_each_transition[stage_id]
                # Only keep masks for objects that existed in the previous frame
                targets.segments = gt_masks_per_frame[stage_id][prev_objects].squeeze(1)
            else:
                targets.segments = gt_masks_per_frame[stage_id].squeeze(1)
            # Ensure that we are averaging the loss correctly.
            # Although this is called num_boxes, it actually stores an array of ones with length=number of objects in the VOS setting.
            targets.num_boxes = targets.num_boxes[: targets.segments.shape[0]]

        backbone_out["valid_idx_per_frame"] = valid_idx_per_frame
        backbone_out["new_idx_per_transition"] = new_idx_per_transition
        backbone_out["valid_objects_prior_to_each_transition"] = (
            valid_idx_prior_to_each_transition
        )
        backbone_out["transition_points"] = set(transition_points)
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        backbone_out["object_appearance_order"] = object_appearance_order

        backbone_out = self._prepare_conditional_frames(backbone_out)

        return backbone_out

    def add_new_masks_to_existing_state(
        self,
        *,
        interactive_pix_feat: torch.Tensor,
        interactive_high_res_features: list[torch.Tensor],
        propagation_vision_feats: Optional[
            list[torch.Tensor]
        ],  # needed when add_mask_to_memory=True
        propagation_feat_sizes: Optional[
            list[tuple[int, int]]
        ],  # needed when add_mask_to_memory=True
        new_masks: torch.Tensor,
        obj_idxs_in_mask: list[
            int
        ],  # len(obj_idxs_in_mask) == new_masks.shape[0]; object idx internal to this state
        obj_ids_in_mask: Optional[
            list[int]
        ],  # len(obj_ids_in_mask) == new_masks.shape[0]; global object ids
        prev_output: StageOutput,  # this state will be modified in-place
        multiplex_state: MultiplexState,
        add_mask_to_memory: bool = True,
        are_masks_from_pts: bool = False,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ) -> None:
        """
        Add new objects to an existing output/multiplex state.

        This function encodes the input masks as new masks and merges them with the existing state.
        The new object entries are always appended to the existing objects.

        This is because, in the dense tracking scenario, we should always propagate (existing state)
        to the current frame first before introducing the new objects.
        """
        assert self.use_mask_input_as_output_without_sam
        assert new_masks.shape[0] == len(obj_idxs_in_mask)

        num_new_objects = new_masks.shape[0]

        if obj_ids_in_mask is not None:
            assert len(obj_ids_in_mask) == num_new_objects

        if self.use_obj_ptrs_in_encoder:
            # demux the existing pointers before we change the multiplex state
            existing_pointers = multiplex_state.demux(prev_output["obj_ptr"])

        # Step 1: Inform the multiplex state that we are adding new objects
        new_object_idx = multiplex_state.find_next_batch_of_available_indices(
            num_objects=num_new_objects,
            allow_new_buckets=allow_new_buckets,
            prefer_new_buckets=prefer_new_buckets,
        )
        multiplex_state.add_objects(
            object_indices=new_object_idx,
            object_ids=obj_ids_in_mask,
            allow_new_buckets=allow_new_buckets,
            prefer_new_buckets=prefer_new_buckets,
        )

        # Step 2: Encode the incoming masks
        mask_output = self._use_mask_as_output(
            backbone_features=interactive_pix_feat,
            high_res_features=interactive_high_res_features,
            mask_inputs=new_masks,
            multiplex_state=multiplex_state,
            objects_in_mask=new_object_idx,
        )

        # Step 3: Merge the existing state with new encoded features
        # Handle resolution mismatch between propagation (e.g., 1008) and interactive (e.g., 288) features
        # Determine target resolution from interactive features (newly generated masks)
        interactive_resolution = mask_output["high_res_masks"].shape[-1]

        # Check if prev_output needs resolution adjustment
        if (
            "pred_masks_high_res" in prev_output
            and prev_output["pred_masks_high_res"] is not None
        ):
            existing_resolution = prev_output["pred_masks_high_res"].shape[-1]

            if existing_resolution != interactive_resolution:
                # Resize existing outputs to match interactive resolution
                # This happens when frame was bootstrapped with propagation features (1008)
                # but we're now adding interactive masks (288)
                prev_output["pred_masks_high_res"] = F.interpolate(
                    prev_output["pred_masks_high_res"],
                    size=(interactive_resolution, interactive_resolution),
                    mode="bilinear",
                    align_corners=False,
                )

        # Resize low_res_masks to match prev_output resolution
        h, w = prev_output["pred_masks"].shape[-2:]
        mask_output["low_res_masks"] = F.interpolate(
            mask_output["low_res_masks"],
            size=(h, w),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )

        _append(prev_output, mask_output, "pred_masks", "low_res_masks")
        _append(
            prev_output,
            mask_output,
            "pred_masks_high_res",
            "high_res_masks",
            strict=False,
        )
        _append(prev_output, mask_output, "object_score_logits", "object_score_logits")
        if self.use_memory_selection:
            mask_output["ious"] = mask_output["ious"].squeeze(-1)
            _append(prev_output, mask_output, "iou_score", "ious")

        # Merge the input masks
        if "input_masks" in prev_output:
            prev_output["input_masks"] = torch.cat(
                [prev_output["input_masks"], new_masks], dim=0
            )

        if self.use_obj_ptrs_in_encoder:
            # Merge the object pointers. Note that the pointers in SAMOutput are in the data space,
            # while those in StageOutput are in the mux space.
            new_pointers = mask_output["obj_ptr"].to(existing_pointers.dtype)
            combined_pointers = torch.cat([existing_pointers, new_pointers], dim=0)
            prev_output["obj_ptr"] = multiplex_state.mux(combined_pointers)

        # Step 4: Update the set of conditioning objects at this frame.
        prev_output["conditioning_objects"].update(new_object_idx)

        # Step 5: Re-encode the spatial memory if needed
        if add_mask_to_memory:
            assert (
                prev_output["pred_masks_high_res"].shape[0]
                == multiplex_state.total_valid_entries
            )
            # Add the new masks to the memory
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                image=None,
                current_vision_feats=propagation_vision_feats,
                feat_sizes=propagation_feat_sizes,
                pred_masks_high_res=prev_output["pred_masks_high_res"],
                object_score_logits=prev_output["object_score_logits"],
                conditioning_objects=prev_output["conditioning_objects"],
                is_mask_from_pts=are_masks_from_pts,
                multiplex_state=multiplex_state,
            )
            prev_output["maskmem_features"] = maskmem_features
            prev_output["maskmem_pos_enc"] = maskmem_pos_enc
            if self.save_image_features:
                # They should already be in the state; no modification is needed
                assert "image_features" in prev_output
                assert "image_pos_enc" in prev_output

    def recondition_masks_in_existing_state(
        self,
        *,
        interactive_pix_feat: torch.Tensor,
        interactive_high_res_features: list[torch.Tensor],
        propagation_vision_feats: Optional[
            list[torch.Tensor]
        ],  # needed when add_mask_to_memory=True
        propagation_feat_sizes: Optional[
            list[tuple[int, int]]
        ],  # needed when add_mask_to_memory=True
        new_masks: torch.Tensor,
        obj_idxs_in_mask: list[
            int
        ],  # len(obj_idxs_in_mask) == new_masks.shape[0]; object idx internal to this state
        obj_ids_in_mask: Optional[
            list[int]
        ],  # len(obj_ids_in_mask) == new_masks.shape[0]; global object ids
        prev_output: StageOutput,  # this state will be modified in-place
        multiplex_state: MultiplexState,
        add_mask_to_memory: bool = True,
    ) -> None:
        """
        Recondition existing objects in an existing output/multiplex state.

        This function encodes the input masks and merges them with the existing state.
        """
        assert self.use_mask_input_as_output_without_sam
        assert new_masks.shape[0] == len(obj_idxs_in_mask)

        num_new_objects = new_masks.shape[0]

        if obj_ids_in_mask is not None:
            assert len(obj_ids_in_mask) == num_new_objects

        if self.use_obj_ptrs_in_encoder:
            # demux the existing pointers before we change the multiplex state
            existing_pointers = multiplex_state.demux(prev_output["obj_ptr"])

        # Step 1: Encode the incoming masks
        mask_output = self._use_mask_as_output(
            backbone_features=interactive_pix_feat,
            high_res_features=interactive_high_res_features,
            mask_inputs=new_masks,
            multiplex_state=multiplex_state,
            objects_in_mask=obj_idxs_in_mask,
        )

        # Step 2: Merge the existing state with new encoded features
        # TODO: Remove this and fix the resolution mismatch
        h, w = prev_output["pred_masks"].shape[-2:]
        mask_output["low_res_masks"] = F.interpolate(
            mask_output["low_res_masks"],
            size=(h, w),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )

        _merge(
            prev_output, mask_output, "pred_masks", "low_res_masks", obj_idxs_in_mask
        )
        _merge(
            prev_output,
            mask_output,
            "pred_masks_high_res",
            "high_res_masks",
            obj_idxs_in_mask,
            strict=False,
        )
        _merge(
            prev_output,
            mask_output,
            "object_score_logits",
            "object_score_logits",
            obj_idxs_in_mask,
        )
        if self.use_memory_selection:
            mask_output["ious"] = mask_output["ious"].squeeze(-1)
            _merge(
                prev_output,
                mask_output,
                "iou_score",
                "ious",
                obj_idxs_in_mask,
            )

        # Merge the input masks
        if "input_masks" in prev_output:
            prev_output["input_masks"][obj_idxs_in_mask] = new_masks

        if self.use_obj_ptrs_in_encoder:
            # Merge the object pointers. Note that the pointers in SAMOutput are in the data space,
            # while those in StageOutput are in the mux space.
            new_pointers = mask_output["obj_ptr"].to(existing_pointers.dtype)
            existing_pointers[obj_idxs_in_mask] = new_pointers
            prev_output["obj_ptr"] = multiplex_state.mux(existing_pointers)

        # Step 3: Update the set of conditioning objects at this frame
        prev_output["conditioning_objects"].update(obj_idxs_in_mask)

        # Step 4: Re-encode the spatial memory if needed
        if add_mask_to_memory:
            assert (
                prev_output["pred_masks_high_res"].shape[0]
                == multiplex_state.total_valid_entries
            )
            # Add the new masks to the memory
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                image=None,
                current_vision_feats=propagation_vision_feats,
                feat_sizes=propagation_feat_sizes,
                pred_masks_high_res=prev_output["pred_masks_high_res"],
                object_score_logits=prev_output["object_score_logits"],
                conditioning_objects=prev_output["conditioning_objects"],
                is_mask_from_pts=False,
                multiplex_state=multiplex_state,
            )
            prev_output["maskmem_features"] = maskmem_features
            prev_output["maskmem_pos_enc"] = maskmem_pos_enc
            if self.save_image_features:
                # They should already be in the state; no modification is needed
                assert "image_features" in prev_output
                assert "image_pos_enc" in prev_output

    def track_step(
        self,
        *,
        frame_idx,
        is_init_cond_frame,
        backbone_features_interactive,
        backbone_features_propagation,
        image,
        point_inputs,
        mask_inputs,
        gt_masks,
        frames_to_add_correction_pt,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
        multiplex_state: MultiplexState,
        # The list of object IDs that point_inputs correspond to; only this set of objects will
        # be interacted with in the correction stage
        objects_to_interact: Optional[list[int]] = None,
        # The following parameters are specific to the dynamic multiplexing model
        new_object_masks: Optional[torch.Tensor] = None,
        new_object_idxs: Optional[list[int]] = None,
        new_object_ids: Optional[list[int]] = None,
        are_new_masks_from_pts: bool = False,
    ) -> StageOutput:
        # First, run track_step_aux.
        # This includes propagation, interaction, and correction.
        current_out, aux_out = self._track_step_aux(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            backbone_features_interactive=backbone_features_interactive,
            backbone_features_propagation=backbone_features_propagation,
            image=image,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            gt_masks=gt_masks,
            frames_to_add_correction_pt=frames_to_add_correction_pt,
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            run_mem_encoder=(run_mem_encoder and new_object_masks is None),
            prev_sam_mask_logits=prev_sam_mask_logits,
            multiplex_state=multiplex_state,
            objects_to_interact=objects_to_interact,
            need_aux_output=(new_object_masks is not None),
        )

        # If new masks are provided, merge them into the existing state
        if new_object_masks is not None:
            assert new_object_idxs is not None
            self.add_new_masks_to_existing_state(
                interactive_pix_feat=aux_out["interactive_pix_feat"],
                interactive_high_res_features=aux_out["interactive_high_res_features"],
                propagation_vision_feats=aux_out["propagation_vision_feats"],
                propagation_feat_sizes=aux_out["propagation_feat_sizes"],
                new_masks=new_object_masks,
                obj_idxs_in_mask=new_object_idxs,
                obj_ids_in_mask=new_object_ids,
                prev_output=current_out,
                multiplex_state=multiplex_state,
                add_mask_to_memory=run_mem_encoder,
                are_masks_from_pts=are_new_masks_from_pts,
            )

        # lastly, trim the output
        current_out = self._trim_output_and_memory(
            frame_idx=frame_idx,
            output_dict=output_dict,
            current_out=current_out,
            memory_encoder_was_used=run_mem_encoder,
        )

        return current_out

    def forward_tracking(
        self,
        backbone_out,
        input,
        return_dict=False,
        objects_to_interact: Optional[list[int]] = None,
    ):
        """Forward video tracking on each frame (and sample correction clicks)."""
        img_feats_already_computed = (
            "interactive" in backbone_out or "sam2_backbone_out" in backbone_out
        )
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            # - vision_masks are in B(HW) format, dtype=bool (False is valid, True is padding)
            backbone_features = self._prepare_backbone_features(backbone_out)

        # Starting the stage loop
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        # First process all the initial conditioning frames to encode them as memory,
        # And then condition on them to track the remaining frames
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]

        new_idx_per_transition = backbone_out["new_idx_per_transition"]
        valid_objects_prior_to_each_transition = backbone_out[
            "valid_objects_prior_to_each_transition"
        ]
        transition_points = backbone_out["transition_points"]

        cond_frame_outputs: dict[int, StageOutput] = {}
        non_cond_frame_outputs: dict[int, StageOutput] = {}
        output_dict = {
            "cond_frame_outputs": cond_frame_outputs,
            "non_cond_frame_outputs": non_cond_frame_outputs,
        }
        multiplex_state = self.multiplex_controller.get_state(
            backbone_out["gt_masks_per_frame"][processing_order[0]].shape[0],
            device=backbone_out["gt_masks_per_frame"][processing_order[0]].device,
            dtype=torch.float,
            random=self.training,
        )

        for stage_id in processing_order:
            # Get the image features for the current frame
            img_ids = input.find_inputs[stage_id].img_ids
            # The image ids are for the entire batch
            assert all(
                [img_id == img_ids[0] for img_id in img_ids]
            )  # should be all the same
            # force this to have a batch size of 1
            img_ids = torch.tensor(
                [img_ids[0]], device=img_ids.device, dtype=img_ids.dtype
            )

            if img_feats_already_computed:
                # Retrieve image features according to img_ids (if they are already computed).
                current_image = input.img_batch.tensors[img_ids]
                current_backbone_features = {}
                for neck_k, neck_out in backbone_features.items():
                    current_backbone_features[neck_k] = {
                        "vision_feats": [
                            x[:, img_ids] for x in neck_out["vision_feats"]
                        ],
                        "vision_masks": [
                            x[img_ids] if x is not None else None
                            for x in neck_out["vision_masks"]
                        ],
                        "vision_pos_embeds": [
                            x[:, img_ids] for x in neck_out["vision_pos_embeds"]
                        ],
                        "feat_sizes": neck_out["feat_sizes"],
                    }
            else:
                # Otherwise, compute the image features on the fly for the given img_ids
                # (this might be used for evaluation on long videos to avoid backbone OOM).
                need_interactive_out = (
                    (stage_id in frames_to_add_correction_pt)
                    or (stage_id in init_cond_frames)
                    or (stage_id in transition_points)
                )
                (current_image, current_backbone_features) = (
                    self._prepare_backbone_features_per_frame(
                        input.img_batch,
                        img_ids,
                        need_interactive_out=need_interactive_out,
                        need_propagation_out=True,
                    )
                )

            gt_masks = backbone_out["gt_masks_per_frame"].get(stage_id, None)
            if stage_id in transition_points:
                assert gt_masks is not None

                # Figure out new object masks / idxs
                new_object_idxs = new_idx_per_transition[stage_id]
                # Get the new object masks, ensure correct ordering
                assert sorted(new_object_idxs) == new_object_idxs
                assert new_object_idxs[0] == len(
                    valid_objects_prior_to_each_transition[stage_id]
                ), (
                    f"{new_object_idxs=}; {gt_masks.shape=}; {valid_objects_prior_to_each_transition[stage_id]=}"
                )
                assert new_object_idxs[-1] == (len(gt_masks) - 1), (
                    f"{new_object_idxs=}; {gt_masks.shape=}"
                )
                new_object_masks = gt_masks[new_object_idxs]

                # Remove the new objects from the gt masks
                gt_masks = gt_masks[: new_object_idxs[0]]
            else:
                new_object_masks = None
                new_object_idxs = None

            # Get output masks based on this frame's prompts and previous memory
            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                backbone_features_interactive=current_backbone_features.get(
                    "interactive"
                ),
                backbone_features_propagation=current_backbone_features.get(
                    "sam2_backbone_out"
                ),
                image=current_image,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
                gt_masks=gt_masks,
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                output_dict=output_dict,
                num_frames=num_frames,
                multiplex_state=multiplex_state,
                objects_to_interact=objects_to_interact,
                new_object_masks=new_object_masks,
                new_object_idxs=new_object_idxs,
            )
            # Append the output, depending on whether it's a conditioning frame
            add_output_as_cond_frame = (
                stage_id in init_cond_frames
                or (
                    self.add_all_frames_to_correct_as_cond
                    and stage_id in frames_to_add_correction_pt
                )
                or (
                    self.add_all_transition_frames_as_cond
                    and stage_id in transition_points
                )
            )

            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        output_dict["multiplex_state"] = multiplex_state

        if return_dict:
            return output_dict
        # turn `output_dict` into a list for loss function
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        if self.is_dynamic_vos_evaluation:
            all_frame_outputs = [all_frame_outputs.get(t) for t in range(num_frames)]
        else:
            all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        # Make DDP happy with activation checkpointing by removing unused keys
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} if d is not None else None
            for d in all_frame_outputs
        ]

        if self.is_dynamic_vos_evaluation:
            object_appearance_order = backbone_out["object_appearance_order"]
            num_objects = len(input.find_metadatas[0].coco_image_id)

            # since we have remapped the object appearance order, we would need to map it back here
            inverse_object_appearance_order = [None for _ in object_appearance_order]
            for idx, obj_id in enumerate(object_appearance_order):
                inverse_object_appearance_order[obj_id] = idx
            assert all(i is not None for i in inverse_object_appearance_order)

            # this is for a rare case where the dataloader thinks that there is an object
            # (is in input.find_metadatas[0].coco_image_id)
            # but it is not visible anywhere in the frames
            # I suspect this is due to mask resizing (the object is so small that it got lost)
            # but I am not 100% sure; haven't investigated yet.
            # This only happens if we evaluate on the new (fully annotated) YouTubeVOS set.
            if len(inverse_object_appearance_order) < num_objects:
                inverse_object_appearance_order.extend(
                    list(range(len(inverse_object_appearance_order), num_objects))
                )

            # we need to pad the outputs with zeros (for the frames before the object appears)
            last_mask = all_frame_outputs[-1]["pred_masks"]

            shape = last_mask.shape[1:]
            dtype = last_mask.dtype
            device = last_mask.device
            for stage_i, frame_out in enumerate(all_frame_outputs):
                if frame_out is None:
                    all_frame_outputs[stage_i] = {
                        "pred_masks": torch.zeros(
                            (num_objects, *shape), device=device, dtype=dtype
                        )
                    }
                    continue

                pred_mask = frame_out["pred_masks"]
                if pred_mask.shape[0] < num_objects:
                    shape = pred_mask.shape[
                        1:
                    ]  # might have a different shape, e.g., input mask
                    frame_out["pred_masks"] = torch.cat(
                        [
                            pred_mask,
                            torch.zeros(
                                (num_objects - pred_mask.shape[0], *shape),
                                device=device,
                                dtype=dtype,
                            ),
                        ],
                        dim=0,
                    )[inverse_object_appearance_order]

        return all_frame_outputs
