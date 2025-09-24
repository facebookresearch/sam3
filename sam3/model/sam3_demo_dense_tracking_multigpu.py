# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

import gc
import logging
from collections import defaultdict
from functools import reduce

import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes
from tqdm import tqdm

from sam3 import perflib
from sam3.model.act_ckpt_utils import clone_output_wrapper
from sam3.model.box_ops import box_xywh_to_cxcywh, box_xyxy_to_xywh
from sam3.model.data_misc import (
    BatchedDatapoint,
    BatchedPointer,
    convert_my_tensors,
    FindStage,
    recursive_to,
)
from sam3.model.geometry_encoders import Prompt
from sam3.model.model_misc import NestedTensor

from sam3.model.sam3_dense_shared_multigpu import Sam3DenseTrackingMultiGPU
from sam3.model.sam3_image_on_video_multigpu_utils import mask_iou

# Sam3DemoMixin functionality - using sam3 imports
from sam3.model.video_tracking_with_prompt_utils import load_resource_as_video_frames
from sam3.perflib.compile import compile_wrapper
from sam3.perflib.masks_to_boxes import masks_to_boxes as perf_masks_to_boxes

logger = logging.getLogger(__name__)


class Sam3DemoMixin:
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.inference_mode()
    def init_state(
        self,
        resource_path,
        offload_video_to_cpu=False,
        async_loading_frames=False,
        use_torchcodec=False,
        use_cv2=False,
    ):
        """Initialize an inference state from `resource_path` (an image or a video)."""
        images, orig_height, orig_width = load_resource_as_video_frames(
            resource_path=resource_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=self.image_mean,
            img_std=self.image_std,
            async_loading_frames=async_loading_frames,
            use_torchcodec=use_torchcodec,
            use_cv2=use_cv2,
        )
        inference_state = {}
        inference_state["image_size"] = self.image_size
        inference_state["num_frames"] = len(images)
        inference_state["device"] = torch.device("cuda")
        # the original video height and width, used for resizing final output scores
        inference_state["orig_height"] = orig_height
        inference_state["orig_width"] = orig_width
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # inputs on each frame
        self._construct_initial_input_batch(inference_state, images)
        return inference_state

    def _construct_initial_input_batch(self, inference_state, images):
        """Construct an initial `BatchedDatapoint` instance as input."""
        # 1) img_batch
        num_frames = len(images)
        device = inference_state["device"]
        img_batch = NestedTensor(tensors=images, mask=None)

        # 2) find_text_batch
        # "<text placeholder>" will be replaced by the actual text prompt when adding prompts
        find_text_batch = ["<text placeholder>", "visual", "geometric"]

        # 3) find_inputs
        input_box_embedding_dim = 258  # historical default
        input_points_embedding_dim = 257  # historical default
        dummy_ptrs = BatchedPointer(
            stage_ids=[], query_ids=[], object_ids=[], ptr_mask=[], ptr_types=[]
        )
        stages = [
            FindStage(
                img_ids=[stage_id],
                text_ids=[0],
                input_boxes=[torch.zeros(input_box_embedding_dim)],
                input_boxes_before_embed=[torch.empty(0, 4)],
                input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
                input_boxes_label=[torch.empty(0, dtype=torch.long)],
                input_points=[torch.empty(0, input_points_embedding_dim)],
                input_points_before_embed=[torch.empty(0, 3)],
                input_points_mask=[torch.empty(0)],
                ptrs=dummy_ptrs,
                ptrs_seg=dummy_ptrs,
                object_ids=[],
            )
            for stage_id in range(num_frames)
        ]
        for i in range(len(stages)):
            stages[i] = convert_my_tensors(stages[i])

        # construct the final `BatchedDatapoint` and cast to GPU
        input_batch = BatchedDatapoint(
            img_batch=img_batch,
            find_text_batch=find_text_batch,
            find_inputs=stages,
            find_targets=[None] * num_frames,
            get_queries=None,
            find_metadatas=[None] * num_frames,
        )
        input_batch = recursive_to(input_batch, device, non_blocking=True)
        inference_state["input_batch"] = input_batch

        # construct the placeholder interactive prompts and tracking queries
        bs = 1
        inference_state["constants"]["empty_geometric_prompt"] = Prompt(
            box_embeddings=torch.zeros(0, bs, 4, device=device),
            box_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
            box_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
            point_embeddings=torch.zeros(0, bs, 2, device=device),
            point_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
            point_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
        )

        # constructing an output list in inference state (we start with an empty list)
        inference_state["previous_stages_out"] = [None] * num_frames
        inference_state["text_prompt"] = None
        inference_state["per_frame_raw_point_input"] = [None] * num_frames
        inference_state["per_frame_raw_box_input"] = [None] * num_frames
        inference_state["per_frame_visual_prompt"] = [None] * num_frames
        inference_state["per_frame_geometric_prompt"] = [None] * num_frames
        inference_state["per_frame_cur_step"] = [0] * num_frames

        # placeholders for cached outputs
        # (note: currently, a single visual prompt embedding is shared for all frames)
        inference_state["backbone_out"] = None
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None

    @torch.inference_mode()
    def reset_state(self, inference_state):
        """Revert `inference_state` to what it was right after initialization."""
        inference_state["input_batch"].find_text_batch[0] = "<text placeholder>"
        inference_state["text_prompt"] = None
        for t in range(inference_state["num_frames"]):
            inference_state["input_batch"].find_inputs[t].text_ids[...] = 0
            # constructing an output list in inference state (we start with an empty list)
            inference_state["previous_stages_out"][t] = None
            inference_state["per_frame_raw_point_input"][t] = None
            inference_state["per_frame_raw_box_input"][t] = None
            inference_state["per_frame_visual_prompt"][t] = None
            inference_state["per_frame_geometric_prompt"][t] = None
            inference_state["per_frame_cur_step"][t] = 0

        inference_state["backbone_out"] = None
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None
        gc.collect()
        torch.cuda.empty_cache()

    def _get_visual_prompt(self, inference_state, frame_idx, boxes_cxcywh, box_labels):
        """
        Handle the case of visual prompt. Currently, in the inference API we do not
        explicitly distinguish between initial box as visual prompt vs subsequent boxes
        or boxes after inference for refinement.
        """
        # If the frame hasn't had any inference results before (prompting or propagation),
        # we treat the first added box prompt as a visual prompt; otherwise, we treat
        # the first box just as a refinement prompt.
        is_new_visual_prompt = (
            inference_state["per_frame_visual_prompt"][frame_idx] is None
            and inference_state["previous_stages_out"][frame_idx] is None
        )
        if is_new_visual_prompt:
            if boxes_cxcywh.size(0) != 1:
                raise RuntimeError(
                    "visual prompts (box as an initial prompt) should only have one box, "
                    f"but got {boxes_cxcywh.shape=}"
                )
            if not box_labels.item():
                logging.warning("A negative box is added as a visual prompt.")
            # take the first box prompt as a visual prompt
            device = inference_state["device"]
            new_visual_prompt = Prompt(
                box_embeddings=boxes_cxcywh[None, 0:1, :].to(device),  # (seq, bs, 4)
                box_mask=None,
                box_labels=box_labels[None, 0:1].to(device),  # (seq, bs)
                point_embeddings=None,
                point_mask=None,
                point_labels=None,
            )
            inference_state["per_frame_visual_prompt"][frame_idx] = new_visual_prompt
        else:
            new_visual_prompt = None

        # `boxes_cxcywh` and `box_labels` contains all the raw box inputs added so far
        # strip any visual prompt from the input boxes (for geometric prompt encoding)
        if inference_state["per_frame_visual_prompt"][frame_idx] is not None:
            boxes_cxcywh = boxes_cxcywh[1:]
            box_labels = box_labels[1:]

        return boxes_cxcywh, box_labels, new_visual_prompt


class Sam3DenseTrackingDemoMultiGPU(Sam3DemoMixin, Sam3DenseTrackingMultiGPU):
    def __init__(
        self,
        image_size=1008,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        compile_model=False,
        **kwargs,
    ):
        """
        hotstart_delay: int, the delay (in #frames) before the model starts to yield output, 0 to disable hotstart delay.
        hotstart_unmatch_thresh: int, remove the object if it has this many unmatched frames within its hotstart_delay period.
            If `hotstart_delay` is set to 0, this parameter is ignored.
        hotstart_dup_thresh: int, remove the object if it has overlapped with another object this many frames within its hotstart_delay period.
        """
        super().__init__(**kwargs)
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.compile_model = compile_model

        self.bf16_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        self.bf16_context.__enter__()

    @torch.inference_mode()
    def init_state(
        self,
        resource_path,
        offload_video_to_cpu=False,
        async_loading_frames=False,
        use_torchcodec=False,
        use_cv2=False,
    ):
        inference_state = super().init_state(
            resource_path=resource_path,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            use_torchcodec=use_torchcodec,
            use_cv2=use_cv2,
        )
        # initialize extra states
        # sam2_inference_states will contain separate inference_states for each frame having new objects if
        # self.sam2_predictor.per_obj_inference is False (bucketized batching), or a single inference_state
        # containing all objects if self.sam2_predictor.per_obj_inference is True (no batching at all).
        inference_state["sam2_inference_states"] = []
        inference_state["sam2_metadata"] = {}
        inference_state["feature_cache"] = {}
        return inference_state

    def reset_state(self, inference_state):
        super().reset_state(inference_state)
        # reset extra states
        inference_state["sam2_inference_states"].clear()
        inference_state["sam2_metadata"].clear()
        inference_state["feature_cache"].clear()

    def _get_processing_order(
        self, inference_state, start_frame_idx, max_frame_num_to_track, reverse
    ):
        num_frames = inference_state["num_frames"]
        previous_stages_out = inference_state["previous_stages_out"]
        if all(out is None for out in previous_stages_out) and start_frame_idx is None:
            raise RuntimeError(
                "No prompts are received on any frames. Please add prompt on at least one frame before propagation."
            )
        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(
                t for t, out in enumerate(previous_stages_out) if out is not None
            )
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = start_frame_idx - max_frame_num_to_track
            end_frame_idx = max(end_frame_idx, 0)
            processing_order = range(start_frame_idx - 1, end_frame_idx - 1, -1)
        else:
            end_frame_idx = start_frame_idx + max_frame_num_to_track
            end_frame_idx = min(end_frame_idx, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        return processing_order, end_frame_idx

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        output_prob_thresh=0.5,
        compute_stability_score=False,
        is_instance_processing=False,
    ):
        """
        Propagate the prompts to get grounding results for the entire video. This method
        is a generator and yields inference outputs for all frames in the range specified
        by `start_frame_idx`, `max_frame_num_to_track`, and `reverse`.
        """
        # compile the model (it's a no-op if the model is already compiled)
        # note that it's intentionally added to `self.propagate_in_video`, so that the first
        # `self.add_prompt` call will be done in eager mode to fill in the decoder buffers
        # such as positional encoding cache)
        self._compile_model()

        processing_order, end_frame_idx = self._get_processing_order(
            inference_state,
            start_frame_idx,
            max_frame_num_to_track,
            reverse=reverse,
        )

        hotstart_buffer = []
        hotstart_removed_obj_ids = set()
        for frame_idx in tqdm(
            processing_order, desc="propagate_in_video", disable=self.rank > 0
        ):
            out = self._run_single_frame_inference(
                inference_state,
                frame_idx,
                reverse,
                is_instance_processing=is_instance_processing,
            )

            if self.hotstart_delay > 0:
                # accumulate the outputs for the first `hotstart_delay` frames
                hotstart_buffer.append([frame_idx, out])
                # update the object IDs removed by hotstart so that we don't output them
                if self.rank == 0:
                    hotstart_removed_obj_ids.update(out["removed_obj_ids"])

                if frame_idx == end_frame_idx:
                    # we reached the end of propagation -- yield all frames in the buffer
                    yield_list = hotstart_buffer
                    hotstart_buffer = []
                elif len(hotstart_buffer) >= self.hotstart_delay:
                    # we have enough frames -- yield and remove the first (oldest) frame from the buffer
                    yield_list = hotstart_buffer[:1]
                    hotstart_buffer = hotstart_buffer[1:]
                else:
                    # not enough frames yet -- skip yielding
                    yield_list = []
            else:
                yield_list = [(frame_idx, out)]  # output the current frame

            for yield_frame_idx, yield_out in yield_list:
                # post-process the output and yield it
                with torch.profiler.record_function(
                    "Sam3DenseTrackingDemoMultiGPU.postprocess_output"
                ):
                    if self.rank == 0:
                        suppressed_obj_ids = yield_out["suppressed_obj_ids"]
                        postprocessed_out = self._postprocess_output(
                            inference_state,
                            yield_out,
                            hotstart_removed_obj_ids,
                            suppressed_obj_ids,
                        )
                    else:
                        postprocessed_out = None  # no output on other GPUs
                    yield yield_frame_idx, postprocessed_out

    def _run_single_frame_inference(
        self, inference_state, frame_idx, reverse, is_instance_processing=False
    ):
        """
        Perform inference on a single frame and get its inference results. This would
        also update `inference_state`.
        """
        # prepare inputs
        input_batch = inference_state["input_batch"]
        sam2_states_local = inference_state["sam2_inference_states"]
        geometric_prompt = (
            inference_state["constants"]["empty_geometric_prompt"]
            if inference_state["per_frame_geometric_prompt"][frame_idx] is None
            else inference_state["per_frame_geometric_prompt"][frame_idx]
        )
        # run inference for the current frame
        (
            obj_id_to_mask,
            obj_id_to_score,
            sam2_states_local_new,
            sam2_metadata_new,
            frame_stats,
        ) = self._det_track_one_frame(
            frame_idx=frame_idx,
            num_frames=inference_state["num_frames"],
            reverse=reverse,
            input_batch=input_batch,
            geometric_prompt=geometric_prompt,
            sam2_states_local=sam2_states_local,
            sam2_metadata_prev=inference_state["sam2_metadata"],
            feature_cache=inference_state["feature_cache"],
            orig_vid_height=inference_state["orig_height"],
            orig_vid_width=inference_state["orig_width"],
        )
        # update inference state
        inference_state["sam2_inference_states"] = sam2_states_local_new
        inference_state["sam2_metadata"] = sam2_metadata_new
        # use a dummy string in "previous_stages_out" to indicate this frame has outputs
        inference_state["previous_stages_out"][frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"
        out = {"obj_id_to_mask": obj_id_to_mask, "obj_id_to_score": obj_id_to_score}
        # removed_obj_ids is only needed on rank 0 to handle hotstart delay buffer
        if self.rank == 0:
            removed_obj_ids = sam2_metadata_new["rank0_metadata"]["removed_obj_ids"]
            out["removed_obj_ids"] = removed_obj_ids
            out["suppressed_obj_ids"] = sam2_metadata_new["rank0_metadata"][
                "suppressed_obj_ids"
            ][frame_idx]
            out["frame_stats"] = frame_stats
        return out

    def _postprocess_output(
        self, inference_state, out, removed_obj_ids=None, suppressed_obj_ids=None
    ):
        obj_id_to_mask = out["obj_id_to_mask"]
        curr_obj_ids = sorted(obj_id_to_mask.keys())
        H_video, W_video = inference_state["orig_height"], inference_state["orig_width"]
        if len(curr_obj_ids) == 0:
            out_obj_ids = torch.zeros(0, dtype=torch.int64)
            out_probs = torch.zeros(0, dtype=torch.float32)
            out_binary_masks = torch.zeros(0, H_video, W_video, dtype=torch.bool)
            out_boxes_xywh = torch.zeros(0, 4, dtype=torch.float32)
        else:
            out_obj_ids = torch.tensor(curr_obj_ids, dtype=torch.int64)
            out_probs = torch.tensor(
                [out["obj_id_to_score"][obj_id] for obj_id in curr_obj_ids]
            )
            out_binary_masks = torch.cat(
                [obj_id_to_mask[obj_id] for obj_id in curr_obj_ids], dim=0
            )

            to_suppress = torch.zeros_like(out_obj_ids, dtype=torch.bool)
            if suppressed_obj_ids is not None and len(suppressed_obj_ids) > 0:
                suppressed_obj_ids = torch.tensor(
                    list(suppressed_obj_ids), dtype=torch.int64
                )
                to_suppress = torch.isin(out_obj_ids, suppressed_obj_ids)
            # remove masks with zero areas
            assert out_binary_masks.dtype == torch.bool
            keep = out_binary_masks.any(dim=(1, 2)).cpu()
            keep &= ~to_suppress
            # remove outputs for those object IDs in `removed_obj_ids`
            if removed_obj_ids is not None and len(removed_obj_ids) > 0:
                removed = torch.tensor(list(removed_obj_ids), dtype=torch.int64)
                keep &= ~torch.isin(out_obj_ids, removed)
            # slice those valid entries from the original outputs
            keep_idx = torch.nonzero(keep, as_tuple=True)[0]
            keep_idx_gpu = keep_idx.pin_memory().to(
                device=out_binary_masks.device, non_blocking=True
            )

            out_obj_ids = torch.index_select(out_obj_ids, 0, keep_idx)
            out_probs = torch.index_select(out_probs, 0, keep_idx)
            out_binary_masks = torch.index_select(out_binary_masks, 0, keep_idx_gpu)

            if perflib.is_enabled:
                out_boxes_xyxy = perf_masks_to_boxes(
                    out_binary_masks, out_obj_ids.tolist()
                )
            else:
                out_boxes_xyxy = masks_to_boxes(out_binary_masks)

            out_boxes_xywh = box_xyxy_to_xywh(out_boxes_xyxy)  # convert to xywh format
            # normalize boxes
            out_boxes_xywh[..., 0] /= W_video
            out_boxes_xywh[..., 1] /= H_video
            out_boxes_xywh[..., 2] /= W_video
            out_boxes_xywh[..., 3] /= H_video

        outputs = {
            "out_obj_ids": out_obj_ids.cpu().numpy(),
            "out_probs": out_probs.cpu().numpy(),
            "out_boxes_xywh": out_boxes_xywh.cpu().numpy(),
            "out_binary_masks": out_binary_masks.cpu().numpy(),
            "frame_stats": out.get("frame_stats", None),
        }
        return outputs

    def _compile_model(self):
        """Compile the SAM model with torch.compile for speedup."""
        # TODO: compile SAM2 model components
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

        # Compile module components following https://www.internalfb.com/diff/D70935785
        # skip compilation of `_encode_prompt` since it sometimes tiggger SymInt errors
        # self._encode_prompt = clone_output_wrapper(
        #     torch.compile(self._encode_prompt, fullgraph=True, mode="max-autotune")
        # )

        ## Compile SAM3 model components
        self.sam3_model.backbone.vision_backbone.forward = clone_output_wrapper(
            torch.compile(
                self.sam3_model.backbone.vision_backbone.forward,
                fullgraph=True,
                mode="max-autotune",
            )
        )
        self.sam3_model.transformer.encoder.forward = clone_output_wrapper(
            torch.compile(
                self.sam3_model.transformer.encoder.forward,
                fullgraph=True,
                mode="max-autotune",
            )
        )
        self.sam3_model.transformer.decoder.forward = clone_output_wrapper(
            torch.compile(
                self.sam3_model.transformer.decoder.forward,
                fullgraph=True,
                mode="max-autotune",
                dynamic=False,  # note: FA decoder uses static shapes
            )
        )

        self.sam3_model.segmentation_head.forward = clone_output_wrapper(
            torch.compile(
                self.sam3_model.segmentation_head.forward,
                fullgraph=True,
                mode="max-autotune",
            )
        )

        ## Compile SAM2 model components
        self.sam2_predictor.maskmem_backbone.forward = compile_wrapper(
            self.sam2_predictor.maskmem_backbone.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

        self.sam2_predictor.transformer.encoder.forward = compile_wrapper(
            self.sam2_predictor.transformer.encoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=True,  # Num. of memories varies
        )

        self.sam2_predictor.sam_mask_decoder.forward = compile_wrapper(
            self.sam2_predictor.sam_mask_decoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,  # Accuracy regression on True
        )

        self._model_is_compiled = True

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def warm_up_compilation(self):
        """
        Warm up the model by running a dummy inference to compile the model. This is
        useful to avoid the compilation overhead in the first inference call.
        """
        if not self.compile_model:
            return
        self._warm_up_complete = False
        if self.device.type != "cuda":
            raise RuntimeError(
                f"The model must be on CUDA for warm-up compilation, got {self.device=}."
            )

        # temporally set to single GPU temporarily for warm-up compilation
        orig_rank = self.rank
        orig_world_size = self.world_size
        self.rank = self.sam3_model.rank = 0
        self.world_size = self.sam3_model.world_size = 1

        # Get a random video
        orig_new_det_thresh = self.new_det_thresh
        inference_state = self.init_state(resource_path="<load-dummy-video-30>")
        # use different tracking score thresholds for each round to simulate different number of output objects
        num_objects_list = range(self.num_obj_for_compile + 1)
        new_det_score_thresh_list = [0.3, 0.5, 0.7]
        num_rounds = len(new_det_score_thresh_list)
        for i, thresh in enumerate(new_det_score_thresh_list):
            self.new_det_thresh = thresh
            for num_objects in num_objects_list:
                # start at different locations for each round
                start_frame_idx = 0
                logger.info(f"{i+1}/{num_rounds} warming up model compilation")
                self.add_prompt(
                    inference_state, frame_idx=start_frame_idx, text_str="cat"
                )
                logger.info(
                    f"{i+1}/{num_rounds} warming up model compilation -- simulating {num_objects} objects"
                )
                new_det_obj_ids_local = np.arange(num_objects)
                new_det_masks = torch.ones(
                    len(new_det_obj_ids_local),
                    self.sam2_predictor.low_res_mask_size,
                    self.sam2_predictor.low_res_mask_size,
                ).to(self.device)
                sam2_states_local = self._sam2_add_new_objects(
                    frame_idx=start_frame_idx,
                    num_frames=inference_state["num_frames"],
                    new_obj_ids=new_det_obj_ids_local,
                    new_obj_masks=new_det_masks,
                    sam2_states_local=inference_state["sam2_inference_states"],
                    orig_vid_height=inference_state["orig_height"],
                    orig_vid_width=inference_state["orig_width"],
                    feature_cache=inference_state["feature_cache"],
                )
                inference_state["sam2_inference_states"] = sam2_states_local

                inference_state["sam2_metadata"].update(
                    {
                        "obj_ids_per_gpu": [np.array(list(range(num_objects)))],
                        "obj_ids_all_gpu": np.array(
                            list(range(num_objects))
                        ),  # Same as 1 GPU
                        "num_obj_per_gpu": [num_objects],
                        "obj_id_to_score": [num_objects] * (num_objects),
                        "max_obj_id": num_objects,
                    }
                )
                for _ in self.propagate_in_video(
                    inference_state, start_frame_idx, reverse=False
                ):
                    pass
                for _ in self.propagate_in_video(
                    inference_state, start_frame_idx, reverse=True
                ):
                    pass
                self.reset_state(inference_state)
                logger.info(
                    f"{i+1}/{num_rounds} warming up model compilation -- completed round {i+1} out of {num_rounds}"
                )

        self.new_det_thresh = orig_new_det_thresh
        logger.info("Warm-up compilation completed.")

        # revert to the original GPU and rank
        self.rank = self.sam3_model.rank = orig_rank
        self.world_size = self.sam3_model.world_size = orig_world_size
        self._warm_up_complete = True

    @torch.inference_mode()
    def add_prompt(
        self,
        inference_state,
        frame_idx,
        text_str=None,
        clear_old_points=True,
        points=None,
        point_labels=None,
        boxes_xywh=None,
        box_labels=None,
        clear_old_boxes=True,
        output_prob_thresh=0.5,
    ):
        """
        Add text, point or box prompts on a single frame. This method returns the inference
        outputs only on the prompted frame.

        Note that text prompts are NOT associated with a particular frame (i.e. they apply
        to all frames). However, we only run inference on the frame specified in `frame_idx`.

        Copied from sam3_demo.Sam3DemoMixin.add_prompt, simplified to support only text prompts.
        """
        logger.info("Running add_prompt on frame %d", frame_idx)

        device = inference_state["device"]
        num_frames = inference_state["num_frames"]
        assert (
            text_str is not None or points is not None or boxes_xywh is not None
        ), "at least one type of prompt (text, points, boxes) must be provided"
        assert (
            0 <= frame_idx < num_frames
        ), f"{frame_idx=} is out of range for a total of {num_frames} frames"

        # 1) add text prompt
        if text_str is not None:
            inference_state["text_prompt"] = text_str
            # add the text prompt into the input batch (to be applied to *all* frames)
            inference_state["input_batch"].find_text_batch[0] = text_str
            for t in range(inference_state["num_frames"]):
                text_id = self.TEXT_ID_FOR_TEXT
                inference_state["input_batch"].find_inputs[t].text_ids[...] = text_id

        # 2) add geometric prompt (points or boxes)
        # start with an empty geometric_prompt (we later add previous point and box prompts
        # from "per_frame_raw_point_input" and "per_frame_raw_box_input" below)
        geometric_prompt = inference_state["constants"][
            "empty_geometric_prompt"
        ].clone()

        # 2.1) handle point prompt
        assert (points is not None) == (point_labels is not None)
        if points is not None:
            points = torch.as_tensor(points, dtype=torch.float32)
            point_labels = torch.as_tensor(point_labels, dtype=torch.long)
            assert points.dim() == 2
            assert points.size(0) > 0 and points.size(-1) == 2
            assert point_labels.dim() == 1 and point_labels.size(0) == points.size(0)
            assert torch.all(points >= 0).item() and torch.all(points <= 1).item()
            # append previous points under `clear_old_points=False`
            prev_point_input = inference_state["per_frame_raw_point_input"][frame_idx]
            if prev_point_input is not None and not clear_old_points:
                prev_points, prev_point_labels = prev_point_input
                points = torch.cat([prev_points, points], dim=0)
                point_labels = torch.cat([prev_point_labels, point_labels], dim=0)
            new_point_input = points, point_labels
            inference_state["per_frame_raw_point_input"][frame_idx] = new_point_input
            # add a batch dimensions (note that it's sequence first)
            points = points.unsqueeze(1).to(device)
            point_labels = point_labels.unsqueeze(1).to(device)
            geometric_prompt.append_points(points=points, labels=point_labels)

        # 2.2) handle box prompt
        assert (boxes_xywh is not None) == (box_labels is not None)
        if boxes_xywh is not None:
            boxes_xywh = torch.as_tensor(boxes_xywh, dtype=torch.float32)
            box_labels = torch.as_tensor(box_labels, dtype=torch.long)
            # input boxes are expected to be [xmin, ymin, width, height] format
            # in normalized coordinates of range 0~1, similar to FA
            assert boxes_xywh.dim() == 2
            assert boxes_xywh.size(0) > 0 and boxes_xywh.size(-1) == 4
            assert box_labels.dim() == 1 and box_labels.size(0) == boxes_xywh.size(0)
            boxes_cxcywh = box_xywh_to_cxcywh(boxes_xywh)
            assert (boxes_xywh >= 0).all().item() and (boxes_xywh <= 1).all().item()
            assert (boxes_cxcywh >= 0).all().item() and (boxes_cxcywh <= 1).all().item()
            # append previous boxes under `clear_old_boxes=False`
            prev_box_input = inference_state["per_frame_raw_box_input"][frame_idx]
            if prev_box_input is not None and not clear_old_boxes:
                prev_boxes_cxcywh, prev_box_labels = prev_box_input
                boxes_cxcywh = torch.cat([prev_boxes_cxcywh, boxes_cxcywh], dim=0)
                box_labels = torch.cat([prev_box_labels, box_labels], dim=0)
            new_box_input = boxes_cxcywh, box_labels
            inference_state["per_frame_raw_box_input"][frame_idx] = new_box_input

            # handle the case of visual prompt (also added as an input box from the UI)
            boxes_cxcywh, box_labels, new_visual_prompt = self._get_visual_prompt(
                inference_state, frame_idx, boxes_cxcywh, box_labels
            )
            # add a batch dimensions (note that it's sequence first)
            boxes_cxcywh = boxes_cxcywh.unsqueeze(1).to(device)
            box_labels = box_labels.unsqueeze(1).to(device)
            geometric_prompt.append_boxes(boxes=boxes_cxcywh, labels=box_labels)
        else:
            new_visual_prompt = None

        inference_state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt

        # 3) run inference on this frame
        inference_state["backbone_out"] = self._init_backbone_out(inference_state)
        if new_visual_prompt is not None:
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
                )
            )
            inference_state["visual_prompt_embed"] = visual_prompt_embed
            inference_state["visual_prompt_mask"] = visual_prompt_mask

        out = self._run_single_frame_inference(
            inference_state, frame_idx, reverse=False
        )
        return frame_idx, self._postprocess_output(inference_state, out)

    def _init_backbone_out(self, inference_state):
        """
        Initialize a backbone_out dictionary and extract the text features.

        Note that the visual features of each frame are not extracted here. They will be
        extracted on the fly when running inference on each frame.
        """
        input = inference_state["input_batch"]
        device = self.device
        backbone_out = {"img_batch_all_stages": input.img_batch}
        text_outputs = self.sam3_model.backbone.forward_text(
            input.find_text_batch, device=device
        )
        backbone_out.update(text_outputs)
        return backbone_out

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(self, input: BatchedDatapoint, is_inference: bool = False):
        """This method is only used for benchmark eval (not used in the demo)."""
        # set the model to single GPU for benchmark evaluation (to be compatible with trainer)
        orig_rank = self.rank
        orig_world_size = self.world_size
        self.rank = self.sam3_model.rank = 0
        self.world_size = self.sam3_model.world_size = 1

        # get data
        text_prompt_ids = input.find_metadatas[0].original_category_id
        text_prompt_list = input.find_text_batch

        # loop over txt prompts
        tracking_res = defaultdict(dict)  # frame_idx --> {obj_id: mask}
        scores_labels = defaultdict(tuple)  # obj_id --> (score, text_prompt_id)
        inference_state = self.init_state(resource_path=input.raw_images)
        for prompt_id, prompt in zip(text_prompt_ids, text_prompt_list):
            self.add_prompt(inference_state, frame_idx=0, text_str=prompt)
            start_obj_id = max(scores_labels.keys(), default=-1) + 1  # prev max + 1

            # propagate the prompts
            obj_ids_this_prompt = set()
            for frame_idx, out in self.propagate_in_video(
                inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=inference_state["num_frames"],
                reverse=False,
            ):
                current_frame_res = tracking_res[frame_idx]
                for obj_id, mask in zip(out["out_obj_ids"], out["out_binary_masks"]):
                    mask_tensor = torch.tensor(mask[None], dtype=torch.bool)
                    current_frame_res[obj_id + start_obj_id] = mask_tensor
                obj_ids_this_prompt.update(current_frame_res.keys())

            obj_id_to_score = inference_state["sam2_metadata"]["obj_id_to_score"]
            for obj_id, score in obj_id_to_score.items():
                if obj_id + start_obj_id in obj_ids_this_prompt:
                    score_tensor = torch.tensor(score, dtype=torch.float32)
                    scores_labels[obj_id + start_obj_id] = (score_tensor, prompt_id)

            self.reset_state(inference_state)

        video_id = input.find_metadatas[0].original_image_id[0].cpu().item()
        preds = self.prep_for_evaluator(input.raw_images, tracking_res, scores_labels)

        # revert the model to the original GPU and rank
        self.rank = self.sam3_model.rank = orig_rank
        self.world_size = self.sam3_model.world_size = orig_world_size
        return {video_id: preds}


class Sam3DenseTrackingDemoMultiGPUWithInstanceInteractivity(
    Sam3DenseTrackingDemoMultiGPU
):
    def __init__(
        self,
        use_prev_mem_frame=False,
        refinement_removal_iou_thd=0.5,
        **kwargs,
    ):
        """
        use_prev_mem_frame: bool, whether to condition on previous memory frames for adding points
        refinement_removal_iou_thd: float, in range (0, 1), remove an object if it overlaps
            (iou > `refinement_removal_iou_thd`) with a refined object
        """
        super().__init__(**kwargs)
        self.use_prev_mem_frame = use_prev_mem_frame
        self.refinement_removal_iou_thd = refinement_removal_iou_thd

    @torch.inference_mode()
    def init_state(
        self,
        resource_path,
        offload_video_to_cpu=False,
        async_loading_frames=False,
        use_torchcodec=False,
        use_cv2=False,
    ):
        inference_state = super().init_state(
            resource_path=resource_path,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            use_torchcodec=use_torchcodec,
            use_cv2=use_cv2,
        )
        # initialize extra states
        inference_state["action_history"] = []  # for logging user actions
        if self.sam2_predictor.per_obj_inference:
            # in per_obj mode only 1 inference state is needed, we init it here.
            inference_state["sam2_inference_states"] = [
                self._init_new_sam2_state(inference_state)
            ]
        return inference_state

    def reset_state(self, inference_state):
        super().reset_state(inference_state)
        # reset extra states
        inference_state["action_history"].clear()
        if self.sam2_predictor.per_obj_inference:
            inference_state["sam2_inference_states"] = [
                self._init_new_sam2_state(inference_state)
            ]

    def _init_new_sam2_state(self, inference_state):
        return self.sam2_predictor.init_state(
            cached_features=inference_state["feature_cache"],
            video_height=inference_state["orig_height"],
            video_width=inference_state["orig_width"],
            num_frames=inference_state["num_frames"],
        )

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        output_prob_thresh=0.5,
        compute_stability_score=False,
        is_instance_processing=False,
    ):
        # step 1: check which type of propagation to run, should be the same for all GPUs.
        propagation_type, obj_ids = self.parse_action_history_for_propagation(
            inference_state
        )
        self.add_action_history(
            inference_state,
            action_type=propagation_type,
            obj_ids=obj_ids,
            frame_idx=start_frame_idx,
        )

        # step 2: run full VG propagation
        if propagation_type == "propagation_full":
            logger.info(f"Running full VG propagation (reverse={reverse}).")
            yield from super().propagate_in_video(
                inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=reverse,
            )
            return

        # step 3: run SAM2 partial propagation or direct fetch existing predictions
        assert propagation_type in ["propagation_partial", "propagation_fetch"]
        logger.info(
            f"Running SAM2 propagation for objects {obj_ids} and merging it with existing VG predictions (reverse={reverse})."
            if propagation_type == "propagation_partial"
            else f"Fetching existing VG predictions without running any propagation (reverse={reverse})."
        )
        processing_order = self.sam2_predictor._get_processing_order(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        )
        # get SAM2 inference states containing selected obj_ids
        if propagation_type == "propagation_partial":
            assert (
                obj_ids is not None
            ), "obj_ids must be provided for partial propagation"
            # can be empty for GPUs where objects are not in their inference states
            sam2_states_local = self._get_sam2_inference_states_by_obj_ids(
                inference_state, obj_ids
            )
            for sam2_state in sam2_states_local:
                self.sam2_predictor.propagate_in_video_preflight(
                    sam2_state, run_mem_encoder=True
                )

        all_overlap_obj_ids = set()
        sam2_metadata = inference_state["sam2_metadata"]
        for frame_idx in tqdm(processing_order):
            # get existing VG outputs for the current frame
            obj_id_to_mask_local = self.get_sam2_output_per_frame(
                inference_state, frame_idx, resize_to_video_res=False
            )
            # run SAM2 propagation
            if propagation_type == "propagation_partial":
                self._prepare_backbone_feats(inference_state, frame_idx, reverse)
                # TODO: to be optimized by running inference only on selected objects
                _obj_ids, _low_res_masks = self._propogate_sam2_one_frame_local_gpu(
                    sam2_states_local,
                    frame_idx=frame_idx,
                    reverse=reverse,
                    run_mem_encoder=True,
                )
                if len(_obj_ids) > 0:
                    obj_id_to_mask_local.update(dict(zip(_obj_ids, _low_res_masks)))

            # concatenate the output masklets from all local inference states
            H_mask = W_mask = self.sam2_predictor.low_res_mask_size
            obj_ids_local = sam2_metadata["obj_ids_per_gpu"][self.rank]
            low_res_masks_local = []
            for obj_id in obj_ids_local:
                if obj_id in obj_id_to_mask_local:
                    low_res_masks_local.append(obj_id_to_mask_local[obj_id])
                else:
                    low_res_masks_local.append(
                        torch.full((H_mask, W_mask), -1024.0, device=self.device)
                    )
            if len(low_res_masks_local) > 0:
                low_res_masks_local = torch.stack(
                    low_res_masks_local, dim=0
                )  # (N, H, W)
                assert low_res_masks_local.shape[1:] == (H_mask, W_mask)
            else:
                low_res_masks_local = torch.zeros(0, H_mask, W_mask, device=self.device)

            # all-gather `low_res_masks_local` into `low_res_masks_global`
            # - low_res_masks_global: Tensor -- (num_global_obj, H_mask, W_mask)
            if self.world_size > 1:
                low_res_masks_local = low_res_masks_local.contiguous()
                low_res_masks_peers = [
                    low_res_masks_local.new_empty(num_obj, H_mask, W_mask)
                    for num_obj in sam2_metadata["num_obj_per_gpu"]
                ]
                dist.all_gather(low_res_masks_peers, low_res_masks_local)
                low_res_masks_global = torch.cat(low_res_masks_peers, dim=0)
            else:
                low_res_masks_global = low_res_masks_local

            # format data for output
            existing_masklet_obj_ids = sam2_metadata["obj_ids_all_gpu"]
            existing_masklet_video_res_masks = F.interpolate(
                low_res_masks_global.unsqueeze(1),
                size=(inference_state["orig_height"], inference_state["orig_width"]),
                mode="bilinear",
                align_corners=False,
            )  # (num_obj, 1, H_video, W_video)
            existing_masklet_binary = existing_masklet_video_res_masks > 0
            assert len(existing_masklet_obj_ids) == len(existing_masklet_binary)
            obj_id_to_mask = dict(
                zip(existing_masklet_obj_ids, existing_masklet_binary)
            )  # obj_id -> (1, H_video, W_video)
            obj_id_to_score = sam2_metadata["obj_id_to_score"]

            # remove overlapping objects (with high IoU) with refined objects to avoid duplicates
            if obj_ids is not None and len(obj_ids) > 0:
                obj_id_to_low_res_mask = dict(
                    zip(existing_masklet_obj_ids, low_res_masks_global.unsqueeze(1) > 0)
                )  # obj_id -> (1, H_mask, W_mask)
                overlap_obj_ids = self.get_overlap_obj_ids(
                    obj_id_to_low_res_mask,
                    obj_ids,
                    iou_thd=self.refinement_removal_iou_thd,
                )
                all_overlap_obj_ids.update(overlap_obj_ids)
                # remove obj_id in all the following frames if an obj_id has been removed before.
                obj_id_to_mask = {
                    k: v
                    for k, v in obj_id_to_mask.items()
                    if k not in all_overlap_obj_ids
                }
                obj_id_to_score = {
                    k: v
                    for k, v in obj_id_to_score.items()
                    if k not in all_overlap_obj_ids
                }

            # pack results
            out = {"obj_id_to_mask": obj_id_to_mask, "obj_id_to_score": obj_id_to_score}

            yield (
                frame_idx,
                self._postprocess_output(
                    inference_state,
                    out,
                ),
            )

        all_overlap_obj_ids, sam2_metadata = self._gather_all_overlap_obj_ids(
            all_overlap_obj_ids, sam2_metadata
        )
        inference_state["sam2_metadata"] = sam2_metadata
        if len(all_overlap_obj_ids) > 0:
            # Remove the overlapped objects. There is a chance that this would cause a discrepancy
            # between online displayed masks and later on fetched masks, when the objects does not
            # overlap with refined objects at its first appearance.
            for obj_id in all_overlap_obj_ids:
                self.remove_object(inference_state, obj_id, is_user_action=False)
            logger.info(
                f"Removed overlapping object IDs under thd={self.refinement_removal_iou_thd}: {all_overlap_obj_ids}"
            )

    def _gather_all_overlap_obj_ids(self, all_overlap_obj_ids, sam2_metadata):
        # gather overlap_obj_ids across all GPUs
        assert isinstance(all_overlap_obj_ids, set)
        if self.world_size > 1:
            obj_list = [None] * self.world_size
            self.all_gather_python_obj_cpu(obj_list, all_overlap_obj_ids)
            all_overlap_obj_ids = reduce((lambda x, y: x | y), obj_list)

        # update the object list in metadata if we have new objects to remove
        for rank in range(self.world_size):
            obj_ids_this_gpu = sam2_metadata["obj_ids_per_gpu"][rank]
            is_removed = np.isin(obj_ids_this_gpu, all_overlap_obj_ids)
            obj_ids_this_gpu = obj_ids_this_gpu[~is_removed]
            sam2_metadata["obj_ids_per_gpu"][rank] = obj_ids_this_gpu
            sam2_metadata["num_obj_per_gpu"][rank] = len(obj_ids_this_gpu)
        sam2_metadata["obj_ids_all_gpu"] = np.concatenate(
            sam2_metadata["obj_ids_per_gpu"]
        )
        return all_overlap_obj_ids, sam2_metadata

    def get_overlap_obj_ids(self, obj_id_to_mask, refined_obj_ids, iou_thd=0.75):
        """Get obj_ids with high mask iou with `refined_obj_ids`.

        obj_id_to_mask: {obj_id: torch.bool tensor)}
        refined_obj_ids: list of object ids that has updated masks.
        iou_thd: IoU threshold to determine if a previous mask should be deleted.
        """
        refined_obj_ids = sorted(refined_obj_ids)
        non_refined_obj_ids = sorted(set(obj_id_to_mask.keys()) - set(refined_obj_ids))

        # if one of the obj_ids is empty, return
        if len(refined_obj_ids) == 0 or len(non_refined_obj_ids) == 0:
            return obj_id_to_mask

        # compute IoUs between refined masks and non_refined masks
        refined_masks = torch.cat(
            [obj_id_to_mask[obj_id] for obj_id in refined_obj_ids], dim=0
        )  # (M, H, W)
        non_refined_masks = torch.cat(
            [obj_id_to_mask[obj_id] for obj_id in non_refined_obj_ids], dim=0
        )  # (N, H, W)
        iou = mask_iou(non_refined_masks, refined_masks)  # (N, M)

        # find the indices of masks in non_refined_masks that have high IoU with any refined_mask
        high_iou_indices = torch.where((iou > iou_thd).any(dim=1))[0].tolist()  # (N,)
        high_iou_obj_ids = [non_refined_obj_ids[e] for e in high_iou_indices]
        return high_iou_obj_ids

    def add_action_history(
        self, inference_state, action_type, frame_idx=None, obj_ids=None
    ):
        """
        action_history is used to automatically decide what to do during propagation.
        action_type: one of ["add", "remove", "refine"] + ["propagation_full", "propagation_partial", "propagation_fetch"]
        """
        instance_actions = ["add", "remove", "refine"]
        propagation_actions = [
            "propagation_full",
            "propagation_partial",
            "propagation_fetch",
        ]
        assert (
            action_type in instance_actions + propagation_actions
        ), f"Invalid action type: {action_type}, must be one of {instance_actions + propagation_actions}"
        action = {
            "type": action_type,
            "frame_idx": frame_idx,
            "obj_ids": obj_ids,
        }
        inference_state["action_history"].append(action)

    def parse_action_history_for_propagation(self, inference_state):
        """
        Parse the actions in history before the last propagation and prepare for the next propagation.
        We support multiple actions (add/remove/refine) between two propagations. If we had an action
        history similar to this ["propagate", "add", "refine", "remove", "add"], the next propagation
        would remove the removed object, and also propagate the two added/refined objects.

        Returns:
            propagation_type: one of ["propagation_full", "propagation_partial", "propagation_fetch"]
                - "propagation_full": run VG propagation for all objects
                - "propagation_partial": run SAM2 propagation for selected objects, useful for add/refine actions
                - "propagation_fetch": fetch existing VG predictions without running any propagation
            obj_ids: list of object ids to run SAM2 propagation on if propagation_type is "propagation_partial".

        TODO: (Jie) this function works for our current workflows, but may need more tests to ensure it works
        correctly with different action histories for future workflows.
        """
        action_history = inference_state["action_history"]
        if len(action_history) == 0:
            # we run propagation for the first time
            return "propagation_full", None

        if "propagation" in action_history[-1]["type"]:
            if action_history[-1]["type"] in ["propagation_fetch"]:
                # last propagation is direct fetch, we fetch existing predictions
                return "propagation_fetch", None
            elif action_history[-1]["type"] in [
                "propagation_partial",
                "propagation_full",
            ]:
                # we do fetch prediction if we have already run propagation twice or we have run
                # propagation once and it is from the first frame or last frame.
                if (
                    len(action_history) > 1
                    and action_history[-2]["type"]
                    in ["propagation_partial", "propagation_full"]
                ) or action_history[-1]["frame_idx"] in [
                    0,
                    inference_state["num_frames"] - 1,
                ]:
                    # we have run both forward and backward partial/full propagation
                    return "propagation_fetch", None
                else:
                    # we have run partial/full forward or backward propagation once, need run it for the rest of the frames
                    return action_history[-1]["type"], action_history[-1]["obj_ids"]

        # parse actions since last propagation
        obj_ids = []
        for action in action_history[::-1]:
            if "propagation" in action["type"]:
                # we reached the last propagation action, stop parsing
                break
            if action["type"] in ["add", "refine"]:
                obj_ids.extend(action["obj_ids"])
            # else action["type"] == "remove": noop
        obj_ids = list(set(obj_ids)) if len(obj_ids) > 0 else None
        propagation_type = (
            "propagation_partial" if obj_ids is not None else "propagation_fetch"
        )
        return propagation_type, obj_ids

    def remove_object(self, inference_state, obj_id, is_user_action=False):
        """
        We try to remove object from sam2 states on every GPU, it will do nothing
        for states without this object.
        """
        obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)
        assert obj_rank is not None, f"Object {obj_id} not found in any GPU."

        sam2_states_local = inference_state["sam2_inference_states"]
        if self.rank == obj_rank:
            self._sam2_remove_object(sam2_states_local, obj_id)

        if is_user_action:
            self.add_action_history(
                inference_state, action_type="remove", obj_ids=[obj_id]
            )

        # update metadata
        sam2_metadata = inference_state["sam2_metadata"]
        _obj_ids = sam2_metadata["obj_ids_per_gpu"][obj_rank]
        sam2_metadata["obj_ids_per_gpu"][obj_rank] = _obj_ids[_obj_ids != obj_id]
        sam2_metadata["num_obj_per_gpu"][obj_rank] = len(
            sam2_metadata["obj_ids_per_gpu"][obj_rank]
        )
        sam2_metadata["obj_ids_all_gpu"] = np.concatenate(
            sam2_metadata["obj_ids_per_gpu"]
        )
        sam2_metadata["obj_id_to_score"].pop(obj_id, None)
        # sam2_metadata["max_obj_id"] # we do not reuse the object id, so we do not update it here

    def _get_gpu_id_by_obj_id(self, inference_state, obj_id):
        """
        Locate GPU ID for a given object.
        """
        obj_ids_per_gpu = inference_state["sam2_metadata"]["obj_ids_per_gpu"]
        for rank, obj_ids in enumerate(obj_ids_per_gpu):
            if obj_id in obj_ids:
                return rank
        return None  # object not found in any GPU

    def _get_sam2_inference_states_by_obj_ids(self, inference_state, obj_ids):
        """
        Get the SAM2 inference states that contain the given object ids.
        This is used to run partial SAM2 propagation on a single object/bucket.
        Possibly multiple or zero states can be returned.
        """
        states = [
            state
            for state in inference_state["sam2_inference_states"]
            if set(obj_ids) & set(state["obj_ids"])
        ]
        return states

    def _prepare_backbone_feats(self, inference_state, frame_idx, reverse):
        input_batch = inference_state["input_batch"]
        feature_cache = inference_state["feature_cache"]
        num_frames = inference_state["num_frames"]
        geometric_prompt = (
            inference_state["constants"]["empty_geometric_prompt"]
            if inference_state["per_frame_geometric_prompt"][frame_idx] is None
            else inference_state["per_frame_geometric_prompt"][frame_idx]
        )
        _ = self.run_backbone_and_detection(
            frame_idx=frame_idx,
            num_frames=num_frames,
            reverse=reverse,
            input_batch=input_batch,
            geometric_prompt=geometric_prompt,
            feature_cache=feature_cache,
        )

    @torch.inference_mode()
    def add_prompt(
        self,
        inference_state,
        frame_idx,
        text_str=None,
        clear_old_points=True,
        points=None,
        point_labels=None,
        boxes_xywh=None,
        box_labels=None,
        clear_old_boxes=True,
        output_prob_thresh=0.5,
        obj_id=None,
        rel_coordinates=True,
    ):
        if points is not None:
            # SAM2 instance prompts
            assert (
                text_str is None and boxes_xywh is None
            ), "When points are provided, text_str and boxes_xywh must be None."
            assert (
                obj_id is not None
            ), "When points are provided, obj_id must be provided."
            return self.add_sam2_new_points(
                inference_state,
                frame_idx,
                obj_id=obj_id,
                points=points,
                labels=point_labels,
                clear_old_points=clear_old_points,
                rel_coordinates=rel_coordinates,
                use_prev_mem_frame=self.use_prev_mem_frame,
            )
        else:
            # SAM3 prompts
            return super().add_prompt(
                inference_state,
                frame_idx,
                text_str=text_str,
                clear_old_points=clear_old_points,
                points=points,
                point_labels=point_labels,
                boxes_xywh=boxes_xywh,
                box_labels=box_labels,
                clear_old_boxes=clear_old_boxes,
                output_prob_thresh=output_prob_thresh,
            )

    @torch.inference_mode()
    def add_sam2_new_points(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points,
        labels,
        clear_old_points,
        rel_coordinates=True,
        use_prev_mem_frame=False,
    ):
        """Add a new point prompt to SAM2. Suppporting instance refinement to existing
        objects by passing existing obj_id or adding a new object by passing a new obj_id.
        use_prev_mem_frame=False to disable cross attention to previous memory frames.
        """
        assert obj_id is not None, "obj_id must be provided to add new points"
        sam2_metadata = inference_state["sam2_metadata"]
        if sam2_metadata == {}:
            # initialize masklet metadata if it's uninitialized (empty dict)
            sam2_metadata.update(self._initialize_metadata())

        obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)

        # prepare feature
        self._prepare_backbone_feats(inference_state, frame_idx, reverse=False)

        if obj_rank is None:
            num_prev_obj = np.sum(sam2_metadata["num_obj_per_gpu"])
            if num_prev_obj >= self.max_num_objects:
                logger.warning(
                    f"add_sam2_new_points: cannot add a new object as we are already tracking {num_prev_obj=} "
                    f"masklets (under {self.max_num_objects=})"
                )
                obj_ids = []
                H_low_res = W_low_res = self.sam2_predictor.low_res_mask_size
                H_video_res = inference_state["orig_height"]
                W_video_res = inference_state["orig_width"]
                low_res_masks = torch.zeros(0, 1, H_low_res, W_low_res)
                video_res_masks = torch.zeros(0, 1, H_video_res, W_video_res)
                return frame_idx, obj_ids, low_res_masks, video_res_masks

            # new object, we assign it a GPU and create a new inference state
            new_det_gpu_ids = self._assign_new_det_to_gpus(
                new_det_num=1,
                num_prev_obj_per_gpu=sam2_metadata["num_obj_per_gpu"],
            )
            obj_rank = new_det_gpu_ids[0]

            # get sam2 inference state for the new object
            if self.rank == obj_rank:
                if self.sam2_predictor.per_obj_inference:
                    new_sam2_state = inference_state["sam2_inference_states"][0]
                else:
                    # for batched inference, we create a new inference state
                    new_sam2_state = self._init_new_sam2_state(inference_state)
                    inference_state["sam2_inference_states"].append(new_sam2_state)

            # update metadata
            sam2_metadata["obj_ids_per_gpu"][obj_rank] = np.concatenate(
                [
                    sam2_metadata["obj_ids_per_gpu"][obj_rank],
                    np.array([obj_id], dtype=np.int64),
                ]
            )
            sam2_metadata["num_obj_per_gpu"][obj_rank] = len(
                sam2_metadata["obj_ids_per_gpu"][obj_rank]
            )
            sam2_metadata["obj_ids_all_gpu"] = np.concatenate(
                sam2_metadata["obj_ids_per_gpu"]
            )
            sam2_metadata["max_obj_id"] = max(sam2_metadata["max_obj_id"], obj_id)
            sam2_metadata["obj_id_to_score"][
                obj_id
            ] = 1.0  # assign a high score to user added object

            logger.info(
                f"[rank={self.rank}] Adding new object with id {obj_id} at frame {frame_idx}."
            )
            self.add_action_history(
                inference_state, "add", frame_idx=frame_idx, obj_ids=[obj_id]
            )
        else:
            # existing object
            if self.rank == obj_rank:
                inference_states = self._get_sam2_inference_states_by_obj_ids(
                    inference_state, [obj_id]
                )
                assert (
                    len(inference_states) == 1
                ), f"[rank={self.rank}] Multiple SAM2 inference states found for the same object id."
                new_sam2_state = inference_states[0]

            # log
            logger.info(
                f"[rank={self.rank}] Refining existing object with id {obj_id} at frame {frame_idx}."
            )
            self.add_action_history(
                inference_state, "refine", frame_idx=frame_idx, obj_ids=[obj_id]
            )

        if self.rank == obj_rank:
            frame_idx, obj_ids, low_res_masks, video_res_masks = (
                self.sam2_predictor.add_new_points(
                    inference_state=new_sam2_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                    clear_old_points=clear_old_points,
                    rel_coordinates=rel_coordinates,
                    use_prev_mem_frame=use_prev_mem_frame,
                )
            )

        # broadcast results to all GPUs
        if self.rank == obj_rank and self.world_size > 1:
            output = [frame_idx, obj_ids, low_res_masks, video_res_masks]
            self.broadcast_python_obj_cpu(output, src=obj_rank)
        elif self.rank != obj_rank and self.world_size > 1:
            output = [None] * 4
            self.broadcast_python_obj_cpu(output, src=obj_rank)
            frame_idx, obj_ids, low_res_masks, video_res_masks = output
        # every GPU returns the same
        return frame_idx, obj_ids, low_res_masks, video_res_masks

    def get_sam2_output_per_frame(
        self, inference_state, frame_idx, resize_to_video_res=False
    ):
        """Get the SAM2 output for a particular frame."""
        output = {}  # obj_id --> mask
        for sam2_inference_state in inference_state["sam2_inference_states"]:
            output.update(
                self._get_sam2_output_per_frame(sam2_inference_state, frame_idx)
            )

        if resize_to_video_res and len(output) > 0:
            # resize the masks to the original video resolution
            all_obj_ids = list(output.keys())
            if len(all_obj_ids) > 1:
                all_pred_masks = torch.cat(
                    [output[obj_id] for obj_id in all_obj_ids], dim=0
                )
            else:
                all_pred_masks = output[all_obj_ids[0]]
            _, video_res_masks = self.sam2_predictor._get_orig_video_res_output(
                sam2_inference_state, all_pred_masks
            )  # any sam2_inference_state works since they have the same resolution
            output = {
                obj_id: self._convert_mask(video_res_masks[idx])
                for idx, obj_id in enumerate(all_obj_ids)
            }
        return output

    def _get_sam2_output_per_frame(self, sam2_inference_state, frame_idx):
        """Get the SAM2 output for a particular frame."""
        output = {}  # obj_id --> mask
        output_dict_per_obj = sam2_inference_state["output_dict_per_obj"]
        for obj_idx, obj_out in output_dict_per_obj.items():
            obj_id = sam2_inference_state["obj_idx_to_id"][obj_idx]
            for storage_key in ["non_cond_frame_outputs", "cond_frame_outputs"]:
                if storage_key not in obj_out:
                    continue
                if frame_idx not in obj_out[storage_key]:
                    continue
                output[obj_id] = obj_out[storage_key][frame_idx]["pred_masks"].squeeze(
                    0, 1
                )
                break
        return output
