from collections import defaultdict
from functools import reduce
from typing import Dict

import numpy as np
import sam3.model.sam3_multiplex_base
import sam3.model.sam3_video_base
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sam3 import perflib
from sam3.logger import get_logger
from sam3.model.box_ops import box_xywh_to_cxcywh, box_xyxy_to_xywh
from sam3.model.data_misc import BatchedDatapoint
from sam3.model.sam3_multiplex_base import MaskletConfirmationStatus, Sam3MultiplexBase
from sam3.model.sam3_tracker_utils import fill_holes_in_mask_scores
from sam3.model.sam3_video_inference import is_image_type
from sam3.perflib.compile import (
    clone_output_wrapper,
    compile_wrapper,
    shape_logging_wrapper,
)
from sam3.perflib.masks_ops import mask_iou, masks_to_boxes as perf_masks_to_boxes
from torch import Tensor
from torchvision.ops import masks_to_boxes
from tqdm.auto import tqdm

logger = get_logger(__name__)

import gc
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import List

from sam3.model.data_misc import (
    BatchedPointer,
    convert_my_tensors,
    FindStage,
    NestedTensor,
)
from sam3.model.geometry_encoders import Prompt
from sam3.model.io_utils import load_resource_as_video_frames


def recursive_to(data, *args, **kwargs):
    if isinstance(data, torch.Tensor):
        ret = data.to(*args, **kwargs)
    elif isinstance(data, np.ndarray):
        ret = data
    elif isinstance(data, Mapping):
        ret = type(data)()
        for key in data:
            ret[key] = recursive_to(data[key], *args, **kwargs)
    elif isinstance(data, tuple):
        ret = ()
        for value in data:
            ret += (recursive_to(value, *args, **kwargs),)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        ret = type(data)()
        for value in data:
            ret.append(recursive_to(value, *args, **kwargs))
    elif is_dataclass(data):
        ret_cls = type(data)
        ret_fields = {
            field.name: recursive_to(getattr(data, field.name), *args, **kwargs)
            for field in fields(data)
        }
        ret = ret_cls(**ret_fields)
    else:
        ret = data
    return ret


DUMMY_OUTPUT = "DUMMY_OUTPUT"


class Sam3MultiplexTracking(Sam3MultiplexBase):
    def __init__(
        self,
        image_size=1008,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        compile_model=False,
        postprocess_batch_size=1,
        **kwargs,
    ):
        """
        hotstart_delay: int, the delay (in #frames) before the model starts to yield output, 0 to disable hotstart delay.
        hotstart_unmatch_thresh: int, remove the object if it has this many unmatched frames within its hotstart_delay period.
            If `hotstart_delay` is set to 0, this parameter is ignored.
        hotstart_dup_thresh: int, remove the object if it has overlapped with another object this many frames within its hotstart_delay period.
        postprocess_batch_size: int, the number of frames to accumulate before running postprocessing. Set to 1 to disable batching.
        """
        super().__init__(**kwargs)
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.compile_model = compile_model
        self.detector.compile_model = self.compile_model
        self.postprocess_batch_size = postprocess_batch_size

    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2

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
                img_ids_np=np.array([stage_id]),
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
        with torch.profiler.record_function(
            "Sam3MultiplexTracking._construct_initial_input_batch"
        ):
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
        with torch.profiler.record_function("Sam3MultiplexTracking.recursive_to"):
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

    def _get_visual_prompt(self, inference_state, frame_idx, boxes_cxcywh, box_labels):
        batch_size = 1
        geometric_prompt = Prompt(
            box_embeddings=torch.zeros(
                0, batch_size, 4, device=inference_state["device"]
            ),
            box_mask=torch.zeros(
                batch_size, 0, device=inference_state["device"], dtype=torch.bool
            ),
            point_embeddings=None,
            point_mask=None,
        )

        geometric_prompt.append_boxes(
            boxes=boxes_cxcywh.view(-1, batch_size, 4).to(inference_state["device"]),
            labels=box_labels.view(-1, batch_size).to(inference_state["device"]),
        )

        return boxes_cxcywh, box_labels, geometric_prompt

    @torch.inference_mode()
    def init_state(
        self,
        resource_path,
        offload_video_to_cpu=False,
        async_loading_frames=False,
        use_torchcodec=False,
        use_cv2=False,
        input_is_mp4=False,
    ):
        # Initialize inference state (inlined from Sam3DemoMixin.init_state)
        if use_torchcodec:
            video_loader_type = "torchcodec"
        elif use_cv2:
            video_loader_type = "cv2"
        else:
            video_loader_type = "cv2"
        images, orig_height, orig_width = load_resource_as_video_frames(
            resource_path=resource_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=self.image_mean,
            img_std=self.image_std,
            async_loading_frames=async_loading_frames,
            video_loader_type=video_loader_type,
        )
        inference_state = {}
        inference_state["image_size"] = self.image_size
        inference_state["num_frames"] = len(images)
        inference_state["device"] = torch.device("cuda")
        inference_state["orig_height"] = orig_height
        inference_state["orig_width"] = orig_width
        inference_state["constants"] = {}
        self._construct_initial_input_batch(inference_state, images)
        # initialize extra states
        # sam2_inference_states will contain separate inference_states for each frame having new objects if
        # self.tracker.per_obj_inference is False (bucketized batching), or a single inference_state
        # containing all objects if self.tracker.per_obj_inference is True (no batching at all).
        inference_state["sam2_inference_states"] = []
        inference_state["tracker_metadata"] = {}
        inference_state["feature_cache"] = {}
        inference_state["cached_frame_outputs"] = {}
        inference_state["is_image_only"] = is_image_type(resource_path)
        return inference_state

    def reset_state(self, inference_state):
        # Inlined from Sam3DemoMixin.reset_state
        inference_state["input_batch"].find_text_batch[0] = "<text placeholder>"
        inference_state["text_prompt"] = None
        for t in range(inference_state["num_frames"]):
            inference_state["input_batch"].find_inputs[t].text_ids[...] = 0
            inference_state["previous_stages_out"][t] = None
            inference_state["per_frame_raw_point_input"][t] = None
            inference_state["per_frame_raw_box_input"][t] = None
            inference_state["per_frame_visual_prompt"][t] = None
            inference_state["per_frame_geometric_prompt"][t] = None
            inference_state["per_frame_cur_step"][t] = 0
        inference_state["backbone_out"] = None
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None
        # reset extra states
        inference_state["sam2_inference_states"].clear()
        inference_state["tracker_metadata"].clear()
        inference_state["feature_cache"].clear()
        inference_state["cached_frame_outputs"] = {}

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
        **kwargs,  # To support passing extra args to child classes
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

        # Store max_frame_num_to_track in feature_cache for downstream methods
        inference_state["feature_cache"]["tracking_bounds"] = {
            "max_frame_num_to_track": max_frame_num_to_track,
            "propagate_in_video_start_frame_idx": start_frame_idx,
        }

        hotstart_buffer = []
        hotstart_removed_obj_ids = set()
        # when deciding whether to output a masklet on `yield_frame_idx`, we check whether the object is confirmed
        # in a future frame (`unconfirmed_frame_delay` frames after the current frame). For example, if we require
        # an object to be detected in 3 consecutive frames to be confirmed, then we look 2 frames in the future --
        # e.g., we output an object on frame 4 only if it becomes confirmed on frame 6.
        unconfirmed_status_delay = self.masklet_confirmation_consecutive_det_thresh - 1
        unconfirmed_obj_ids_per_frame = {}  # frame_idx -> hidden_obj_ids

        # Batch postprocessing: accumulate yield_list entries and process every postprocess_batch_size frames
        postprocess_yield_list = []

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
                    unconfirmed_obj_ids = out.get("unconfirmed_obj_ids", None)
                    if unconfirmed_obj_ids is not None:
                        unconfirmed_obj_ids_per_frame[frame_idx] = unconfirmed_obj_ids

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

            # Accumulate yield_list into postprocess_yield_list
            # Snapshot hotstart_removed_obj_ids at the time of accumulation to preserve
            # the correct state for each frame (important: this set is mutated over time)
            for yield_frame_idx, yield_out in yield_list:
                postprocess_yield_list.append(
                    (yield_frame_idx, yield_out, set(hotstart_removed_obj_ids))
                )

            # Process batch when we have enough frames
            while len(postprocess_yield_list) >= self.postprocess_batch_size:
                batch_to_process = postprocess_yield_list[: self.postprocess_batch_size]
                postprocess_yield_list = postprocess_yield_list[
                    self.postprocess_batch_size :
                ]

                with torch.profiler.record_function(
                    "Sam3MultiplexTracking.postprocess_output_batched"
                ):
                    if self.rank == 0:
                        # Prepare batched inputs for postprocessing
                        H_video, W_video = (
                            inference_state["orig_height"],
                            inference_state["orig_width"],
                        )
                        num_frames = inference_state["num_frames"]

                        batched_outs = []
                        frame_indices = []
                        for (
                            yield_frame_idx,
                            yield_out,
                            removed_obj_ids_snapshot,
                        ) in batch_to_process:
                            suppressed_obj_ids = yield_out["suppressed_obj_ids"]
                            unconfirmed_status_frame_idx = (
                                yield_frame_idx + unconfirmed_status_delay
                                if not reverse
                                else yield_frame_idx - unconfirmed_status_delay
                            )
                            unconfirmed_status_frame_idx = max(
                                0, min(unconfirmed_status_frame_idx, num_frames - 1)
                            )
                            unconfirmed_obj_ids = unconfirmed_obj_ids_per_frame.get(
                                unconfirmed_status_frame_idx, None
                            )

                            batched_outs.append(
                                (
                                    yield_out,
                                    removed_obj_ids_snapshot,
                                    suppressed_obj_ids,
                                    unconfirmed_obj_ids,
                                )
                            )
                            frame_indices.append(yield_frame_idx)

                            # Cache frame outputs
                            self._cache_frame_outputs(
                                inference_state,
                                yield_frame_idx,
                                yield_out["obj_id_to_mask"],
                                suppressed_obj_ids=suppressed_obj_ids,
                                removed_obj_ids=removed_obj_ids_snapshot,
                                unconfirmed_obj_ids=unconfirmed_obj_ids,
                            )

                        if self.postprocess_batch_size > 1:
                            # Process all frames in batch
                            postprocessed_outs = self._postprocess_output_batched(
                                H_video, W_video, batched_outs
                            )
                        else:
                            # Process each frame individually but output together
                            postprocessed_outs = []
                            for (
                                yield_out,
                                removed_obj_ids_snapshot,
                                suppressed_obj_ids,
                                unconfirmed_obj_ids,
                            ) in batched_outs:
                                postprocessed_out = self._postprocess_output(
                                    inference_state,
                                    yield_out,
                                    removed_obj_ids_snapshot,
                                    suppressed_obj_ids,
                                    unconfirmed_obj_ids,
                                )
                                postprocessed_outs.append(postprocessed_out)

                        # Yield results
                        for yield_frame_idx, postprocessed_out in zip(
                            frame_indices, postprocessed_outs
                        ):
                            yield yield_frame_idx, postprocessed_out
                    else:
                        # No output on other GPUs
                        for yield_frame_idx, _, _ in batch_to_process:
                            yield yield_frame_idx, DUMMY_OUTPUT

        # Flush any remaining frames in the postprocess buffer
        if len(postprocess_yield_list) > 0:
            with torch.profiler.record_function(
                "Sam3MultiplexTracking.postprocess_output_batched"
            ):
                if self.rank == 0:
                    H_video, W_video = (
                        inference_state["orig_height"],
                        inference_state["orig_width"],
                    )
                    num_frames = inference_state["num_frames"]

                    batched_outs = []
                    frame_indices = []
                    for (
                        yield_frame_idx,
                        yield_out,
                        removed_obj_ids_snapshot,
                    ) in postprocess_yield_list:
                        suppressed_obj_ids = yield_out["suppressed_obj_ids"]
                        unconfirmed_status_frame_idx = (
                            yield_frame_idx + unconfirmed_status_delay
                            if not reverse
                            else yield_frame_idx - unconfirmed_status_delay
                        )
                        unconfirmed_status_frame_idx = max(
                            0, min(unconfirmed_status_frame_idx, num_frames - 1)
                        )
                        unconfirmed_obj_ids = unconfirmed_obj_ids_per_frame.get(
                            unconfirmed_status_frame_idx, None
                        )

                        batched_outs.append(
                            (
                                yield_out,
                                removed_obj_ids_snapshot,
                                suppressed_obj_ids,
                                unconfirmed_obj_ids,
                            )
                        )
                        frame_indices.append(yield_frame_idx)

                        self._cache_frame_outputs(
                            inference_state,
                            yield_frame_idx,
                            yield_out["obj_id_to_mask"],
                            suppressed_obj_ids=suppressed_obj_ids,
                            removed_obj_ids=removed_obj_ids_snapshot,
                            unconfirmed_obj_ids=unconfirmed_obj_ids,
                        )

                    if self.postprocess_batch_size > 1:
                        postprocessed_outs = self._postprocess_output_batched(
                            H_video, W_video, batched_outs
                        )
                    else:
                        # Process each frame individually but output together
                        postprocessed_outs = []
                        for (
                            yield_out,
                            removed_obj_ids_snapshot,
                            suppressed_obj_ids,
                            unconfirmed_obj_ids,
                        ) in batched_outs:
                            postprocessed_out = self._postprocess_output(
                                inference_state,
                                yield_out,
                                removed_obj_ids_snapshot,
                                suppressed_obj_ids,
                                unconfirmed_obj_ids,
                            )
                            postprocessed_outs.append(postprocessed_out)

                    for yield_frame_idx, postprocessed_out in zip(
                        frame_indices, postprocessed_outs
                    ):
                        yield yield_frame_idx, postprocessed_out
                else:
                    for yield_frame_idx, _, _ in postprocess_yield_list:
                        yield yield_frame_idx, DUMMY_OUTPUT

        if self.is_multiplex:
            # log the bucket utilization stats
            # bucket utilization rate is total valid objects / total capacity -> represents rooms for improvement
            # subscription rate is total valid objects / total number of buckets -> represents speedup
            total_valid_objects = 0
            total_num_buckets = 0
            for state in inference_state["sam2_inference_states"]:
                assert (
                    len(state["obj_ids"])
                    == state["multiplex_state"].total_valid_entries
                )
                total_valid_objects += len(state["obj_ids"])
                total_num_buckets += state["multiplex_state"].num_buckets
            if total_num_buckets > 0:
                bucket_utilization_rate = (
                    total_valid_objects / (total_num_buckets * self.bucket_capacity)
                ) * 100
                subscription_rate = (total_valid_objects / total_num_buckets) * 100
                logger.info(
                    f"Bucket utilization rate: {bucket_utilization_rate:.2f}%, subscription rate: {subscription_rate:.2f}%"
                )

    def _run_single_frame_inference(
        self,
        inference_state,
        frame_idx,
        reverse,
        is_instance_processing=False,
    ):
        """
        Perform inference on a single frame and get its inference results. This would
        also update `inference_state`.
        """
        # prepare inputs
        input_batch = inference_state["input_batch"]
        tracker_states_local = inference_state["sam2_inference_states"]
        geometric_prompt = (
            inference_state["constants"]["empty_geometric_prompt"]
            if inference_state["per_frame_geometric_prompt"][frame_idx] is None
            else inference_state["per_frame_geometric_prompt"][frame_idx]
        )
        text_batch_key = tuple(input_batch.find_text_batch)
        inference_state["feature_cache"]["text"] = {
            text_batch_key: {
                "language_features": inference_state["backbone_out"][
                    "language_features"
                ],
                "language_mask": inference_state["backbone_out"]["language_mask"],
            }
        }
        # run inference for the current frame
        (
            obj_id_to_mask,
            obj_id_to_score,
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            _,
        ) = self._det_track_one_frame(
            frame_idx=frame_idx,
            num_frames=inference_state["num_frames"],
            reverse=reverse,
            input_batch=input_batch,
            geometric_prompt=geometric_prompt,
            tracker_states_local=tracker_states_local,
            tracker_metadata_prev=inference_state["tracker_metadata"],
            feature_cache=inference_state["feature_cache"],
            orig_vid_height=inference_state["orig_height"],
            orig_vid_width=inference_state["orig_width"],
            is_image_only=inference_state["is_image_only"],
        )
        # update inference state
        inference_state["sam2_inference_states"] = tracker_states_local_new
        inference_state["tracker_metadata"] = tracker_metadata_new
        # use a dummy string in "previous_stages_out" to indicate this frame has outputs
        inference_state["previous_stages_out"][frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"

        if self.rank == 0:
            self._cache_frame_outputs(inference_state, frame_idx, obj_id_to_mask)

        out = {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": obj_id_to_score,  # first frame detection score
            "obj_id_to_sam2_score": tracker_metadata_new[
                "obj_id_to_sam2_score_frame_wise"
            ][frame_idx],
        }
        # removed_obj_ids is only needed on rank 0 to handle hotstart delay buffer
        if self.rank == 0:
            rank0_metadata = tracker_metadata_new["rank0_metadata"]
            removed_obj_ids = rank0_metadata["removed_obj_ids"]
            out["removed_obj_ids"] = removed_obj_ids
            out["suppressed_obj_ids"] = rank0_metadata["suppressed_obj_ids"][frame_idx]
            out["frame_stats"] = frame_stats
            if self.masklet_confirmation_enable:
                status = rank0_metadata["masklet_confirmation"]["status"]
                is_unconfirmed = status == MaskletConfirmationStatus.UNCONFIRMED.value
                out["unconfirmed_obj_ids"] = tracker_metadata_new["obj_ids_all_gpu"][
                    is_unconfirmed
                ].tolist()
            else:
                out["unconfirmed_obj_ids"] = []

        return out

    def _postprocess_output(
        self,
        inference_state,
        out,
        removed_obj_ids=None,
        suppressed_obj_ids=None,
        unconfirmed_obj_ids=None,
    ):
        obj_id_to_mask = out["obj_id_to_mask"]  # low res masks
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
            out_sam2_probs = torch.tensor(
                [
                    (
                        out["obj_id_to_sam2_score"][obj_id]
                        if obj_id in out["obj_id_to_sam2_score"]
                        else 0.0
                    )
                    for obj_id in curr_obj_ids
                ]
            )
            out_binary_masks = torch.cat(
                [obj_id_to_mask[obj_id] for obj_id in curr_obj_ids], dim=0
            )

            assert out_binary_masks.dtype == torch.bool
            keep = out_binary_masks.any(dim=(1, 2)).cpu()  # remove masks with 0 areas
            # hide outputs for those object IDs in `obj_ids_to_hide`
            obj_ids_to_hide = []
            if suppressed_obj_ids is not None:
                obj_ids_to_hide.extend(suppressed_obj_ids)
            if removed_obj_ids is not None:
                obj_ids_to_hide.extend(removed_obj_ids)
            if unconfirmed_obj_ids is not None:
                obj_ids_to_hide.extend(unconfirmed_obj_ids)
            if len(obj_ids_to_hide) > 0:
                obj_ids_to_hide_t = torch.tensor(obj_ids_to_hide, dtype=torch.int64)
                keep &= ~torch.isin(out_obj_ids, obj_ids_to_hide_t)

            # slice those valid entries from the original outputs
            keep_idx = torch.nonzero(keep, as_tuple=True)[0]
            keep_idx_gpu = keep_idx.pin_memory().to(
                device=out_binary_masks.device, non_blocking=True
            )

            out_obj_ids = torch.index_select(out_obj_ids, 0, keep_idx)
            out_probs = torch.index_select(out_probs, 0, keep_idx)
            out_sam2_probs = torch.index_select(out_sam2_probs, 0, keep_idx)
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

        # apply non-overlapping constraints on the existing masklets
        if out_binary_masks.shape[0] > 1:
            assert len(out_binary_masks) == len(out_sam2_probs)
            out_binary_masks = (
                self.tracker._apply_object_wise_non_overlapping_constraints(
                    out_binary_masks.unsqueeze(1),
                    out_sam2_probs.unsqueeze(1).to(out_binary_masks.device),
                    background_value=0,
                ).squeeze(1)
            ) > 0

        prod_outputs = {}
        if self.running_in_prod:
            with torch.profiler.record_function(
                "Sam3MultiplexTracking._postprocess_output.prod_outputs"
            ):
                out_centers = torch.zeros(
                    out_binary_masks.shape[0],
                    2,
                    dtype=torch.float32,
                    device=out_binary_masks.device,
                )

                y_coords = torch.arange(
                    H_video, device=out_binary_masks.device, dtype=torch.float32
                )
                x_coords = torch.arange(
                    W_video, device=out_binary_masks.device, dtype=torch.float32
                )
                y_grid = y_coords.view(1, H_video, 1)
                x_grid = x_coords.view(1, 1, W_video)
                with torch.profiler.record_function(
                    "Sam3MultiplexTracking._postprocess_output.prod_outputs.center"
                ):
                    weighted_y_sum = (out_binary_masks * y_grid).sum(dim=(1, 2))
                    weighted_x_sum = (out_binary_masks * x_grid).sum(dim=(1, 2))
                    total_mass = out_binary_masks.sum(dim=(1, 2)).clamp_min(1e-6)
                    center_y = weighted_y_sum / total_mass / H_video
                    center_x = weighted_x_sum / total_mass / W_video
                    out_centers[:, 0] = center_x
                    out_centers[:, 1] = center_y

                with torch.profiler.record_function(
                    "Sam3MultiplexTracking._postprocess_output.prod_outputs.to_cpu"
                ):
                    prod_outputs["out_centers"] = out_centers.cpu().numpy()

        outputs = {
            "out_obj_ids": out_obj_ids.cpu().numpy(),
            "out_probs": out_probs.cpu().numpy(),
            "out_boxes_xywh": out_boxes_xywh.cpu().numpy(),
            "out_binary_masks": out_binary_masks.cpu().numpy(),
            "frame_stats": out.get("frame_stats", None),
        } | prod_outputs

        return outputs

    def _postprocess_output_batched(
        self,
        H_video,
        W_video,
        batched_outs,
    ):
        """
        Batched version of _postprocess_output that batches GPU computations
        (keep filtering, box computation) across frames for efficiency.

        Args:
            H_video: Video height
            W_video: Video width
            batched_outs: List of tuples, each containing:
                (out, removed_obj_ids, suppressed_obj_ids, unconfirmed_obj_ids)
                where out is the output dict from _run_single_frame_inference

        Returns:
            List of output dicts, one per frame in batched_outs
        """
        batch_size = len(batched_outs)
        if batch_size == 0:
            return []

        # ========== Phase 1: Collect per-frame data ==========
        # We'll track: frame_data[i] = (obj_ids, probs, sam2_probs, masks, keep_mask, frame_stats)
        # or None if frame has no objects
        frame_data = []
        device = None

        for (
            out,
            removed_obj_ids,
            suppressed_obj_ids,
            unconfirmed_obj_ids,
        ) in batched_outs:
            obj_id_to_mask = out["obj_id_to_mask"]
            curr_obj_ids = sorted(obj_id_to_mask.keys())
            frame_stats = out.get("frame_stats", None)

            if len(curr_obj_ids) == 0:
                frame_data.append((None, None, None, None, None, frame_stats))
                continue

            out_obj_ids = torch.tensor(curr_obj_ids, dtype=torch.int64)
            obj_id_to_score_dict = out["obj_id_to_score"]
            obj_id_to_sam2_score = out["obj_id_to_sam2_score"]

            if device is None:
                device = obj_id_to_mask[curr_obj_ids[0]].device
            default_sam2_score = torch.zeros((), dtype=torch.float32, device=device)

            probs_list = []
            sam2_probs_list = []
            binary_masks_list = []

            for obj_id in curr_obj_ids:
                probs_list.append(obj_id_to_score_dict[obj_id])
                sam2_probs_list.append(
                    obj_id_to_sam2_score.get(obj_id, default_sam2_score)
                )
                binary_masks_list.append(obj_id_to_mask[obj_id])

            out_probs = torch.tensor(probs_list, dtype=torch.float32)
            out_sam2_probs_gpu = torch.stack(sam2_probs_list)
            out_binary_masks = torch.cat(binary_masks_list, dim=0)

            # Compute keep mask (which objects to hide)
            obj_ids_to_hide = []
            if suppressed_obj_ids is not None:
                obj_ids_to_hide.extend(suppressed_obj_ids)
            if removed_obj_ids is not None:
                obj_ids_to_hide.extend(removed_obj_ids)
            if unconfirmed_obj_ids is not None:
                obj_ids_to_hide.extend(unconfirmed_obj_ids)

            if len(obj_ids_to_hide) > 0:
                obj_ids_to_hide_t = torch.tensor(obj_ids_to_hide, dtype=torch.int64)
                hide_mask = torch.isin(out_obj_ids, obj_ids_to_hide_t)
            else:
                hide_mask = torch.zeros(len(out_obj_ids), dtype=torch.bool)

            frame_data.append(
                (
                    out_obj_ids,
                    out_probs,
                    out_sam2_probs_gpu,
                    out_binary_masks,
                    hide_mask,
                    frame_stats,
                )
            )

        # ========== Phase 2: Batch concatenate masks for GPU operations ==========
        # Collect frames with objects
        frames_with_objects = []
        frame_obj_counts = []  # Number of objects per frame (for frames with objects only)
        all_masks_list = []
        all_hide_masks_list = []

        for i, data in enumerate(frame_data):
            if data[0] is not None:
                frames_with_objects.append(i)
                frame_obj_counts.append(data[0].shape[0])
                all_masks_list.append(data[3])  # binary_masks
                all_hide_masks_list.append(data[4])  # hide_mask

        # Handle case where all frames have 0 objects
        if len(frames_with_objects) == 0:
            outputs = []
            for data in frame_data:
                output_dict = {
                    "out_obj_ids": np.zeros(0, dtype=np.int64),
                    "out_probs": np.zeros(0, dtype=np.float32),
                    "out_boxes_xywh": np.zeros((0, 4), dtype=np.float32),
                    "out_binary_masks": np.zeros((0, H_video, W_video), dtype=bool),
                    "frame_stats": data[5],
                }
                if self.running_in_prod:
                    output_dict["out_centers"] = np.zeros((0, 2), dtype=np.float32)
                outputs.append(output_dict)
            return outputs

        # Concatenate all masks for batched GPU operations
        all_masks = torch.cat(all_masks_list, dim=0)
        all_hide_masks = torch.cat(all_hide_masks_list, dim=0)

        # ========== Phase 3: Batched keep mask computation on GPU ==========
        # Compute which masks have non-zero area (batched on GPU)
        has_area = all_masks.any(dim=(1, 2))  # GPU operation

        # Combine with hide mask (move hide_mask to GPU for the operation)
        all_hide_masks_gpu = all_hide_masks.to(device=all_masks.device)
        keep_mask_gpu = has_area & ~all_hide_masks_gpu

        # Get keep indices
        keep_indices = torch.nonzero(keep_mask_gpu, as_tuple=True)[0]

        if len(keep_indices) == 0:
            # All objects filtered out
            outputs = []
            for data in frame_data:
                output_dict = {
                    "out_obj_ids": np.zeros(0, dtype=np.int64),
                    "out_probs": np.zeros(0, dtype=np.float32),
                    "out_boxes_xywh": np.zeros((0, 4), dtype=np.float32),
                    "out_binary_masks": np.zeros((0, H_video, W_video), dtype=bool),
                    "frame_stats": data[5],
                }
                if self.running_in_prod:
                    output_dict["out_centers"] = np.zeros((0, 2), dtype=np.float32)
                outputs.append(output_dict)
            return outputs

        # ========== Phase 4: Batched filtering and box computation ==========
        # Filter masks on GPU
        kept_masks = torch.index_select(all_masks, 0, keep_indices)

        # Compute bounding boxes in batch on GPU
        if perflib.is_enabled:
            # Need to gather obj_ids for perflib
            all_obj_ids_list = [frame_data[i][0] for i in frames_with_objects]
            all_obj_ids_cat = torch.cat(all_obj_ids_list, dim=0)
            kept_obj_ids_for_perf = torch.index_select(
                all_obj_ids_cat, 0, keep_indices.cpu()
            )
            kept_boxes_xyxy = perf_masks_to_boxes(
                kept_masks, kept_obj_ids_for_perf.tolist()
            )
        else:
            kept_boxes_xyxy = masks_to_boxes(kept_masks)

        kept_boxes_xywh = box_xyxy_to_xywh(kept_boxes_xyxy)
        kept_boxes_xywh[..., 0] /= W_video
        kept_boxes_xywh[..., 1] /= H_video
        kept_boxes_xywh[..., 2] /= W_video
        kept_boxes_xywh[..., 3] /= H_video

        # ========== Phase 5: Split back to per-frame for non-overlapping ==========
        # Compute how many objects were kept per frame
        keep_indices_cpu = keep_indices.cpu()
        keep_set = set(keep_indices_cpu.tolist())

        kept_counts = []
        offset = 0
        for count in frame_obj_counts:
            kept_in_frame = sum(
                1 for j in range(offset, offset + count) if j in keep_set
            )
            kept_counts.append(kept_in_frame)
            offset += count

        # Split the kept tensors back to per-frame
        split_masks = torch.split(kept_masks, kept_counts)
        split_boxes = torch.split(kept_boxes_xywh, kept_counts)

        # Also need to split obj_ids, probs, sam2_probs (filtering from original frame_data)
        # We need to track which original indices were kept per frame
        frame_kept_indices = []  # List of (local_kept_indices) per frame
        offset = 0
        for count in frame_obj_counts:
            local_kept = []
            for j in range(offset, offset + count):
                if j in keep_set:
                    local_kept.append(j - offset)  # Local index within frame
            frame_kept_indices.append(local_kept)
            offset += count

        # ========== Phase 6: Apply non-overlapping per frame, collect final results ==========
        final_results = []  # List of (frame_idx, obj_ids, probs, boxes, masks)

        for idx, frame_i in enumerate(frames_with_objects):
            data = frame_data[frame_i]
            local_kept = frame_kept_indices[idx]

            if len(local_kept) == 0:
                continue

            # Get the filtered data for this frame
            local_kept_t = torch.tensor(local_kept, dtype=torch.int64)
            out_obj_ids = torch.index_select(data[0], 0, local_kept_t)
            out_probs = torch.index_select(data[1], 0, local_kept_t)
            out_sam2_probs = torch.index_select(
                data[2], 0, local_kept_t.to(data[2].device)
            )
            out_masks = split_masks[idx]
            out_boxes = split_boxes[idx]

            # Apply non-overlapping constraints (per-frame operation)
            if out_masks.shape[0] > 1:
                # Copy sam2_probs to CPU pinned memory then back to GPU for the operation
                out_sam2_probs_cpu = torch.empty(
                    out_sam2_probs.shape, dtype=out_sam2_probs.dtype, pin_memory=True
                )
                out_sam2_probs_cpu.copy_(out_sam2_probs, non_blocking=True)
                out_masks = (
                    self.tracker._apply_object_wise_non_overlapping_constraints(
                        out_masks.unsqueeze(1),
                        out_sam2_probs_cpu.unsqueeze(1).to(out_masks.device),
                        background_value=0,
                    ).squeeze(1)
                ) > 0

            final_results.append(
                (frame_i, out_obj_ids, out_probs, out_boxes, out_masks)
            )

        # ========== Phase 6.5: Compute centers for prod ==========
        all_centers = None
        if self.running_in_prod and len(final_results) > 0:
            with torch.profiler.record_function(
                "Sam3MultiplexTracking._postprocess_output_batched.prod_outputs"
            ):
                # Concatenate all masks for batched center computation
                all_masks = torch.cat([r[4] for r in final_results], dim=0)
                if all_masks.shape[0] > 0:
                    y_coords = torch.arange(
                        H_video, device=all_masks.device, dtype=torch.float32
                    )
                    x_coords = torch.arange(
                        W_video, device=all_masks.device, dtype=torch.float32
                    )
                    y_grid = y_coords.view(1, H_video, 1)
                    x_grid = x_coords.view(1, 1, W_video)

                    weighted_y_sum = (all_masks * y_grid).sum(dim=(1, 2))
                    weighted_x_sum = (all_masks * x_grid).sum(dim=(1, 2))
                    total_mass = all_masks.sum(dim=(1, 2)).clamp_min(1e-6)
                    center_y = weighted_y_sum / total_mass / H_video
                    center_x = weighted_x_sum / total_mass / W_video
                    all_centers = torch.stack([center_x, center_y], dim=1)

        # Handle case where all filtered out
        if len(final_results) == 0:
            outputs = []
            for data in frame_data:
                output_dict = {
                    "out_obj_ids": np.zeros(0, dtype=np.int64),
                    "out_probs": np.zeros(0, dtype=np.float32),
                    "out_boxes_xywh": np.zeros((0, 4), dtype=np.float32),
                    "out_binary_masks": np.zeros((0, H_video, W_video), dtype=bool),
                    "frame_stats": data[5],
                }
                if self.running_in_prod:
                    output_dict["out_centers"] = np.zeros((0, 2), dtype=np.float32)
                outputs.append(output_dict)
            return outputs

        # ========== Phase 7: Concatenate for batched GPU→CPU copy ==========
        final_obj_ids = torch.cat([r[1] for r in final_results], dim=0)
        final_probs = torch.cat([r[2] for r in final_results], dim=0)
        final_boxes = torch.cat([r[3] for r in final_results], dim=0)
        final_masks = torch.cat([r[4] for r in final_results], dim=0)

        total_objects = final_obj_ids.shape[0]

        # Initialize or resize batched CPU buffer
        batched_buffer_size = self.postprocess_batch_size * self.max_num_objects
        needs_buffer_init = not hasattr(self, "buffer_cpu_batched")
        needs_buffer_resize = not needs_buffer_init and (
            self.buffer_cpu_batched["out_binary_masks"].shape[0] != batched_buffer_size
            or self.buffer_cpu_batched["out_binary_masks"].shape[1] != H_video
            or self.buffer_cpu_batched["out_binary_masks"].shape[2] != W_video
        )

        if needs_buffer_init or needs_buffer_resize:
            self.buffer_cpu_batched = {
                "out_obj_ids": torch.zeros(
                    batched_buffer_size,
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=True,
                ),
                "out_probs": torch.zeros(
                    batched_buffer_size,
                    dtype=torch.float32,
                    device="cpu",
                    pin_memory=True,
                ),
                "out_boxes_xywh": torch.zeros(
                    batched_buffer_size,
                    4,
                    dtype=torch.float32,
                    device="cpu",
                    pin_memory=True,
                ),
                "out_binary_masks": torch.zeros(
                    batched_buffer_size,
                    H_video,
                    W_video,
                    dtype=bool,
                    device="cpu",
                    pin_memory=True,
                ),
            }
            if self.running_in_prod:
                self.buffer_cpu_batched["out_centers"] = torch.zeros(
                    batched_buffer_size,
                    2,
                    dtype=torch.float32,
                    device="cpu",
                    pin_memory=True,
                )

        self.buffer_cpu_batched["out_obj_ids"][:total_objects].copy_(final_obj_ids)
        self.buffer_cpu_batched["out_probs"][:total_objects].copy_(final_probs)
        self.buffer_cpu_batched["out_boxes_xywh"][:total_objects].copy_(final_boxes)
        self.buffer_cpu_batched["out_binary_masks"][:total_objects].copy_(final_masks)

        if all_centers is not None:
            self.buffer_cpu_batched["out_centers"][:total_objects].copy_(all_centers)

        # ========== Phase 8: Build output list ==========
        # Create mapping from frame index to (offset, count) in the buffer
        frame_to_offset_count = {}
        offset = 0
        for frame_i, obj_ids, _, _, _ in final_results:
            count = obj_ids.shape[0]
            frame_to_offset_count[frame_i] = (offset, count)
            offset += count

        outputs = []
        for i, data in enumerate(frame_data):
            frame_stats = data[5]
            if i not in frame_to_offset_count:
                # Frame has no objects (either originally or after filtering)
                output_dict = {
                    "out_obj_ids": np.zeros(0, dtype=np.int64),
                    "out_probs": np.zeros(0, dtype=np.float32),
                    "out_boxes_xywh": np.zeros((0, 4), dtype=np.float32),
                    "out_binary_masks": np.zeros((0, H_video, W_video), dtype=bool),
                    "frame_stats": frame_stats,
                }
                if all_centers is not None:
                    output_dict["out_centers"] = np.zeros((0, 2), dtype=np.float32)
                outputs.append(output_dict)
            else:
                buf_offset, num_objects = frame_to_offset_count[i]
                output_dict = {
                    "out_obj_ids": self.buffer_cpu_batched["out_obj_ids"][
                        buf_offset : buf_offset + num_objects
                    ]
                    .numpy()
                    .copy(),
                    "out_probs": self.buffer_cpu_batched["out_probs"][
                        buf_offset : buf_offset + num_objects
                    ]
                    .numpy()
                    .copy(),
                    "out_boxes_xywh": self.buffer_cpu_batched["out_boxes_xywh"][
                        buf_offset : buf_offset + num_objects
                    ]
                    .numpy()
                    .copy(),
                    "out_binary_masks": self.buffer_cpu_batched["out_binary_masks"][
                        buf_offset : buf_offset + num_objects
                    ]
                    .numpy()
                    .copy(),
                    "frame_stats": frame_stats,
                }
                if all_centers is not None:
                    output_dict["out_centers"] = (
                        self.buffer_cpu_batched["out_centers"][
                            buf_offset : buf_offset + num_objects
                        ]
                        .numpy()
                        .copy()
                    )
                outputs.append(output_dict)

        return outputs

    def _cache_frame_outputs(
        self,
        inference_state,
        frame_idx,
        obj_id_to_mask,
        suppressed_obj_ids=None,
        removed_obj_ids=None,
        unconfirmed_obj_ids=None,
    ):
        if "cached_frame_outputs" not in inference_state:
            inference_state["cached_frame_outputs"] = {}

        # Filter out suppressed, removed, and unconfirmed objects from the cache
        filtered_obj_id_to_mask = obj_id_to_mask.copy()

        objects_to_exclude = set()
        if suppressed_obj_ids is not None:
            objects_to_exclude.update(suppressed_obj_ids)
        if removed_obj_ids is not None:
            objects_to_exclude.update(removed_obj_ids)
        if unconfirmed_obj_ids is not None:
            objects_to_exclude.update(unconfirmed_obj_ids)

        if objects_to_exclude:
            for obj_id in objects_to_exclude:
                if obj_id in filtered_obj_id_to_mask:
                    del filtered_obj_id_to_mask[obj_id]

        inference_state["cached_frame_outputs"][frame_idx] = filtered_obj_id_to_mask

    def _build_sam2_output(
        self, inference_state, frame_idx, refined_obj_id_to_mask=None
    ):
        if not frame_idx in inference_state["cached_frame_outputs"]:
            return {}

        cached_outputs = inference_state["cached_frame_outputs"][frame_idx]
        obj_id_to_mask = cached_outputs.copy()

        # Update with refined masks if provided
        if refined_obj_id_to_mask is not None:
            for obj_id, refined_mask in refined_obj_id_to_mask.items():
                assert refined_mask is not None, (
                    f"Refined mask data must be provided for obj_id {obj_id}"
                )
                obj_id_to_mask[obj_id] = refined_mask

        return obj_id_to_mask

    def _compile_model(self):
        """Compile the SAM model with torch.compile for speedup."""
        # TODO: compile SAM2 model components
        is_compiled = getattr(self, "_model_is_compiled", False)
        if is_compiled or not self.compile_model:
            return

        import torch._dynamo

        # a larger cache size to hold varying number of shapes for torch.compile
        # see https://github.com/pytorch/pytorch/blob/v2.5.1/torch/_dynamo/config.py#L42-L49
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True

        # Compile module components following https://www.internalfb.com/diff/D70935785
        # skip compilation of `_encode_prompt` since it sometimes tiggger SymInt errors
        # self._encode_prompt = clone_output_wrapper(
        #     torch.compile(self._encode_prompt, fullgraph=True, mode="max-autotune")
        # )

        ## Compile SAM3 model components (matching OV: clone_output_wrapper(torch.compile(fn)))
        self.detector.backbone.language_backbone.encoder.forward = clone_output_wrapper(
            torch.compile(
                self.detector.backbone.language_backbone.encoder.forward,
                fullgraph=True,
                mode="max-autotune",
            )
        )

        self.detector.backbone.vision_backbone.forward = clone_output_wrapper(
            torch.compile(
                self.detector.backbone.vision_backbone.forward,
                fullgraph=True,
                mode="max-autotune",
            )
        )
        self.detector.transformer.encoder.forward = clone_output_wrapper(
            torch.compile(
                self.detector.transformer.encoder.forward,
                fullgraph=True,
                mode="max-autotune",
            )
        )
        self.detector.transformer.decoder.forward = clone_output_wrapper(
            torch.compile(
                self.detector.transformer.decoder.forward,
                fullgraph=True,
                mode="max-autotune",
                dynamic=False,  # note: FA decoder uses static shapes
            )
        )

        self.detector.segmentation_head.forward = clone_output_wrapper(
            torch.compile(
                self.detector.segmentation_head.forward,
                fullgraph=True,
                mode="max-autotune",
            )
        )

        ## Compile SAM2 model components
        self.tracker.maskmem_backbone.forward = compile_wrapper(
            self.tracker.maskmem_backbone.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

        self.tracker.transformer.encoder.forward = shape_logging_wrapper(
            compile_wrapper(
                self.tracker.transformer.encoder.forward,
                mode="max-autotune-no-cudagraphs",
                fullgraph=True,
                dynamic=True,
            ),
            keep_kwargs=["src", "src_pos", "prompt", "prompt_pos"],
        )

        self.tracker.sam_mask_decoder.forward = compile_wrapper(
            self.tracker.sam_mask_decoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,  # Accuracy regression on True
        )

        sam3.model.sam3_video_base._associate_det_trk_compilable = compile_wrapper(
            sam3.model.sam3_video_base._associate_det_trk_compilable,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
            dynamic=False,
        )

        self.tracker._suppress_object_pw_area_shrinkage = compile_wrapper(
            self.tracker._suppress_object_pw_area_shrinkage,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
            dynamic=False,
        )

        self._model_is_compiled = True

    def _warm_up_vg_propagation(self, inference_state, start_frame_idx=0):
        # use different tracking score thresholds for each round to simulate different number of output objects
        num_objects_list = range(self.num_obj_for_compile + 1)
        num_rounds = 3
        orig_new_det_thresh = self.new_det_thresh
        for i in range(num_rounds):
            for num_objects in num_objects_list:
                logger.info(
                    f"round {i + 1}/{num_rounds} warming up model compilation -- simulating {num_objects}/{self.num_obj_for_compile} objects"
                )
                # Initialize text prompt and cache image features
                self.add_prompt(
                    inference_state, frame_idx=start_frame_idx, text_str="cat"
                )
                if num_objects > 0:
                    inference_state = self.add_fake_objects_to_inference_state(
                        inference_state, num_objects, frame_idx=start_frame_idx
                    )
                inference_state["tracker_metadata"]["rank0_metadata"].update(
                    {
                        "masklet_confirmation": {
                            "status": np.zeros(num_objects, dtype=np.int64),
                            "consecutive_det_num": np.zeros(
                                num_objects, dtype=np.int64
                            ),
                        }
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
                    f"{i + 1}/{num_rounds} warming up model compilation -- completed round {i + 1} out of {num_rounds}"
                )

        # Warm up SAM2 memory encoder with varying input shapes
        num_iters = 3
        feat_size = self.tracker.sam_image_embedding_size**2  # 72 * 72 = 5184
        hidden_dim = self.tracker.hidden_dim  # 256
        mem_dim = self.tracker.mem_dim  # 64 for non-multiplex, 256 for multiplex
        is_multiplex = self.tracker.is_multiplex

        for _ in tqdm(range(num_iters)):
            for b in range(1, self.num_obj_for_compile + 1):
                for i in range(
                    1,
                    self.tracker.max_cond_frames_in_attn + self.tracker.num_maskmem,
                ):
                    for j in range(
                        self.tracker.max_cond_frames_in_attn
                        + self.tracker.max_obj_ptrs_in_encoder
                    ):
                        if is_multiplex:
                            # Multiplex encoder: mem_dim == hidden_dim, uses decoupled cross-attention
                            # num_obj_ptr_tokens = j (since hidden_dim // mem_dim = 1)
                            num_obj_ptr_tokens = j
                            memory_seq_len = feat_size * i + num_obj_ptr_tokens

                            # src and memory have batch=num_buckets (b)
                            src = torch.randn(
                                feat_size, b, hidden_dim, device=self.device
                            )
                            src_pos = torch.randn(
                                feat_size, b, hidden_dim, device=self.device
                            )
                            memory = torch.randn(
                                memory_seq_len, b, hidden_dim, device=self.device
                            )
                            memory_pos = torch.randn(
                                memory_seq_len, b, hidden_dim, device=self.device
                            )

                            # image and memory_image always have batch=1 (shared image features)
                            image = torch.randn(
                                feat_size, 1, hidden_dim, device=self.device
                            )
                            image_pos = torch.randn(
                                feat_size, 1, hidden_dim, device=self.device
                            )
                            memory_image = torch.randn(
                                feat_size * i, 1, hidden_dim, device=self.device
                            )
                            memory_image_pos = torch.randn(
                                feat_size * i, 1, hidden_dim, device=self.device
                            )

                            self.tracker.transformer.encoder.forward(
                                image=image,
                                src=src,
                                memory_image=memory_image,
                                memory=memory,
                                image_pos=image_pos,
                                src_pos=src_pos,
                                memory_image_pos=memory_image_pos,
                                memory_pos=memory_pos,
                                num_obj_ptr_tokens=num_obj_ptr_tokens,
                            )
                        else:
                            # Non-multiplex encoder: mem_dim = 64, uses standard cross-attention
                            # num_obj_ptr_tokens = (hidden_dim // mem_dim) * j = 4 * j
                            num_obj_ptr_tokens = (hidden_dim // mem_dim) * j
                            src = torch.randn(
                                feat_size, b, hidden_dim, device=self.device
                            )
                            src_pos = torch.randn(
                                feat_size, b, hidden_dim, device=self.device
                            )
                            prompt = torch.randn(
                                feat_size * i + num_obj_ptr_tokens,
                                b,
                                mem_dim,
                                device=self.device,
                            )
                            prompt_pos = torch.randn(
                                feat_size * i + num_obj_ptr_tokens,
                                b,
                                mem_dim,
                                device=self.device,
                            )

                            self.tracker.transformer.encoder.forward(
                                src=src,
                                src_pos=src_pos,
                                prompt=prompt,
                                prompt_pos=prompt_pos,
                                num_obj_ptr_tokens=num_obj_ptr_tokens,
                            )

        # Warm up different number of kbox
        for _ in tqdm(range(num_iters)):
            for i in range(1, self.max_num_kboxes + 1):
                kboxes = (
                    torch.rand(i, 4, dtype=torch.float32) * 0.5
                )  # Generate positive values between 0 and 1
                print(
                    "Warming up masks_to_boxes with",
                    i,
                    f"kboxes.shape={kboxes.shape}",
                )
                self.add_prompt(
                    inference_state,
                    frame_idx=start_frame_idx,
                    text_str="cat",
                    boxes_xywh=kboxes,
                    box_labels=[1] * len(kboxes),
                )

                for _ in self.propagate_in_video(
                    inference_state, start_frame_idx, reverse=False
                ):
                    pass

        self.new_det_thresh = orig_new_det_thresh
        return inference_state

    def add_fake_objects_to_inference_state(
        self, inference_state, num_objects, frame_idx
    ):
        new_det_obj_ids_local = np.arange(num_objects)
        high_res_H, high_res_W = (
            self.tracker.maskmem_backbone.mask_downsampler.interpol_size
        )
        new_det_masks = torch.ones(
            len(new_det_obj_ids_local), high_res_H, high_res_W
        ).to(self.device)

        inference_state["sam2_inference_states"] = self._tracker_add_new_objects(
            frame_idx=frame_idx,
            num_frames=inference_state["num_frames"],
            new_obj_ids=new_det_obj_ids_local,
            new_obj_masks=new_det_masks,
            tracker_states_local=inference_state["sam2_inference_states"],
            orig_vid_height=inference_state["orig_height"],
            orig_vid_width=inference_state["orig_width"],
            feature_cache=inference_state["feature_cache"],
        )

        # Synthesize obj_id_to_mask data for cached_frame_outputs to support _build_sam2_output during warmup
        obj_id_to_mask = {}
        if num_objects > 0:
            H_video = inference_state["orig_height"]
            W_video = inference_state["orig_width"]

            video_res_masks = F.interpolate(
                new_det_masks.unsqueeze(1),  # Add channel dimension for interpolation
                size=(H_video, W_video),
                mode="bilinear",
                align_corners=False,
            )  # (num_objects, 1, H_video, W_video)
            for i, obj_id in enumerate(new_det_obj_ids_local):
                obj_id_to_mask[obj_id] = (video_res_masks[i] > 0.0).to(torch.bool)
        if self.rank == 0:
            for fidx in range(inference_state["num_frames"]):
                self._cache_frame_outputs(inference_state, fidx, obj_id_to_mask)

        inference_state["tracker_metadata"] = {
            "obj_ids_per_gpu": [np.arange(num_objects)],
            "obj_ids_all_gpu": np.arange(num_objects),  # Same as 1 GPU
            "num_obj_per_gpu": [num_objects],
            "obj_id_to_score": {i: 1.0 for i in range(num_objects)},
            "obj_id_to_sam2_score_frame_wise": defaultdict(dict),
            "obj_id_to_last_occluded": {},
            "max_obj_id": num_objects,
            "rank0_metadata": {
                "masklet_confirmation": {
                    "status": np.zeros(num_objects, dtype=np.int64),
                    "consecutive_det_num": np.zeros(num_objects, dtype=np.int64),
                },
                "removed_obj_ids": set(),
                "suppressed_obj_ids": defaultdict(set),
            },
            # gpu_metadata for hotstart tracking on GPU
            "gpu_metadata": {
                "N_obj": num_objects,
                "obj_first_frame": torch.zeros(
                    num_objects, dtype=torch.long, device=self.device
                ),
                "consecutive_unmatch_count": torch.zeros(
                    num_objects, dtype=torch.long, device=self.device
                ),
                "trk_keep_alive": torch.ones(
                    num_objects, dtype=torch.bool, device=self.device
                ),
                "removed_mask": torch.zeros(
                    num_objects, dtype=torch.bool, device=self.device
                ),
                "overlap_pair_counts": torch.zeros(
                    (num_objects, num_objects), dtype=torch.long, device=self.device
                ),
                "last_occluded_tensor": torch.zeros(
                    num_objects, dtype=torch.long, device=self.device
                ),
            },
        }
        # Add num_buc_per_gpu for multiplex mode
        if self.is_multiplex:
            # Count actual buckets from the inference states
            num_buc = self._count_buckets_in_states(
                inference_state["sam2_inference_states"]
            )
            inference_state["tracker_metadata"]["num_buc_per_gpu"] = np.array(
                [num_buc], dtype=np.int64
            )

        return inference_state

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
        self.rank = self.detector.rank = 0
        self.world_size = self.detector.world_size = 1
        orig_recondition_every_nth_frame = self.recondition_every_nth_frame
        # self.recondition_every_nth_frame = 2

        # Get a random video
        inference_state = self.init_state(resource_path="<load-zero-video-30>")
        start_frame_idx = 0

        # Run basic propagation warm-up
        inference_state = self._warm_up_vg_propagation(inference_state, start_frame_idx)

        logger.info("Warm-up compilation completed.")

        # revert to the original GPU and rank
        self.rank = self.detector.rank = orig_rank
        self.world_size = self.detector.world_size = orig_world_size
        self.recondition_every_nth_frame = orig_recondition_every_nth_frame
        self._warm_up_complete = True
        self.tracker.transformer.encoder.forward.set_logging(True)

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
        assert text_str is not None or points is not None or boxes_xywh is not None, (
            "at least one type of prompt (text, points, boxes) must be provided"
        )
        assert 0 <= frame_idx < num_frames, (
            f"{frame_idx=} is out of range for a total of {num_frames} frames"
        )

        assert clear_old_boxes, "clear old boxes must be True"

        assert points is None and clear_old_points is True and point_labels is None, (
            "Point prompts not accepted"
        )

        # since it's a semantic prompt, we start over
        self.reset_state(inference_state)

        # 1) add text prompt
        if text_str is not None:
            inference_state["text_prompt"] = text_str
            # add the text prompt into the input batch (to be applied to *all* frames)
            inference_state["input_batch"].find_text_batch[0] = text_str
            for t in range(inference_state["num_frames"]):
                text_id = self.TEXT_ID_FOR_TEXT
                inference_state["input_batch"].find_inputs[t].text_ids[...] = text_id

        # 2) handle box prompt
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

            new_box_input = boxes_cxcywh, box_labels
            inference_state["per_frame_raw_box_input"][frame_idx] = new_box_input

            # handle the case of visual prompt (also added as an input box from the UI)
            boxes_cxcywh, box_labels, geometric_prompt = self._get_visual_prompt(
                inference_state, frame_idx, boxes_cxcywh, box_labels
            )

            inference_state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt

        with torch.profiler.record_function("add_prompt._init_backbone_out"):
            inference_state["backbone_out"] = self._init_backbone_out(inference_state)
        out = self._run_single_frame_inference(
            inference_state,
            frame_idx,
            reverse=False,
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
        text_outputs = self.detector.backbone.forward_text(
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
        self.rank = self.detector.rank = 0
        self.world_size = self.detector.world_size = 1

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
                out_obj_ids = (
                    out["out_obj_ids"].numpy()
                    if isinstance(out["out_obj_ids"], torch.Tensor)
                    else out["out_obj_ids"]
                )
                out_binary_masks = (
                    out["out_binary_masks"].numpy()
                    if isinstance(out["out_binary_masks"], torch.Tensor)
                    else out["out_binary_masks"]
                )

                current_frame_res = tracking_res[frame_idx]
                for obj_id, mask in zip(out_obj_ids, out_binary_masks):
                    mask_tensor = torch.tensor(mask[None], dtype=torch.bool)
                    current_frame_res[obj_id + start_obj_id] = mask_tensor
                obj_ids_this_prompt.update(current_frame_res.keys())

            obj_id_to_score = inference_state["tracker_metadata"]["obj_id_to_score"]
            for obj_id, score in obj_id_to_score.items():
                if obj_id + start_obj_id in obj_ids_this_prompt:
                    score_tensor = torch.tensor(score, dtype=torch.float32)
                    scores_labels[obj_id + start_obj_id] = (score_tensor, prompt_id)

            self.reset_state(inference_state)

        video_id = input.find_metadatas[0].original_image_id[0].cpu().item()
        preds = self.prep_for_evaluator(input.raw_images, tracking_res, scores_labels)

        # revert the model to the original GPU and rank
        self.rank = self.detector.rank = orig_rank
        self.world_size = self.detector.world_size = orig_world_size
        return {video_id: preds}


class Sam3MultiplexTrackingProd(Sam3MultiplexTracking):
    """
    Subclass of Sam3MultiplexTracking with support for batched processing.

    This class enables processing videos in batches rather than all at once by:
    1. Adding an `is_last_batch` parameter to control buffer flushing
    2. Persisting generator state (hotstart_buffer, hotstart_removed_obj_ids,
       unconfirmed_obj_ids_per_frame) in inference_state across generator instantiations

    This is useful for processing large videos in smaller chunks to manage memory
    or distribute processing across multiple calls.
    """

    @torch.inference_mode()
    def init_state(
        self,
        resource_path,
        offload_video_to_cpu=False,
        async_loading_frames=False,
        use_torchcodec=False,
        use_cv2=False,
        input_is_mp4=False,
    ):
        inference_state = super().init_state(
            resource_path=resource_path,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            use_torchcodec=use_torchcodec,
            use_cv2=use_cv2,
            input_is_mp4=input_is_mp4,
        )
        # Initialize generator state for batched processing
        inference_state["generator_state"] = {
            "hotstart_buffer": [],
            "hotstart_removed_obj_ids": set(),
            "unconfirmed_obj_ids_per_frame": {},
            "postprocess_yield_list": [],
        }
        return inference_state

    def reset_state(self, inference_state):
        super().reset_state(inference_state)
        # Reset generator state for batched processing
        inference_state["generator_state"] = {
            "hotstart_buffer": [],
            "hotstart_removed_obj_ids": set(),
            "unconfirmed_obj_ids_per_frame": {},
            "postprocess_yield_list": [],
        }

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
        is_last_batch=True,
    ):
        """
        Propagate the prompts to get grounding results for the entire video. This method
        is a generator and yields inference outputs for all frames in the range specified
        by `start_frame_idx`, `max_frame_num_to_track`, and `reverse`.

        Args:
            is_last_batch: Whether this is the last batch in a batched processing scenario.
                When True (default), the hotstart buffer will be flushed at end_frame_idx.
                When False, the buffer is preserved in inference_state for the next batch.
                This flag should be set to False for all batches except the last one when
                processing a video in multiple batches.
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

        # Store max_frame_num_to_track in feature_cache for downstream methods
        inference_state["feature_cache"]["tracking_bounds"] = {
            "max_frame_num_to_track": max_frame_num_to_track,
            "propagate_in_video_start_frame_idx": start_frame_idx,
        }

        # Initialize or retrieve generator state from inference_state to persist across batches
        if "generator_state" not in inference_state:
            inference_state["generator_state"] = {
                "hotstart_buffer": [],
                "hotstart_removed_obj_ids": set(),
                "unconfirmed_obj_ids_per_frame": {},
                "postprocess_yield_list": [],
            }

        generator_state = inference_state["generator_state"]
        hotstart_buffer = generator_state["hotstart_buffer"]
        hotstart_removed_obj_ids = generator_state["hotstart_removed_obj_ids"]
        unconfirmed_obj_ids_per_frame = generator_state["unconfirmed_obj_ids_per_frame"]
        postprocess_yield_list = generator_state.get("postprocess_yield_list", [])

        # when deciding whether to output a masklet on `yield_frame_idx`, we check whether the object is confirmed
        # in a future frame (`unconfirmed_frame_delay` frames after the current frame). For example, if we require
        # an object to be detected in 3 consecutive frames to be confirmed, then we look 2 frames in the future --
        # e.g., we output an object on frame 4 only if it becomes confirmed on frame 6.
        unconfirmed_status_delay = self.masklet_confirmation_consecutive_det_thresh - 1

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
                    unconfirmed_obj_ids = out.get("unconfirmed_obj_ids", None)
                    if unconfirmed_obj_ids is not None:
                        unconfirmed_obj_ids_per_frame[frame_idx] = unconfirmed_obj_ids

                if frame_idx == end_frame_idx and is_last_batch:
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

            # Accumulate yield_list into postprocess_yield_list
            # Snapshot hotstart_removed_obj_ids at the time of accumulation to preserve
            # the correct state for each frame (important: this set is mutated over time)
            for yield_frame_idx, yield_out in yield_list:
                postprocess_yield_list.append(
                    (yield_frame_idx, yield_out, set(hotstart_removed_obj_ids))
                )

            # Process batch when we have enough frames
            while len(postprocess_yield_list) >= self.postprocess_batch_size:
                batch_to_process = postprocess_yield_list[: self.postprocess_batch_size]
                postprocess_yield_list = postprocess_yield_list[
                    self.postprocess_batch_size :
                ]

                with torch.profiler.record_function(
                    "Sam3MultiplexTrackingProd.postprocess_output_batched"
                ):
                    if self.rank == 0:
                        # Prepare batched inputs for postprocessing
                        H_video, W_video = (
                            inference_state["orig_height"],
                            inference_state["orig_width"],
                        )
                        num_frames = inference_state["num_frames"]

                        batched_outs = []
                        frame_indices = []
                        for (
                            yield_frame_idx,
                            yield_out,
                            removed_obj_ids_snapshot,
                        ) in batch_to_process:
                            suppressed_obj_ids = yield_out["suppressed_obj_ids"]
                            unconfirmed_status_frame_idx = (
                                yield_frame_idx + unconfirmed_status_delay
                                if not reverse
                                else yield_frame_idx - unconfirmed_status_delay
                            )
                            unconfirmed_status_frame_idx = max(
                                0, min(unconfirmed_status_frame_idx, num_frames - 1)
                            )
                            unconfirmed_obj_ids = unconfirmed_obj_ids_per_frame.get(
                                unconfirmed_status_frame_idx, None
                            )

                            batched_outs.append(
                                (
                                    yield_out,
                                    removed_obj_ids_snapshot,
                                    suppressed_obj_ids,
                                    unconfirmed_obj_ids,
                                )
                            )
                            frame_indices.append(yield_frame_idx)

                            # Cache frame outputs
                            self._cache_frame_outputs(
                                inference_state,
                                yield_frame_idx,
                                yield_out["obj_id_to_mask"],
                                suppressed_obj_ids=suppressed_obj_ids,
                                removed_obj_ids=removed_obj_ids_snapshot,
                                unconfirmed_obj_ids=unconfirmed_obj_ids,
                            )

                        # Process all frames in batch
                        if self.postprocess_batch_size > 1:
                            postprocessed_outs = self._postprocess_output_batched(
                                H_video, W_video, batched_outs
                            )
                        else:
                            # Process each frame individually but output together
                            postprocessed_outs = []
                            for (
                                yield_out,
                                removed_obj_ids_snapshot,
                                suppressed_obj_ids,
                                unconfirmed_obj_ids,
                            ) in batched_outs:
                                postprocessed_out = self._postprocess_output(
                                    inference_state,
                                    yield_out,
                                    removed_obj_ids_snapshot,
                                    suppressed_obj_ids,
                                    unconfirmed_obj_ids,
                                )
                                postprocessed_outs.append(postprocessed_out)

                        # Yield results
                        for yield_frame_idx, postprocessed_out in zip(
                            frame_indices, postprocessed_outs
                        ):
                            yield yield_frame_idx, postprocessed_out
                    else:
                        # No output on other GPUs
                        for yield_frame_idx, _, _ in batch_to_process:
                            yield yield_frame_idx, DUMMY_OUTPUT

        # Handle remaining frames in hotstart buffer at end of last batch
        if is_last_batch and len(hotstart_buffer) > 0:
            for yield_frame_idx, yield_out in hotstart_buffer:
                postprocess_yield_list.append(
                    (yield_frame_idx, yield_out, set(hotstart_removed_obj_ids))
                )
            hotstart_buffer = []

        # Flush any remaining frames in the postprocess buffer (even partial
        # batches) so that the caller gets results as soon as possible. This is
        # especially important for the first batch where hotstart_delay causes
        # only a few frames to exit the hotstart buffer — without this flush
        # the client would have to wait for the next batch before receiving any
        # output, hurting time-to-first-frame.
        if len(postprocess_yield_list) > 0:
            with torch.profiler.record_function(
                "Sam3MultiplexTrackingProd.postprocess_output_batched"
            ):
                if self.rank == 0:
                    H_video, W_video = (
                        inference_state["orig_height"],
                        inference_state["orig_width"],
                    )
                    num_frames = inference_state["num_frames"]

                    batched_outs = []
                    frame_indices = []
                    for (
                        yield_frame_idx,
                        yield_out,
                        removed_obj_ids_snapshot,
                    ) in postprocess_yield_list:
                        suppressed_obj_ids = yield_out["suppressed_obj_ids"]
                        unconfirmed_status_frame_idx = (
                            yield_frame_idx + unconfirmed_status_delay
                            if not reverse
                            else yield_frame_idx - unconfirmed_status_delay
                        )
                        unconfirmed_status_frame_idx = max(
                            0, min(unconfirmed_status_frame_idx, num_frames - 1)
                        )
                        unconfirmed_obj_ids = unconfirmed_obj_ids_per_frame.get(
                            unconfirmed_status_frame_idx, None
                        )

                        batched_outs.append(
                            (
                                yield_out,
                                removed_obj_ids_snapshot,
                                suppressed_obj_ids,
                                unconfirmed_obj_ids,
                            )
                        )
                        frame_indices.append(yield_frame_idx)

                        self._cache_frame_outputs(
                            inference_state,
                            yield_frame_idx,
                            yield_out["obj_id_to_mask"],
                            suppressed_obj_ids=suppressed_obj_ids,
                            removed_obj_ids=removed_obj_ids_snapshot,
                            unconfirmed_obj_ids=unconfirmed_obj_ids,
                        )

                    if self.postprocess_batch_size > 1:
                        postprocessed_outs = self._postprocess_output_batched(
                            H_video, W_video, batched_outs
                        )
                    else:
                        # Process each frame individually but output together
                        postprocessed_outs = []
                        for (
                            yield_out,
                            removed_obj_ids_snapshot,
                            suppressed_obj_ids,
                            unconfirmed_obj_ids,
                        ) in batched_outs:
                            postprocessed_out = self._postprocess_output(
                                inference_state,
                                yield_out,
                                removed_obj_ids_snapshot,
                                suppressed_obj_ids,
                                unconfirmed_obj_ids,
                            )
                            postprocessed_outs.append(postprocessed_out)

                    for yield_frame_idx, postprocessed_out in zip(
                        frame_indices, postprocessed_outs
                    ):
                        yield yield_frame_idx, postprocessed_out
                else:
                    for yield_frame_idx, _, _ in postprocess_yield_list:
                        yield yield_frame_idx, DUMMY_OUTPUT

            postprocess_yield_list = []

        # Store the generator state back to inference_state for persistence across batches
        generator_state["postprocess_yield_list"] = postprocess_yield_list
        generator_state["hotstart_buffer"] = hotstart_buffer
        generator_state["hotstart_removed_obj_ids"] = hotstart_removed_obj_ids
        generator_state["unconfirmed_obj_ids_per_frame"] = unconfirmed_obj_ids_per_frame

        if self.is_multiplex:
            # log the bucket utilization stats
            # bucket utilization rate is total valid objects / total capacity -> represents rooms for improvement
            # subscription rate is total valid objects / total number of buckets -> represents speedup
            total_valid_objects = 0
            total_num_buckets = 0
            for state in inference_state["sam2_inference_states"]:
                assert (
                    len(state["obj_ids"])
                    == state["multiplex_state"].total_valid_entries
                )
                total_valid_objects += len(state["obj_ids"])
                total_num_buckets += state["multiplex_state"].num_buckets
            if total_num_buckets > 0:
                bucket_utilization_rate = (
                    total_valid_objects / (total_num_buckets * self.bucket_capacity)
                ) * 100
                subscription_rate = (total_valid_objects / total_num_buckets) * 100
                logger.info(
                    f"Bucket utilization rate: {bucket_utilization_rate:.2f}%, subscription rate: {subscription_rate:.2f}%"
                )


class Sam3MultiplexTrackingWithInteractivity(Sam3MultiplexTracking):
    def __init__(
        self,
        use_prev_mem_frame=False,
        use_stateless_refinement=False,
        refinement_detector_cond_frame_removal_window=30 * 4,
        **kwargs,
    ):
        """
        use_prev_mem_frame: bool, whether to condition on previous memory frames for adding points
        use_stateless_refinement: bool, whether to enable stateless refinement behavior
        refinement_detector_cond_frame_removal_window: int, we remove a detector conditioning frame if it
            is within this many frames of a user refined frame. Set to a large value (e.g. 10000) to
            always remove detector conditioning frames if there is any user refinement in the video.
        """
        super().__init__(**kwargs)
        self.use_prev_mem_frame = use_prev_mem_frame
        self.use_stateless_refinement = use_stateless_refinement
        self.refinement_detector_cond_frame_removal_window = (
            refinement_detector_cond_frame_removal_window
        )

    @torch.inference_mode()
    def init_state(
        self,
        resource_path,
        offload_video_to_cpu=False,
        async_loading_frames=False,
        use_torchcodec=False,
        use_cv2=False,
        input_is_mp4=False,
    ):
        inference_state = super().init_state(
            resource_path=resource_path,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            use_torchcodec=use_torchcodec,
            use_cv2=use_cv2,
            input_is_mp4=input_is_mp4,
        )
        # initialize extra states
        inference_state["action_history"] = []  # for logging user actions
        if self.tracker.per_obj_inference:
            # in per_obj mode only 1 inference state is needed, we init it here.
            inference_state["sam2_inference_states"] = [
                self._init_new_sam2_state(inference_state)
            ]
        return inference_state

    def reset_state(self, inference_state):
        super().reset_state(inference_state)
        # reset extra states
        inference_state["action_history"].clear()
        if self.tracker.per_obj_inference:
            inference_state["sam2_inference_states"] = [
                self._init_new_sam2_state(inference_state)
            ]

    def _init_new_sam2_state(self, inference_state):
        return self.tracker.init_state(
            cached_features=inference_state["feature_cache"],
            video_height=inference_state["orig_height"],
            video_width=inference_state["orig_width"],
            num_frames=inference_state["num_frames"],
        )

    def cancel_propagation(self, inference_state):
        """
        Cancel any ongoing propagation and reset the model state.
        """
        logger.info("Cancelling ongoing propagation.")
        self.add_action_history(
            inference_state,
            action_type="propagation_cancel",
            obj_ids=None,
            frame_idx=None,
        )

    def fetch_and_process_single_frame_results(self, inference_state, frame_idx):
        tracker_metadata = inference_state["tracker_metadata"]
        obj_id_to_mask = inference_state["cached_frame_outputs"][frame_idx]
        # post processing - remove suppressed obj_ids
        obj_id_to_score = tracker_metadata["obj_id_to_score"]
        suppressed_obj_ids = tracker_metadata["rank0_metadata"]["suppressed_obj_ids"][
            frame_idx
        ]
        obj_id_to_sam2_score = tracker_metadata["obj_id_to_sam2_score_frame_wise"][
            frame_idx
        ]

        out = {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": obj_id_to_score,
            "obj_id_to_sam2_score": obj_id_to_sam2_score,
        }
        return frame_idx, self._postprocess_output(
            inference_state, out, suppressed_obj_ids=suppressed_obj_ids
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
        is_last_batch: bool = False,
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
                is_last_batch=is_last_batch,
            )
            return

        # step 3: run SAM2 partial propagation or direct fetch existing predictions
        assert propagation_type in ["propagation_partial", "propagation_fetch"]
        logger.info(
            f"Running SAM2 propagation for objects {obj_ids} and merging it with existing VG predictions (reverse={reverse})."
            if propagation_type == "propagation_partial"
            else f"Fetching existing VG predictions without running any propagation (reverse={reverse})."
        )
        processing_order, _end_frame_idx = self._get_processing_order(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        )

        tracker_metadata = inference_state["tracker_metadata"]

        # if fetch just return from output
        if propagation_type == "propagation_fetch":
            for frame_idx in tqdm(processing_order):
                if self.rank == 0:
                    frame_idx, out = self.fetch_and_process_single_frame_results(
                        inference_state, frame_idx
                    )
                    yield frame_idx, out
                else:
                    yield frame_idx, DUMMY_OUTPUT  # no output for other GPUs

            return

        # get SAM2 inference states containing selected obj_ids
        if propagation_type == "propagation_partial":
            # can be empty for GPUs where objects are not in their inference states
            tracker_states_local = self._get_sam2_inference_states_by_obj_ids(
                inference_state, obj_ids
            )
            for sam2_state in tracker_states_local:
                self.tracker.propagate_in_video_preflight(
                    sam2_state, run_mem_encoder=True
                )

        for frame_idx in tqdm(processing_order):
            # run SAM2 propagation
            if propagation_type == "propagation_partial":
                self._prepare_backbone_feats(inference_state, frame_idx, reverse)
                obj_ids_local, low_res_masks_local, sam2_scores_local = (
                    self._propogate_tracker_one_frame_local_gpu(
                        tracker_states_local,
                        frame_idx=frame_idx,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                )

                # broadcast refined object sam2 scores and masks to all GPUs
                # handle multiple objects that can be located on different GPUs
                refined_obj_data = {}  # obj_id -> (score, mask_video_res)

                # Collect data for objects on this GPU
                local_obj_data = {}
                for obj_id in obj_ids:
                    obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)
                    if self.rank == obj_rank and obj_id in obj_ids_local:
                        refined_obj_idx = obj_ids_local.index(obj_id)
                        refined_mask_low_res = low_res_masks_local[
                            refined_obj_idx
                        ]  # (H_low_res, W_low_res)
                        refined_score = sam2_scores_local[refined_obj_idx]

                        # Keep low resolution for broadcasting to reduce communication cost
                        local_obj_data[obj_id] = (refined_score, refined_mask_low_res)

                # Broadcast data from each GPU that has refined objects
                if self.world_size > 1:
                    for obj_id in obj_ids:
                        obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)
                        if self.rank == obj_rank:
                            # This GPU has the object, broadcast its data
                            data_to_broadcast = local_obj_data.get(obj_id, None)
                            data_list = [data_to_broadcast]
                            self.broadcast_python_obj_cpu(data_list, src=obj_rank)
                            if data_to_broadcast is not None:
                                refined_obj_data[obj_id] = data_to_broadcast
                        elif self.rank != obj_rank:
                            # This GPU doesn't have the object, receive data
                            data_list = [None]
                            self.broadcast_python_obj_cpu(data_list, src=obj_rank)
                            if data_list[0] is not None:
                                refined_obj_data[obj_id] = data_list[0]
                else:
                    # Single GPU case
                    refined_obj_data = local_obj_data

                # Update SAM2 scores for all refined objects
                for obj_id, (refined_score, _) in refined_obj_data.items():
                    # After broadcast_python_obj_cpu in multi-GPU, tensors may become numpy scalars
                    # Ensure it's a GPU tensor for consistency with base class behavior
                    if not isinstance(refined_score, torch.Tensor):
                        refined_score = torch.tensor(
                            refined_score, dtype=torch.float32, device=self.device
                        )
                    tracker_metadata["obj_id_to_sam2_score_frame_wise"][
                        frame_idx
                    ].update({obj_id: refined_score})

                if self.rank == 0:
                    # get predictions from SAM2 inference states, it includes the original
                    # VG predictions and the refined predictions from interactivity.

                    # Prepare refined masks dictionary - upscale to video resolution after broadcast
                    refined_obj_id_to_mask = {}
                    for obj_id, (_, refined_mask_low_res) in refined_obj_data.items():
                        refined_mask_video_res = (
                            self._convert_low_res_mask_to_video_res(
                                refined_mask_low_res, inference_state
                            )
                        )  # (1, H_video, W_video) bool
                        refined_obj_id_to_mask[obj_id] = refined_mask_video_res

                    obj_id_to_mask = self._build_sam2_output(
                        inference_state, frame_idx, refined_obj_id_to_mask
                    )
                    out = {
                        "obj_id_to_mask": obj_id_to_mask,
                        "obj_id_to_score": tracker_metadata["obj_id_to_score"],
                        "obj_id_to_sam2_score": tracker_metadata[
                            "obj_id_to_sam2_score_frame_wise"
                        ][frame_idx],
                    }
                    suppressed_obj_ids = tracker_metadata["rank0_metadata"][
                        "suppressed_obj_ids"
                    ][frame_idx]
                    self._cache_frame_outputs(
                        inference_state,
                        frame_idx,
                        obj_id_to_mask,
                        suppressed_obj_ids=suppressed_obj_ids,
                    )
                    suppressed_obj_ids = tracker_metadata["rank0_metadata"][
                        "suppressed_obj_ids"
                    ][frame_idx]
                    yield (
                        frame_idx,
                        self._postprocess_output(
                            inference_state, out, suppressed_obj_ids=suppressed_obj_ids
                        ),
                    )
                else:
                    yield frame_idx, DUMMY_OUTPUT  # no output for other GPUs

    def add_action_history(
        self, inference_state, action_type, frame_idx=None, obj_ids=None
    ):
        """
        action_history is used to automatically decide what to do during propagation.
        action_type: one of ["add", "remove", "refine"] + ["propagation_full", "propagation_partial", "propagation_fetch", "propagation_cancel"]
        """
        instance_actions = ["add", "remove", "refine"]
        propagation_actions = [
            "propagation_full",
            "propagation_partial",
            "propagation_fetch",
            "propagation_cancel",
        ]
        assert action_type in instance_actions + propagation_actions, (
            f"Invalid action type: {action_type}, must be one of {instance_actions + propagation_actions}"
        )
        action = {
            "type": action_type,
            "frame_idx": frame_idx,
            "obj_ids": obj_ids,
        }
        inference_state["action_history"].append(action)

    def _has_object_been_refined(self, inference_state, obj_id):
        if "action_history" not in inference_state:
            return False
        action_history = inference_state["action_history"]
        for action in action_history:
            if action["type"] in ["add", "refine"] and action.get("obj_ids"):
                if obj_id in action["obj_ids"]:
                    return True
        return False

    def parse_action_history_for_propagation(self, inference_state):
        action_history = inference_state["action_history"]
        if (
            len(action_history) == 1
            and action_history[0]["type"] == "propagation_cancel"
        ):
            # only one action and it is cancel, we do full propagation
            return "propagation_full", None
        elif (
            len(action_history) >= 2
            and action_history[-1]["type"] == "propagation_cancel"
        ):
            # last action is cancel, we go back to the action before cancel
            action_before_cancelation = inference_state["action_history"][-2]
            # the action before cancellation can be a propagation_fetch from running both forward
            # and backward propagation as in webdemo interface, in that case we go back one more step
            if action_before_cancelation["type"] == "propagation_fetch":
                action_before_cancelation = inference_state["action_history"][-3]
            return action_before_cancelation["type"], action_before_cancelation.get(
                "obj_ids", None
            )
        return self._parse_action_history_for_propagation(
            inference_state["action_history"], inference_state["num_frames"]
        )

    def _parse_action_history_for_propagation(self, action_history, num_frames):
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
                - "propagation_cancel": this will be handled in parse_action_history_for_propagation() not this function.
            obj_ids: list of object ids to run SAM2 propagation on if propagation_type is "propagation_partial".

        TODO: (Jie) this function works for our current workflows, but may need more tests to ensure it works
        correctly with different action histories for future workflows.
        """
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
                    num_frames - 1,
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

    def remove_object(self, inference_state, obj_id, frame_idx, is_user_action=False):
        """
        We try to remove object from sam2 states on every GPU, it will do nothing
        for states without this object.
        """
        obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)
        if obj_rank is None:
            # Object was already removed (e.g., by hotstart heuristics during
            # propagation). Log a warning and skip SAM2 state and metadata
            # removal, but still record action history and clean up cached outputs.
            logger.warning(
                f"Object {obj_id} not found in any GPU (already removed). "
                f"Skipping SAM2 state and metadata removal."
            )
        else:
            tracker_states_local = inference_state["sam2_inference_states"]
            if self.rank == obj_rank:
                self._tracker_remove_objects(tracker_states_local, [obj_id])

            # update metadata
            tracker_metadata = inference_state["tracker_metadata"]
            _obj_ids = tracker_metadata["obj_ids_per_gpu"][obj_rank]
            tracker_metadata["obj_ids_per_gpu"][obj_rank] = _obj_ids[_obj_ids != obj_id]
            tracker_metadata["num_obj_per_gpu"][obj_rank] = len(
                tracker_metadata["obj_ids_per_gpu"][obj_rank]
            )
            tracker_metadata["obj_ids_all_gpu"] = np.concatenate(
                tracker_metadata["obj_ids_per_gpu"]
            )
            tracker_metadata["obj_id_to_score"].pop(obj_id, None)
            # tracker_metadata["max_obj_id"] # we do not reuse the object id, so we do not update it here

        if is_user_action:
            self.add_action_history(
                inference_state, action_type="remove", obj_ids=[obj_id]
            )

        # Clean up cached frame outputs to remove references to the deleted object
        if "cached_frame_outputs" in inference_state:
            for _frame_idx in inference_state["cached_frame_outputs"]:
                frame_cache = inference_state["cached_frame_outputs"][_frame_idx]
                if obj_id in frame_cache:
                    del frame_cache[obj_id]

        out = None
        if frame_idx is not None and self.rank == 0:
            frame_idx, out = self.fetch_and_process_single_frame_results(
                inference_state, frame_idx
            )
        return frame_idx, out

    def _get_gpu_id_by_obj_id(self, inference_state, obj_id):
        """
        Locate GPU ID for a given object.
        """
        obj_ids_per_gpu = inference_state["tracker_metadata"]["obj_ids_per_gpu"]
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
            assert text_str is None and boxes_xywh is None, (
                "When points are provided, text_str and boxes_xywh must be None."
            )
            assert obj_id is not None, (
                "When points are provided, obj_id must be provided."
            )
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
            # SAM3 prompts — disable batched grounding for single-frame add_prompt
            _orig_batched = self.use_batched_grounding
            self.use_batched_grounding = False
            try:
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
            finally:
                self.use_batched_grounding = _orig_batched

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
        Every GPU returns the same results, and results should contain all masks including
        these masks not refined or not added by the current user points.
        """
        assert obj_id is not None, "obj_id must be provided to add new points"
        tracker_metadata = inference_state["tracker_metadata"]
        if tracker_metadata == {}:
            # initialize masklet metadata if it's uninitialized (empty dict)
            tracker_metadata.update(self._initialize_metadata())

        obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)

        # prepare feature
        self._prepare_backbone_feats(inference_state, frame_idx, reverse=False)

        object_has_been_refined = self._has_object_been_refined(inference_state, obj_id)
        if (
            obj_rank is not None
            and self.use_stateless_refinement
            and not object_has_been_refined
        ):
            # The first time we start refinement on the object, we remove it.
            logger.info(
                f"[rank={self.rank}] Removing object {obj_id} before refinement."
            )
            self.remove_object(inference_state, obj_id, is_user_action=False)
            obj_rank = None
        elif obj_rank is not None and not object_has_been_refined:
            # Extract the object into its own singleton inference state if it belongs to a batch
            if self.rank == obj_rank and not self.tracker.per_obj_inference:
                tracker_states = self._get_sam2_inference_states_by_obj_ids(
                    inference_state, [obj_id]
                )
                assert len(tracker_states) == 1
                # Check if this is a batched state (contains multiple objects)
                sam2_state = tracker_states[0]
                if len(sam2_state["obj_ids"]) > 1:
                    logger.info(
                        f"[rank={self.rank}] Extracting object {obj_id} into singleton inference state."
                    )
                    self._extract_object_to_singleton_state(
                        inference_state, obj_id, obj_rank
                    )

        if obj_rank is None:
            # new object, we assign it a GPU and create a new inference state if limit allows
            num_prev_obj = np.sum(tracker_metadata["num_obj_per_gpu"])
            if num_prev_obj >= self.max_num_objects:
                logger.warning(
                    f"add_sam2_new_points: cannot add a new object as we are already tracking {num_prev_obj=} "
                    f"masklets (under {self.max_num_objects=})"
                )
                return frame_idx, None

            new_det_gpu_ids = self._assign_new_det_to_gpus(
                new_det_num=1,
                prev_workload_per_gpu=tracker_metadata["num_obj_per_gpu"],
            )
            obj_rank = new_det_gpu_ids[0]

            # get sam2 inference state for the new object
            if self.rank == obj_rank:
                if self.tracker.per_obj_inference:
                    sam2_state = inference_state["sam2_inference_states"][0]
                else:
                    # for batched inference, we create a new inference state
                    sam2_state = self._init_new_sam2_state(inference_state)
                    inference_state["sam2_inference_states"].append(sam2_state)

            # update metadata
            tracker_metadata["obj_ids_per_gpu"][obj_rank] = np.concatenate(
                [
                    tracker_metadata["obj_ids_per_gpu"][obj_rank],
                    np.array([obj_id], dtype=np.int64),
                ]
            )
            tracker_metadata["num_obj_per_gpu"][obj_rank] = len(
                tracker_metadata["obj_ids_per_gpu"][obj_rank]
            )
            tracker_metadata["obj_ids_all_gpu"] = np.concatenate(
                tracker_metadata["obj_ids_per_gpu"]
            )
            tracker_metadata["max_obj_id"] = max(tracker_metadata["max_obj_id"], obj_id)

            logger.info(
                f"[rank={self.rank}] Adding new object with id {obj_id} at frame {frame_idx}."
            )
            self.add_action_history(
                inference_state, "add", frame_idx=frame_idx, obj_ids=[obj_id]
            )
        else:
            # existing object, for refinement
            if self.rank == obj_rank:
                tracker_states = self._get_sam2_inference_states_by_obj_ids(
                    inference_state, [obj_id]
                )
                assert len(tracker_states) == 1, (
                    f"[rank={self.rank}] Multiple SAM2 inference states found for the same object id."
                )
                sam2_state = tracker_states[0]

            # log
            logger.info(
                f"[rank={self.rank}] Refining existing object with id {obj_id} at frame {frame_idx}."
            )
            self.add_action_history(
                inference_state, "refine", frame_idx=frame_idx, obj_ids=[obj_id]
            )

        # assign higher score to added/refined object
        tracker_metadata["obj_id_to_score"][obj_id] = 1.0
        tracker_metadata["obj_id_to_sam2_score_frame_wise"][frame_idx][obj_id] = (
            torch.tensor(1.0, dtype=torch.float32, device=self.device)
        )

        if self.rank == 0:
            rank0_metadata = tracker_metadata.get("rank0_metadata", {})

            if "removed_obj_ids" in rank0_metadata:
                rank0_metadata["removed_obj_ids"].discard(obj_id)

            if "suppressed_obj_ids" in rank0_metadata:
                for frame_id in rank0_metadata["suppressed_obj_ids"]:
                    rank0_metadata["suppressed_obj_ids"][frame_id].discard(obj_id)

            if "masklet_confirmation" in rank0_metadata:
                obj_ids_all_gpu = tracker_metadata["obj_ids_all_gpu"]
                obj_indices = np.where(obj_ids_all_gpu == obj_id)[0]
                if len(obj_indices) > 0:
                    obj_idx = obj_indices[0]
                    if obj_idx < len(rank0_metadata["masklet_confirmation"]["status"]):
                        rank0_metadata["masklet_confirmation"]["status"][obj_idx] = 1
                        rank0_metadata["masklet_confirmation"]["consecutive_det_num"][
                            obj_idx
                        ] = self.masklet_confirmation_consecutive_det_thresh

        if self.rank == obj_rank:
            should_fallback_to_original_mask = (
                len(points) == 0 and inference_state["is_image_only"]
            )
            if should_fallback_to_original_mask:
                mask_input = self._get_mask_input(sam2_state, frame_idx, obj_id)
                if mask_input is None or 0 in mask_input.shape:
                    logger.warning(
                        f"Cannot retrieve original mask input for obj_id {obj_id} at frame {frame_idx} to fallback."
                    )
                    should_fallback_to_original_mask = False
            if should_fallback_to_original_mask:
                # When user cancels all points on an image, we recover the original mask
                # by re-feeding the detector mask to SAM2.
                mask_input = self._get_mask_input(sam2_state, frame_idx, obj_id)
                # clear out states related to this object to have a fresh start
                self.tracker.clear_all_points_in_frame(
                    sam2_state, frame_idx, obj_id, need_output=False
                )
                frame_idx, obj_ids, low_res_masks, video_res_masks = (
                    self.tracker.add_new_mask(
                        sam2_state,
                        frame_idx,
                        obj_id,
                        mask_input,
                    )
                )
            else:
                frame_idx, obj_ids, low_res_masks, video_res_masks = (
                    self.tracker.add_new_points(
                        inference_state=sam2_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=points,
                        labels=labels,
                        clear_old_points=clear_old_points,
                        rel_coordinates=rel_coordinates,
                        use_prev_mem_frame=use_prev_mem_frame,
                    )
                )

            if video_res_masks is not None and len(video_res_masks) > 0:
                video_res_masks = fill_holes_in_mask_scores(
                    video_res_masks,  # shape (N, 1, H_video, W_video)
                    fill_hole_area=self.fill_hole_area,
                    sprinkle_removal_area=self.sprinkle_removal_area,
                    fill_holes=True,
                    remove_sprinkles=True,
                )

            # TODO: will this cause issue when user switching to refine another object?
            # Since the mem encoder has already run for the current input points?
            # FIX: Synchronize consolidated_frame_inds with actual point/mask
            # inputs before propagate_in_video_preflight. Two issues can cause
            # the `all_consolidated_frame_inds == input_frames_inds` assertion
            # to fail:
            #  1) VG detector conditioning frames in mask_inputs_per_obj without
            #     corresponding point inputs (stale VG entries).
            #  2) Previously consolidated point-input frames (from earlier
            #     add_points) whose consolidated_frame_inds entries were lost
            #     during subsequent propagation.
            # We fix both by: (a) clearing mask-only inputs, (b) rebuilding
            # consolidated_frame_inds from the remaining inputs, excluding
            # temp output frames (which preflight will add itself).

            # (a) Clear detector-only mask inputs
            for _obj_idx in list(sam2_state["mask_inputs_per_obj"].keys()):
                _point_frames = set(
                    sam2_state["point_inputs_per_obj"].get(_obj_idx, {}).keys()
                )
                _mask_only_frames = [
                    f
                    for f in list(sam2_state["mask_inputs_per_obj"][_obj_idx].keys())
                    if f not in _point_frames
                ]
                for f in _mask_only_frames:
                    sam2_state["mask_inputs_per_obj"][_obj_idx].pop(f, None)

            # (b) Rebuild consolidated_frame_inds from remaining inputs
            _input_frames = set()
            for _oi in sam2_state["point_inputs_per_obj"]:
                _input_frames.update(sam2_state["point_inputs_per_obj"][_oi].keys())
            for _oi in sam2_state["mask_inputs_per_obj"]:
                _input_frames.update(sam2_state["mask_inputs_per_obj"][_oi].keys())
            # Exclude temp output frames — preflight will consolidate those
            _temp_frames = set()
            for _obj_temp in sam2_state["temp_output_dict_per_obj"].values():
                _temp_frames.update(_obj_temp["cond_frame_outputs"].keys())
                _temp_frames.update(_obj_temp["non_cond_frame_outputs"].keys())
            _prev_frames = _input_frames - _temp_frames
            _cond = set()
            _non_cond = set()
            for f in _prev_frames:
                if f in sam2_state["output_dict"].get("cond_frame_outputs", {}):
                    _cond.add(f)
                else:
                    _non_cond.add(f)
            sam2_state["consolidated_frame_inds"] = {
                "cond_frame_outputs": _cond,
                "non_cond_frame_outputs": _non_cond,
            }
            self.tracker.propagate_in_video_preflight(sam2_state, run_mem_encoder=True)
            if not inference_state["is_image_only"]:
                # Clear detector conditioning frames when user clicks are received to allow
                # model updating masks on these frames. It is a noop if user is refining on the
                # detector conditioning frames or adding new objects.
                self.clear_detector_added_cond_frame_in_sam2(
                    sam2_state, obj_id, frame_idx
                )

        # fetch results from states and gather across GPUs
        # Use optimized caching approach to avoid reprocessing unmodified objects
        if self.rank == obj_rank and len(obj_ids) > 0:
            new_mask_data = (video_res_masks[obj_ids.index(obj_id)] > 0.0).to(
                torch.bool
            )
        else:
            new_mask_data = None

        # Broadcast the new mask data across all ranks for consistency
        if self.world_size > 1:
            data_list = [new_mask_data]
            self.broadcast_python_obj_cpu(data_list, src=obj_rank)
            new_mask_data = data_list[0]

        if self.rank == 0:
            obj_id_to_mask = self._build_sam2_output(
                inference_state,
                frame_idx,
                {obj_id: new_mask_data} if new_mask_data is not None else None,
            )
            # post processing - remove suppressed obj_ids
            obj_id_to_score = tracker_metadata["obj_id_to_score"]
            suppressed_obj_ids = tracker_metadata["rank0_metadata"][
                "suppressed_obj_ids"
            ][frame_idx]
            obj_id_to_sam2_score = tracker_metadata["obj_id_to_sam2_score_frame_wise"][
                frame_idx
            ]

            out = {
                "obj_id_to_mask": obj_id_to_mask,
                "obj_id_to_score": obj_id_to_score,
                "obj_id_to_sam2_score": obj_id_to_sam2_score,
            }
            self._cache_frame_outputs(
                inference_state,
                frame_idx,
                obj_id_to_mask,
                suppressed_obj_ids=suppressed_obj_ids,
            )
            return frame_idx, self._postprocess_output(
                inference_state, out, suppressed_obj_ids=suppressed_obj_ids
            )
        else:
            return frame_idx, None  # no output on other GPUs

    def _get_mask_input(self, inference_state, frame_idx, obj_id):
        """Get the mask input for a specific object on a specific frame."""
        obj_idx = self.tracker._obj_id_to_idx(inference_state, obj_id)
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]
        if frame_idx not in mask_inputs_per_frame:
            logger.info(
                f"frame {frame_idx} not in mask_inputs_per_frame for obj_id {obj_id}"
            )
            return None

        mask_inputs_orig = mask_inputs_per_frame[frame_idx].squeeze(0, 1)  # (H, W)
        return mask_inputs_orig

    def _gather_obj_id_to_mask_across_gpus(self, inference_state, obj_id_to_mask_local):
        """Gather obj_id_to_mask from all GPUs. Optionally resize the masks to the video resolution."""
        tracker_metadata = inference_state["tracker_metadata"]

        # concatenate the output masklets from all local inference states
        H_mask = W_mask = self.tracker.low_res_mask_size
        obj_ids_local = tracker_metadata["obj_ids_per_gpu"][self.rank]
        low_res_masks_local = []
        for obj_id in obj_ids_local:
            if obj_id in obj_id_to_mask_local:
                low_res_masks_local.append(obj_id_to_mask_local[obj_id])
            else:
                low_res_masks_local.append(
                    torch.full((H_mask, W_mask), -1024.0, device=self.device)
                )
        if len(low_res_masks_local) > 0:
            low_res_masks_local = torch.stack(low_res_masks_local, dim=0)  # (N, H, W)
            assert low_res_masks_local.shape[1:] == (H_mask, W_mask)
        else:
            low_res_masks_local = torch.zeros(0, H_mask, W_mask, device=self.device)

        # all-gather `low_res_masks_local` into `low_res_masks_global`
        # - low_res_masks_global: Tensor -- (num_global_obj, H_mask, W_mask)
        if self.world_size > 1:
            low_res_masks_local = low_res_masks_local.float().contiguous()
            low_res_masks_peers = [
                low_res_masks_local.new_empty(num_obj, H_mask, W_mask)
                for num_obj in tracker_metadata["num_obj_per_gpu"]
            ]
            dist.all_gather(low_res_masks_peers, low_res_masks_local)
            low_res_masks_global = torch.cat(low_res_masks_peers, dim=0)
        else:
            low_res_masks_global = low_res_masks_local
        return low_res_masks_global

    def _convert_low_res_mask_to_video_res(self, low_res_mask, inference_state):
        """
        Convert a low-res mask to video resolution, matching the format expected by _build_sam2_output.

        Args:
            low_res_mask: Tensor of shape (H_low_res, W_low_res)
            inference_state: Contains video dimensions

        Returns:
            video_res_mask: Tensor of shape (1, H_video, W_video) bool
        """
        if low_res_mask is None:
            return None

        # Convert to 3D for interpolation: (H_low_res, W_low_res) -> (1, H_low_res, W_low_res)
        low_res_mask_3d = low_res_mask.unsqueeze(0).unsqueeze(0)

        # Get video dimensions
        H_video = inference_state["orig_height"]
        W_video = inference_state["orig_width"]

        video_res_mask = F.interpolate(
            low_res_mask_3d.float(),
            size=(H_video, W_video),
            mode="bilinear",
            align_corners=False,
        )  # (1, H_video, W_video)

        # Convert to boolean - already in the right shape!
        return (video_res_mask.squeeze(0) > 0.0).to(torch.bool)

    def clear_detector_added_cond_frame_in_sam2(
        self, sam2_state, obj_id, refined_frame_idx
    ):
        """Clear detector added conditioning frame if it is within a predefined window
        of the refined frame. This allow model to update masks on these frames."""
        obj_idx = self.tracker._obj_id_to_idx(sam2_state, obj_id)

        mask_only_cond_frame_indices = []
        window = self.refinement_detector_cond_frame_removal_window
        for frame_idx in sam2_state["mask_inputs_per_obj"][obj_idx]:
            if frame_idx not in sam2_state["point_inputs_per_obj"][obj_idx]:
                # clear conditioning frames within a window of the refined frame
                if abs(frame_idx - refined_frame_idx) <= window:
                    mask_only_cond_frame_indices.append(frame_idx)

        # clear
        if len(mask_only_cond_frame_indices) > 0:
            for frame_idx in mask_only_cond_frame_indices:
                # obj_ids_on_this_frame is essentially all obj_ids in the state
                # since they are bucket batched
                obj_ids_on_this_frame = sam2_state["obj_id_to_idx"].keys()
                for obj_id2 in obj_ids_on_this_frame:
                    self.tracker.clear_all_points_in_frame(
                        sam2_state, frame_idx, obj_id2, need_output=False
                    )
            logger.info(
                f"Cleared detector mask only conditioning frames ({mask_only_cond_frame_indices}) in SAM2."
            )
        return

    def _extract_object_to_singleton_state(self, inference_state, obj_id, obj_rank):
        """
        Extract an object from a batched inference state into its own singleton state.
        """
        if self.rank != obj_rank:
            return

        tracker_states_local = inference_state["sam2_inference_states"]

        # Find the inference state containing this object
        source_state = None
        source_state_idx = None
        for idx, state in enumerate(tracker_states_local):
            if obj_id in state["obj_ids"]:
                source_state = state
                source_state_idx = idx
                break

        assert source_state is not None

        if len(source_state["obj_ids"]) <= 1:
            # Object not found or already in singleton state
            return

        # Step 1: Extract all the object's state data before removing it
        obj_idx_in_source = source_state["obj_id_to_idx"][obj_id]
        multiplex_state = source_state.get("multiplex_state")

        # Extract consolidated outputs (obj_ptr, maskmem_features, etc.) BEFORE
        # remove_object modifies the source tensors.
        singleton_consolidated_outputs = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        if "output_dict" in source_state:
            for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                source_outputs = source_state["output_dict"].get(storage_key, {})
                for f_idx, source_frame_out in source_outputs.items():
                    if source_frame_out["pred_masks"].shape[0] < obj_idx_in_source + 1:
                        continue
                    singleton_frame_out = {
                        "pred_masks": source_frame_out["pred_masks"][
                            obj_idx_in_source : obj_idx_in_source + 1
                        ].clone(),
                        "object_score_logits": source_frame_out["object_score_logits"][
                            obj_idx_in_source : obj_idx_in_source + 1
                        ].clone(),
                        "image_features": source_frame_out.get("image_features"),
                        "image_pos_enc": source_frame_out.get("image_pos_enc"),
                        "local_obj_id_to_idx": {obj_id: 0},
                    }
                    # Extract maskmem_features (demux from multiplex space)
                    maskmem_features = source_frame_out.get("maskmem_features")
                    if maskmem_features is not None and multiplex_state is not None:
                        try:
                            demuxed = multiplex_state.demux(maskmem_features)
                            maskmem_features = demuxed[
                                obj_idx_in_source : obj_idx_in_source + 1
                            ].clone()
                        except (AssertionError, IndexError):
                            maskmem_features = None
                    elif maskmem_features is not None:
                        maskmem_features = maskmem_features[
                            obj_idx_in_source : obj_idx_in_source + 1
                        ].clone()
                    singleton_frame_out["maskmem_features"] = maskmem_features
                    # Extract maskmem_pos_enc (demux level by level)
                    maskmem_pos_enc = source_frame_out.get("maskmem_pos_enc")
                    if maskmem_pos_enc is not None:
                        remapped = []
                        for level_enc in maskmem_pos_enc:
                            if level_enc is None:
                                remapped.append(None)
                                continue
                            if multiplex_state is not None:
                                try:
                                    demuxed = multiplex_state.demux(level_enc)
                                    remapped.append(
                                        demuxed[
                                            obj_idx_in_source : obj_idx_in_source + 1
                                        ].clone()
                                    )
                                except (AssertionError, IndexError):
                                    remapped.append(None)
                            else:
                                remapped.append(
                                    level_enc[
                                        obj_idx_in_source : obj_idx_in_source + 1
                                    ].clone()
                                )
                        maskmem_pos_enc = remapped
                    singleton_frame_out["maskmem_pos_enc"] = maskmem_pos_enc
                    # Extract obj_ptr (demux from multiplex space)
                    if (
                        "obj_ptr" in source_frame_out
                        and self.tracker.use_obj_ptrs_in_encoder
                    ):
                        source_obj_ptr = source_frame_out["obj_ptr"]
                        if multiplex_state is not None:
                            obj_ptr_data = multiplex_state.demux(source_obj_ptr)
                            singleton_frame_out["obj_ptr"] = obj_ptr_data[
                                obj_idx_in_source : obj_idx_in_source + 1
                            ].clone()
                        else:
                            singleton_frame_out["obj_ptr"] = source_obj_ptr[
                                obj_idx_in_source : obj_idx_in_source + 1
                            ].clone()
                    # Extract conditioning_objects
                    if "conditioning_objects" in source_frame_out:
                        if (
                            obj_idx_in_source
                            in source_frame_out["conditioning_objects"]
                        ):
                            singleton_frame_out["conditioning_objects"] = {0}
                        else:
                            singleton_frame_out["conditioning_objects"] = set()
                    singleton_consolidated_outputs[storage_key][f_idx] = (
                        singleton_frame_out
                    )

        # Extract point and mask inputs for this object
        extracted_point_inputs = {}
        extracted_mask_inputs = {}

        if (
            "point_inputs_per_obj" in source_state
            and obj_idx_in_source in source_state["point_inputs_per_obj"]
        ):
            extracted_point_inputs = source_state["point_inputs_per_obj"][
                obj_idx_in_source
            ].copy()

        if (
            "mask_inputs_per_obj" in source_state
            and obj_idx_in_source in source_state["mask_inputs_per_obj"]
        ):
            extracted_mask_inputs = source_state["mask_inputs_per_obj"][
                obj_idx_in_source
            ].copy()

        # Extract per-object outputs - these are already properly sliced for the object
        extracted_obj_cond_outputs = {}
        extracted_obj_non_cond_outputs = {}
        extracted_temp_cond_outputs = {}
        extracted_temp_non_cond_outputs = {}

        if (
            "output_dict_per_obj" in source_state
            and obj_idx_in_source in source_state["output_dict_per_obj"]
        ):
            obj_output_dict = source_state["output_dict_per_obj"][obj_idx_in_source]
            extracted_obj_cond_outputs = obj_output_dict.get(
                "cond_frame_outputs", {}
            ).copy()
            cond_input_keys = (
                extracted_point_inputs.keys() | extracted_mask_inputs.keys()
            )
            # we may have obj cond outputs for other objects in a batch, so limit to cond inputs for only this object
            extracted_obj_cond_outputs = {
                k: v
                for k, v in extracted_obj_cond_outputs.items()
                if k in cond_input_keys
            }

            extracted_obj_non_cond_outputs = obj_output_dict.get(
                "non_cond_frame_outputs", {}
            ).copy()

        if (
            "temp_output_dict_per_obj" in source_state
            and obj_idx_in_source in source_state["temp_output_dict_per_obj"]
        ):
            temp_obj_output_dict = source_state["temp_output_dict_per_obj"][
                obj_idx_in_source
            ]
            extracted_temp_cond_outputs = temp_obj_output_dict.get(
                "cond_frame_outputs", {}
            ).copy()
            extracted_temp_non_cond_outputs = temp_obj_output_dict.get(
                "non_cond_frame_outputs", {}
            ).copy()

        # Step 2: Remove the object from the source state
        remaining_obj_ids, _ = self.tracker.remove_object(
            source_state, obj_id, strict=False, need_output=False
        )

        # Step 3: Create a new singleton inference state
        new_sam2_state = self.tracker.init_state(
            cached_features=inference_state["feature_cache"],
            video_height=inference_state["orig_height"],
            video_width=inference_state["orig_width"],
            num_frames=inference_state["num_frames"],
        )

        # Step 4: Set up the singleton state structure for the extracted object
        # Map the object to index 0 in the new singleton state
        new_sam2_state["obj_id_to_idx"] = {obj_id: 0}
        new_sam2_state["obj_idx_to_id"] = {0: obj_id}
        new_sam2_state["obj_ids"] = [obj_id]

        # Step 5: Restore all the extracted state
        # Restore point and mask inputs
        new_sam2_state["point_inputs_per_obj"] = {0: extracted_point_inputs}
        new_sam2_state["mask_inputs_per_obj"] = {0: extracted_mask_inputs}

        # Restore per-object output dictionaries (already properly sliced)
        new_sam2_state["output_dict_per_obj"] = {
            0: {
                "cond_frame_outputs": extracted_obj_cond_outputs,
                "non_cond_frame_outputs": extracted_obj_non_cond_outputs,
            }
        }

        # Restore temporary outputs
        new_sam2_state["temp_output_dict_per_obj"] = {
            0: {
                "cond_frame_outputs": extracted_temp_cond_outputs,
                "non_cond_frame_outputs": extracted_temp_non_cond_outputs,
            }
        }

        # Step 6: Rebuild the consolidated output_dict for the singleton state
        # Use the extracted consolidated outputs which include obj_ptr,
        # maskmem_features, maskmem_pos_enc (not just pred_masks/object_score_logits)

        # Create singleton multiplex state and remux extracted tensors
        new_multiplex_state = self.tracker.multiplex_controller.get_state(
            num_valid_entries=1,
            device=source_state.get("device", "cuda"),
            dtype=torch.float32,
            random=False,
            object_ids=[obj_id],
        )
        new_sam2_state["multiplex_state"] = new_multiplex_state

        for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
            for f_idx, frame_out in singleton_consolidated_outputs[storage_key].items():
                if frame_out.get("maskmem_features") is not None:
                    frame_out["maskmem_features"] = frame_out[
                        "maskmem_features"
                    ].clone()
                if frame_out.get("maskmem_pos_enc") is not None:
                    frame_out["maskmem_pos_enc"] = [
                        level.clone() if level is not None else None
                        for level in frame_out["maskmem_pos_enc"]
                    ]
                if "obj_ptr" in frame_out and self.tracker.use_obj_ptrs_in_encoder:
                    frame_out["obj_ptr"] = new_multiplex_state.mux(frame_out["obj_ptr"])

        new_sam2_state["output_dict"] = singleton_consolidated_outputs

        # Step 7: Copy other important state if it exists
        for key in [
            "first_ann_frame_idx",
            "tracking_has_started",
        ]:
            if key in source_state:
                new_sam2_state[key] = source_state[key]

        # Leave consolidated_frame_inds empty so preflight reconstructs from per-obj data
        new_sam2_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        }

        # Step 8: Add the new singleton state to the list
        tracker_states_local.append(new_sam2_state)

        # Step 9: If the source state is now empty, remove it
        if len(remaining_obj_ids) == 0:
            tracker_states_local.pop(source_state_idx)
            logger.info(
                f"Removed empty inference state after extracting object {obj_id}"
            )

        logger.info(f"Object {obj_id} successfully extracted to singleton state")
