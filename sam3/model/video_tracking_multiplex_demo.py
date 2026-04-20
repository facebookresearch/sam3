import logging
from collections import OrderedDict
from copy import deepcopy
from typing import Iterable, Optional

import numpy as np
import torch
from sam3.model.data_misc import NestedTensor
from sam3.model.io_utils import load_video_frames
from sam3.model.multiplex_utils import MultiplexState
from sam3.model.sam3_tracker_utils import fill_holes_in_mask_scores
from sam3.model.video_tracking_multiplex import (
    concat_points,
    NO_OBJ_SCORE,
    VideoTrackingDynamicMultiplex,
)
from tqdm import tqdm


class VideoTrackingMultiplexDemo(VideoTrackingDynamicMultiplex):
    """
    The demo class that extends the `VideoTrackingDynamicMultiplex` to handle user interactions
    and manage inference states, with support for multi-object tracking.

    Interactions are not yet implemented.
    """

    def __init__(
        self,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # whether to also clear non-conditioning memory of the surrounding frames (only effective when `clear_non_cond_mem_around_input` is True).
        clear_non_cond_mem_for_multi_obj=False,
        # if fill_hole_area > 0, we fill small holes in the final masks up to this area (after resizing them to the original video resolution)
        fill_hole_area=0,
        # if always_start_from_first_ann_frame is True, we always start tracking from the frame where we receive the first annotation (clicks or mask)
        # and ignore the `start_frame_idx` passed to `propagate_in_video`
        always_start_from_first_ann_frame=False,
        # the maximum number of points to be used in the prompt encoder, which reduce the domain gap between training (that only has 8 points)
        # - if it's set to a positive integer, we only take the `max_point_num_in_prompt_enc//2` points and
        #   the last `(max_point_num_in_prompt_enc - max_point_num_in_prompt_enc//2)` points in the prompt encoder
        # - if it's set to 0 or negative, this option is turned off and we use all points in the prompt encoder
        max_point_num_in_prompt_enc=16,
        non_overlap_masks_for_output=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        self.fill_hole_area = fill_hole_area
        self.always_start_from_first_ann_frame = always_start_from_first_ann_frame
        self.max_point_num_in_prompt_enc = max_point_num_in_prompt_enc
        self.non_overlap_masks_for_output = non_overlap_masks_for_output

    @torch.inference_mode()
    def init_state(
        self,
        video_path,
        offload_video_to_cpu,
        offload_state_to_cpu,
        async_loading_frames=False,
        use_torchcodec=False,
        use_cv2=False,
    ):
        """Initialize a inference state."""
        # Make sure that sigmoid is used on mask logits (should be True for all our recent models).
        # Since we rely on large negative values as scores for missing objects, the raw logits
        # cannot be consumed directly and must be converted into 0~1 range via sigmoid first.
        if not self.apply_sigmoid_to_mask_logits_for_mem_enc:
            raise NotImplementedError(
                "Multi-object tracking requires sigmoid in memory encoder for non-overlapping constraints."
            )

        images, video_height, video_width = load_video_frames(
            video_path=video_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            use_torchcodec=use_torchcodec,
            use_cv2=use_cv2,
        )
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = torch.device("cuda")
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = torch.device("cuda")
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # The index of the frame that received the first annotation
        inference_state["first_ann_frame_idx"] = None
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        inference_state["multiplex_state"] = None
        # Track which frames have been refined by user interaction (per object)
        # This is used to distinguish first refinement (fresh) vs subsequent refinements (incremental)
        inference_state["user_refined_frames_per_obj"] = {}
        # # Warm up the whole model and cache the image feature on frame 0
        # # by making a dummy click on the first frame (and then cleaning it up)
        # self.add_new_points(
        #     inference_state=inference_state,
        #     frame_idx=0,
        #     obj_id=1,
        #     points=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        #     labels=torch.tensor([1], dtype=torch.int32),
        #     clear_old_points=True,
        #     rel_coordinates=True,
        # )
        # self.clear_all_points_in_video(inference_state)
        return inference_state

    def _obj_id_to_idx(self, inference_state, obj_id, error_if_new=False):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        if (
            self.is_dynamic_model or not inference_state["tracking_has_started"]
        ) and not error_if_new:
            # get the next object slot
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            # set up input and output structures for this object
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id}. "
                f"All existing object ids: {inference_state['obj_ids']}."
            )

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        """Get the total number of unique object ids received so far in this session."""
        # return len(inference_state["obj_idx_to_id"])
        return inference_state["multiplex_state"].total_valid_entries

    @torch.inference_mode()
    def _extract_object_for_interaction(self, inference_state, obj_id, frame_idx):
        """
        Extract a single object from multiplex state for singleton interaction.
        Adapted from sam3_multiplex_tracking._extract_object_to_singleton_state()

        Returns:
            singleton_state: New inference state containing only this object
            obj_idx_in_source: Original object index before removal (for merging back)
        """
        source_state = inference_state
        obj_idx_in_source = source_state["obj_id_to_idx"][obj_id]

        # Step 1: Extract all object data BEFORE removing it
        multiplex_state = source_state.get("multiplex_state")

        # Extract consolidated outputs (slice NOW before remove_object modifies tensors)
        singleton_consolidated_outputs = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }

        if "output_dict" in source_state:
            for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                source_outputs = source_state["output_dict"].get(storage_key, {})

                for f_idx, source_frame_out in source_outputs.items():
                    # Check if this frame has valid data for this object
                    has_valid_data = (
                        source_frame_out["pred_masks"].shape[0] >= obj_idx_in_source + 1
                    )

                    if has_valid_data:
                        # Create singleton frame output by slicing
                        singleton_frame_out = {
                            "pred_masks": source_frame_out["pred_masks"][
                                obj_idx_in_source : obj_idx_in_source + 1
                            ].clone(),
                            "object_score_logits": source_frame_out[
                                "object_score_logits"
                            ][obj_idx_in_source : obj_idx_in_source + 1].clone(),
                            # image_features and image_pos_enc remain shared (not in multiplex space)
                            "image_features": source_frame_out.get("image_features"),
                            "image_pos_enc": source_frame_out.get("image_pos_enc"),
                            "local_obj_id_to_idx": {obj_id: 0},
                        }

                        # Handle maskmem_features by converting from multiplex space to data space
                        maskmem_features = source_frame_out.get("maskmem_features")
                        if maskmem_features is not None:
                            if multiplex_state is not None:
                                expected_buckets = multiplex_state.num_buckets
                                expected_multiplex = multiplex_state.multiplex_count
                                if (
                                    maskmem_features.dim() >= 2
                                    and maskmem_features.shape[0] == expected_buckets
                                    and maskmem_features.shape[1] == expected_multiplex
                                ):
                                    try:
                                        demuxed_features = multiplex_state.demux(
                                            maskmem_features
                                        )
                                    except AssertionError as exc:
                                        logging.warning(
                                            "[EXTRACT] demux failed for maskmem_features shape %s: %s",
                                            tuple(maskmem_features.shape),
                                            exc,
                                        )
                                        demuxed_features = None
                                    if demuxed_features is not None:
                                        maskmem_features = demuxed_features[
                                            obj_idx_in_source : obj_idx_in_source + 1
                                        ].clone()
                                    else:
                                        maskmem_features = maskmem_features[
                                            obj_idx_in_source : obj_idx_in_source + 1
                                        ].clone()
                                elif maskmem_features.shape[0] == 0:
                                    # No entries for this object yet; treat as missing without warning
                                    maskmem_features = None
                                elif maskmem_features.shape[0] >= obj_idx_in_source + 1:
                                    # Already in data space; slice directly
                                    maskmem_features = maskmem_features[
                                        obj_idx_in_source : obj_idx_in_source + 1
                                    ].clone()
                                else:
                                    logging.warning(
                                        "[EXTRACT] maskmem_features shape %s incompatible with multiplex state; dropping tensor",
                                        tuple(maskmem_features.shape),
                                    )
                                    maskmem_features = None
                            else:
                                maskmem_features = maskmem_features[
                                    obj_idx_in_source : obj_idx_in_source + 1
                                ].clone()
                        singleton_frame_out["maskmem_features"] = maskmem_features

                        # Handle maskmem_pos_enc similarly, level by level
                        maskmem_pos_enc = source_frame_out.get("maskmem_pos_enc")
                        if maskmem_pos_enc is not None:
                            remapped_pos_enc = []
                            for level_enc in maskmem_pos_enc:
                                if level_enc is None:
                                    remapped_pos_enc.append(None)
                                    continue
                                if multiplex_state is not None:
                                    expected_buckets = multiplex_state.num_buckets
                                    expected_multiplex = multiplex_state.multiplex_count
                                    if (
                                        level_enc.dim() >= 2
                                        and level_enc.shape[0] == expected_buckets
                                        and level_enc.shape[1] == expected_multiplex
                                    ):
                                        try:
                                            demuxed_level = multiplex_state.demux(
                                                level_enc
                                            )
                                        except AssertionError as exc:
                                            logging.warning(
                                                "[EXTRACT] demux failed for maskmem_pos_enc level shape %s: %s",
                                                tuple(level_enc.shape),
                                                exc,
                                            )
                                            demuxed_level = None
                                        if demuxed_level is not None:
                                            remapped_pos_enc.append(
                                                demuxed_level[
                                                    obj_idx_in_source : obj_idx_in_source
                                                    + 1
                                                ].clone()
                                            )
                                        elif (
                                            level_enc.shape[0] >= obj_idx_in_source + 1
                                        ):
                                            remapped_pos_enc.append(
                                                level_enc[
                                                    obj_idx_in_source : obj_idx_in_source
                                                    + 1
                                                ].clone()
                                            )
                                        else:
                                            logging.warning(
                                                "[EXTRACT] maskmem_pos_enc level shape %s incompatible with multiplex state; dropping level",
                                                tuple(level_enc.shape),
                                            )
                                            remapped_pos_enc.append(None)
                                    elif level_enc.shape[0] >= obj_idx_in_source + 1:
                                        remapped_pos_enc.append(
                                            level_enc[
                                                obj_idx_in_source : obj_idx_in_source
                                                + 1
                                            ].clone()
                                        )
                                    else:
                                        logging.warning(
                                            "[EXTRACT] maskmem_pos_enc level shape %s incompatible with multiplex state; dropping level",
                                            tuple(level_enc.shape),
                                        )
                                        remapped_pos_enc.append(None)
                                else:
                                    remapped_pos_enc.append(
                                        level_enc[
                                            obj_idx_in_source : obj_idx_in_source + 1
                                        ].clone()
                                    )
                            maskmem_pos_enc = remapped_pos_enc
                        singleton_frame_out["maskmem_pos_enc"] = maskmem_pos_enc

                        # Handle obj_ptr (must demux from multiplex space first)
                        if (
                            "obj_ptr" in source_frame_out
                            and self.use_obj_ptrs_in_encoder
                        ):
                            source_obj_ptr = source_frame_out["obj_ptr"]
                            if multiplex_state is not None:
                                # Demux: multiplex space → data space
                                obj_ptr_data_space = multiplex_state.demux(
                                    source_obj_ptr
                                )
                                # Slice for this object
                                singleton_frame_out["obj_ptr"] = obj_ptr_data_space[
                                    obj_idx_in_source : obj_idx_in_source + 1
                                ].clone()
                            else:
                                singleton_frame_out["obj_ptr"] = source_obj_ptr[
                                    obj_idx_in_source : obj_idx_in_source + 1
                                ].clone()

                        # Convert conditioning_objects
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

        # Extract point and mask inputs
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

        # Extract per-object outputs
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

        # Step 2: Remove the object from source state
        remaining_obj_ids, _ = self.remove_object(
            source_state,
            obj_id,
            strict=False,
            need_output=False,
            clear_user_refined_map=False,
        )

        # If multiplex state became empty, reset it so downstream code can reinitialize
        updated_multiplex_state = source_state.get("multiplex_state")
        if updated_multiplex_state is not None:
            if (
                getattr(updated_multiplex_state, "assignments", None) is None
                or updated_multiplex_state.total_valid_entries == 0
            ):
                source_state["multiplex_state"] = None

        # Step 3: Create new singleton inference state
        singleton_state = self.init_state(
            cached_features=source_state["cached_features"],
            video_height=source_state["video_height"],
            video_width=source_state["video_width"],
            num_frames=source_state["num_frames"],
        )

        # Step 4: Set up singleton state structure
        singleton_state["obj_id_to_idx"] = {obj_id: 0}
        singleton_state["obj_idx_to_id"] = {0: obj_id}
        singleton_state["obj_ids"] = [obj_id]
        singleton_state["point_inputs_per_obj"] = {0: extracted_point_inputs}
        singleton_state["mask_inputs_per_obj"] = {0: extracted_mask_inputs}
        singleton_state["output_dict_per_obj"] = {
            0: {
                "cond_frame_outputs": extracted_obj_cond_outputs,
                "non_cond_frame_outputs": extracted_obj_non_cond_outputs,
            }
        }
        singleton_state["temp_output_dict_per_obj"] = {
            0: {
                "cond_frame_outputs": extracted_temp_cond_outputs,
                "non_cond_frame_outputs": extracted_temp_non_cond_outputs,
            }
        }
        singleton_state["frames_already_tracked"] = source_state[
            "frames_already_tracked"
        ].copy()

        # Step 5: Create new singleton multiplex state (even for 1 object, needed for obj_ptr)
        new_multiplex_state = self.multiplex_controller.get_state(
            num_valid_entries=1,
            device=source_state["device"],
            dtype=torch.float32,
            random=False,
            object_ids=[obj_id],
        )
        singleton_state["multiplex_state"] = new_multiplex_state

        # Step 6: Remux extracted tensors into the singleton multiplex space
        for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
            for f_idx, frame_out in singleton_consolidated_outputs[storage_key].items():
                # mask memory features
                if frame_out.get("maskmem_features") is not None:
                    # Keep mask memory features in data space (num_objects, C, H, W)
                    frame_out["maskmem_features"] = frame_out[
                        "maskmem_features"
                    ].clone()

                if frame_out.get("maskmem_pos_enc") is not None:
                    remapped_levels = []
                    for level_enc in frame_out["maskmem_pos_enc"]:
                        if level_enc is None:
                            remapped_levels.append(None)
                            continue
                        remapped_levels.append(level_enc.clone())
                    frame_out["maskmem_pos_enc"] = remapped_levels

                # object pointers
                if "obj_ptr" in frame_out and self.use_obj_ptrs_in_encoder:
                    # Mux: data space [1, D] → singleton multiplex space [1, 1, D]
                    frame_out["obj_ptr"] = new_multiplex_state.mux(frame_out["obj_ptr"])

        singleton_state["output_dict"] = singleton_consolidated_outputs

        return singleton_state, obj_idx_in_source

    @torch.inference_mode()
    def _merge_singleton_interaction_result(
        self,
        inference_state,
        singleton_state,
        obj_id,
        original_obj_idx,
    ):
        """
        Merge singleton interaction result back into multiplex state.

        SIMPLIFIED APPROACH: Add object back at the END (new index), not at original position.
        This avoids complex index shifting and works with multiplex controller's add_objects() API.

        Args:
            inference_state: The main multiplex inference state
            singleton_state: The singleton state with interaction results
            obj_id: The object ID
            original_obj_idx: The original index before extraction (unused - we add at end instead)
        """
        # Determine new index (add at end)
        new_obj_idx = len(inference_state["obj_ids"])

        # Step 1: Add object mappings at new index
        inference_state["obj_ids"].append(obj_id)
        inference_state["obj_id_to_idx"][obj_id] = new_obj_idx

        # Create entry in output_dict_per_obj and temp_output_dict_per_obj for new index
        # These are DICTIONARIES indexed by obj_idx, not lists!
        inference_state["output_dict_per_obj"][new_obj_idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        inference_state["temp_output_dict_per_obj"][new_obj_idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }

        inference_state["obj_idx_to_id"][new_obj_idx] = obj_id

        # Step 2: Add object to multiplex state buckets using proper API
        multiplex_state = inference_state.get("multiplex_state")

        assignments = (
            getattr(multiplex_state, "assignments", None)
            if multiplex_state is not None
            else None
        )
        total_valid_entries = (
            getattr(multiplex_state, "total_valid_entries", 0)
            if multiplex_state is not None and assignments is not None
            else 0
        )
        need_state_reinit = (
            multiplex_state is None or assignments is None or total_valid_entries == 0
        )

        if not need_state_reinit and getattr(multiplex_state, "object_ids", None):
            if obj_id in multiplex_state.object_ids:
                old_idx = multiplex_state.object_ids.index(obj_id)
                multiplex_state.remove_objects(object_indices=[old_idx], strict=False)
                assignments = getattr(multiplex_state, "assignments", None)
                total_valid_entries = (
                    getattr(multiplex_state, "total_valid_entries", 0)
                    if assignments is not None
                    else 0
                )
                need_state_reinit = assignments is None or total_valid_entries == 0

        if need_state_reinit:
            inference_state["multiplex_state"] = self.multiplex_controller.get_state(
                num_valid_entries=len(inference_state["obj_ids"]),
                device=inference_state["device"],
                dtype=torch.float32,
                random=False,
                object_ids=list(inference_state["obj_ids"]),
            )
            multiplex_state = inference_state["multiplex_state"]
        else:
            # Allow new buckets since we're adding at a new index (the old bucket slot may have been removed)
            multiplex_state.add_objects(
                object_indices=[new_obj_idx],
                object_ids=[obj_id],
                allow_new_buckets=True,  # May need new bucket if old slot was compacted
            )

        # Step 3: Restore point and mask inputs at new index
        singleton_obj_idx = 0  # Object is always at index 0 in singleton state
        if (
            "point_inputs_per_obj" in singleton_state
            and singleton_obj_idx in singleton_state["point_inputs_per_obj"]
        ):
            if "point_inputs_per_obj" not in inference_state:
                inference_state["point_inputs_per_obj"] = {}
            inference_state["point_inputs_per_obj"][new_obj_idx] = singleton_state[
                "point_inputs_per_obj"
            ][singleton_obj_idx].copy()

        if (
            "mask_inputs_per_obj" in singleton_state
            and singleton_obj_idx in singleton_state["mask_inputs_per_obj"]
        ):
            if "mask_inputs_per_obj" not in inference_state:
                inference_state["mask_inputs_per_obj"] = {}
            inference_state["mask_inputs_per_obj"][new_obj_idx] = singleton_state[
                "mask_inputs_per_obj"
            ][singleton_obj_idx].copy()

        # Step 4: Restore per-object outputs at new index
        if (
            "output_dict_per_obj" in singleton_state
            and singleton_obj_idx in singleton_state["output_dict_per_obj"]
        ):
            if "output_dict_per_obj" not in inference_state:
                inference_state["output_dict_per_obj"] = {}
            inference_state["output_dict_per_obj"][new_obj_idx] = singleton_state[
                "output_dict_per_obj"
            ][singleton_obj_idx].copy()

        if (
            "temp_output_dict_per_obj" in singleton_state
            and singleton_obj_idx in singleton_state["temp_output_dict_per_obj"]
        ):
            if "temp_output_dict_per_obj" not in inference_state:
                inference_state["temp_output_dict_per_obj"] = {}
            inference_state["temp_output_dict_per_obj"][new_obj_idx] = singleton_state[
                "temp_output_dict_per_obj"
            ][singleton_obj_idx].copy()

        # Step 5: Merge consolidated outputs back into multiplex (append at new_obj_idx)
        # Preserve each frame's original storage key from the singleton state so that
        # conditioning frames remain in cond_frame_outputs after the merge.
        if "output_dict" in singleton_state:
            singleton_multiplex_state = singleton_state.get("multiplex_state")
            for singleton_storage_key in [
                "cond_frame_outputs",
                "non_cond_frame_outputs",
            ]:
                singleton_outputs = singleton_state["output_dict"].get(
                    singleton_storage_key, {}
                )

                # Skip if singleton doesn't have any frames in this storage_key
                if not singleton_outputs:
                    continue

                for frame_idx, singleton_frame_out in singleton_outputs.items():
                    # Get or create frame output in main state at the EXPECTED storage_key
                    if "output_dict" not in inference_state:
                        inference_state["output_dict"] = {
                            "cond_frame_outputs": {},
                            "non_cond_frame_outputs": {},
                        }

                    if (
                        frame_idx
                        not in inference_state["output_dict"][singleton_storage_key]
                    ):
                        # Frame doesn't exist - create with singleton results at new_obj_idx
                        num_objs = len(inference_state["obj_ids"])

                        # Ensure num_objs is at least new_obj_idx + 1
                        # (in case obj_ids list is somehow inconsistent)
                        if num_objs <= new_obj_idx:
                            num_objs = new_obj_idx + 1

                        new_maskmem_features = None
                        new_maskmem_pos_enc = None
                        if (
                            singleton_frame_out.get("maskmem_features") is not None
                            and multiplex_state is not None
                        ):
                            # Check if singleton features are in multiplexed format and demux if needed
                            singleton_features_muxed = singleton_frame_out[
                                "maskmem_features"
                            ]
                            if singleton_features_muxed.shape[:2] == (
                                singleton_multiplex_state.num_buckets,
                                singleton_multiplex_state.multiplex_count,
                            ):
                                # Singleton features are multiplexed, need to demux
                                singleton_features_data = (
                                    singleton_multiplex_state.demux(
                                        singleton_features_muxed
                                    )
                                )
                            else:
                                # Singleton features are in data space
                                singleton_features_data = singleton_features_muxed

                            feature_shape = (num_objs,) + singleton_features_data.shape[
                                1:
                            ]
                            maskmem_features_data = torch.zeros(
                                feature_shape,
                                dtype=singleton_features_data.dtype,
                                device=singleton_features_data.device,
                            )
                            maskmem_features_data[new_obj_idx : new_obj_idx + 1] = (
                                singleton_features_data
                            )
                            # Mux using destination multiplex state
                            new_maskmem_features = multiplex_state.mux(
                                maskmem_features_data
                            )

                        if (
                            singleton_frame_out.get("maskmem_pos_enc") is not None
                            and multiplex_state is not None
                        ):
                            new_maskmem_pos_enc = []
                            for level_enc in singleton_frame_out["maskmem_pos_enc"]:
                                if level_enc is None:
                                    new_maskmem_pos_enc.append(None)
                                    continue
                                # Check if singleton pos_enc is in multiplexed format and demux if needed
                                if level_enc.shape[:2] == (
                                    singleton_multiplex_state.num_buckets,
                                    singleton_multiplex_state.multiplex_count,
                                ):
                                    # Singleton pos_enc is multiplexed, need to demux
                                    level_data = singleton_multiplex_state.demux(
                                        level_enc
                                    )
                                else:
                                    # Singleton pos_enc is in data space
                                    level_data = level_enc

                                level_shape = (num_objs,) + level_data.shape[1:]
                                level_tensor = torch.zeros(
                                    level_shape,
                                    dtype=level_data.dtype,
                                    device=level_data.device,
                                )
                                level_tensor[new_obj_idx : new_obj_idx + 1] = level_data
                                # Mux using destination multiplex state to store in multiplex format
                                new_maskmem_pos_enc.append(
                                    multiplex_state.mux(level_tensor)
                                )

                        inference_state["output_dict"][singleton_storage_key][
                            frame_idx
                        ] = {
                            "maskmem_features": new_maskmem_features,
                            "maskmem_pos_enc": new_maskmem_pos_enc,
                            "image_features": singleton_frame_out.get("image_features"),
                            "image_pos_enc": singleton_frame_out.get("image_pos_enc"),
                            "local_obj_id_to_idx": {obj_id: new_obj_idx},
                            "conditioning_objects": (
                                set([new_obj_idx])
                                if singleton_obj_idx
                                in singleton_frame_out.get(
                                    "conditioning_objects", set()
                                )
                                else set()
                            ),
                            "pred_masks": torch.zeros(
                                (
                                    num_objs,
                                    1,
                                    singleton_frame_out["pred_masks"].shape[2],
                                    singleton_frame_out["pred_masks"].shape[3],
                                ),
                                dtype=singleton_frame_out["pred_masks"].dtype,
                                device=singleton_frame_out["pred_masks"].device,
                            ),
                            "object_score_logits": torch.full(
                                (num_objs, 1),
                                NO_OBJ_SCORE,
                                dtype=singleton_frame_out["object_score_logits"].dtype,
                                device=singleton_frame_out[
                                    "object_score_logits"
                                ].device,
                            ),
                        }
                        # Set singleton results at new_obj_idx
                        inference_state["output_dict"][singleton_storage_key][
                            frame_idx
                        ]["pred_masks"][
                            new_obj_idx : new_obj_idx + 1
                        ] = singleton_frame_out["pred_masks"]
                        inference_state["output_dict"][singleton_storage_key][
                            frame_idx
                        ]["object_score_logits"][
                            new_obj_idx : new_obj_idx + 1
                        ] = singleton_frame_out["object_score_logits"]

                        # Also copy pred_masks_video_res if it exists in singleton output
                        if "pred_masks_video_res" in singleton_frame_out:
                            inference_state["output_dict"][singleton_storage_key][
                                frame_idx
                            ]["pred_masks_video_res"] = torch.zeros(
                                (
                                    num_objs,
                                    1,
                                    singleton_frame_out["pred_masks_video_res"].shape[
                                        2
                                    ],
                                    singleton_frame_out["pred_masks_video_res"].shape[
                                        3
                                    ],
                                ),
                                dtype=singleton_frame_out["pred_masks_video_res"].dtype,
                                device=singleton_frame_out[
                                    "pred_masks_video_res"
                                ].device,
                            )
                            inference_state["output_dict"][singleton_storage_key][
                                frame_idx
                            ]["pred_masks_video_res"][
                                new_obj_idx : new_obj_idx + 1
                            ] = singleton_frame_out["pred_masks_video_res"]

                        # Handle obj_ptr if present
                        if (
                            "obj_ptr" in singleton_frame_out
                            and self.use_obj_ptrs_in_encoder
                        ):
                            singleton_obj_ptr_data = singleton_multiplex_state.demux(
                                singleton_frame_out["obj_ptr"]
                            )
                            obj_ptr_data = torch.zeros(
                                (num_objs, singleton_obj_ptr_data.shape[1]),
                                dtype=singleton_obj_ptr_data.dtype,
                                device=singleton_obj_ptr_data.device,
                            )
                            obj_ptr_data[new_obj_idx : new_obj_idx + 1] = (
                                singleton_obj_ptr_data
                            )
                            inference_state["output_dict"][singleton_storage_key][
                                frame_idx
                            ]["obj_ptr"] = multiplex_state.mux(obj_ptr_data)
                    else:
                        # Frame exists - expand tensors and add singleton results
                        main_frame_out = inference_state["output_dict"][
                            singleton_storage_key
                        ][frame_idx]

                        num_objs_total = len(inference_state["obj_ids"])

                        if (
                            singleton_frame_out.get("maskmem_features") is not None
                            and multiplex_state is not None
                        ):
                            # Check if singleton features are in multiplexed format and demux if needed
                            singleton_features_muxed = singleton_frame_out[
                                "maskmem_features"
                            ]
                            if singleton_features_muxed.shape[:2] == (
                                singleton_multiplex_state.num_buckets,
                                singleton_multiplex_state.multiplex_count,
                            ):
                                # Singleton features are multiplexed, need to demux
                                singleton_features_data = (
                                    singleton_multiplex_state.demux(
                                        singleton_features_muxed
                                    )
                                )
                            else:
                                # Singleton features are in data space
                                singleton_features_data = singleton_features_muxed

                            existing_features_muxed = main_frame_out.get(
                                "maskmem_features"
                            )
                            if existing_features_muxed is not None:
                                # Check if features are in multiplex format before demuxing
                                if existing_features_muxed.shape[:2] == (
                                    multiplex_state.num_buckets,
                                    multiplex_state.multiplex_count,
                                ):
                                    # Features are in multiplex format, demux them
                                    existing_features_data = multiplex_state.demux(
                                        existing_features_muxed
                                    )
                                else:
                                    # Features are already in data space, use directly
                                    existing_features_data = existing_features_muxed
                            else:
                                existing_features_data = None

                            if existing_features_data is None:
                                feature_shape = (
                                    num_objs_total,
                                ) + singleton_features_data.shape[1:]
                                existing_features_data = torch.zeros(
                                    feature_shape,
                                    dtype=singleton_features_data.dtype,
                                    device=singleton_features_data.device,
                                )
                            elif existing_features_data.shape[0] < num_objs_total:
                                pad_size = (
                                    num_objs_total - existing_features_data.shape[0]
                                )
                                pad = torch.zeros(
                                    (pad_size,) + existing_features_data.shape[1:],
                                    dtype=existing_features_data.dtype,
                                    device=existing_features_data.device,
                                )
                                existing_features_data = torch.cat(
                                    [existing_features_data, pad], dim=0
                                )

                            existing_features_data[new_obj_idx : new_obj_idx + 1] = (
                                singleton_features_data
                            )
                            main_frame_out["maskmem_features"] = multiplex_state.mux(
                                existing_features_data
                            )

                        if (
                            singleton_frame_out.get("maskmem_pos_enc") is not None
                            and multiplex_state is not None
                        ):
                            existing_pos_enc_list = (
                                main_frame_out.get("maskmem_pos_enc") or []
                            )
                            new_maskmem_pos_enc = []
                            max_levels = max(
                                len(singleton_frame_out["maskmem_pos_enc"]),
                                len(existing_pos_enc_list),
                            )
                            for level_idx in range(max_levels):
                                singleton_level_muxed = (
                                    singleton_frame_out["maskmem_pos_enc"][level_idx]
                                    if level_idx
                                    < len(singleton_frame_out["maskmem_pos_enc"])
                                    else None
                                )
                                existing_level_muxed = (
                                    existing_pos_enc_list[level_idx]
                                    if level_idx < len(existing_pos_enc_list)
                                    else None
                                )

                                if singleton_level_muxed is None:
                                    # Keep existing entry (which may also be None)
                                    new_maskmem_pos_enc.append(existing_level_muxed)
                                    continue

                                # Check if singleton pos_enc is in multiplexed format and demux if needed
                                if singleton_level_muxed.shape[:2] == (
                                    singleton_multiplex_state.num_buckets,
                                    singleton_multiplex_state.multiplex_count,
                                ):
                                    # Singleton pos_enc is multiplexed, need to demux
                                    singleton_level_data = (
                                        singleton_multiplex_state.demux(
                                            singleton_level_muxed
                                        )
                                    )
                                else:
                                    # Singleton pos_enc is in data space
                                    singleton_level_data = singleton_level_muxed

                                if existing_level_muxed is not None:
                                    # Check if pos_enc is in multiplex format before demuxing
                                    if existing_level_muxed.shape[:2] == (
                                        multiplex_state.num_buckets,
                                        multiplex_state.multiplex_count,
                                    ):
                                        # Positional encoding is in multiplex format, demux it
                                        existing_level_data = multiplex_state.demux(
                                            existing_level_muxed
                                        )
                                    else:
                                        # Positional encoding is already in data space, use directly
                                        existing_level_data = existing_level_muxed
                                else:
                                    existing_level_data = None

                                if existing_level_data is None:
                                    level_shape = (
                                        num_objs_total,
                                    ) + singleton_level_data.shape[1:]
                                    existing_level_data = torch.zeros(
                                        level_shape,
                                        dtype=singleton_level_data.dtype,
                                        device=singleton_level_data.device,
                                    )
                                elif existing_level_data.shape[0] < num_objs_total:
                                    pad_size = (
                                        num_objs_total - existing_level_data.shape[0]
                                    )
                                    pad = torch.zeros(
                                        (pad_size,) + existing_level_data.shape[1:],
                                        dtype=existing_level_data.dtype,
                                        device=existing_level_data.device,
                                    )
                                    existing_level_data = torch.cat(
                                        [existing_level_data, pad], dim=0
                                    )

                                existing_level_data[new_obj_idx : new_obj_idx + 1] = (
                                    singleton_level_data
                                )
                                new_maskmem_pos_enc.append(
                                    multiplex_state.mux(existing_level_data)
                                )

                            main_frame_out["maskmem_pos_enc"] = new_maskmem_pos_enc

                        singleton_pred_masks = singleton_frame_out[
                            "pred_masks"
                        ]  # [1, 1, H, W]
                        singleton_scores = singleton_frame_out[
                            "object_score_logits"
                        ]  # [1, 1]

                        # Expand tensors if needed
                        num_existing_objs = main_frame_out["pred_masks"].shape[0]
                        if new_obj_idx >= num_existing_objs:
                            num_objs_needed = new_obj_idx + 1
                            pad_size = num_objs_needed - num_existing_objs

                            main_frame_out["pred_masks"] = torch.cat(
                                [
                                    main_frame_out["pred_masks"],
                                    torch.zeros(
                                        (
                                            pad_size,
                                            1,
                                            singleton_pred_masks.shape[2],
                                            singleton_pred_masks.shape[3],
                                        ),
                                        dtype=singleton_pred_masks.dtype,
                                        device=singleton_pred_masks.device,
                                    ),
                                ],
                                dim=0,
                            )

                            main_frame_out["object_score_logits"] = torch.cat(
                                [
                                    main_frame_out["object_score_logits"],
                                    torch.full(
                                        (pad_size, 1),
                                        NO_OBJ_SCORE,
                                        dtype=singleton_scores.dtype,
                                        device=singleton_scores.device,
                                    ),
                                ],
                                dim=0,
                            )

                        # Set singleton results at new_obj_idx
                        main_frame_out["pred_masks"][new_obj_idx : new_obj_idx + 1] = (
                            singleton_pred_masks
                        )
                        main_frame_out["object_score_logits"][
                            new_obj_idx : new_obj_idx + 1
                        ] = singleton_scores
                        # Initialize local_obj_id_to_idx if missing (e.g., frame
                        # output was created by VG propagation's track_step which
                        # does not populate this field).
                        if "local_obj_id_to_idx" not in main_frame_out:
                            main_frame_out["local_obj_id_to_idx"] = deepcopy(
                                inference_state["obj_id_to_idx"]
                            )
                        main_frame_out["local_obj_id_to_idx"][obj_id] = new_obj_idx

                        # Also expand and copy pred_masks_video_res if it exists in singleton output
                        if "pred_masks_video_res" in singleton_frame_out:
                            if "pred_masks_video_res" in main_frame_out:
                                # Expand existing video_res masks
                                if (
                                    main_frame_out["pred_masks_video_res"].shape[0]
                                    < new_obj_idx + 1
                                ):
                                    pad_size = (
                                        new_obj_idx
                                        + 1
                                        - main_frame_out["pred_masks_video_res"].shape[
                                            0
                                        ]
                                    )
                                    main_frame_out["pred_masks_video_res"] = torch.cat(
                                        [
                                            main_frame_out["pred_masks_video_res"],
                                            torch.zeros(
                                                (
                                                    pad_size,
                                                    1,
                                                    singleton_frame_out[
                                                        "pred_masks_video_res"
                                                    ].shape[2],
                                                    singleton_frame_out[
                                                        "pred_masks_video_res"
                                                    ].shape[3],
                                                ),
                                                dtype=singleton_frame_out[
                                                    "pred_masks_video_res"
                                                ].dtype,
                                                device=singleton_frame_out[
                                                    "pred_masks_video_res"
                                                ].device,
                                            ),
                                        ],
                                        dim=0,
                                    )
                            else:
                                # Create new video_res masks tensor
                                num_objs = len(inference_state["obj_ids"])
                                main_frame_out["pred_masks_video_res"] = torch.zeros(
                                    (
                                        num_objs,
                                        1,
                                        singleton_frame_out[
                                            "pred_masks_video_res"
                                        ].shape[2],
                                        singleton_frame_out[
                                            "pred_masks_video_res"
                                        ].shape[3],
                                    ),
                                    dtype=singleton_frame_out[
                                        "pred_masks_video_res"
                                    ].dtype,
                                    device=singleton_frame_out[
                                        "pred_masks_video_res"
                                    ].device,
                                )
                            # Set singleton video_res mask
                            main_frame_out["pred_masks_video_res"][
                                new_obj_idx : new_obj_idx + 1
                            ] = singleton_frame_out["pred_masks_video_res"]

                        # Handle obj_ptr
                        if (
                            "obj_ptr" in singleton_frame_out
                            and self.use_obj_ptrs_in_encoder
                        ):
                            singleton_obj_ptr_data = singleton_multiplex_state.demux(
                                singleton_frame_out["obj_ptr"]
                            )  # [1, D]

                            if "obj_ptr" in main_frame_out:
                                # The existing obj_ptr may have been created with a DIFFERENT number of buckets
                                # (before we called multiplex_state.add_objects() which may have created new buckets).
                                # We need to infer the OLD bucket count from the tensor shape to demux it correctly.

                                old_obj_ptr_muxed = main_frame_out["obj_ptr"]
                                # Infer old bucket count: shape is [B_old, M_old, D]
                                old_num_buckets = old_obj_ptr_muxed.shape[1]

                                # Create temporary multiplex state with old bucket count to demux
                                if old_num_buckets != multiplex_state.num_buckets:
                                    # Bucket count changed - cannot safely demux old obj_ptr
                                    # Instead, create new obj_ptr from scratch for all objects
                                    num_objs = len(inference_state["obj_ids"])
                                    obj_ptr_data = torch.zeros(
                                        (num_objs, singleton_obj_ptr_data.shape[1]),
                                        dtype=singleton_obj_ptr_data.dtype,
                                        device=singleton_obj_ptr_data.device,
                                    )
                                    # Only set the singleton object's ptr, leave others as zeros
                                    obj_ptr_data[new_obj_idx : new_obj_idx + 1] = (
                                        singleton_obj_ptr_data
                                    )
                                    main_frame_out["obj_ptr"] = multiplex_state.mux(
                                        obj_ptr_data
                                    )
                                else:
                                    # Bucket count matches - safe to demux
                                    main_obj_ptr_data = multiplex_state.demux(
                                        old_obj_ptr_muxed
                                    )

                                    # Expand if needed
                                    if main_obj_ptr_data.shape[0] < new_obj_idx + 1:
                                        pad_size = (
                                            new_obj_idx + 1 - main_obj_ptr_data.shape[0]
                                        )
                                        main_obj_ptr_data = torch.cat(
                                            [
                                                main_obj_ptr_data,
                                                torch.zeros(
                                                    (
                                                        pad_size,
                                                        main_obj_ptr_data.shape[1],
                                                    ),
                                                    dtype=main_obj_ptr_data.dtype,
                                                    device=main_obj_ptr_data.device,
                                                ),
                                            ],
                                            dim=0,
                                        )

                                    main_obj_ptr_data[new_obj_idx : new_obj_idx + 1] = (
                                        singleton_obj_ptr_data
                                    )
                                    main_frame_out["obj_ptr"] = multiplex_state.mux(
                                        main_obj_ptr_data
                                    )
                            else:
                                # Create new obj_ptr
                                num_objs = len(inference_state["obj_ids"])
                                obj_ptr_data = torch.zeros(
                                    (num_objs, singleton_obj_ptr_data.shape[1]),
                                    dtype=singleton_obj_ptr_data.dtype,
                                    device=singleton_obj_ptr_data.device,
                                )
                                obj_ptr_data[new_obj_idx : new_obj_idx + 1] = (
                                    singleton_obj_ptr_data
                                )
                                main_frame_out["obj_ptr"] = multiplex_state.mux(
                                    obj_ptr_data
                                )

                        # Update conditioning_objects
                        if singleton_obj_idx in singleton_frame_out.get(
                            "conditioning_objects", set()
                        ):
                            main_frame_out["conditioning_objects"].add(new_obj_idx)

    @torch.inference_mode()
    def add_new_points(
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
        """
        Add new points to create a new object in the multiplex model.

        This method converts point inputs to masks via the interactivity head and adds
        the new object to the existing multiplex bucket (for dynamic models).

        Args:
            inference_state: Current inference state
            frame_idx: Frame index to add points
            obj_id: Object ID (will be auto-created if new)
            points: Point coordinates tensor
            labels: Point labels tensor (1 for positive, 0 for negative)
            clear_old_points: Whether to clear old points on this frame
            rel_coordinates: Whether points are in relative coordinates [0, 1]
            use_prev_mem_frame: Whether to use previous memory frames (for compatibility)

        Returns:
            Tuple of (frame_idx, obj_ids, low_res_masks, video_res_masks)
        """
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        obj_idxs = [obj_idx]
        obj_ids = [obj_id]

        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if points.dim() == 2:
            points = points.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        if rel_coordinates:
            points = points * self.image_size

        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])

        if not clear_old_points:
            old_point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            old_point_inputs = None

        point_inputs = concat_points(old_point_inputs, points, labels)
        point_inputs_per_frame[frame_idx] = point_inputs

        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]

        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]

        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        multiplex_state = inference_state["multiplex_state"]
        is_new_state = multiplex_state is None

        if is_new_state:
            multiplex_state = self.multiplex_controller.get_state(
                num_valid_entries=1,
                device=inference_state["device"],
                dtype=torch.float32,
                random=False,
                object_ids=obj_ids,
            )
            inference_state["multiplex_state"] = multiplex_state

        # Determine interaction case:
        # - New object: never seen before
        # - Refine: existing mask on tracked frame
        # - Gap fill: object exists but frame has no output
        is_existing_object = (
            not is_new_state
            and multiplex_state is not None
            and obj_id in multiplex_state.object_ids
        )

        if is_existing_object:
            if is_init_cond_frame:
                is_new_obj = False
                is_refine = False
                is_gap_fill_case = True
            else:
                is_new_obj = False
                is_refine = True
                is_gap_fill_case = False
        else:
            is_new_obj = True
            is_refine = False
            is_gap_fill_case = False

        if is_new_obj:
            should_add_to_existing = not is_new_state
            allow_new_buckets_local = True
            prefer_new_buckets_local = True

            current_out, _ = self._run_single_frame_inference(
                inference_state=inference_state,
                output_dict=inference_state["output_dict"],
                frame_idx=frame_idx,
                batch_size=1,
                is_init_cond_frame=True,
                point_inputs=point_inputs,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=False,
                prev_sam_mask_logits=None,
                add_to_existing_state=should_add_to_existing,
                new_obj_idxs=obj_idxs,
                new_obj_ids=obj_ids,
                allow_new_buckets=allow_new_buckets_local,
                prefer_new_buckets=prefer_new_buckets_local,
                objects_to_interact=None,
            )
        elif is_refine:
            singleton_state, original_obj_idx = self._extract_object_for_interaction(
                inference_state, obj_id, frame_idx
            )

            user_refined_frames_map = inference_state.get(
                "user_refined_frames_per_obj", {}
            )
            user_refined_frames = user_refined_frames_map.get(obj_id)
            if user_refined_frames is None:
                user_refined_frames = set()
            is_first_refinement = frame_idx not in user_refined_frames

            prev_sam_mask_logits_singleton = None
            if not is_first_refinement:
                singleton_obj_idx = 0
                singleton_output_dict = singleton_state["output_dict_per_obj"][
                    singleton_obj_idx
                ]
                singleton_temp_output_dict = singleton_state[
                    "temp_output_dict_per_obj"
                ][singleton_obj_idx]

                # Check BOTH storage keys since previous refinement might be in a different key
                # (e.g., first refinement creates cond_frame, but after propagation,
                # second refinement on same frame would look for non_cond_frame)
                prev_out = None

                storage_key_current = (
                    "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
                )
                prev_out = singleton_temp_output_dict[storage_key_current].get(
                    frame_idx
                )

                if prev_out is None:
                    prev_out = singleton_output_dict["cond_frame_outputs"].get(
                        frame_idx
                    )
                if prev_out is None:
                    prev_out = singleton_output_dict["non_cond_frame_outputs"].get(
                        frame_idx
                    )

                if prev_out is not None and prev_out["pred_masks"] is not None:
                    prev_sam_mask_logits_singleton = prev_out["pred_masks"].cuda(
                        non_blocking=True
                    )
                    prev_sam_mask_logits_singleton = torch.clamp(
                        prev_sam_mask_logits_singleton, -32.0, 32.0
                    )

            if is_first_refinement:
                # ALWAYS use is_init_cond_frame=True to force interaction_only mode
                # for fresh segmentation from points (not refinement of propagated mask).
                singleton_is_init_cond = True
                singleton_objects_to_interact = None
            else:
                # Second+ refinement: Incremental refinement for quality improvement
                singleton_is_init_cond = False
                singleton_objects_to_interact = (
                    [0] if prev_sam_mask_logits_singleton is not None else None
                )

            singleton_obj_idx = 0
            singleton_obj_idxs = [singleton_obj_idx]
            singleton_obj_ids = [obj_id]

            current_out, _ = self._run_single_frame_inference(
                inference_state=singleton_state,
                output_dict=singleton_state["output_dict"],
                frame_idx=frame_idx,
                batch_size=1,
                is_init_cond_frame=singleton_is_init_cond,
                point_inputs=point_inputs,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=False,
                prev_sam_mask_logits=prev_sam_mask_logits_singleton,
                add_to_existing_state=False,
                new_obj_idxs=singleton_obj_idxs,
                new_obj_ids=singleton_obj_ids,
                allow_new_buckets=False,
                objects_to_interact=singleton_objects_to_interact,
            )

            singleton_storage_key = (
                "cond_frame_outputs"
                if singleton_is_init_cond
                else "non_cond_frame_outputs"
            )

            _, singleton_video_res_masks = self._get_orig_video_res_output(
                singleton_state, current_out["pred_masks"]
            )
            current_out["pred_masks_video_res"] = singleton_video_res_masks

            singleton_state["output_dict"][singleton_storage_key][frame_idx] = (
                current_out
            )

            self._merge_singleton_interaction_result(
                inference_state, singleton_state, obj_id, original_obj_idx
            )

            obj_idx = inference_state["obj_id_to_idx"][obj_id]
            obj_idxs = [obj_idx]

            if "user_refined_frames_per_obj" not in inference_state:
                inference_state["user_refined_frames_per_obj"] = {}
            if obj_id not in inference_state["user_refined_frames_per_obj"]:
                inference_state["user_refined_frames_per_obj"][obj_id] = set()

            inference_state["user_refined_frames_per_obj"][obj_id].add(frame_idx)

            merged_frame_out = inference_state["output_dict"][singleton_storage_key][
                frame_idx
            ]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]

            if "pred_masks_video_res" in merged_frame_out:
                pred_masks_video_res_slice = merged_frame_out["pred_masks_video_res"][
                    obj_idx : obj_idx + 1
                ]
            else:
                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state, merged_frame_out["pred_masks"]
                )
                pred_masks_video_res_slice = video_res_masks[obj_idx : obj_idx + 1]

            pred_masks_slice = merged_frame_out["pred_masks"][obj_idx : obj_idx + 1]

            obj_temp_output_dict[singleton_storage_key][frame_idx] = {
                "pred_masks": pred_masks_slice,
                "pred_masks_video_res": pred_masks_video_res_slice,
                "object_score_logits": merged_frame_out["object_score_logits"][
                    obj_idx : obj_idx + 1
                ],
            }
            obj_output_dict[singleton_storage_key][frame_idx] = obj_temp_output_dict[
                singleton_storage_key
            ][frame_idx]

        elif is_gap_fill_case:
            # Gap fill: Run inference directly in multiplex mode (no singleton extraction)
            # Even though is_init_cond_frame=True, we use add_to_existing_state=False
            # because the object ALREADY EXISTS in multiplex state.
            obj_idx = inference_state["obj_id_to_idx"][obj_id]
            obj_idxs = [obj_idx]
            batch_size = self._get_obj_num(inference_state)

            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]

            current_out, _ = self._run_single_frame_inference(
                inference_state=inference_state,
                output_dict=inference_state["output_dict"],
                frame_idx=frame_idx,
                batch_size=batch_size,
                is_init_cond_frame=True,
                point_inputs=point_inputs,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=False,
                prev_sam_mask_logits=None,
                add_to_existing_state=False,
                new_obj_idxs=[obj_idx],
                new_obj_ids=[obj_id],
                allow_new_buckets=False,
                prefer_new_buckets=False,
                objects_to_interact=[obj_idx],
            )

            current_out["local_obj_id_to_idx"] = deepcopy(
                inference_state["obj_id_to_idx"]
            )

            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, current_out["pred_masks"]
            )
            current_out["pred_masks_video_res"] = video_res_masks

            is_cond = storage_key == "cond_frame_outputs"
            if (
                is_cond
                and frame_idx
                in inference_state["output_dict"]["non_cond_frame_outputs"]
            ):
                del inference_state["output_dict"]["non_cond_frame_outputs"][frame_idx]
                if "consolidated_frame_inds" in inference_state:
                    inference_state["consolidated_frame_inds"][
                        "non_cond_frame_outputs"
                    ].discard(frame_idx)

            # Store consolidated output (has obj_ptr, maskmem_features, etc.)
            inference_state["output_dict"][storage_key][frame_idx] = current_out

            # Mark as consolidated
            if "consolidated_frame_inds" in inference_state:
                inference_state["consolidated_frame_inds"][storage_key].add(frame_idx)

            # Also store per-object slices in temp_output_dict_per_obj
            obj_temp_output_dict[storage_key][frame_idx] = {
                "pred_masks": current_out["pred_masks"][obj_idx : obj_idx + 1],
                "pred_masks_video_res": video_res_masks[obj_idx : obj_idx + 1],
                "object_score_logits": current_out["object_score_logits"][
                    obj_idx : obj_idx + 1
                ],
            }
            obj_output_dict[storage_key][frame_idx] = obj_temp_output_dict[storage_key][
                frame_idx
            ]

        # Store outputs and prepare return values
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]

        # For refinement/gap fill (singleton extraction), handle singleton output specially
        if is_refine or is_gap_fill_case:
            # Singleton case: The merge already updated the consolidated output_dict during merge.
            # However, we need to ensure the frame is properly stored and marked.

            singleton_obj_idx = 0

            # Get video resolution masks from singleton output
            _, video_res_masks_singleton = self._get_orig_video_res_output(
                inference_state, current_out["pred_masks"]
            )

            # Mark frame as consolidated (prevents double consolidation in preflight)
            if "consolidated_frame_inds" in inference_state:
                inference_state["consolidated_frame_inds"][storage_key].add(frame_idx)

            # For return value, use singleton masks
            video_res_masks_to_return = video_res_masks_singleton[
                singleton_obj_idx : singleton_obj_idx + 1
            ]
        else:
            # Standard multiplex output - use obj_idx
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, current_out["pred_masks"]
            )

            current_out["pred_masks_video_res"] = video_res_masks
            current_out["local_obj_id_to_idx"] = deepcopy(
                inference_state["obj_id_to_idx"]
            )

            # Remove from non_cond if this becomes a cond frame
            if (
                is_cond
                and frame_idx
                in inference_state["output_dict"]["non_cond_frame_outputs"]
            ):
                del inference_state["output_dict"]["non_cond_frame_outputs"][frame_idx]
                # Also update consolidated_frame_inds
                if "consolidated_frame_inds" in inference_state:
                    inference_state["consolidated_frame_inds"][
                        "non_cond_frame_outputs"
                    ].discard(frame_idx)

            inference_state["output_dict"][storage_key][frame_idx] = current_out

            # Update consolidated_frame_inds to track this frame
            if "consolidated_frame_inds" in inference_state:
                inference_state["consolidated_frame_inds"][storage_key].add(frame_idx)

            # Store per-object outputs (slice from the full multiplex output)
            obj_temp_output_dict[storage_key][frame_idx] = {
                "pred_masks_video_res": current_out["pred_masks_video_res"][
                    obj_idx : obj_idx + 1
                ],
                "pred_masks": current_out["pred_masks"][obj_idx : obj_idx + 1],
                "object_score_logits": current_out["object_score_logits"][
                    obj_idx : obj_idx + 1
                ],
            }

            obj_output_dict[storage_key][frame_idx] = obj_temp_output_dict[storage_key][
                frame_idx
            ]

            video_res_masks_to_return = video_res_masks[obj_idx : obj_idx + 1]

        low_res_masks = None
        return frame_idx, obj_ids, low_res_masks, video_res_masks_to_return

    @torch.inference_mode()
    def add_new_masks(
        self,
        inference_state,
        frame_idx,
        obj_ids,
        masks,
        # for compatibility with per_obj_inference class, not used here
        add_mask_to_memory=False,
        # for object reconditioning; do not update the multiplex state
        reconditioning=False,
    ):
        """Add new mask to a frame."""
        if isinstance(obj_ids, np.ndarray):
            obj_ids = obj_ids.tolist()
        obj_idxs = [
            self._obj_id_to_idx(inference_state, obj_id, error_if_new=reconditioning)
            for obj_id in obj_ids
        ]
        point_inputs_per_frame = [
            inference_state["point_inputs_per_obj"][obj_idx] for obj_idx in obj_idxs
        ]
        mask_inputs_per_frame = [
            inference_state["mask_inputs_per_obj"][obj_idx] for obj_idx in obj_idxs
        ]

        assert masks.dim() == 3
        num_objects, mask_H, mask_W = masks.shape
        assert num_objects == len(obj_ids)
        masks_inputs_orig = masks[:, None, :, :]  # add channel dimension
        masks_inputs_orig = masks_inputs_orig.float().to(inference_state["device"])

        # resize the mask if it doesn't match the model's input mask size
        if mask_H != self.input_mask_size or mask_W != self.input_mask_size:
            mask_inputs = torch.nn.functional.interpolate(
                masks_inputs_orig,
                size=(self.input_mask_size, self.input_mask_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
        else:
            mask_inputs = masks_inputs_orig

        # also get the mask at the original video resolution (for outputting)
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        if mask_H != video_H or mask_W != video_W:
            mask_inputs_video_res = torch.nn.functional.interpolate(
                masks_inputs_orig,
                size=(video_H, video_W),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for potential downsampling
            )
        else:
            mask_inputs_video_res = masks_inputs_orig
        # convert mask_inputs_video_res to binary (threshold at 0.5 as it is in range 0~1)
        mask_inputs_video_res = mask_inputs_video_res > 0.5

        multiplex_state = inference_state["multiplex_state"]
        is_new_state = multiplex_state is None

        if not reconditioning:
            if is_new_state:
                multiplex_state = self.multiplex_controller.get_state(
                    num_valid_entries=num_objects,
                    device=inference_state["device"],
                    dtype=torch.float32,  # lower precision is also fine
                    random=False,
                    object_ids=obj_ids,
                )
                inference_state["multiplex_state"] = multiplex_state
            else:
                assert self.is_dynamic_model, (
                    "New objects are not allowed after state creation"
                )

        for i in range(num_objects):
            mask_inputs_per_frame[i][frame_idx] = mask_inputs_video_res[i : i + 1]
            point_inputs_per_frame[i].pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dicts = [
            inference_state["output_dict_per_obj"][obj_idx] for obj_idx in obj_idxs
        ]
        obj_temp_output_dicts = [
            inference_state["temp_output_dict_per_obj"][obj_idx] for obj_idx in obj_idxs
        ]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Allow creating a new bucket only when existing buckets cannot fit the new objects
        allow_new_buckets_local = False
        if not is_new_state and not reconditioning and multiplex_state is not None:
            if multiplex_state.available_slots < num_objects:
                allow_new_buckets_local = True

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=inference_state["output_dict"],
            frame_idx=frame_idx,
            batch_size=num_objects,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            add_to_existing_state=not is_new_state and not reconditioning,
            new_obj_idxs=obj_idxs,
            new_obj_ids=obj_ids,
            allow_new_buckets=allow_new_buckets_local,
            reconditioning=reconditioning,
        )
        # We directly use the input mask at video resolution as the output mask for a better
        # video editing experience (so that the masks don't change after each brushing).
        # Here NO_OBJ_SCORE is a large negative value to represent the background and
        # similarly -NO_OBJ_SCORE is a large positive value to represent the foreground.
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, current_out["pred_masks"]
        )
        obj_idxs_t = torch.as_tensor(obj_idxs, device=video_res_masks.device)
        video_res_masks[obj_idxs_t] = torch.where(
            mask_inputs_video_res, -NO_OBJ_SCORE, NO_OBJ_SCORE
        )

        current_out["pred_masks_video_res"] = video_res_masks
        with torch.profiler.record_function("add_new_masks._deepcopy"):
            current_out["local_obj_id_to_idx"] = deepcopy(
                inference_state["obj_id_to_idx"]
            )
        if (
            is_cond
            and frame_idx in inference_state["output_dict"]["non_cond_frame_outputs"]
        ):
            del inference_state["output_dict"]["non_cond_frame_outputs"][frame_idx]
            # Also update consolidated_frame_inds
            if "consolidated_frame_inds" in inference_state:
                inference_state["consolidated_frame_inds"][
                    "non_cond_frame_outputs"
                ].discard(frame_idx)

        inference_state["output_dict"][storage_key][frame_idx] = current_out

        # Update consolidated_frame_inds to track this frame
        if "consolidated_frame_inds" in inference_state:
            inference_state["consolidated_frame_inds"][storage_key].add(frame_idx)

        with torch.profiler.record_function("add_new_masks.obj_loop"):
            # Step 1: Set all new object masks first (batched)
            for i, obj_idx in enumerate(obj_idxs):
                # Add the predicted masks to the output dict
                # NOTE: object ordering matters here but I guess this is the same for the per-object implementation
                obj_temp_output_dicts[i][storage_key][frame_idx] = {
                    "pred_masks_video_res": current_out["pred_masks_video_res"][
                        obj_idx : obj_idx + 1
                    ]
                }
                obj_output_dicts[i][storage_key][frame_idx] = obj_temp_output_dicts[i][
                    storage_key
                ][frame_idx]

            # Step 2: Precompute suppress masks to avoid O(n*m) torch.where calls
            # Combined mask of all new objects (for existing objects)
            combined_new_mask = mask_inputs_video_res.any(
                dim=0, keepdim=True
            )  # (1, 1, H, W)

            # Precompute exclude-self masks for new objects (if there are multiple new objects)
            num_new = len(obj_idxs)
            exclude_self_masks = {}
            if num_new > 1:
                for i in range(num_new):
                    other_indices = torch.cat(
                        [
                            torch.arange(i, device=mask_inputs_video_res.device),
                            torch.arange(
                                i + 1, num_new, device=mask_inputs_video_res.device
                            ),
                        ]
                    )
                    exclude_self_masks[obj_idxs[i]] = mask_inputs_video_res[
                        other_indices
                    ].any(dim=0, keepdim=True)

            # Step 3: Apply suppression to all objects in a single pass
            temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
            obj_idxs_set = set(obj_idxs)

            for obj_idx2, obj_temp_output_dict2 in temp_output_dict_per_obj.items():
                current_out2 = obj_temp_output_dict2[storage_key].get(frame_idx, None)
                if current_out2 is None:
                    continue

                if obj_idx2 not in obj_idxs_set:
                    # Existing object: suppress by all new masks
                    suppress_mask = combined_new_mask
                elif obj_idx2 in exclude_self_masks:
                    # New object: suppress by other new objects' masks
                    suppress_mask = exclude_self_masks[obj_idx2]
                else:
                    # Only one new object - nothing to suppress for itself
                    continue

                current_out2["pred_masks_video_res"] = torch.where(
                    suppress_mask,
                    NO_OBJ_SCORE,
                    current_out2["pred_masks_video_res"],
                )

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        low_res_masks = None  # not needed by the demo

        consolidated_out["local_obj_id_to_idx"] = current_out["local_obj_id_to_idx"]

        return frame_idx, obj_ids, low_res_masks, video_res_masks

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks_for_output:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        # potentially fill holes in the predicted masks
        if self.fill_hole_area > 0:
            video_res_masks = fill_holes_in_mask_scores(
                video_res_masks, self.fill_hole_area
            )
        return any_res_masks, video_res_masks

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        run_mem_encoder,
        consolidate_at_video_res=False,
    ):
        """
        Consolidate the per-object temporary outputs in `temp_output_dict_per_obj` on
        a frame into a single output for all objects, including
        1) fill any missing objects either from `output_dict_per_obj` (if they exist in
           `output_dict_per_obj` for this frame) or leave them as placeholder values
           (if they don't exist in `output_dict_per_obj` for this frame);
        2) if specified, rerun memory encoder after apply non-overlapping constraints
           on the object scores.
        """
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # After singleton merge, objects can be added at indices beyond batch_size
        # We need to find the maximum object index that has temp or regular outputs to size the tensor correctly
        max_obj_idx = batch_size - 1  # Default to batch_size - 1

        # Check both temp and regular output dicts to find max index
        for obj_idx in inference_state["temp_output_dict_per_obj"].keys():
            if obj_idx > max_obj_idx:
                max_obj_idx = obj_idx
        for obj_idx in inference_state["output_dict_per_obj"].keys():
            if obj_idx > max_obj_idx:
                max_obj_idx = obj_idx

        # Size the consolidated tensor to accommodate all object indices (not just count)
        consolidated_batch_size = max(max_obj_idx + 1, 0)  # Ensure non-negative

        # Optionally, we allow consolidating the temporary outputs at the original
        # video resolution (to provide a better editing experience for mask prompts).
        if consolidate_at_video_res:
            assert not run_mem_encoder, "memory encoder cannot run at video resolution"
            consolidated_H = inference_state["video_height"]
            consolidated_W = inference_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = self.low_res_mask_size
            consolidated_mask_key = "pred_masks"

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc"
        # will be added when rerunning the memory encoder after applying non-overlapping
        # constraints to object scores. Its "pred_masks" are prefilled with a large
        # negative value (NO_OBJ_SCORE) to represent missing objects.

        consolidated_out = {
            "conditioning_objects": None,
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            "image_features": None,
            "image_pos_enc": None,
            "obj_ptr": None,
            consolidated_mask_key: torch.full(
                size=(
                    consolidated_batch_size,
                    1,
                    consolidated_H,
                    consolidated_W,
                ),  # Use consolidated_batch_size, not batch_size!
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["storage_device"],
            ),
        }

        all_out = inference_state["output_dict"]["cond_frame_outputs"].get(
            frame_idx, None
        )
        if all_out is None:
            all_out = inference_state["output_dict"]["non_cond_frame_outputs"].get(
                frame_idx, None
            )

        # Handle the case where output_dict is empty (e.g., during demo VG propagation)
        # In this case, we'll reconstruct the consolidated output from per-object outputs
        need_to_reconstruct_from_per_obj = all_out is None

        if need_to_reconstruct_from_per_obj:
            # Initialize fields that will be populated from per-object outputs or later
            # Determine which objects are conditioned by checking if they have point/mask inputs on this frame
            conditioning_objects = set()
            for obj_idx in range(batch_size):
                # Check if this object has point inputs on this frame
                if obj_idx in inference_state["point_inputs_per_obj"]:
                    point_inputs = inference_state["point_inputs_per_obj"][obj_idx]
                    if (
                        frame_idx in point_inputs
                        and point_inputs[frame_idx] is not None
                    ):
                        conditioning_objects.add(obj_idx)
                        continue

                # Check if this object has mask inputs on this frame
                if obj_idx in inference_state["mask_inputs_per_obj"]:
                    mask_inputs = inference_state["mask_inputs_per_obj"][obj_idx]
                    if frame_idx in mask_inputs and mask_inputs[frame_idx] is not None:
                        conditioning_objects.add(obj_idx)

            consolidated_out["conditioning_objects"] = conditioning_objects
            # Shared features will be populated when running memory encoder
            # Note: obj_ptr and object_score_logits will be populated from per-object outputs below
        else:
            # Normal case: populate from existing consolidated output
            consolidated_out["conditioning_objects"] = all_out.get(
                "conditioning_objects", set()
            )
            consolidated_out["obj_ptr"] = all_out["obj_ptr"]
            consolidated_out["object_score_logits"] = all_out["object_score_logits"]
            if self.use_memory_selection:
                consolidated_out["iou_score"] = all_out["iou_score"]
            # These fields might not exist in per-object outputs (e.g., after singleton extraction)
            consolidated_out["maskmem_features"] = all_out.get("maskmem_features")
            consolidated_out["maskmem_pos_enc"] = all_out.get("maskmem_pos_enc")
            consolidated_out["image_features"] = all_out.get("image_features")
            consolidated_out["image_pos_enc"] = all_out.get("image_pos_enc")
            consolidated_out["local_obj_id_to_idx"] = all_out.get(
                "local_obj_id_to_idx", {}
            )
            consolidated_out["obj_ptr"] = all_out["obj_ptr"]
            consolidated_out["object_score_logits"] = all_out["object_score_logits"]
            if self.use_memory_selection:
                consolidated_out["iou_score"] = all_out["iou_score"]
            # These fields might not exist in per-object outputs (e.g., after singleton extraction)
            consolidated_out["maskmem_features"] = all_out.get("maskmem_features")
            consolidated_out["maskmem_pos_enc"] = all_out.get("maskmem_pos_enc")
            consolidated_out["image_features"] = all_out.get("image_features")
            consolidated_out["image_pos_enc"] = all_out.get("image_pos_enc")
            consolidated_out["local_obj_id_to_idx"] = all_out.get(
                "local_obj_id_to_idx", {}
            )
            all_mask = all_out.get("pred_masks_video_res", all_out["pred_masks"])
            # Ensure masks are at the correct consolidated resolution
            # This handles the case where all_out has interactive resolution (288) masks
            # that need to be resized to SAM2's low_res_mask_size (256) for consistency
            if all_mask.shape[-2:] == (consolidated_H, consolidated_W):
                consolidated_out[consolidated_mask_key] = all_mask
            else:
                # Resize first if mask has a different resolution (e.g., 288 from interactive)
                # Determine if we're downsampling or upsampling
                is_downsampling = all_mask.shape[-1] > consolidated_W
                resized_mask = torch.nn.functional.interpolate(
                    all_mask,
                    size=(consolidated_H, consolidated_W),
                    mode="bilinear",
                    align_corners=False,
                    antialias=is_downsampling,  # use antialias for downsampling
                )
                consolidated_out[consolidated_mask_key] = resized_mask

        # Collect per-object outputs (masks and scores) to build consolidated output
        # When reconstructing from per-object outputs, we also need to collect obj_ptr and object_score_logits
        obj_score_logits_list = []
        obj_ptr_list = [] if need_to_reconstruct_from_per_obj else None
        iou_scores_list = (
            []
            if need_to_reconstruct_from_per_obj and self.use_memory_selection
            else None
        )

        # When reconstructing from per-object outputs, initialize the mask tensor
        # with the correct size (consolidated_batch_size, not batch_size)
        if (
            need_to_reconstruct_from_per_obj
            and consolidated_mask_key not in consolidated_out
        ):
            # Initialize with zeros - will be populated from per-object outputs below
            consolidated_out[consolidated_mask_key] = torch.zeros(
                (consolidated_batch_size, 1, consolidated_H, consolidated_W),
                dtype=torch.float32,
                device=inference_state["storage_device"],
            )
            consolidated_out["object_score_logits"] = torch.full(
                (consolidated_batch_size, 1),
                NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["storage_device"],
            )

        for obj_idx in range(
            consolidated_batch_size
        ):  # Use consolidated_batch_size instead of batch_size
            # Check if this object index exists in temp/output dicts (it may not if object was just added)
            if obj_idx not in inference_state["temp_output_dict_per_obj"]:
                continue
            if obj_idx not in inference_state["output_dict_per_obj"]:
                continue
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            # If the object doesn't appear in "temp_output_dict_per_obj" on this frame,
            # we fall back and look up its previous output in "output_dict_per_obj".
            # We look up both "cond_frame_outputs" and "non_cond_frame_outputs" in
            # "output_dict_per_obj" to find a previous output for this object.
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                # object pointers are filled globally above; we don't need empty_mask_ptr
                continue
            # Add the temporary object output mask to consolidated output mask
            # (use "pred_masks_video_res" if it's available)
            obj_mask = out.get("pred_masks_video_res")
            if obj_mask is None:
                obj_mask = out.get("pred_masks")
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]

            # If obj_idx is beyond the consolidated_pred_masks size,
            # we need to expand it (can happen after singleton merge adds object at end)
            if obj_idx >= consolidated_pred_masks.shape[0]:
                pad_size = obj_idx + 1 - consolidated_pred_masks.shape[0]
                consolidated_pred_masks = torch.cat(
                    [
                        consolidated_pred_masks,
                        torch.zeros(
                            (
                                pad_size,
                                1,
                                consolidated_pred_masks.shape[-2],
                                consolidated_pred_masks.shape[-1],
                            ),
                            dtype=consolidated_pred_masks.dtype,
                            device=consolidated_pred_masks.device,
                        ),
                    ],
                    dim=0,
                )
                consolidated_out[consolidated_mask_key] = consolidated_pred_masks
                # Also expand object_score_logits if present
                if "object_score_logits" in consolidated_out:
                    consolidated_scores = consolidated_out["object_score_logits"]
                    consolidated_scores = torch.cat(
                        [
                            consolidated_scores,
                            torch.full(
                                (pad_size, 1),
                                NO_OBJ_SCORE,
                                dtype=consolidated_scores.dtype,
                                device=consolidated_scores.device,
                            ),
                        ],
                        dim=0,
                    )
                    consolidated_out["object_score_logits"] = consolidated_scores

            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                # Ensure dtype match between source and destination before assignment
                if obj_mask.dtype != consolidated_pred_masks.dtype:
                    obj_mask = obj_mask.to(consolidated_pred_masks.dtype)
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask
            else:
                # Resize first if temporary object mask has a different resolution
                is_downsampling = "pred_masks_video_res" in out
                resized_obj_mask = torch.nn.functional.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                    antialias=is_downsampling,  # use antialias for downsampling
                )
                # Ensure dtype match between source and destination before assignment
                if resized_obj_mask.dtype != consolidated_pred_masks.dtype:
                    resized_obj_mask = resized_obj_mask.to(
                        consolidated_pred_masks.dtype
                    )
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask

            # When reconstructing from per-object outputs, also collect scores
            if need_to_reconstruct_from_per_obj:
                if "object_score_logits" in out:
                    obj_score_logits_list.append(out["object_score_logits"])
                if self.use_memory_selection and "iou_score" in out:
                    iou_scores_list.append(out["iou_score"])

        # If we reconstructed from per-object outputs, consolidate the score fields
        if need_to_reconstruct_from_per_obj:
            # Check if we have ANY valid per-object outputs
            # If not, we're trying to consolidate a VG-propagated frame that was never
            # stored in output_dict (only in cached_frame_outputs)
            # In this case, we SKIP memory encoding during preflight and will do it
            # during the first propagation step instead
            if not obj_score_logits_list and run_mem_encoder:
                run_mem_encoder = False  # Skip for now, will encode during propagation

            if obj_score_logits_list:
                consolidated_out["object_score_logits"] = torch.cat(
                    obj_score_logits_list, dim=0
                )
            else:
                # Create placeholder scores - these will be replaced when memory encoder runs
                device = inference_state["device"]
                consolidated_out["object_score_logits"] = torch.zeros(
                    (batch_size, 1),
                    dtype=torch.float32,
                    device=device,
                )

            if self.use_memory_selection:
                if iou_scores_list:
                    consolidated_out["iou_score"] = torch.cat(iou_scores_list, dim=0)
                else:
                    consolidated_out["iou_score"] = None

            # obj_ptr will be populated by memory encoder, set to None for now
            consolidated_out["obj_ptr"] = None

        # Optionally, apply non-overlapping constraints on the consolidated scores
        # and rerun the memory encoder
        if run_mem_encoder:
            device = inference_state["device"]
            high_res_masks = torch.nn.functional.interpolate(
                consolidated_out["pred_masks"].to(device, non_blocking=True),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            high_res_masks = self._apply_non_overlapping_constraints(high_res_masks)
            maskmem_features, maskmem_pos_enc, image_features, image_pos_enc = (
                self._run_memory_encoder(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    high_res_masks=high_res_masks,
                    object_score_logits=consolidated_out["object_score_logits"],
                    is_mask_from_pts=True,  # these frames are what the user interacted with
                    conditioning_objects=consolidated_out[
                        "conditioning_objects"
                    ],  # Pass conditioning_objects
                )
            )
            consolidated_out["maskmem_features"] = maskmem_features
            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc
            consolidated_out["image_features"] = image_features
            consolidated_out["image_pos_enc"] = image_pos_enc

        return consolidated_out

    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state, run_mem_encoder=True):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        inference_state["tracking_has_started"] = True
        batch_size = self._get_obj_num(inference_state)

        # Consolidate per-object temporary outputs in "temp_output_dict_per_obj" and
        # add them into "output_dict".
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        # "consolidated_frame_inds" contains indices of those frames where consolidated
        # temporary outputs have been added (either in this call or any previous calls
        # to `propagate_in_video_preflight`).
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        for is_cond in [False, True]:
            # Separately consolidate conditioning and non-conditioning temp outptus
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            # Find all the frames that contain temporary outputs for any objects
            # (these should be the frames that have just received clicks for mask inputs
            # via `add_new_points` or `add_new_mask`)
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            # consolidate the temprary output across all objects on this frame
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    run_mem_encoder=run_mem_encoder,
                )
                # merge them into "output_dict" and also create per-object slices
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(
                    inference_state, frame_idx, consolidated_out, storage_key
                )
                clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
                    self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                )
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)

            # clear temporary outputs in `temp_output_dict_per_obj`
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # edge case: if an output is added to "cond_frame_outputs", we remove any prior
        # output on the same frame in "non_cond_frame_outputs"
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        # Make sure that the frame indices in "consolidated_frame_inds" are exactly those frames
        # with either points or mask inputs (which should be true under a correct demo workflow).
        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"]
            | consolidated_frame_inds["non_cond_frame_outputs"]
        )

        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds
        # Record the first interacted frame index (for tracking start)
        if inference_state["first_ann_frame_idx"] is None:
            inference_state["first_ann_frame_idx"] = min(
                input_frames_inds, default=None
            )
        # In case `first_ann_frame_idx` is not in the conditioning frames (e.g. because
        # we cleared the input points on that frame), pick the first conditioning frame
        if (
            inference_state["first_ann_frame_idx"]
            not in output_dict["cond_frame_outputs"]
        ):
            inference_state["first_ann_frame_idx"] = min(
                output_dict["cond_frame_outputs"], default=None
            )

    def _get_processing_order(
        self, inference_state, start_frame_idx, max_frame_num_to_track, reverse
    ):
        num_frames = inference_state["num_frames"]
        # set start index, end index, and processing order
        if self.always_start_from_first_ann_frame:
            # in this case, we always start tracking from the frame where we receive
            # the initial annotation and ignore the provided start_frame_idx
            start_frame_idx = inference_state["first_ann_frame_idx"]
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(inference_state["output_dict"]["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                # TODO: Jie - this is the edge case that we start from frame 0 and track in reverse order;
                # and in the case we track a single frame for dense tracking, it should still run 1 frame (idx=0).
                # Not sure if this has any side effect.
                # processing_order = []  # skip reverse tracking if starting from frame 0 <-- original behaviour
                processing_order = [0]
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        return processing_order

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx,
        max_frame_num_to_track,
        reverse,
        tqdm_disable=False,
        obj_ids=None,
        run_mem_encoder=True,
    ):
        """Propagate the input points across frames to track in the entire video."""
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        if obj_ids is not None:
            raise NotImplementedError(
                "Per-object tracking yet for batched inference if not implemented."
            )
        obj_ids = inference_state["obj_ids"]
        batch_size = self._get_obj_num(inference_state)
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )
        assert clear_non_cond_mem is False, "Not implemented"

        processing_order = self._get_processing_order(
            inference_state,
            start_frame_idx,
            max_frame_num_to_track,
            reverse,
        )

        for frame_idx in tqdm(
            processing_order, desc="propagate in video", disable=tqdm_disable
        ):
            # We skip those frames already in consolidated outputs (these are frames
            # that received input clicks or mask). Note that we cannot directly run
            # batched forward on them via `_run_single_frame_inference` because the
            # number of clicks on each object might be different.
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            else:
                storage_key = "non_cond_frame_outputs"
                with torch.profiler.record_function(
                    "VideoTrackingMultiplexDemo._run_single_frame_inference"
                ):
                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=output_dict,
                        frame_idx=frame_idx,
                        batch_size=batch_size,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=run_mem_encoder,
                    )
                current_out["local_obj_id_to_idx"] = deepcopy(
                    inference_state["obj_id_to_idx"]
                )
                output_dict[storage_key][frame_idx] = current_out
            # Create slices of per-object outputs for subsequent interaction with each
            # individual object after tracking.
            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            low_res_masks, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )
            yield frame_idx, obj_ids, low_res_masks, video_res_masks

    def _add_output_per_object(
        self, inference_state, frame_idx, current_out, storage_key
    ):
        """
        Split a multi-object output into per-object output slices and add them into
        `output_dict_per_obj`. The resulting slices share the same tensor storage.
        """
        # Note for the multiplex model: we don't store the maskmem features
        # because we don't use the memory during interaction

        output_dict_per_obj = inference_state["output_dict_per_obj"]
        for obj_idx, obj_output_dict in output_dict_per_obj.items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "pred_masks": current_out["pred_masks"][obj_slice],
                "object_score_logits": current_out["object_score_logits"][obj_slice],
            }
            if self.use_memory_selection:
                obj_out["iou_score"] = current_out["iou_score"][obj_slice]
            obj_output_dict[storage_key][frame_idx] = obj_out

    @torch.inference_mode()
    def clear_all_points_in_frame(
        self,
        inference_state,
        frame_idx,
        obj_id,
        need_output=True,
        preserve_user_refined: bool = False,
    ):
        """Remove all input points or mask in a specific frame for a given object."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)

        # Clear the conditioning information on the given frame
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        # Clear user refinement tracking for this frame and object unless preserving it
        if (
            not preserve_user_refined
            and "user_refined_frames_per_obj" in inference_state
        ):
            user_refined_map = inference_state["user_refined_frames_per_obj"]
            if obj_id in user_refined_map:
                user_refined_map[obj_id].discard(frame_idx)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        # Check and see if there are still any inputs left on this frame
        batch_size = self._get_obj_num(inference_state)
        frame_has_input = False
        for obj_idx2 in range(batch_size):
            # Skip if this object doesn't exist in the input dictionaries
            if obj_idx2 not in inference_state["point_inputs_per_obj"]:
                continue
            if obj_idx2 not in inference_state["mask_inputs_per_obj"]:
                continue
            if frame_idx in inference_state["point_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break
            if frame_idx in inference_state["mask_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break

        # If this frame has no remaining inputs for any objects, we further clear its
        # conditioning frame status
        if not frame_has_input:
            output_dict = inference_state["output_dict"]
            consolidated_frame_inds = inference_state["consolidated_frame_inds"]
            consolidated_frame_inds["cond_frame_outputs"].discard(frame_idx)
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
            # Remove the frame's conditioning output (possibly downgrading it to non-conditioning)
            out = output_dict["cond_frame_outputs"].pop(frame_idx, None)
            if out is not None:
                # The frame is not a conditioning frame anymore since it's not receiving inputs,
                # so we "downgrade" its output (if exists) to a non-conditioning frame output.
                output_dict["non_cond_frame_outputs"][frame_idx] = out
                inference_state["frames_already_tracked"].pop(frame_idx, None)
            # Similarly, do it for the sliced output on each object.
            for obj_idx2 in range(batch_size):
                # Skip if this object doesn't exist in the output dictionary
                if obj_idx2 not in inference_state["output_dict_per_obj"]:
                    continue
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx2]
                obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
                if obj_out is not None:
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out

            # If all the conditioning frames have been removed, we also clear the tracking outputs
            if len(output_dict["cond_frame_outputs"]) == 0:
                self._reset_tracking_results(inference_state)

        if not need_output:
            return
        # Finally, output updated masks per object (after removing the inputs above)
        obj_ids = inference_state["obj_ids"]
        is_cond = any(
            frame_idx in obj_temp_output_dict["cond_frame_outputs"]
            for obj_temp_output_dict in temp_output_dict_per_obj.values()
        )
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        low_res_masks = None  # not needed by the demo
        return frame_idx, obj_ids, low_res_masks, video_res_masks

    @torch.inference_mode()
    def clear_all_points_in_video(self, inference_state):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results(inference_state)
        # Remove all object ids
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()
        inference_state["multiplex_state"] = None

    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results across the videos."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()
        inference_state["first_ann_frame_idx"] = None

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            # Cache miss -- we will run inference on a single image
            image = inference_state["images"][frame_idx].cuda().float().unsqueeze(0)
            # TODO: We should optimize this because we don't always need all three outs
            backbone_out = self.forward_image(
                NestedTensor(tensors=image, mask=None),
                need_sam3_out=True,
                need_interactive_out=True,
                need_propagation_out=True,
            )
            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        features = self._prepare_backbone_features(backbone_out)
        return image, features

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
        add_to_existing_state: bool = False,
        new_obj_idxs: Optional[list[int]] = None,
        new_obj_ids: Optional[list[int]] = None,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
        reconditioning: bool = False,
        objects_to_interact: Optional[list[int]] = None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        # Retrieve correct image features
        with torch.profiler.record_function(
            "VideoTrackingMultiplexDemo._get_image_feature"
        ):
            image, backbone_features = self._get_image_feature(
                inference_state, frame_idx, batch_size
            )

        if add_to_existing_state or reconditioning:
            assert new_obj_idxs is not None
            assert new_obj_ids is not None

        backbone_features_interactive = backbone_features["interactive"]
        backbone_features_propagation = backbone_features["sam2_backbone_out"]

        if add_to_existing_state or reconditioning:
            with torch.profiler.record_function(
                "VideoTrackingMultiplexDemo.add_new_masks_to_existing_state"
            ):
                # Get existing output from current frame to modify in-place
                # Try both storage keys since the output could be in either location
                existing_out = output_dict["cond_frame_outputs"].get(frame_idx)
                if existing_out is None:
                    existing_out = output_dict["non_cond_frame_outputs"].get(frame_idx)
                if existing_out is None:
                    raise RuntimeError(
                        f"No existing output found for frame {frame_idx} in either storage"
                    )

                # Prepare interactive features
                interactive_pix_feat = self._get_interactive_pix_mem(
                    backbone_features_interactive["vision_feats"],
                    backbone_features_interactive["feat_sizes"],
                )

                # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
                interactive_high_res_features = [
                    x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                    for x, s in zip(
                        backbone_features_interactive["vision_feats"][:-1],
                        backbone_features_interactive["feat_sizes"][:-1],
                    )
                ]

                # Prepare propagation features for memory encoding
                propagation_vision_feats = (
                    backbone_features_propagation["vision_feats"]
                    if run_mem_encoder
                    else None
                )
                propagation_feat_sizes = (
                    backbone_features_propagation["feat_sizes"]
                    if run_mem_encoder
                    else None
                )

                # Add new masks to existing state
                if reconditioning:
                    self.recondition_masks_in_existing_state(
                        interactive_pix_feat=interactive_pix_feat,
                        interactive_high_res_features=interactive_high_res_features,
                        propagation_vision_feats=propagation_vision_feats,
                        propagation_feat_sizes=propagation_feat_sizes,
                        new_masks=mask_inputs,
                        obj_idxs_in_mask=new_obj_idxs,
                        obj_ids_in_mask=new_obj_ids,
                        prev_output=existing_out,
                        multiplex_state=inference_state["multiplex_state"],
                        add_mask_to_memory=run_mem_encoder,
                    )
                else:
                    # If we are adding to existing state using points (mask_inputs is None),
                    # first convert points -> masks via the interactivity head.
                    new_masks_from_points = None
                    if mask_inputs is None and point_inputs is not None:
                        with torch.profiler.record_function(
                            "VideoTrackingMultiplexDemo.points_to_masks"
                        ):
                            multimask_output = self._use_multimask(
                                is_init_cond_frame, point_inputs=point_inputs
                            )
                            interaction_out = self._forward_sam_heads(
                                backbone_features=interactive_pix_feat,
                                point_inputs=point_inputs,
                                mask_inputs=None,
                                interactive_high_res_features=interactive_high_res_features,
                                multimask_output=multimask_output,
                                objects_to_interact=new_obj_idxs,
                                multiplex_state=inference_state["multiplex_state"],
                            )
                            new_masks_from_points = interaction_out["low_res_masks"]

                    self.add_new_masks_to_existing_state(
                        interactive_pix_feat=interactive_pix_feat,
                        interactive_high_res_features=interactive_high_res_features,
                        propagation_vision_feats=propagation_vision_feats,
                        propagation_feat_sizes=propagation_feat_sizes,
                        new_masks=(
                            mask_inputs
                            if mask_inputs is not None
                            else new_masks_from_points
                        ),
                        obj_idxs_in_mask=new_obj_idxs,
                        obj_ids_in_mask=new_obj_ids,
                        prev_output=existing_out,
                        multiplex_state=inference_state["multiplex_state"],
                        add_mask_to_memory=run_mem_encoder,
                        are_masks_from_pts=(mask_inputs is None),
                        allow_new_buckets=allow_new_buckets,
                        prefer_new_buckets=prefer_new_buckets,
                    )

                # Return the modified existing output
                current_out = existing_out
        else:
            # point and mask should not appear as input simultaneously on the same frame
            assert point_inputs is None or mask_inputs is None
            with torch.profiler.record_function(
                "VideoTrackingMultiplexDemo.track_step"
            ):
                current_out = self.track_step(
                    frame_idx=frame_idx,
                    is_init_cond_frame=is_init_cond_frame,
                    backbone_features_interactive=backbone_features_interactive,
                    backbone_features_propagation=backbone_features_propagation,
                    image=image,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    gt_masks=None,
                    frames_to_add_correction_pt=[],
                    output_dict=output_dict,
                    num_frames=inference_state["num_frames"],
                    track_in_reverse=reverse,
                    run_mem_encoder=run_mem_encoder,
                    prev_sam_mask_logits=prev_sam_mask_logits,
                    multiplex_state=inference_state["multiplex_state"],
                    objects_to_interact=objects_to_interact,
                )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        if current_out.get("maskmem_features") is not None:
            maskmem_features = current_out["maskmem_features"]
            maskmem_features = maskmem_features.to(
                device=storage_device, dtype=torch.bfloat16, non_blocking=True
            )
        else:
            maskmem_features = None

        if current_out.get("image_features") is not None:
            assert "image_pos_enc" in current_out
            image_features = current_out["image_features"].to(
                storage_device, non_blocking=True
            )
            image_pos_enc = current_out["image_pos_enc"].to(
                storage_device, non_blocking=True
            )
        else:
            image_features = image_pos_enc = None

        pred_masks_gpu = current_out["pred_masks"]
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        with torch.profiler.record_function(
            "VideoTrackingMultiplexDemo.maskmem_pos_enc"
        ):
            maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        conditioning_objects = current_out["conditioning_objects"]
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "image_features": image_features,
            "image_pos_enc": image_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
            "conditioning_objects": conditioning_objects,
        }
        if self.use_memory_selection:
            with torch.profiler.record_function(
                "VideoTrackingMultiplexDemo.use_memory_selection"
            ):
                compact_current_out["iou_score"] = current_out["iou_score"]
                compact_current_out["eff_iou_score"] = self.cal_mem_score(
                    object_score_logits, current_out["iou_score"]
                )
        return compact_current_out, pred_masks_gpu

    def _run_memory_encoder(
        self,
        inference_state,
        frame_idx,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
        conditioning_objects=None,  # Accept as parameter
    ):
        """
        Run the memory encoder on `high_res_masks`. This is usually after applying
        non-overlapping constraints to object scores. Since their scores changed, their
        memory also need to be computed again with the memory encoder.
        """
        # Retrieve correct image features
        image, backbone_features = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        backbone_features_propagation = backbone_features["sam2_backbone_out"]
        propagation_vision_feats = backbone_features_propagation["vision_feats"]
        propagation_vision_pos_embeds = backbone_features_propagation[
            "vision_pos_embeds"
        ]
        propagation_feat_sizes = backbone_features_propagation["feat_sizes"]

        # If conditioning_objects is not provided, look it up from output_dict
        if conditioning_objects is None:
            output_dict = inference_state["output_dict"]
            for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                storage = output_dict[storage_key]
                if frame_idx not in storage:
                    continue
                conditioning_objects = storage[frame_idx]["conditioning_objects"]
                break
            else:
                raise ValueError(f"conditioning objects not found at {frame_idx=}")

        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            image=image,
            current_vision_feats=propagation_vision_feats,
            feat_sizes=propagation_feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
            conditioning_objects=conditioning_objects,
            multiplex_state=inference_state["multiplex_state"],
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = maskmem_features.to(torch.bfloat16)
        maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )

        image_features = propagation_vision_feats[-1]
        image_features = image_features.to(storage_device, non_blocking=True)
        image_pos_enc = propagation_vision_pos_embeds[-1]
        image_pos_enc = image_pos_enc.to(storage_device, non_blocking=True)
        return maskmem_features, maskmem_pos_enc, image_features, image_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """
        `maskmem_pos_enc` is the same across frames and objects, so we cache it as
        a constant in the inference session to reduce session storage size.
        """
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc" should be either a list of tensors or None
        out_maskmem_pos_enc = current_out.get("maskmem_pos_enc")
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # only take the slice for one object, since it's same across objects
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # expand the cached maskmem_pos_enc to the actual batch size
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc

    @torch.inference_mode()
    def remove_object(
        self,
        inference_state,
        obj_id: int,
        strict=False,
        need_output=True,
        clear_user_refined_map: bool = True,
    ):
        """
        Remove a single object from the tracking state.

        This is a convenience wrapper around remove_objects() for removing a single object.

        Args:
            inference_state: Current inference state
            obj_id: Object ID to remove
            strict: If True, raise error if object doesn't exist
            need_output: Whether to return updated frames

        Returns:
            Tuple of (remaining_obj_ids, updated_frames)
        """
        return self.remove_objects(
            inference_state,
            obj_ids=[obj_id],
            strict=strict,
            need_output=need_output,
            clear_user_refined_map=clear_user_refined_map,
        )

    @torch.inference_mode()
    def remove_objects(
        self,
        inference_state,
        obj_ids: Iterable[int],
        strict=False,
        need_output=True,
        clear_user_refined_map: bool = True,
    ):
        """
        Remove a list of object ids from the tracking state. If strict is True, we check whether
        the object ids actually exist and raise an error if any of them don't exist.
        """
        obj_ids = list(obj_ids)
        old_obj_idxs_to_rm = [
            inference_state["obj_id_to_idx"].get(obj_id, None) for obj_id in obj_ids
        ]
        updated_frames = []
        actually_used_obj_ids = []
        removing_any = False
        for old_obj_idx_to_rm, obj_id in zip(old_obj_idxs_to_rm, obj_ids, strict=True):
            if old_obj_idx_to_rm is None:
                if strict:
                    raise ValueError(
                        f"Object id {obj_id} does not exist in the tracking state."
                    )
            else:
                actually_used_obj_ids.append(obj_id)
                removing_any = True
        if not removing_any:
            return inference_state["obj_ids"], updated_frames

        # ignore any object IDs that don't exist
        old_obj_idxs_to_rm = [x for x in old_obj_idxs_to_rm if x is not None]
        obj_ids = actually_used_obj_ids
        removed_obj_ids = list(obj_ids)

        # There are still remaining objects after removing this object id. In this case,
        # we need to delete the object storage from inference state tensors.
        # Step 0: clear the input on those frames where this object id has point or mask input
        # (note that this step is required as it might downgrade conditioning frames to
        # non-conditioning ones)
        if clear_user_refined_map and "user_refined_frames_per_obj" in inference_state:
            user_refined_map = inference_state["user_refined_frames_per_obj"]
            for removed_obj_id in removed_obj_ids:
                if removed_obj_id in user_refined_map:
                    user_refined_map.pop(removed_obj_id, None)

        all_obj_input_frames_inds = set()
        for old_obj_idx_to_rm, obj_id in zip(old_obj_idxs_to_rm, obj_ids, strict=True):
            obj_input_frames_inds = set()
            obj_input_frames_inds.update(
                inference_state["point_inputs_per_obj"][old_obj_idx_to_rm]
            )
            obj_input_frames_inds.update(
                inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm]
            )
            for frame_idx in obj_input_frames_inds:
                self.clear_all_points_in_frame(
                    inference_state,
                    frame_idx,
                    obj_id,
                    need_output=False,
                    preserve_user_refined=not clear_user_refined_map,
                )
            all_obj_input_frames_inds.update(obj_input_frames_inds)

        # Step 1: Update the object id mapping (note that it must be done after Step 0,
        # since Step 0 still requires the old object id mappings in inference_state)
        old_obj_ids = inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        for old_obj_idx_to_rm in old_obj_idxs_to_rm:
            remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        # build new mappings
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        if len(new_obj_ids) == 0:
            return new_obj_ids, updated_frames

        # Step 2: For per-object tensor storage, we shift their obj_idx in the dict keys.
        # (note that "consolidated_frame_inds" doesn't need to be updated in this step as
        # it's already handled in Step 0)
        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])

        multiplex_state: MultiplexState = inference_state["multiplex_state"]
        # strict is set to True because we have done the filtering above
        buckets_to_keep = multiplex_state.remove_objects(
            old_obj_idxs_to_rm, strict=True
        )
        obj_ids = set(obj_ids)

        # Step 3: For packed tensor storage, we index the remaining ids and rebuild the per-bucket/per-object slices.
        def _slice_state(output_dict, storage_key):
            for frame_idx, out in output_dict[storage_key].items():
                out["maskmem_features"] = out["maskmem_features"][buckets_to_keep]
                out["maskmem_pos_enc"] = [
                    x[buckets_to_keep] for x in out["maskmem_pos_enc"]
                ]
                # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
                out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(inference_state, out)
                out["obj_ptr"] = out["obj_ptr"][buckets_to_keep]

                # Note that pred_maks and score_logits are stored in a per-object manner
                # When we add new objects, obj_id_to_idx mapping could be different
                # locally (at this past frame) versus globally (at the current frame),
                # so we need to use a local copy of this mapping
                local_obj_id_to_idx = out["local_obj_id_to_idx"]

                # Find which local indices correspond to the remaining old object indices
                local_remain_old_obj_inds = [
                    obj_idx
                    for obj_id, obj_idx in local_obj_id_to_idx.items()
                    if obj_id not in obj_ids
                ]

                # Guard against stale indices by intersecting with available rows
                max_pred = out["pred_masks"].shape[0]
                max_scores = out["object_score_logits"].shape[0]
                keep_indices = [
                    idx
                    for idx in local_remain_old_obj_inds
                    if 0 <= idx < max_pred and 0 <= idx < max_scores
                ]
                out["pred_masks"] = out["pred_masks"][keep_indices]
                out["object_score_logits"] = out["object_score_logits"][keep_indices]
                if self.use_memory_selection:
                    out["iou_score"] = out["iou_score"][keep_indices]
                    out["eff_iou_score"] = self.cal_mem_score(
                        out["object_score_logits"], out["iou_score"]
                    )  # recalculate the memory frame score
                sliced_conditioning_objects = set()

                # Update local_obj_id_to_idx to reflect the new indices after removal
                new_local_obj_id_to_idx = {}
                old_to_new = {
                    old_idx: new_i for new_i, old_idx in enumerate(keep_indices)
                }
                for obj_id, old_idx in local_obj_id_to_idx.items():
                    if obj_id not in obj_ids:  # Keep objects not being removed
                        # Find the new index for this object if it was kept
                        if old_idx in old_to_new:
                            new_idx = old_to_new[old_idx]
                            new_local_obj_id_to_idx[obj_id] = new_idx
                            if old_idx in out["conditioning_objects"]:
                                sliced_conditioning_objects.add(new_idx)

                out["local_obj_id_to_idx"] = new_local_obj_id_to_idx
                out["conditioning_objects"] = sliced_conditioning_objects

                # also update the per-object slices
                self._add_output_per_object(
                    inference_state, frame_idx, out, storage_key
                )

        _slice_state(inference_state["output_dict"], "cond_frame_outputs")
        _slice_state(inference_state["output_dict"], "non_cond_frame_outputs")

        # Step 4: Further collect the outputs on those frames in `obj_input_frames_inds`, which
        # could show an updated mask for objects previously occluded by the object being removed
        if need_output:
            temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
            for frame_idx in all_obj_input_frames_inds:
                is_cond = any(
                    frame_idx in obj_temp_output_dict["cond_frame_outputs"]
                    for obj_temp_output_dict in temp_output_dict_per_obj.values()
                )
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    run_mem_encoder=False,
                    consolidate_at_video_res=True,
                )
                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state, consolidated_out["pred_masks_video_res"]
                )
                updated_frames.append((frame_idx, video_res_masks))

        return inference_state["obj_ids"], updated_frames

    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        """
        Remove the non-conditioning memory around the input frame. When users provide
        correction clicks, the surrounding frames' non-conditioning memories can still
        contain outdated object appearance information and could confuse the model.

        This function clears those non-conditioning memories surrounding the interacted
        frame to avoid giving the model both old and new information about the object.
        """
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        output_dict = inference_state["output_dict"]
        non_cond_frame_outputs = output_dict["non_cond_frame_outputs"]
        for t in range(frame_idx_begin, frame_idx_end + 1):
            non_cond_frame_outputs.pop(t, None)
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def warm_up_compilation(
        self, offload_video_to_cpu=False, offload_state_to_cpu=False
    ):
        """
        Warm up the model by running a dummy inference to compile the model. This is
        useful to avoid the compilation overhead in the first inference call.
        """
        if not self.compile_all_components:
            return

        raise NotImplementedError(
            "Please use `VideoTrackingMultiplexDemoPerBucketInference` instead for full model compilation."
        )


class Sam3VideoTrackingMultiplexDemo(VideoTrackingMultiplexDemo):
    @torch.inference_mode()
    def init_state(
        self,
        video_height,
        video_width,
        num_frames,
        cached_features=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        """Initialize a inference state."""
        # Make sure that sigmoid is used on mask logits (should be True for all our recent models).
        # Since we rely on large negative values as scores for missing objects, the raw logits
        # cannot be consumed directly and must be converted into 0~1 range via sigmoid first.
        if not self.apply_sigmoid_to_mask_logits_for_mem_enc:
            raise NotImplementedError(
                "Multi-object tracking requires sigmoid in memory encoder for non-overlapping constraints."
            )
        inference_state = {}
        # inference_state["images"] = images
        inference_state["num_frames"] = num_frames
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = torch.device("cuda")
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = torch.device("cuda")
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = (
            {} if cached_features is None else cached_features
        )
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # The index of the frame that received the first annotation
        inference_state["first_ann_frame_idx"] = None
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        inference_state["multiplex_state"] = None
        # Warm up the whole model and cache the image feature on frame 0
        # by making a dummy click on the first frame (and then cleaning it up)
        # self.add_new_points(
        #     inference_state=inference_state,
        #     frame_idx=0,
        #     obj_id=1,
        #     points=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        #     labels=torch.tensor([1], dtype=torch.int32),
        #     clear_old_points=True,
        #     rel_coordinates=True,
        # )
        self.clear_all_points_in_video(inference_state)
        return inference_state

    def _suppress_shrinked_masks(
        self, pred_masks, new_pred_masks, shrink_threshold=0.3
    ):
        area_before = (pred_masks > 0).sum(dim=(-1, -2))
        area_after = (new_pred_masks > 0).sum(dim=(-1, -2))
        area_before = torch.clamp(area_before, min=1.0)
        area_ratio = area_after / area_before
        keep = area_ratio >= shrink_threshold
        keep_mask = keep[..., None, None].expand_as(pred_masks)
        pred_masks_after = torch.where(
            keep_mask, pred_masks, torch.clamp(pred_masks, max=-10.0)
        )
        return pred_masks_after

    @staticmethod
    def _suppress_object_pw_area_shrinkage(pred_masks):
        """
        This function suppresses masks that shrink in area after applying pixelwise non-overlapping constriants.
        Note that the final output can still be overlapping.
        """
        # Apply pixel-wise non-overlapping constraint based on mask scores
        # pixel_level_non_overlapping_masks = super()._apply_non_overlapping_constraints(
        #     pred_masks
        # )

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
        pixel_level_non_overlapping_masks = torch.where(
            keep, pred_masks, torch.clamp(pred_masks, max=-10.0)
        )

        # Fully suppress masks with high shrinkage (probably noisy) based on the pixel wise non-overlapping constraints
        # NOTE: The output of this function can be a no op if none of the masks shrinked by a large factor.
        # pred_masks = self._suppress_shrinked_masks(
        #     pred_masks, pixel_level_non_overlapping_masks
        # )

        shrink_threshold = 0.3
        area_before = (pred_masks > 0).sum(dim=(-1, -2))
        area_after = (pixel_level_non_overlapping_masks > 0).sum(dim=(-1, -2))
        area_before = torch.clamp(area_before, min=1.0)
        area_ratio = area_after / area_before
        keep = area_ratio >= shrink_threshold
        keep_mask = keep[..., None, None].expand_as(pred_masks)
        pred_masks_after = torch.where(
            keep_mask, pred_masks, torch.clamp(pred_masks, max=-10.0)
        )

        return pred_masks_after

    def _apply_object_wise_non_overlapping_constraints(
        self, pred_masks, obj_scores, background_value=-10.0
    ):
        """
        Applies non-overlapping constraints object wise (i.e. only one object can claim the overlapping region)
        """
        # TODO: Try suppression based on IoM here as well.
        # Replace pixel scores with object scores
        pred_masks_single_score = torch.where(
            pred_masks > 0, obj_scores[..., None, None], background_value
        )
        # Apply pixel-wise non-overlapping constraint based on mask scores
        pixel_level_non_overlapping_masks = super()._apply_non_overlapping_constraints(
            pred_masks_single_score
        )
        # Replace object scores with pixel scores. Note, that now only one object can claim the overlapping region
        pred_masks = torch.where(
            pixel_level_non_overlapping_masks > 0,
            pred_masks,
            torch.clamp(pred_masks, max=background_value),
        )
        return pred_masks

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx,
        max_frame_num_to_track,
        reverse,
        tqdm_disable=False,
        obj_ids=None,
        run_mem_encoder=True,
    ):
        """Propagate the input points across frames to track in the entire video."""
        # NOTE: This is a copy from the parent class, except that we return object scores as well.
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        if obj_ids is not None:
            raise NotImplementedError(
                "Per-object tracking yet for batched inference if not implemented."
            )
        obj_ids = inference_state["obj_ids"]
        batch_size = self._get_obj_num(inference_state)
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )

        processing_order = self._get_processing_order(
            inference_state,
            start_frame_idx,
            max_frame_num_to_track,
            reverse,
        )

        for frame_idx in tqdm(
            processing_order, desc="propagate in video", disable=tqdm_disable
        ):
            # We skip those frames already in consolidated outputs (these are frames
            # that received input clicks or mask). Note that we cannot directly run
            # batched forward on them via `_run_single_frame_inference` because the
            # number of clicks on each object might be different.
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                obj_scores = current_out["object_score_logits"]
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                obj_scores = current_out["object_score_logits"]
            else:
                storage_key = "non_cond_frame_outputs"
                with torch.profiler.record_function(
                    "VideoTrackingMultiplexDemo._run_single_frame_inference"
                ):
                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=output_dict,
                        frame_idx=frame_idx,
                        batch_size=batch_size,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=run_mem_encoder,
                    )
                    obj_scores = current_out["object_score_logits"]
                    current_out["local_obj_id_to_idx"] = deepcopy(
                        inference_state["obj_id_to_idx"]
                    )
                output_dict[storage_key][frame_idx] = current_out

            # Create slices of per-object outputs for subsequent interaction with each
            # individual object after tracking.
            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            low_res_masks, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )
            yield frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores
