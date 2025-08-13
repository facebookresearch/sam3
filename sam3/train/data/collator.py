from dataclasses import dataclass, field as field_ptr_behaviour, fields, is_dataclass
from enum import Enum
from typing import Any, get_args, get_origin, List, Mapping, Optional, Sequence, Union

import torch
from sam3.model.data_misc import (
    BatchedDatapoint,
    BatchedFindTarget,
    BatchedInferenceMetadata,
    BatchedPointer,
    FindStage,
    GetStage,
    PointerExtractBehaviour,
)

from sam3.model.model_misc import NestedTensor

from .modulated_detection_api_v2 import Datapoint, QueryType


class PtrType(Enum):
    """Enum representing the various possible pointer types"""

    # - 0=a "x" pointer for a "find/get" query
    FindGetX = 0
    # - 1=a "x" pointer for a "find again close" query
    FindAgainCloseX = 1
    # - 2=a "x" pointer for a "find again far" query
    FindAgainFarX = 2
    # - 3=a "y" pointer for a "find/get" query
    FindGetY = 3
    # - 4=a "x" pointer to memory for TA
    FindAgainMemory = 4


MyTensor = Union[torch.Tensor, List[Any]]


def convert_my_tensors(obj):
    def is_optional_field(field) -> bool:
        return get_origin(field) is Union and type(None) in get_args(field)

    for field in fields(obj):
        if is_dataclass(getattr(obj, field.name)):
            convert_my_tensors(getattr(obj, field.name))
            continue

        field_type = field.type
        if is_optional_field(field.type):
            field_type = Union[get_args(field.type)[:-1]]  # Get the Optional field type

        if field_type != MyTensor or getattr(obj, field.name) is None:
            continue

        elif len(getattr(obj, field.name)) and isinstance(
            getattr(obj, field.name)[0], torch.Tensor
        ):
            stack_dim = 0
            if field.name in [
                "input_boxes_before_embed",
                "input_boxes",
                "input_boxes_label",
            ]:
                stack_dim = 1
            setattr(
                obj,
                field.name,
                torch.stack(getattr(obj, field.name), dim=stack_dim).to(
                    getattr(obj, field.name + "__type")
                ),
            )
        else:
            setattr(
                obj,
                field.name,
                torch.as_tensor(
                    getattr(obj, field.name), dtype=getattr(obj, field.name + "__type")
                ),
            )
    return obj


def packed_to_padded_naive(boxes_packed, num_boxes, fill_value=0):
    """
    Convert a packed tensor of bounding boxes to a padded tensor of bounding
    boxes. Naive implementation using a loop.

    Inputs:
    - boxes_packed: Tensor of shape (N_1 + ... + N_B, 4)
    - num_boxes: Tensor of shape (B,) where num_boxes[i] = N_i

    Returns:
    - boxes_padded: Tensor of shape (B, N_max, 4) where N_max = max_i N_i
    """
    B = num_boxes.shape[0]
    Ns = num_boxes.tolist()

    boxes_padded = boxes_packed.new_zeros(B, max(Ns), *boxes_packed.shape[1:])
    if fill_value != 0:
        boxes_padded[...] = fill_value
    prev_idx = 0
    for i in range(B):
        next_idx = prev_idx + Ns[i]
        boxes_padded[i, : Ns[i]] = boxes_packed[prev_idx:next_idx]
        prev_idx = next_idx
    return boxes_padded


def pad_tensor_list_to_longest(
    tensors: List[torch.Tensor], dim=0, pad_val=0
) -> List[torch.Tensor]:
    # Edits the list in-place
    if not tensors:
        return tensors
    pad_len = max(t.shape[dim] for t in tensors)
    for i in range(len(tensors)):
        n_dims = len(tensors[i].shape)
        n_right_dims = (n_dims - 1) - (n_dims + dim) % n_dims
        n_pad = pad_len - tensors[i].shape[dim]
        pad_tuple = tuple([0] * 2 * n_right_dims + [0, n_pad])
        tensors[i] = torch.nn.functional.pad(tensors[i], pad_tuple, value=pad_val)
    return tensors


def get_max_ptrs_per_stage(batch, ptr_class="ptr_mem"):
    """Get the maximum number of ptrs of a certain class per stage"""
    res = {}
    for data in batch:
        for q in data.find_queries:
            stage_id = q.query_processing_order
            if stage_id not in res:
                res[stage_id] = 0

            # Get the list of pointers in the query
            if ptr_class == "ptr_mem":
                ptrs = q.ptr_mem
            elif ptr_class == "ptrs_seg":
                ptrs = q.ptrs_seg
            else:
                raise NotImplementedError(ptr_class)

            if ptrs is not None:
                num_ptrs = len(ptrs)
            else:
                num_ptrs = 0
            if num_ptrs > res[stage_id]:
                res[stage_id] = num_ptrs
    return res


def add_ptrs_to_stage(
    query_ptrs,
    stage_ptrs,
    data,
    max_ptrs,
    datapoint_query_id_2_stage_query_id,
    ptr_type=PtrType.FindAgainMemory,
):
    if query_ptrs is not None:
        for ptr_mem in query_ptrs:
            target_stage = data.find_queries[ptr_mem.query_id].query_processing_order
            stage_ptrs.stage_ids.append(target_stage)
            stage_ptrs.query_ids.append(
                datapoint_query_id_2_stage_query_id[ptr_mem.query_id]
            )
            stage_ptrs.object_ids.append(ptr_mem.object_id)
            stage_ptrs.ptr_mask.append(0)
            stage_ptrs.ptr_types.append(ptr_type.value)
        nb_padding_ptrs = max_ptrs - len(query_ptrs)
    else:
        nb_padding_ptrs = max_ptrs
    for _ in range(nb_padding_ptrs):
        stage_ptrs.stage_ids.append(0)
        stage_ptrs.query_ids.append(0)
        stage_ptrs.object_ids.append(0)
        stage_ptrs.ptr_mask.append(1)
        stage_ptrs.ptr_types.append(ptr_type.value)


def collate_fn_api(
    batch: List[Datapoint],
    dict_key,
    with_seg_masks=False,
    input_box_embedding_dim=258,  # Historical default
    input_points_embedding_dim=257,
    repeats: int = 0,
    ptr_behaviour: PointerExtractBehaviour = PointerExtractBehaviour(),
    load_image_in_fp16: bool = False,
):
    # img_batch = torch.stack(sum([[img.data for img in v.images] for v in batch], []))
    img_batch = []
    text_batch = []
    raw_images = None

    num_stages = (
        max(q.query_processing_order for data in batch for q in data.find_queries) + 1
    )

    stages = [
        FindStage(
            img_ids=[],
            text_ids=[],
            input_boxes=[],
            input_boxes_label=[],
            input_boxes_before_embed=[],
            input_boxes_mask=[],
            input_points=[],
            input_points_before_embed=[],
            input_points_mask=[],
            ptrs=BatchedPointer(
                stage_ids=[], query_ids=[], object_ids=[], ptr_mask=[], ptr_types=[]
            ),
            ptrs_seg=BatchedPointer(
                stage_ids=[], query_ids=[], object_ids=[], ptr_mask=[], ptr_types=[]
            ),
            object_ids=[],
        )
        for _ in range(num_stages)
    ]
    find_targets = [
        BatchedFindTarget(
            num_boxes=[],
            boxes=[],
            boxes_padded=[],
            is_exhaustive=[],
            segments=[],
            semantic_segments=[],
            is_valid_segment=[],
            repeated_boxes=[],
            object_ids=[],
            object_ids_padded=[],
        )
        for _ in range(num_stages)
    ]
    get_stage = GetStage(
        text_inputs=[],
        text_output=[],
        ptrs_x=BatchedPointer(
            stage_ids=[], query_ids=[], object_ids=[], ptr_mask=[], ptr_types=[]
        ),
        ptrs_y=BatchedPointer(
            stage_ids=[], query_ids=[], object_ids=[], ptr_mask=[], ptr_types=[]
        ),
    )
    find_metadatas = [
        BatchedInferenceMetadata(
            coco_image_id=[],
            original_size=[],
            object_id=[],
            frame_index=[],
            original_image_id=[],
            original_category_id=[],
            get_text_input=[],
            is_conditioning_only=[],
        )
        for _ in range(num_stages)
    ]

    # Get the maximum number of ptrs of a certain class, for padding purposes
    max_ptrs_mem = get_max_ptrs_per_stage(batch, ptr_class="ptr_mem")
    max_ptrs_seg = get_max_ptrs_per_stage(batch, ptr_class="ptrs_seg")

    offset_img_id = 0
    offset_query_id = [0 for _ in range(num_stages)]
    for i, data in enumerate(batch):
        stage2within_stage_order = [[] for _ in range(num_stages)]
        img_batch.extend([img.data for img in data.images])

        if data.raw_images is not None:
            if raw_images is None:
                raw_images = []
            raw_images.extend(data.raw_images)

        # Conversion of query_ids indexing in a datapoint to query_ids indexing in a stage
        datapoint_query_id_2_stage_query_id = []
        for q in data.find_queries:
            stage_id = q.query_processing_order
            datapoint_query_id_2_stage_query_id.append(offset_query_id[stage_id])
            offset_query_id[stage_id] += 1

        for j, q in enumerate(data.get_queries):
            get_stage.text_inputs.append(q.query_text)
            get_stage.text_output.append(q.text_output)
            assert (
                q.query_type == QueryType.GetQuery
            ), f"get queries must be get queries, got {q.query_type}"
            assert (
                q.ptr_x is not None or q.ptr_y is not None
            ), "get queries must have at least one pointer"

            if q.ptr_x is not None:
                target_stage = data.find_queries[
                    q.ptr_x.query_id
                ].query_processing_order
                get_stage.ptrs_x.stage_ids.append(target_stage)
                get_stage.ptrs_x.query_ids.append(
                    datapoint_query_id_2_stage_query_id[q.ptr_x.query_id]
                )
                get_stage.ptrs_x.object_ids.append(q.ptr_x.object_id)
                get_stage.ptrs_x.ptr_mask.append(0)
                get_stage.ptrs_x.ptr_types.append(PtrType.FindGetX.value)
            else:
                # No pointer, populate with dummy values
                get_stage.ptrs_x.stage_ids.append(-1)
                get_stage.ptrs_x.query_ids.append(0)
                get_stage.ptrs_x.object_ids.append(0)
                get_stage.ptrs_x.ptr_mask.append(1)
                get_stage.ptrs_x.ptr_types.append(PtrType.FindGetX.value)

            if q.ptr_y is not None:
                target_stage = data.find_queries[
                    q.ptr_y.query_id
                ].query_processing_order
                get_stage.ptrs_y.stage_ids.append(target_stage)
                get_stage.ptrs_y.query_ids.append(
                    datapoint_query_id_2_stage_query_id[q.ptr_y.query_id]
                )
                get_stage.ptrs_y.object_ids.append(q.ptr_y.object_id)
                get_stage.ptrs_y.ptr_mask.append(0)
                get_stage.ptrs_y.ptr_types.append(PtrType.FindGetY.value)
            else:
                # No pointer, populate with dummy values
                get_stage.ptrs_y.stage_ids.append(-1)
                get_stage.ptrs_y.query_ids.append(0)
                get_stage.ptrs_y.object_ids.append(0)
                get_stage.ptrs_y.ptr_mask.append(1)
                get_stage.ptrs_y.ptr_types.append(PtrType.FindGetY.value)

        for j, q in enumerate(data.find_queries):
            stage_id = q.query_processing_order
            stages[stage_id].img_ids.append(q.image_id + offset_img_id)
            stage2within_stage_order[stage_id].append(q.within_stage_order)
            if q.query_text not in text_batch:
                text_batch.append(q.query_text)
            stages[stage_id].text_ids.append(text_batch.index(q.query_text))

            assert (
                q.inference_metadata is not None
            ), "inference_metadata must be provided when FindQueryLoaded is created."
            for f in fields(q.inference_metadata):
                getattr(find_metadatas[stage_id], f.name).append(
                    getattr(q.inference_metadata, f.name)
                )

            assert (
                q.ptr_x is None or q.ptr_y is None
            ), "can't provide both x and y pointers for find"

            assert stages[stage_id].ptrs is not None, "ptrs must be initialized"

            # Add the query conditional segments to the stage segment pointers
            add_ptrs_to_stage(
                q.ptrs_seg,
                stages[stage_id].ptrs_seg,
                data,
                max_ptrs_seg[stage_id],
                datapoint_query_id_2_stage_query_id,
            )

            # Add the query memory pointers to the stage pointers
            add_ptrs_to_stage(
                q.ptr_mem,
                stages[stage_id].ptrs,
                data,
                max_ptrs_mem[stage_id],
                datapoint_query_id_2_stage_query_id,
            )

            if q.ptr_x is not None:
                target_stage = data.find_queries[
                    q.ptr_x.query_id
                ].query_processing_order
                stages[stage_id].ptrs.stage_ids.append(target_stage)
                stages[stage_id].ptrs.query_ids.append(
                    datapoint_query_id_2_stage_query_id[q.ptr_x.query_id]
                )
                stages[stage_id].ptrs.object_ids.append(q.ptr_x.object_id)
                stages[stage_id].ptrs.ptr_mask.append(0)
                ptr_type = None
                if q.query_type == QueryType.FindQuery:
                    ptr_type = PtrType.FindGetX
                elif q.query_type == QueryType.FindAgainClose:
                    ptr_type = PtrType.FindAgainCloseX
                elif q.query_type == QueryType.FindAgainFar:
                    ptr_type = PtrType.FindAgainFarX
                else:
                    assert False, "unknown query type"
                stages[stage_id].ptrs.ptr_types.append(ptr_type.value)
            elif q.ptr_y is not None:
                target_stage = data.find_queries[
                    q.ptr_y.query_id
                ].query_processing_order
                stages[stage_id].ptrs.stage_ids.append(target_stage)
                stages[stage_id].ptrs.query_ids.append(
                    datapoint_query_id_2_stage_query_id[q.ptr_y.query_id]
                )
                stages[stage_id].ptrs.object_ids.append(q.ptr_y.object_id)
                stages[stage_id].ptrs.ptr_mask.append(0)
                assert (
                    q.query_type == QueryType.FindQuery
                ), "y pointers are only allowed for find queries"
                stages[stage_id].ptrs.ptr_types.append(PtrType.FindGetY.value)
            else:
                # No pointer, populate with dummy values
                stages[stage_id].ptrs.stage_ids.append(-1)
                stages[stage_id].ptrs.query_ids.append(0)
                stages[stage_id].ptrs.object_ids.append(0)
                stages[stage_id].ptrs.ptr_mask.append(1)
                stages[stage_id].ptrs.ptr_types.append(0)

            if q.input_bbox is not None:
                assert q.input_bbox.shape[-1] == input_box_embedding_dim, (
                    "Mismatch between input bbox's embedding dimension from "
                    "dataset and expected dimension from collator. "
                    f"Dataset: {q.input_bbox.shape[-1]}, "
                    f"Collator: {input_box_embedding_dim}."
                )
                assert q.input_bbox_before_embed is not None
                assert q.input_bbox_before_embed.numel() % 4 == 0
                assert q.input_bbox_label is not None
                nb_boxes = q.input_bbox_before_embed.numel() // 4
                assert len(q.input_bbox_label) == nb_boxes
                stages[stage_id].input_boxes.append(
                    q.input_bbox.view(nb_boxes, input_box_embedding_dim)
                )
                stages[stage_id].input_boxes_before_embed.append(
                    q.input_bbox_before_embed.view(nb_boxes, 4)
                )
                stages[stage_id].input_boxes_label.append(
                    q.input_bbox_label.view(nb_boxes)
                )
                stages[stage_id].input_boxes_mask.append(
                    torch.zeros(nb_boxes, dtype=torch.bool)
                )
            else:
                stages[stage_id].input_boxes.append(
                    torch.zeros(0, input_box_embedding_dim)
                )
                stages[stage_id].input_boxes_before_embed.append(torch.zeros(0, 4))
                stages[stage_id].input_boxes_label.append(
                    torch.zeros(0, dtype=torch.bool)
                )
                stages[stage_id].input_boxes_mask.append(
                    torch.ones(0, dtype=torch.bool)
                )

            if q.input_points is not None:
                assert q.input_points.shape[-1] == input_points_embedding_dim, (
                    "Mismatch between input points's embedding dimension from "
                    "dataset and expected dimension from collator. "
                    f"Dataset: {q.input_points.shape[-1]}, "
                    f"Collator: {input_box_embedding_dim}."
                )
                stages[stage_id].input_points.append(
                    q.input_points.squeeze(0)  # Strip a trivial batch index
                )
                stages[stage_id].input_points_before_embed.append(
                    q.input_points_before_embed.squeeze(0)
                )
                # All masks will be padded up to the longest length
                # with 1s before final conversion to batchd tensors
                stages[stage_id].input_points_mask.append(
                    torch.zeros(q.input_points.shape[1])
                )
            else:
                stages[stage_id].input_points.append(
                    torch.empty(0, input_points_embedding_dim)
                )
                stages[stage_id].input_points_before_embed.append(torch.empty(0, 3))
                stages[stage_id].input_points_mask.append(torch.empty(0))

            current_out_boxes = []
            current_out_object_ids = []
            # Set the object ids referred to by this query
            stages[stage_id].object_ids.append(q.object_ids_output)
            for object_id in q.object_ids_output:
                current_out_boxes.append(
                    data.images[q.image_id].objects[object_id].bbox
                )
                current_out_object_ids.append(object_id)
            find_targets[stage_id].boxes.extend(current_out_boxes)
            find_targets[stage_id].object_ids.extend(current_out_object_ids)
            if repeats > 0:
                for _ in range(repeats):
                    find_targets[stage_id].repeated_boxes.extend(current_out_boxes)
            find_targets[stage_id].num_boxes.append(len(current_out_boxes))
            find_targets[stage_id].is_exhaustive.append(q.is_exhaustive)

            if with_seg_masks:
                current_seg_mask = []
                current_is_valid_segment = []
                for object_id in q.object_ids_output:
                    seg_mask = data.images[q.image_id].objects[object_id].segment
                    if seg_mask is not None:
                        current_seg_mask.append(seg_mask)
                        current_is_valid_segment.append(1)
                    else:
                        dummy_mask = torch.zeros(
                            data.images[q.image_id].data.shape[-2:], dtype=torch.bool
                        )
                        current_seg_mask.append(dummy_mask)
                        current_is_valid_segment.append(0)
                find_targets[stage_id].segments.extend(current_seg_mask)
                find_targets[stage_id].is_valid_segment.extend(current_is_valid_segment)
            else:
                # We are not loading segmentation masks
                find_targets[stage_id].segments = None
                find_targets[stage_id].is_valid_segment = None

            if q.semantic_target is not None:
                find_targets[stage_id].semantic_segments.append(q.semantic_target)

        offset_img_id += len(data.images)

        # Sanity check: we should have the same within stage order in all stages
        for stage_id in range(num_stages):
            assert stage2within_stage_order[stage_id] == stage2within_stage_order[0], (
                f"Within stage order mismatch in stage {stage_id} for datapoint {i}: "
                f"{stage2within_stage_order[stage_id]} vs {stage2within_stage_order[0]}"
            )

    # Pad input points to equal sequence lengths
    for i in range(len(stages)):
        stages[i].input_points = pad_tensor_list_to_longest(
            stages[i].input_points, dim=0, pad_val=0
        )
        stages[i].input_points_before_embed = pad_tensor_list_to_longest(
            stages[i].input_points_before_embed, dim=0, pad_val=0
        )
        # Masked-out regions indicated by 1s.
        stages[i].input_points_mask = pad_tensor_list_to_longest(
            stages[i].input_points_mask, dim=0, pad_val=1
        )

    # Pad input boxes to equal sequence lengths
    for i in range(len(stages)):
        stages[i].input_boxes = pad_tensor_list_to_longest(
            stages[i].input_boxes, dim=0, pad_val=0
        )
        stages[i].input_boxes_before_embed = pad_tensor_list_to_longest(
            stages[i].input_boxes_before_embed, dim=0, pad_val=0
        )
        stages[i].input_boxes_label = pad_tensor_list_to_longest(
            stages[i].input_boxes_label, dim=0, pad_val=0
        )
        # Masked-out regions indicated by 1s.
        stages[i].input_boxes_mask = pad_tensor_list_to_longest(
            stages[i].input_boxes_mask, dim=0, pad_val=1
        )

    # Convert to tensors
    for i in range(len(stages)):
        stages[i] = convert_my_tensors(stages[i])
        find_targets[i] = convert_my_tensors(find_targets[i])
        find_metadatas[i] = convert_my_tensors(find_metadatas[i])
        # get padded representation for the boxes
        find_targets[i].boxes_padded = packed_to_padded_naive(
            find_targets[i].boxes.view(-1, 4), find_targets[i].num_boxes
        )
        find_targets[i].object_ids_padded = packed_to_padded_naive(
            find_targets[i].object_ids, find_targets[i].num_boxes, fill_value=-1
        )

        # Optimization: If all pointers for the stages are dummy, we can delete them
        if torch.all(stages[i].ptrs.ptr_mask == 1).item():
            stages[i].ptrs = None

    get_stage = convert_my_tensors(get_stage)

    # Finalize the image batch
    image_batch = NestedTensor.from_tensor_list(img_batch, rounding=None)
    if load_image_in_fp16:
        # Optionally, cast the image tensors to fp16, which helps save GPU memory on
        # long videos with thousands of frames (where image tensors could be several GBs)
        image_batch.tensors = image_batch.tensors.half()

    return {
        dict_key: BatchedDatapoint(
            img_batch=image_batch,
            find_text_batch=text_batch,
            find_inputs=stages,
            find_targets=find_targets,
            get_queries=get_stage,
            find_metadatas=find_metadatas,
            ptr_behaviour=ptr_behaviour,
            raw_images=raw_images,
        )
    }
