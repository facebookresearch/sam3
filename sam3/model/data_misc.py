# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe
"""
Misc functions, including distributed helpers.
"""

import collections
import re
from dataclasses import dataclass, field as field_ptr_behaviour, fields, is_dataclass
from typing import Any, get_args, get_origin, List, Mapping, Optional, Sequence, Union

import torch


MyTensor = Union[torch.Tensor, List[Any]]


class NestedTensor:
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask)

    def clone(self):
        new_tensors = self.tensors.clone()
        new_mask = None if self.mask is None else self.mask.clone()
        return NestedTensor(new_tensors, new_mask)

    def __getitem__(self, idx):
        return self.tensors[idx]

    def __len__(self):
        return len(self.tensors)

    @property
    def device(self):
        return self.tensors.device

    @property
    def shape(self):
        return self.tensors.shape

    # custom memory pinning method on custom type
    def pin_memory(self, device=None):
        self.tensors = self.tensors.pin_memory(device)
        if self.mask is not None:
            self.mask = self.mask.pin_memory(device)


# Register NestedTensor as a pytree node so tree_map_only can traverse into it
# (matches onevision/utils/misc.py registration)
from torch.utils import _pytree as pytree

pytree.register_pytree_node(
    NestedTensor,
    lambda x: ([x.tensors, x.mask], None),
    lambda values, _: NestedTensor(values[0], values[1]),
)


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty channel sizes.
    """
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners
        )

    assert input.shape[0] != 0 or input.shape[1] != 0, (
        "At least one of the two first dimensions must be non zero"
    )

    if input.shape[1] == 0:
        # Pytorch doesn't support null dimension on the channel dimension, so we transpose to fake a null batch dim
        return torch.nn.functional.interpolate(
            input.transpose(0, 1), size, scale_factor, mode, align_corners
        ).transpose(0, 1)

    # empty batch dimension is now supported in pytorch
    return torch.nn.functional.interpolate(
        input, size, scale_factor, mode, align_corners
    )


@dataclass
class BatchedPointer:
    stage_ids: MyTensor
    stage_ids__type = torch.long
    query_ids: MyTensor
    query_ids__type = torch.long
    object_ids: MyTensor
    object_ids__type = torch.long
    ptr_mask: MyTensor
    ptr_mask__type = torch.bool
    ptr_types: MyTensor
    ptr_types__type = torch.long


@dataclass
class FindStage:
    img_ids: MyTensor
    img_ids__type = torch.long
    text_ids: MyTensor
    text_ids__type = torch.long

    input_boxes: MyTensor
    input_boxes__type = torch.float
    input_boxes_mask: MyTensor
    input_boxes_mask__type = torch.bool
    input_boxes_label: MyTensor
    input_boxes_label__type = torch.long

    input_points: MyTensor
    input_points__type = torch.float
    input_points_mask: MyTensor
    input_points_mask__type = torch.bool

    # We track the object ids referred to by this query.
    # This is beneficial for tracking in videos without the need for pointers.
    object_ids: Optional[List[List]] = None  # List of objects per query

    # Multiplex-specific fields (used by sam3_demo_multiplex)
    img_ids_np: Optional[Any] = None
    input_boxes_before_embed: Optional[MyTensor] = None
    input_boxes_before_embed__type = torch.float
    input_points_before_embed: Optional[MyTensor] = None
    input_points_before_embed__type = torch.float
    ptrs: Optional[Any] = None
    ptrs_seg: Optional[Any] = None


@dataclass
class BatchedFindTarget:
    # The number of boxes in each find query
    num_boxes: MyTensor
    num_boxes__type = torch.long

    # Target boxes in normalized CxCywh format
    boxes: MyTensor
    boxes__type = torch.float
    # Target boxes in normalized CxCywh format but in padded representation
    # as used in BinaryHungarianMatcherV2 (unlike the packed ones in `boxes`)
    boxes_padded: MyTensor
    boxes_padded__type = torch.float

    # For hybrid matching, we repeat the boxes
    repeated_boxes: MyTensor
    repeated_boxes__type = torch.float

    # Target Segmentation masks
    segments: Optional[MyTensor]
    segments__type = torch.bool

    # Target Semantic Segmentation masks
    semantic_segments: Optional[MyTensor]
    semantic_segments__type = torch.bool

    is_valid_segment: Optional[MyTensor]
    is_valid_segment__type = torch.bool

    # Whether annotations are exhaustive for each query
    is_exhaustive: MyTensor
    is_exhaustive__type = torch.bool

    # The object id for each ground-truth box, in both packed and padded representations
    object_ids: MyTensor
    object_ids__type = torch.long
    object_ids_padded: MyTensor
    object_ids_padded__type = torch.long


@dataclass
class BatchedInferenceMetadata:
    """All metadata required to post-process a find stage"""

    # Coco id that corresponds to the "image" for evaluation by the coco evaluator
    coco_image_id: MyTensor
    coco_image_id__type = torch.long

    # id in the original dataset, such that we can use the original evaluator
    original_image_id: MyTensor
    original_image_id__type = torch.long

    # Original category id (if we want to use the original evaluator)
    original_category_id: MyTensor
    original_category_id__type = torch.int

    # Size of the raw image (height, width)
    original_size: MyTensor
    original_size__type = torch.long

    # id of the object in the media (track_id for a video)
    object_id: MyTensor
    object_id__type = torch.long

    # index of the frame in the media (0 in the case of a single-frame media)
    frame_index: MyTensor
    frame_index__type = torch.long

    # Adding for relations inference
    # get_text_input: List[Optional[str]]

    # Adding for TA conditional inference
    is_conditioning_only: List[Optional[bool]]


@dataclass
class BatchedDatapoint:
    img_batch: torch.Tensor
    find_text_batch: List[str]
    find_inputs: List[FindStage]
    find_targets: List[BatchedFindTarget]
    find_metadatas: List[BatchedInferenceMetadata]
    raw_images: Optional[List[Any]] = None
    get_queries: Optional[Any] = None


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
