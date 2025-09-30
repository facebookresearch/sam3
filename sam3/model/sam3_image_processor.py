# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for SAM3.
"""

import gc
import logging
import sys
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

import torch
from PIL import Image, ImageOps

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


# Constants typically from transformers
IMAGENET_DEFAULT_MEAN = [0.5, 0.5, 0.5]
IMAGENET_DEFAULT_STD = [0.5, 0.5, 0.5]


class ChannelDimension(Enum):
    """Enum for channel dimension format."""

    FIRST = "channels_first"
    LAST = "channels_last"


class PILImageResampling(Enum):
    """PIL Image resampling methods."""

    NEAREST = Image.Resampling.NEAREST
    BILINEAR = Image.Resampling.BILINEAR
    BICUBIC = Image.Resampling.BICUBIC
    LANCZOS = Image.Resampling.LANCZOS


# Type aliases
ImageInput = Union[
    Image.Image,
    np.ndarray,
    torch.Tensor,
    List[Union[Image.Image, np.ndarray, torch.Tensor]],
]


def make_list_of_images(
    images: ImageInput,
) -> List[Union[Image.Image, np.ndarray, torch.Tensor]]:
    """Convert input to list of images."""
    if not isinstance(images, list):
        return [images]
    return images


def to_numpy_array(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert image to numpy array."""
    if isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, torch.Tensor):
        return image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def valid_images(images: List[np.ndarray]) -> bool:
    """Check if images are valid."""
    if not images:
        return False
    for image in images:
        if not isinstance(image, np.ndarray):
            return False
        if image.ndim not in [2, 3]:
            return False
    return True


def infer_channel_dimension_format(image: np.ndarray) -> ChannelDimension:
    """Infer channel dimension format from image shape."""
    if image.ndim == 2:
        return ChannelDimension.LAST  # Grayscale
    elif image.ndim == 3:
        if image.shape[-1] in [1, 3, 4]:  # Common channel counts at the end
            return ChannelDimension.LAST
        elif image.shape[0] in [1, 3, 4]:  # Common channel counts at the beginning
            return ChannelDimension.FIRST
        else:
            # Default to channels last for ambiguous cases
            return ChannelDimension.LAST
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")


def get_image_size(image: np.ndarray, channel_dim: ChannelDimension) -> tuple:
    """Get image size (height, width)."""
    if channel_dim == ChannelDimension.FIRST:
        if image.ndim == 3:
            return image.shape[1], image.shape[2]  # (H, W)
        else:
            return image.shape[0], image.shape[1]  # (H, W)
    else:  # LAST
        return image.shape[0], image.shape[1]  # (H, W)


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert image to RGB format."""
    if image.ndim == 2:
        # Grayscale to RGB
        return np.stack([image] * 3, axis=-1)
    elif image.ndim == 3:
        if image.shape[-1] == 4:  # RGBA to RGB
            return image[..., :3]
        elif image.shape[-1] == 1:  # Single channel to RGB
            return np.repeat(image, 3, axis=-1)
        elif image.shape[-1] == 3:  # Already RGB
            return image
        elif image.shape[0] == 3:  # Channels first RGB
            return np.transpose(image, (1, 2, 0))
        elif image.shape[0] == 4:  # Channels first RGBA
            return np.transpose(image[..., :3], (1, 2, 0))
    return image


def convert_list_to_tensor(data: List[Any]) -> torch.Tensor:
    """Convert list to tensor."""
    if isinstance(data, list):
        return torch.stack([convert_list_to_tensor(item) for item in data])
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        return data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def resize(
    image: np.ndarray,
    size: tuple,
    resample: PILImageResampling,
    input_data_format: ChannelDimension,
) -> np.ndarray:
    """Resize image to target size."""
    # Convert numpy array to PIL Image
    if input_data_format == ChannelDimension.FIRST and image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))

    pil_image = Image.fromarray(image.astype(np.uint8))
    resized_pil = pil_image.resize((size[1], size[0]), resample.value)
    resized_array = np.array(resized_pil)

    # Convert back to original format if needed
    if input_data_format == ChannelDimension.FIRST and resized_array.ndim == 3:
        resized_array = np.transpose(resized_array, (2, 0, 1))

    return resized_array.astype(image.dtype)


def normalize(
    image: np.ndarray,
    mean: List[float],
    std: List[float],
    input_data_format: ChannelDimension,
) -> np.ndarray:
    """Normalize image with mean and std."""
    image = image.astype(np.float32) / 255.0

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    if input_data_format == ChannelDimension.FIRST:
        if image.ndim == 3:
            mean = mean.reshape(-1, 1, 1)
            std = std.reshape(-1, 1, 1)
    else:  # LAST
        if image.ndim == 3:
            mean = mean.reshape(1, 1, -1)
            std = std.reshape(1, 1, -1)

    return (image - mean) / std


def to_channel_dimension_format(
    image: np.ndarray,
    data_format: ChannelDimension,
    input_channel_dim: ChannelDimension,
) -> np.ndarray:
    """Convert image to specified channel dimension format."""
    if data_format == input_channel_dim:
        return image

    if image.ndim == 2:
        return image

    if (
        data_format == ChannelDimension.FIRST
        and input_channel_dim == ChannelDimension.LAST
    ):
        return np.transpose(image, (2, 0, 1))
    elif (
        data_format == ChannelDimension.LAST
        and input_channel_dim == ChannelDimension.FIRST
    ):
        return np.transpose(image, (1, 2, 0))

    return image


class ImageProcessor:
    r"""
    SAM3 Image Processor for preprocessing images for the SAM3 model.
    This processor handles image resizing, normalization, and format conversion.
    """

    def __init__(
        self,
        do_resize: bool = True,
        size: List[int] = (1008, 1008),
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample

        self.do_normalize = do_normalize
        self.image_mean = (
            image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        )
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD

        self.do_convert_rgb = do_convert_rgb

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[Union[List[int], tuple]] = None,
        resample: PILImageResampling = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[str] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        TODO
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample

        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = (
            do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        )

        images = make_list_of_images(images)

        images = [to_numpy_array(image) for image in images]

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        original_sizes = []
        for image in images:
            original_sizes.append(get_image_size(image, channel_dim=input_data_format))

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        if do_resize:
            images = [
                resize(
                    image=image,
                    size=size,
                    resample=resample,
                    input_data_format=input_data_format,
                )
                for image in images
            ]

        reshaped_input_sizes = []
        for image in images:
            reshaped_input_sizes.append(
                get_image_size(image, channel_dim=input_data_format)
            )

        if do_normalize:
            images = [
                normalize(
                    image=image,
                    mean=image_mean,
                    std=image_std,
                    input_data_format=input_data_format,
                )
                for image in images
            ]

        images = [
            to_channel_dimension_format(
                image, data_format, input_channel_dim=input_data_format
            )
            for image in images
        ]

        # convert to tensor
        # images = torch.from_numpy(images)

        data = {
            "pixel_values": images,
            "original_sizes": original_sizes,
            "reshaped_input_sizes": reshaped_input_sizes,
        }

        if return_tensors:
            data["pixel_values"] = convert_list_to_tensor(data["pixel_values"])
            data["original_sizes"] = data["original_sizes"]
            data["reshaped_input_sizes"] = data["reshaped_input_sizes"]

        return data
        # return BatchFeature(data=data, tensor_type=return_tensors)


class Sam3Processor:
    r"""
    Constructs a SAM3 processor which wraps a SAM3 image processor and text tokenizer into a single processor.

    [`Sam3Processor`] offers all the functionalities of [`Sam3ImageProcessor`] and [`SimpleTokenizer`]. See the docstring of
    [`~Sam3Processor.__call__`] and [`~Sam3Processor.decode`] for more information.

    Args:
        image_processor ([`Sam3ImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`SimpleTokenizer`], *optional*):
            The tokenizer for text inputs. If not provided, text inputs will not be processed.
    """

    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2

    def __init__(self, image_processor=ImageProcessor(), tokenizer=None, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")

        self.image_processor = image_processor
        self.tokenizer = tokenizer
        assert self.tokenizer is None, "Tokenizer is not supported yet"
        self.current_processor = self.image_processor
        self._in_target_context_manager = False
        self.image_size = image_processor.size

    def __call__(
        self,
        images=None,
        inference_state=None,
        return_tensors: str = "pt",
        instance_prompt: Optional[bool] = False,
        device: Optional[torch.device] = "cuda",
        **kwargs,
    ) -> Dict[str, Any]:
        """ """

        # only image or inference_state can be passed
        if images is None and inference_state is None:
            raise ValueError("You have to specify images or inference_state.")

        # Process images with prompts
        if images is not None:
            assert (
                inference_state is None
            ), "You cannot specify both images and inference_state."
            inputs = self.image_processor.preprocess(
                images=images, return_tensors=return_tensors, **kwargs
            )
            inference_state = self._init_state(inputs, device=device)

        return inference_state

    def _init_state(
        self,
        inputs,
        device: Optional[torch.device] = "cuda",
    ):
        """Initialize an inference state from `resource_path` (an image or a video)."""

        images = inputs["pixel_values"].to(device)
        orig_height = inputs["original_sizes"][0][0]
        orig_width = inputs["original_sizes"][0][1]
        inference_state = {}
        inference_state["device"] = torch.device(device)
        inference_state["image_size"] = self.image_size[0]
        inference_state["num_frames"] = len(images)

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
                input_boxes_before_embed=[torch.zeros(4)],
                input_boxes_mask=[1],
                input_boxes_label=[0],
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
        # if self.use_aot_mem:
        #     inference_state["aot_mem_per_frame"] = {"spatial": {}, "pointer": {}}
        # if self.use_obj_mem_bank:
        #     inference_state["obj_mem_per_frame"] = {
        #         "roi": {},
        #         "roi_zoomed_out": {},
        #         "global": {},
        #     }

        # placeholders for cached outputs
        # (note: currently, a single visual prompt embedding is shared for all frames)
        inference_state["backbone_out"] = None
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None

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

    def add_prompt(
        self,
        inference_state,
        frame_idx=0,
        text_str=None,
        clear_old_points=True,
        points=None,
        point_labels=None,
        boxes_xywh=None,
        box_labels=None,
        clear_old_boxes=True,
        instance_prompt=False,
    ):
        """
        Add text, point or box prompts on a single frame. This method returns the inference
        outputs only on the prompted frame.

        Note that text prompts are NOT associated with a particular frame (i.e. they apply
        to all frames). However, we only run inference on the frame specified in `frame_idx`.
        """
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
            # currently we do not allow simultaneously adding text prompt and visual
            # prompt both as initial prompt (since visual prompt uses the text "visual")
            if any(p is not None for p in inference_state["per_frame_visual_prompt"]):
                raise RuntimeError(
                    "Text and visual prompts (box as an initial prompt) cannot be used together. "
                    "Please reset the session."
                )

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

        if points is not None and boxes_xywh is not None:
            raise RuntimeError(
                "Cannot add both point and box prompts at the same time. "
            )

        if points is not None and not instance_prompt:
            raise RuntimeError(
                "Point prompts are only supported for instance tracking. "
            )

        if instance_prompt and (text_str is not None or boxes_xywh is not None):
            raise RuntimeError(
                "Text and box prompts are not supported for instance tracking. "
            )

        new_visual_prompt = None

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
            new_visual_prompt = None

            for t in range(inference_state["num_frames"]):
                inference_state["input_batch"].find_inputs[t].text_ids[
                    ...
                ] = self.TEXT_ID_FOR_GEOMETRIC

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

        inference_state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt

        inference_state["new_visual_prompt"] = new_visual_prompt
        inference_state["frame_idx"] = frame_idx
        inference_state["model_out"] = None
        inference_state["instance_prompt"] = instance_prompt

        # return out

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
        inference_state["model_out"] = None
        inference_state["frame_idx"] = None
        inference_state["new_visual_prompt"] = None
        inference_state["instance_prompt"] = None

        gc.collect()

    def postprocess_output(
        self,
        inference_state,
        output_prob_thresh=0.5,
        pop=True,
        reset_state=False,
    ):
        """Post-process the single-frame output into the desired numpy result format."""
        prompt_idx = 0
        out = inference_state["model_out"]
        if pop:
            out_scores = out.pop("pred_logits")[prompt_idx].squeeze(-1)
            out_boxes_xyxy = out.pop("pred_boxes_xyxy")[prompt_idx]
            # out_obj_ids = out.pop("pred_object_ids")[prompt_idx]
            out_masks = out.pop("pred_masks")[prompt_idx]

            # remove a few unused keys (to reduce GPU memory usage)
            unused_output_keys = [
                "pred_boxes",
                "pred_is_valid",
                "pred_old_obj_ids",
                "semantic_seg",
                "presence_logit",
            ]
            for k in unused_output_keys:
                out.pop(k, None)
        else:
            out_scores = out["pred_logits"][prompt_idx].squeeze(-1)
            out_boxes_xyxy = out["pred_boxes_xyxy"][prompt_idx]
            # out_obj_ids = out["pred_object_ids"][prompt_idx]
            out_masks = out["pred_masks"][prompt_idx]

        # only take the entries above the score threshold
        out_probs = out_scores.sigmoid()  # output in probabilities in 0~1

        if "presence_logit_dec" in out:
            presence_score = out["presence_logit_dec"][prompt_idx].squeeze(-1).sigmoid()
            out_probs = presence_score * out_probs

        # keep = out_obj_ids >= 0
        if output_prob_thresh is not None:
            # keep = torch.logical_and(out_probs > output_prob_thresh, keep)
            keep = out_probs > output_prob_thresh
        out_probs = out_probs[keep]
        out_boxes_xyxy = out_boxes_xyxy[keep]
        # out_obj_ids = out_obj_ids[keep]
        out_masks = out_masks[keep]

        out_boxes_xywh = box_xyxy_to_xywh(out_boxes_xyxy)  # output in XYWH box format
        num_out_obj = out_masks.size(0)
        if num_out_obj > 0:
            out_masks_orig_size = torch.nn.functional.interpolate(
                out_masks.unsqueeze(0),
                size=(inference_state["orig_height"], inference_state["orig_width"]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            out_binary_masks = out_masks_orig_size > 0.0
        else:
            # in case there is no object, `torch.nn.functional.interpolate` would raise
            # an error on an empty tensor, so we treat it specially here
            out_binary_masks = torch.zeros(
                0,
                inference_state["orig_height"],
                inference_state["orig_width"],
                dtype=torch.bool,
            )

        # We directly convert the outputs to CPU numpy format so that it is easy for
        # the server to send them across processes and construct the final response.
        frame_outputs = {
            "out_probs": out_probs.float().cpu().numpy(),
            "out_boxes_xywh": out_boxes_xywh.float().cpu().numpy(),
            # "out_obj_ids": out_obj_ids.cpu().numpy(),
            "out_binary_masks": out_binary_masks.cpu().numpy(),
        }

        if reset_state:
            self.reset_state(inference_state)

        return frame_outputs
