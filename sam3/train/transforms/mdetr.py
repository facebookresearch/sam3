# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

"""Dataset class for modulated detection"""

from typing import Callable, List, Tuple

import torch
from pycocotools import mask as coco_mask


def create_positive_map(
    tokenized, chars_positive: List[List[Tuple[int, int]]], max_len: int
):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j

    :param tokenized: tokenized text_input
    :param chars_positive: Each element is a list of spans represented as (begining char, end char) tuples (end is exclusive)
    :param max_len: maximum length of the text_input
    :return: a tensor of shape (len(chars_positive), max_len) where positive_map[i,j] = True iff box i is associated to token j

    """
    positive_map = torch.zeros((len(chars_positive), max_len), dtype=torch.float)
    for j, tok_list in enumerate(chars_positive):
        for beg, end in tok_list:
            if tokenized.is_fast:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
                if beg_pos is None:
                    try:
                        beg_pos = tokenized.char_to_token(beg + 1)
                        if beg_pos is None:
                            beg_pos = tokenized.char_to_token(beg + 2)
                    except:
                        beg_pos = None
                if end_pos is None:
                    try:
                        end_pos = tokenized.char_to_token(end - 2)
                        if end_pos is None:
                            end_pos = tokenized.char_to_token(end - 3)
                    except:
                        end_pos = None
                if beg_pos is None or end_pos is None:
                    continue
            else:
                assert (
                    max_len > 2040
                ), "Something is weird, if we are here we expect the canine model hence long sequence length"
                # char based tokenization is easy :)
                beg_pos = beg
                end_pos = end - 1

            assert beg_pos is not None and end_pos is not None
            assert beg_pos >= 0 and end_pos >= beg_pos
            positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMaskModulated(Callable):
    def __init__(
        self,
        return_masks,
        return_tokens,
        vision_tokenizer,
        language_tokenizer,
        do_gen,
        max_len,
        max_ann_per_img,
        return_eval_map,
        training,
        coco_img_keys: List[str],
    ):
        self.coco_img_keys = coco_img_keys
        self.return_masks = return_masks
        self.return_tokens = return_tokens
        self.vision_tokenizer = vision_tokenizer
        self.language_tokenizer = language_tokenizer
        self.do_gen = do_gen
        self.max_len = max_len
        self.max_ann_per_img = max_ann_per_img if not training else -1
        self.return_eval_map = return_eval_map

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        coco_img = target.pop("coco_img")
        for key in self.coco_img_keys:
            if key in coco_img:
                target[key] = coco_img[key]

        anno = target["annotations"]
        text_input = target.get("text_input", None)
        text_output = target.get("text_output", None)
        api_input = target.get("api_input", None)

        anno = [obj for obj in anno]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]

        if boxes.numel() > 0:
            assert (
                boxes.max() < 1.5
            ), f"The boxes don't look normalized... max={boxes.max()}\n, {boxes}"

        boxes[:, 0::2].mul_(w).clamp_(min=0, max=w)
        boxes[:, 1::2].mul_(h).clamp_(min=0, max=h)

        input_boxes = None
        if "input_boxes" in target:
            input_boxes = torch.as_tensor(target["input_boxes"]).view(-1, 4)
            input_boxes[:, 2:] += input_boxes[:, :2]
            if input_boxes.numel() > 0:
                assert input_boxes.max() < 1.5, "The boxes don't look normalized..."
            input_boxes[:, 0::2].mul_(w).clamp_(min=0, max=w)
            input_boxes[:, 1::2].mul_(h).clamp_(min=0, max=h)
        else:
            input_boxes = torch.zeros(0, 4)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        chars_positive_input = [] if self.return_tokens else None
        if self.return_tokens and anno and "chars_positive_input" in anno[0]:
            chars_positive_input = [obj["chars_positive_input"] for obj in anno]
        elif self.return_tokens and anno and "tokens" in anno[0]:
            chars_positive_input = [obj["tokens"] for obj in anno]

        chars_positive_output = [] if self.do_gen else None
        if (
            self.do_gen
            and self.return_tokens
            and anno
            and "chars_positive_output" in anno[0]
        ):
            chars_positive_output = [obj["chars_positive_output"] for obj in anno]

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        if self.max_ann_per_img > 0 and keep.sum() > self.max_ann_per_img:
            k = keep.float().topk(k=self.max_ann_per_img).indices
            keep.zero_()[k] = True
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        # target = {}  # retain all existing keys in `target`
        target["boxes"] = boxes
        target["labels"] = classes
        if text_input is not None:
            target["text_input"] = text_input
        if api_input is not None:
            target["api_input"] = api_input
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        if chars_positive_input is not None:
            target["chars_positive_input"] = []

            for i, k in enumerate(keep):
                if k:
                    target["chars_positive_input"].append(chars_positive_input[i])

        if chars_positive_output is not None:
            target["chars_positive_output"] = []

            for i, k in enumerate(keep):
                if k:
                    target["chars_positive_output"].append(chars_positive_output[i])

        if input_boxes is not None:
            target["input_boxes"] = input_boxes

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        target["area"] = area[keep]

        target["orig_size"] = torch.as_tensor([float(1), float(1)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["orig_size_unnormalized"] = torch.as_tensor([int(h), int(w)])

        if self.return_tokens:
            assert len(target["boxes"]) == len(target["chars_positive_input"])
            tokenized = self.vision_tokenizer(text_input, return_tensors="pt")
            target["positive_map_input"] = create_positive_map(
                tokenized, target["chars_positive_input"], self.max_len
            )
            assert len(target["boxes"]) == len(target["positive_map_input"])

            if chars_positive_output is not None:
                assert len(target["boxes"]) == len(target["chars_positive_output"])
                tokenized = self.language_tokenizer(text_output, return_tensors="pt")
                target["positive_map_output"] = create_positive_map(
                    tokenized, target["chars_positive_output"], self.max_len
                )

        if self.return_eval_map:
            assert "chars_positive_eval" in target
            tokenized = self.vision_tokenizer(text_input, return_tensors="pt")
            target["positive_map_eval"] = create_positive_map(
                tokenized, target["chars_positive_eval"], self.max_len
            )
            target["nb_eval"] = len(target["positive_map_eval"])

        return image, target


class ApplyPosEmbedToBoxes(Callable):
    def __init__(self, pos_embed):
        self.pos_embed = pos_embed

    def __call__(self, image, target):
        if "input_boxes" in target:
            input_boxes = target["input_boxes"]
            cx, cy, w, h = input_boxes.unbind(1)
            target["input_boxes"] = self.pos_embed.encode(cx, cy, w, h)

        return image, target


class ApplyPosEmbedToGeometricInputsAPI(Callable):
    def __init__(self, pos_embed):
        self.pos_embed = pos_embed

    def __call__(self, datapoint, **kwargs):
        for q in datapoint.find_queries:
            if q.input_bbox is not None:
                q.input_bbox_before_embed = q.input_bbox
                input_boxes = q.input_bbox.view(-1, 4)
                cx, cy, w, h = input_boxes.unbind(1)
                q.input_bbox = self.pos_embed.encode(cx, cy, w, h)
            if q.input_points is not None:
                q.input_points_before_embed = q.input_points
                input_points = q.input_points.view(1, -1, 3)
                cx, cy, labels = input_points.unbind(2)
                q.input_points = self.pos_embed.encode_points(cx, cy, labels)
        return datapoint


ApplyPosEmbedToBoxesAPI = ApplyPosEmbedToGeometricInputsAPI  # Backwards compatibility


class ConvertCocoPolysToMask(Callable):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        target["area"] = area[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target
