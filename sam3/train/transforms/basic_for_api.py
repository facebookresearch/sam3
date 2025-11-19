# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Transforms and data augmentation for both image + bbox.
"""

import logging

import numbers
import random
from collections.abc import Sequence
from typing import Iterable

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as Fv2

from PIL import Image as PILImage

from sam3.model.box_ops import box_xyxy_to_cxcywh, masks_to_boxes
from sam3.train.data.sam3_image_dataset import Datapoint
from torchvision.transforms import InterpolationMode


def crop(
    datapoint,
    index,
    region,
    v2=False,
    check_validity=True,
    check_input_validity=True,
    recompute_box_from_mask=False,
):
    if v2:
        rtop, rleft, rheight, rwidth = (int(round(r)) for r in region)
        datapoint.images[index].data = Fv2.crop(
            datapoint.images[index].data,
            top=rtop,
            left=rleft,
            height=rheight,
            width=rwidth,
        )
    else:
        datapoint.images[index].data = F.crop(datapoint.images[index].data, *region)

    i, j, h, w = region

    # should we do something wrt the original size?
    datapoint.images[index].size = (h, w)

    for obj in datapoint.images[index].objects:
        # crop the mask
        if obj.segment is not None:
            obj.segment = F.crop(obj.segment, int(i), int(j), int(h), int(w))

        # crop the bounding box
        if recompute_box_from_mask and obj.segment is not None:
            # here the boxes are still in XYXY format with absolute coordinates (they are
            # converted to CxCyWH with relative coordinates in basic_for_api.NormalizeAPI)
            obj.bbox, obj.area = get_bbox_xyxy_abs_coords_from_mask(obj.segment)
        else:
            if recompute_box_from_mask and obj.segment is None and obj.area > 0:
                logging.warning(
                    "Cannot recompute bounding box from mask since `obj.segment` is None. "
                    "Falling back to directly cropping from the input bounding box."
                )
            boxes = obj.bbox.view(1, 4)
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            cropped_boxes = boxes - torch.as_tensor([j, i, j, i], dtype=torch.float32)
            cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)
            obj.area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
            obj.bbox = cropped_boxes.reshape(-1, 4)

    for query in datapoint.find_queries:
        if query.semantic_target is not None:
            query.semantic_target = F.crop(
                query.semantic_target, int(i), int(j), int(h), int(w)
            )
        if query.image_id == index and query.input_bbox is not None:
            boxes = query.input_bbox
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            cropped_boxes = boxes - torch.as_tensor([j, i, j, i], dtype=torch.float32)
            cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)

            # cur_area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
            # if check_input_validity:
            #     assert (
            #         (cur_area > 0).all().item()
            #     ), "Some input box got cropped out by the crop transform"

            query.input_bbox = cropped_boxes.reshape(-1, 4)
        if query.image_id == index and query.input_points is not None:
            print(
                "Warning! Point cropping with this function may lead to unexpected results"
            )
            points = query.input_points
            # Unlike right-lower box edges, which are exclusive, the
            # point must be in [0, length-1], hence the -1
            max_size = torch.as_tensor([w, h], dtype=torch.float32) - 1
            cropped_points = points - torch.as_tensor([j, i, 0], dtype=torch.float32)
            cropped_points[:, :, :2] = torch.min(cropped_points[:, :, :2], max_size)
            cropped_points[:, :, :2] = cropped_points[:, :, :2].clamp(min=0)
            query.input_points = cropped_points

    if check_validity:
        # Check that all boxes are still valid
        for obj in datapoint.images[index].objects:
            assert obj.area > 0, "Box {} has no area".format(obj.bbox)

    return datapoint


def hflip(datapoint, index):
    datapoint.images[index].data = F.hflip(datapoint.images[index].data)

    w, h = datapoint.images[index].data.size
    for obj in datapoint.images[index].objects:
        boxes = obj.bbox.view(1, 4)
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor(
            [-1, 1, -1, 1]
        ) + torch.as_tensor([w, 0, w, 0])
        obj.bbox = boxes
        if obj.segment is not None:
            obj.segment = F.hflip(obj.segment)

    for query in datapoint.find_queries:
        if query.semantic_target is not None:
            query.semantic_target = F.hflip(query.semantic_target)
        if query.image_id == index and query.input_bbox is not None:
            boxes = query.input_bbox
            boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor(
                [-1, 1, -1, 1]
            ) + torch.as_tensor([w, 0, w, 0])
            query.input_bbox = boxes
        if query.image_id == index and query.input_points is not None:
            points = query.input_points
            points = points * torch.as_tensor([-1, 1, 1]) + torch.as_tensor([w, 0, 0])
            query.input_points = points
    return datapoint


def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = max_size * min_original_size / max_original_size

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = int(round(size))
        oh = int(round(size * h / w))
    else:
        oh = int(round(size))
        ow = int(round(size * w / h))

    return (oh, ow)


def resize(datapoint, index, size, max_size=None, square=False, v2=False):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    if square:
        size = size, size
    else:
        cur_size = (
            datapoint.images[index].data.size()[-2:][::-1]
            if v2
            else datapoint.images[index].data.size
        )
        size = get_size(cur_size, size, max_size)

    old_size = (
        datapoint.images[index].data.size()[-2:][::-1]
        if v2
        else datapoint.images[index].data.size
    )
    if v2:
        datapoint.images[index].data = Fv2.resize(
            datapoint.images[index].data, size, antialias=True
        )
    else:
        datapoint.images[index].data = F.resize(datapoint.images[index].data, size)

    new_size = (
        datapoint.images[index].data.size()[-2:][::-1]
        if v2
        else datapoint.images[index].data.size
    )
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, old_size))
    ratio_width, ratio_height = ratios

    for obj in datapoint.images[index].objects:
        boxes = obj.bbox.view(1, 4)
        scaled_boxes = boxes * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height], dtype=torch.float32
        )
        obj.bbox = scaled_boxes
        obj.area *= ratio_width * ratio_height
        if obj.segment is not None:
            obj.segment = F.resize(obj.segment[None, None], size).squeeze()

    for query in datapoint.find_queries:
        if query.semantic_target is not None:
            query.semantic_target = F.resize(
                query.semantic_target[None, None], size
            ).squeeze()
        if query.image_id == index and query.input_bbox is not None:
            boxes = query.input_bbox
            scaled_boxes = boxes * torch.as_tensor(
                [ratio_width, ratio_height, ratio_width, ratio_height],
                dtype=torch.float32,
            )
            query.input_bbox = scaled_boxes
        if query.image_id == index and query.input_points is not None:
            points = query.input_points
            scaled_points = points * torch.as_tensor(
                [ratio_width, ratio_height, 1],
                dtype=torch.float32,
            )
            query.input_points = scaled_points

    h, w = size
    datapoint.images[index].size = (h, w)
    return datapoint


def pad(datapoint, index, padding, v2=False):
    old_h, old_w = datapoint.images[index].size
    h, w = old_h, old_w
    if len(padding) == 2:
        # assumes that we only pad on the bottom right corners
        if v2:
            datapoint.images[index].data = Fv2.pad(
                datapoint.images[index].data, (0, 0, padding[0], padding[1])
            )
        else:
            datapoint.images[index].data = F.pad(
                datapoint.images[index].data, (0, 0, padding[0], padding[1])
            )
        h += padding[1]
        w += padding[0]
    else:
        if v2:
            # left, top, right, bottom
            datapoint.images[index].data = Fv2.pad(
                datapoint.images[index].data,
                (padding[0], padding[1], padding[2], padding[3]),
            )
        else:
            # left, top, right, bottom
            datapoint.images[index].data = F.pad(
                datapoint.images[index].data,
                (padding[0], padding[1], padding[2], padding[3]),
            )
        h += padding[1] + padding[3]
        w += padding[0] + padding[2]

    datapoint.images[index].size = (h, w)

    for obj in datapoint.images[index].objects:
        if len(padding) != 2:
            obj.bbox += torch.as_tensor(
                [padding[0], padding[1], padding[0], padding[1]], dtype=torch.float32
            )
        if obj.segment is not None:
            if v2:
                if len(padding) == 2:
                    obj.segment = Fv2.pad(
                        obj.segment[None], (0, 0, padding[0], padding[1])
                    ).squeeze(0)
                else:
                    obj.segment = Fv2.pad(obj.segment[None], tuple(padding)).squeeze(0)
            else:
                if len(padding) == 2:
                    obj.segment = F.pad(obj.segment, (0, 0, padding[0], padding[1]))
                else:
                    obj.segment = F.pad(obj.segment, tuple(padding))

    for query in datapoint.find_queries:
        if query.semantic_target is not None:
            if v2:
                if len(padding) == 2:
                    query.semantic_target = Fv2.pad(
                        query.semantic_target[None, None],
                        (0, 0, padding[0], padding[1]),
                    ).squeeze()
                else:
                    query.semantic_target = Fv2.pad(
                        query.semantic_target[None, None], tuple(padding)
                    ).squeeze()
            else:
                if len(padding) == 2:
                    query.semantic_target = F.pad(
                        query.semantic_target[None, None],
                        (0, 0, padding[0], padding[1]),
                    ).squeeze()
                else:
                    query.semantic_target = F.pad(
                        query.semantic_target[None, None], tuple(padding)
                    ).squeeze()
        if query.image_id == index and query.input_bbox is not None:
            if len(padding) != 2:
                query.input_bbox += torch.as_tensor(
                    [padding[0], padding[1], padding[0], padding[1]],
                    dtype=torch.float32,
                )
        if query.image_id == index and query.input_points is not None:
            if len(padding) != 2:
                query.input_points += torch.as_tensor(
                    [padding[0], padding[1], 0], dtype=torch.float32
                )

    return datapoint


class RandomSizeCropAPI:
    def __init__(
        self,
        min_size: int,
        max_size: int,
        respect_boxes: bool,
        consistent_transform: bool,
        respect_input_boxes: bool = True,
        v2: bool = False,
        recompute_box_from_mask: bool = False,
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.respect_boxes = respect_boxes  # if True we can't crop a box out
        self.respect_input_boxes = respect_input_boxes
        self.consistent_transform = consistent_transform
        self.v2 = v2
        self.recompute_box_from_mask = recompute_box_from_mask

    def _sample_no_respect_boxes(self, img):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        return T.RandomCrop.get_params(img, (h, w))

    def _sample_respect_boxes(self, img, boxes, points, min_box_size=10.0):
        """
        Assure that no box or point is dropped via cropping, though portions
        of boxes may be removed.
        """
        if len(boxes) == 0 and len(points) == 0:
            return self._sample_no_respect_boxes(img)

        if self.v2:
            img_height, img_width = img.size()[-2:]
        else:
            img_width, img_height = img.size

        minW, minH, maxW, maxH = (
            min(img_width, self.min_size),
            min(img_height, self.min_size),
            min(img_width, self.max_size),
            min(img_height, self.max_size),
        )

        # The crop box must extend one pixel beyond points to the bottom/right
        # to assure the exclusive box contains the points.
        minX = (
            torch.cat([boxes[:, 0] + min_box_size, points[:, 0] + 1], dim=0)
            .max()
            .item()
        )
        minY = (
            torch.cat([boxes[:, 1] + min_box_size, points[:, 1] + 1], dim=0)
            .max()
            .item()
        )
        minX = min(img_width, minX)
        minY = min(img_height, minY)
        maxX = torch.cat([boxes[:, 2] - min_box_size, points[:, 0]], dim=0).min().item()
        maxY = torch.cat([boxes[:, 3] - min_box_size, points[:, 1]], dim=0).min().item()
        maxX = max(0.0, maxX)
        maxY = max(0.0, maxY)
        minW = max(minW, minX - maxX)
        minH = max(minH, minY - maxY)
        w = random.uniform(minW, max(minW, maxW))
        h = random.uniform(minH, max(minH, maxH))
        if minX > maxX:
            # i = random.uniform(max(0, minX - w + 1), max(maxX, max(0, minX - w + 1)))
            i = random.uniform(max(0, minX - w), max(maxX, max(0, minX - w)))
        else:
            i = random.uniform(
                max(0, minX - w + 1), max(maxX - 1, max(0, minX - w + 1))
            )
        if minY > maxY:
            # j = random.uniform(max(0, minY - h + 1), max(maxY, max(0, minY - h + 1)))
            j = random.uniform(max(0, minY - h), max(maxY, max(0, minY - h)))
        else:
            j = random.uniform(
                max(0, minY - h + 1), max(maxY - 1, max(0, minY - h + 1))
            )

        return [j, i, h, w]

    def __call__(self, datapoint, **kwargs):
        if self.respect_boxes or self.respect_input_boxes:
            if self.consistent_transform:
                # Check that all the images are the same size
                w, h = datapoint.images[0].data.size
                for img in datapoint.images:
                    assert img.data.size == (w, h)

                all_boxes = []
                # Getting all boxes in all the images
                if self.respect_boxes:
                    all_boxes += [
                        obj.bbox.view(-1, 4)
                        for img in datapoint.images
                        for obj in img.objects
                    ]
                # Get all the boxes in the find queries
                if self.respect_input_boxes:
                    all_boxes += [
                        q.input_bbox.view(-1, 4)
                        for q in datapoint.find_queries
                        if q.input_bbox is not None
                    ]
                if all_boxes:
                    all_boxes = torch.cat(all_boxes, 0)
                else:
                    all_boxes = torch.empty(0, 4)

                all_points = [
                    q.input_points.view(-1, 3)[:, :2]
                    for q in datapoint.find_queries
                    if q.input_points is not None
                ]
                if all_points:
                    all_points = torch.cat(all_points, 0)
                else:
                    all_points = torch.empty(0, 2)

                crop_param = self._sample_respect_boxes(
                    datapoint.images[0].data, all_boxes, all_points
                )
                for i in range(len(datapoint.images)):
                    datapoint = crop(
                        datapoint,
                        i,
                        crop_param,
                        v2=self.v2,
                        check_validity=self.respect_boxes,
                        check_input_validity=self.respect_input_boxes,
                        recompute_box_from_mask=self.recompute_box_from_mask,
                    )
                return datapoint
            else:
                for i in range(len(datapoint.images)):
                    all_boxes = []
                    # Get all boxes in the current image
                    if self.respect_boxes:
                        all_boxes += [
                            obj.bbox.view(-1, 4) for obj in datapoint.images[i].objects
                        ]
                    # Get all the boxes in the find queries that correspond to this image
                    if self.respect_input_boxes:
                        all_boxes += [
                            q.input_bbox.view(-1, 4)
                            for q in datapoint.find_queries
                            if q.image_id == i and q.input_bbox is not None
                        ]
                    if all_boxes:
                        all_boxes = torch.cat(all_boxes, 0)
                    else:
                        all_boxes = torch.empty(0, 4)

                    all_points = [
                        q.input_points.view(-1, 3)[:, :2]
                        for q in datapoint.find_queries
                        if q.input_points is not None
                    ]
                    if all_points:
                        all_points = torch.cat(all_points, 0)
                    else:
                        all_points = torch.empty(0, 2)

                    crop_param = self._sample_respect_boxes(
                        datapoint.images[i].data, all_boxes, all_points
                    )
                    datapoint = crop(
                        datapoint,
                        i,
                        crop_param,
                        v2=self.v2,
                        check_validity=self.respect_boxes,
                        check_input_validity=self.respect_input_boxes,
                        recompute_box_from_mask=self.recompute_box_from_mask,
                    )
                return datapoint
        else:
            if self.consistent_transform:
                # Check that all the images are the same size
                w, h = datapoint.images[0].data.size
                for img in datapoint.images:
                    assert img.data.size == (w, h)

                crop_param = self._sample_no_respect_boxes(datapoint.images[0].data)
                for i in range(len(datapoint.images)):
                    datapoint = crop(
                        datapoint,
                        i,
                        crop_param,
                        v2=self.v2,
                        check_validity=self.respect_boxes,
                        check_input_validity=self.respect_input_boxes,
                        recompute_box_from_mask=self.recompute_box_from_mask,
                    )
                return datapoint
            else:
                for i in range(len(datapoint.images)):
                    crop_param = self._sample_no_respect_boxes(datapoint.images[i].data)
                    datapoint = crop(
                        datapoint,
                        i,
                        crop_param,
                        v2=self.v2,
                        check_validity=self.respect_boxes,
                        check_input_validity=self.respect_input_boxes,
                        recompute_box_from_mask=self.recompute_box_from_mask,
                    )
                return datapoint


class CenterCropAPI:
    def __init__(self, size, consistent_transform, recompute_box_from_mask=False):
        self.size = size
        self.consistent_transform = consistent_transform
        self.recompute_box_from_mask = recompute_box_from_mask

    def _sample_crop(self, image_width, image_height):
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop_top, crop_left, crop_height, crop_width

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            # Check that all the images are the same size
            w, h = datapoint.images[0].data.size
            for img in datapoint.images:
                assert img.size == (w, h)

            crop_top, crop_left, crop_height, crop_width = self._sample_crop(w, h)
            for i in range(len(datapoint.images)):
                datapoint = crop(
                    datapoint,
                    i,
                    (crop_top, crop_left, crop_height, crop_width),
                    recompute_box_from_mask=self.recompute_box_from_mask,
                )
            return datapoint

        for i in range(len(datapoint.images)):
            w, h = datapoint.images[i].data.size
            crop_top, crop_left, crop_height, crop_width = self._sample_crop(w, h)
            datapoint = crop(
                datapoint,
                i,
                (crop_top, crop_left, crop_height, crop_width),
                recompute_box_from_mask=self.recompute_box_from_mask,
            )

        return datapoint


class RandomHorizontalFlip:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p
        self.consistent_transform = consistent_transform

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                for i in range(len(datapoint.images)):
                    datapoint = hflip(datapoint, i)
            return datapoint
        for i in range(len(datapoint.images)):
            if random.random() < self.p:
                datapoint = hflip(datapoint, i)
        return datapoint


class RandomResizeAPI:
    def __init__(
        self, sizes, consistent_transform, max_size=None, square=False, v2=False
    ):
        if isinstance(sizes, int):
            sizes = (sizes,)
        assert isinstance(sizes, Iterable)
        self.sizes = list(sizes)
        self.max_size = max_size
        self.square = square
        self.consistent_transform = consistent_transform
        self.v2 = v2

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            size = random.choice(self.sizes)
            for i in range(len(datapoint.images)):
                datapoint = resize(
                    datapoint, i, size, self.max_size, square=self.square, v2=self.v2
                )
            return datapoint
        for i in range(len(datapoint.images)):
            size = random.choice(self.sizes)
            datapoint = resize(
                datapoint, i, size, self.max_size, square=self.square, v2=self.v2
            )
        return datapoint


class ScheduledRandomResizeAPI(RandomResizeAPI):
    def __init__(self, size_scheduler, consistent_transform, square=False):
        self.size_scheduler = size_scheduler
        # Just a meaningful init value for super
        params = self.size_scheduler(epoch_num=0)
        sizes, max_size = params["sizes"], params["max_size"]
        super().__init__(sizes, consistent_transform, max_size=max_size, square=square)

    def __call__(self, datapoint, **kwargs):
        assert "epoch" in kwargs, "Param scheduler needs to know the current epoch"
        params = self.size_scheduler(kwargs["epoch"])
        sizes, max_size = params["sizes"], params["max_size"]
        self.sizes = sizes
        self.max_size = max_size
        datapoint = super(ScheduledRandomResizeAPI, self).__call__(datapoint, **kwargs)
        return datapoint


class RandomPadAPI:
    def __init__(self, max_pad, consistent_transform):
        self.max_pad = max_pad
        self.consistent_transform = consistent_transform

    def _sample_pad(self):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad_x, pad_y

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            pad_x, pad_y = self._sample_pad()
            for i in range(len(datapoint.images)):
                datapoint = pad(datapoint, i, (pad_x, pad_y))
            return datapoint

        for i in range(len(datapoint.images)):
            pad_x, pad_y = self._sample_pad()
            datapoint = pad(datapoint, i, (pad_x, pad_y))
        return datapoint


class PadToSizeAPI:
    def __init__(self, size, consistent_transform, bottom_right=False, v2=False):
        self.size = size
        self.consistent_transform = consistent_transform
        self.v2 = v2
        self.bottom_right = bottom_right

    def _sample_pad(self, w, h):
        pad_x = self.size - w
        pad_y = self.size - h
        assert pad_x >= 0 and pad_y >= 0
        pad_left = random.randint(0, pad_x)
        pad_right = pad_x - pad_left
        pad_top = random.randint(0, pad_y)
        pad_bottom = pad_y - pad_top
        return pad_left, pad_top, pad_right, pad_bottom

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            # Check that all the images are the same size
            w, h = datapoint.images[0].data.size
            for img in datapoint.images:
                assert img.size == (w, h)
            if self.bottom_right:
                pad_right = self.size - w
                pad_bottom = self.size - h
                padding = (pad_right, pad_bottom)
            else:
                padding = self._sample_pad(w, h)
            for i in range(len(datapoint.images)):
                datapoint = pad(datapoint, i, padding, v2=self.v2)
            return datapoint

        for i, img in enumerate(datapoint.images):
            w, h = img.data.size
            if self.bottom_right:
                pad_right = self.size - w
                pad_bottom = self.size - h
                padding = (pad_right, pad_bottom)
            else:
                padding = self._sample_pad(w, h)
            datapoint = pad(datapoint, i, padding, v2=self.v2)
        return datapoint


class RandomMosaicVideoAPI:
    def __init__(self, prob=0.15, grid_h=2, grid_w=2, use_random_hflip=False):
        self.prob = prob
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.use_random_hflip = use_random_hflip

    def __call__(self, datapoint, **kwargs):
        if random.random() > self.prob:
            return datapoint

        # select a random location to place the target mask in the mosaic
        target_grid_y = random.randint(0, self.grid_h - 1)
        target_grid_x = random.randint(0, self.grid_w - 1)
        # whether to flip each grid in the mosaic horizontally
        if self.use_random_hflip:
            should_hflip = torch.rand(self.grid_h, self.grid_w) < 0.5
        else:
            should_hflip = torch.zeros(self.grid_h, self.grid_w, dtype=torch.bool)
        for i in range(len(datapoint.images)):
            datapoint = random_mosaic_frame(
                datapoint,
                i,
                grid_h=self.grid_h,
                grid_w=self.grid_w,
                target_grid_y=target_grid_y,
                target_grid_x=target_grid_x,
                should_hflip=should_hflip,
            )

        return datapoint


def random_mosaic_frame(
    datapoint,
    index,
    grid_h,
    grid_w,
    target_grid_y,
    target_grid_x,
    should_hflip,
):
    # Step 1: downsize the images and paste them into a mosaic
    image_data = datapoint.images[index].data
    is_pil = isinstance(image_data, PILImage.Image)
    if is_pil:
        H_im = image_data.height
        W_im = image_data.width
        image_data_output = PILImage.new("RGB", (W_im, H_im))
    else:
        H_im = image_data.size(-2)
        W_im = image_data.size(-1)
        image_data_output = torch.zeros_like(image_data)

    downsize_cache = {}
    for grid_y in range(grid_h):
        for grid_x in range(grid_w):
            y_offset_b = grid_y * H_im // grid_h
            x_offset_b = grid_x * W_im // grid_w
            y_offset_e = (grid_y + 1) * H_im // grid_h
            x_offset_e = (grid_x + 1) * W_im // grid_w
            H_im_downsize = y_offset_e - y_offset_b
            W_im_downsize = x_offset_e - x_offset_b

            if (H_im_downsize, W_im_downsize) in downsize_cache:
                image_data_downsize = downsize_cache[(H_im_downsize, W_im_downsize)]
            else:
                image_data_downsize = F.resize(
                    image_data,
                    size=(H_im_downsize, W_im_downsize),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,  # antialiasing for downsizing
                )
                downsize_cache[(H_im_downsize, W_im_downsize)] = image_data_downsize
            if should_hflip[grid_y, grid_x].item():
                image_data_downsize = F.hflip(image_data_downsize)

            if is_pil:
                image_data_output.paste(image_data_downsize, (x_offset_b, y_offset_b))
            else:
                image_data_output[:, y_offset_b:y_offset_e, x_offset_b:x_offset_e] = (
                    image_data_downsize
                )

    datapoint.images[index].data = image_data_output

    # Step 2: downsize the masks and paste them into the target grid of the mosaic
    # (note that we don't scale input/target boxes since they are not used in TA)
    for obj in datapoint.images[index].objects:
        if obj.segment is None:
            continue
        assert obj.segment.shape == (H_im, W_im) and obj.segment.dtype == torch.uint8
        segment_output = torch.zeros_like(obj.segment)

        target_y_offset_b = target_grid_y * H_im // grid_h
        target_x_offset_b = target_grid_x * W_im // grid_w
        target_y_offset_e = (target_grid_y + 1) * H_im // grid_h
        target_x_offset_e = (target_grid_x + 1) * W_im // grid_w
        target_H_im_downsize = target_y_offset_e - target_y_offset_b
        target_W_im_downsize = target_x_offset_e - target_x_offset_b

        segment_downsize = F.resize(
            obj.segment[None, None],
            size=(target_H_im_downsize, target_W_im_downsize),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,  # antialiasing for downsizing
        )[0, 0]
        if should_hflip[target_grid_y, target_grid_x].item():
            segment_downsize = F.hflip(segment_downsize[None, None])[0, 0]

        segment_output[
            target_y_offset_b:target_y_offset_e, target_x_offset_b:target_x_offset_e
        ] = segment_downsize
        obj.segment = segment_output

    return datapoint


class ScheduledPadToSizeAPI(PadToSizeAPI):
    def __init__(self, size_scheduler, consistent_transform):
        self.size_scheduler = size_scheduler
        size = self.size_scheduler(epoch_num=0)["sizes"]
        super().__init__(size, consistent_transform)

    def __call__(self, datapoint, **kwargs):
        assert "epoch" in kwargs, "Param scheduler needs to know the current epoch"
        params = self.size_scheduler(kwargs["epoch"])
        self.size = params["resolution"]
        return super(ScheduledPadToSizeAPI, self).__call__(datapoint, **kwargs)


class IdentityAPI:
    def __call__(self, datapoint, **kwargs):
        return datapoint


class RandomSelectAPI:
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1=None, transforms2=None, p=0.5):
        self.transforms1 = transforms1 or IdentityAPI()
        self.transforms2 = transforms2 or IdentityAPI()
        self.p = p

    def __call__(self, datapoint, **kwargs):
        if random.random() < self.p:
            return self.transforms1(datapoint, **kwargs)
        return self.transforms2(datapoint, **kwargs)


class ToTensorAPI:
    def __init__(self, v2=False):
        self.v2 = v2

    def __call__(self, datapoint: Datapoint, **kwargs):
        for img in datapoint.images:
            if self.v2:
                img.data = Fv2.to_image_tensor(img.data)
                # img.data = Fv2.to_dtype(img.data, torch.uint8, scale=True)
                # img.data = Fv2.convert_image_dtype(img.data, torch.uint8)
            else:
                img.data = F.to_tensor(img.data)
        return datapoint


class NormalizeAPI:
    def __init__(self, mean, std, v2=False):
        self.mean = mean
        self.std = std
        self.v2 = v2

    def __call__(self, datapoint: Datapoint, **kwargs):
        for img in datapoint.images:
            if self.v2:
                img.data = Fv2.convert_image_dtype(img.data, torch.float32)
                img.data = Fv2.normalize(img.data, mean=self.mean, std=self.std)
            else:
                img.data = F.normalize(img.data, mean=self.mean, std=self.std)
            for obj in img.objects:
                boxes = obj.bbox
                cur_h, cur_w = img.data.shape[-2:]
                boxes = box_xyxy_to_cxcywh(boxes)
                boxes = boxes / torch.tensor(
                    [cur_w, cur_h, cur_w, cur_h], dtype=torch.float32
                )
                obj.bbox = boxes

        for query in datapoint.find_queries:
            if query.input_bbox is not None:
                boxes = query.input_bbox
                cur_h, cur_w = datapoint.images[query.image_id].data.shape[-2:]
                boxes = box_xyxy_to_cxcywh(boxes)
                boxes = boxes / torch.tensor(
                    [cur_w, cur_h, cur_w, cur_h], dtype=torch.float32
                )
                query.input_bbox = boxes
            if query.input_points is not None:
                points = query.input_points
                cur_h, cur_w = datapoint.images[query.image_id].data.shape[-2:]
                points = points / torch.tensor([cur_w, cur_h, 1.0], dtype=torch.float32)
                query.input_points = points

        return datapoint


class ComposeAPI:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, datapoint, **kwargs):
        for t in self.transforms:
            datapoint = t(datapoint, **kwargs)
        return datapoint

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomGrayscale:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p
        self.consistent_transform = consistent_transform
        self.Grayscale = T.Grayscale(num_output_channels=3)

    def __call__(self, datapoint: Datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                for img in datapoint.images:
                    img.data = self.Grayscale(img.data)
            return datapoint
        for img in datapoint.images:
            if random.random() < self.p:
                img.data = self.Grayscale(img.data)
        return datapoint


class ColorJitter:
    def __init__(self, consistent_transform, brightness, contrast, saturation, hue):
        self.consistent_transform = consistent_transform
        self.brightness = (
            brightness
            if isinstance(brightness, list)
            else [max(0, 1 - brightness), 1 + brightness]
        )
        self.contrast = (
            contrast
            if isinstance(contrast, list)
            else [max(0, 1 - contrast), 1 + contrast]
        )
        self.saturation = (
            saturation
            if isinstance(saturation, list)
            else [max(0, 1 - saturation), 1 + saturation]
        )
        self.hue = hue if isinstance(hue, list) or hue is None else ([-hue, hue])

    def __call__(self, datapoint: Datapoint, **kwargs):
        if self.consistent_transform:
            # Create a color jitter transformation params
            (
                fn_idx,
                brightness_factor,
                contrast_factor,
                saturation_factor,
                hue_factor,
            ) = T.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        for img in datapoint.images:
            if not self.consistent_transform:
                (
                    fn_idx,
                    brightness_factor,
                    contrast_factor,
                    saturation_factor,
                    hue_factor,
                ) = T.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue
                )
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img.data = F.adjust_brightness(img.data, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img.data = F.adjust_contrast(img.data, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img.data = F.adjust_saturation(img.data, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img.data = F.adjust_hue(img.data, hue_factor)
        return datapoint


class RandomAffine:
    def __init__(
        self,
        degrees,
        consistent_transform,
        scale=None,
        translate=None,
        shear=None,
        image_mean=(123, 116, 103),
        log_warning=True,
        num_tentatives=1,
        image_interpolation="bicubic",
    ):
        """
        The mask is required for this transform.
        if consistent_transform if True, then the same random affine is applied to all frames and masks.
        """
        self.degrees = degrees if isinstance(degrees, list) else ([-degrees, degrees])
        self.scale = scale
        self.shear = (
            shear if isinstance(shear, list) else ([-shear, shear] if shear else None)
        )
        self.translate = translate
        self.fill_img = image_mean
        self.consistent_transform = consistent_transform
        self.log_warning = log_warning
        self.num_tentatives = num_tentatives

        if image_interpolation == "bicubic":
            self.image_interpolation = InterpolationMode.BICUBIC
        elif image_interpolation == "bilinear":
            self.image_interpolation = InterpolationMode.BILINEAR
        else:
            raise NotImplementedError

    def __call__(self, datapoint: Datapoint, **kwargs):
        for _tentative in range(self.num_tentatives):
            res = self.transform_datapoint(datapoint)
            if res is not None:
                return res

        if self.log_warning:
            logging.warning(
                f"Skip RandomAffine for zero-area mask in first frame after {self.num_tentatives} tentatives"
            )
        return datapoint

    def transform_datapoint(self, datapoint: Datapoint):
        _, height, width = F.get_dimensions(datapoint.images[0].data)
        img_size = [width, height]

        if self.consistent_transform:
            # Create a random affine transformation
            affine_params = T.RandomAffine.get_params(
                degrees=self.degrees,
                translate=self.translate,
                scale_ranges=self.scale,
                shears=self.shear,
                img_size=img_size,
            )

        for img_idx, img in enumerate(datapoint.images):
            this_masks = [
                obj.segment.unsqueeze(0) if obj.segment is not None else None
                for obj in img.objects
            ]
            if not self.consistent_transform:
                # if not consistent we create a new affine params for every frame&mask pair Create a random affine transformation
                affine_params = T.RandomAffine.get_params(
                    degrees=self.degrees,
                    translate=self.translate,
                    scale_ranges=self.scale,
                    shears=self.shear,
                    img_size=img_size,
                )

            transformed_bboxes, transformed_masks = [], []
            for i in range(len(img.objects)):
                if this_masks[i] is None:
                    transformed_masks.append(None)
                    # Dummy bbox for a dummy target
                    transformed_bboxes.append(torch.tensor([[0, 0, 0, 0]]))
                else:
                    transformed_mask = F.affine(
                        this_masks[i],
                        *affine_params,
                        interpolation=InterpolationMode.NEAREST,
                        fill=0.0,
                    )
                    if img_idx == 0 and transformed_mask.max() == 0:
                        # We are dealing with a video and the object is not visible in the first frame
                        # Return the datapoint without transformation
                        return None
                    transformed_bbox = masks_to_boxes(transformed_mask)
                    transformed_bboxes.append(transformed_bbox)
                    transformed_masks.append(transformed_mask.squeeze())

            for i in range(len(img.objects)):
                img.objects[i].bbox = transformed_bboxes[i]
                img.objects[i].segment = transformed_masks[i]

            img.data = F.affine(
                img.data,
                *affine_params,
                interpolation=self.image_interpolation,
                fill=self.fill_img,
            )
        return datapoint


class RandomResizedCrop:
    def __init__(
        self,
        consistent_transform,
        size,
        scale=None,
        ratio=None,
        log_warning=True,
        num_tentatives=4,
        keep_aspect_ratio=False,
    ):
        """
        The mask is required for this transform.
        if consistent_transform if True, then the same random resized crop is applied to all frames and masks.
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        elif isinstance(size, Sequence) and len(size) == 1:
            self.size = (size[0], size[0])
        elif len(size) != 2:
            raise ValueError("Please provide only two dimensions (h, w) for size.")
        else:
            self.size = size

        self.scale = scale if scale is not None else (0.08, 1.0)
        self.ratio = ratio if ratio is not None else (3.0 / 4.0, 4.0 / 3.0)
        self.consistent_transform = consistent_transform
        self.log_warning = log_warning
        self.num_tentatives = num_tentatives
        self.keep_aspect_ratio = keep_aspect_ratio

    def __call__(self, datapoint: Datapoint, **kwargs):
        for _tentative in range(self.num_tentatives):
            res = self.transform_datapoint(datapoint)
            if res is not None:
                return res

        if self.log_warning:
            logging.warning(
                f"Skip RandomResizeCrop for zero-area mask in first frame after {self.num_tentatives} tentatives"
            )
        return datapoint

    def transform_datapoint(self, datapoint: Datapoint):
        if self.keep_aspect_ratio:
            original_size = datapoint.images[0].size
            original_ratio = original_size[1] / original_size[0]
            ratio = [r * original_ratio for r in self.ratio]
        else:
            ratio = self.ratio

        if self.consistent_transform:
            # Create a random crop transformation
            crop_params = T.RandomResizedCrop.get_params(
                img=datapoint.images[0].data,
                scale=self.scale,
                ratio=ratio,
            )

        for img_idx, img in enumerate(datapoint.images):
            if not self.consistent_transform:
                # Create a random crop transformation
                crop_params = T.RandomResizedCrop.get_params(
                    img=img.data,
                    scale=self.scale,
                    ratio=ratio,
                )

            this_masks = [
                obj.segment.unsqueeze(0) if obj.segment is not None else None
                for obj in img.objects
            ]

            transformed_bboxes, transformed_masks = [], []
            for i in range(len(img.objects)):
                if this_masks[i] is None:
                    transformed_masks.append(None)
                    # Dummy bbox for a dummy target
                    transformed_bboxes.append(torch.tensor([[0, 0, 0, 0]]))
                else:
                    transformed_mask = F.resized_crop(
                        this_masks[i],
                        *crop_params,
                        size=self.size,
                        interpolation=InterpolationMode.NEAREST,
                    )
                    if img_idx == 0 and transformed_mask.max() == 0:
                        # We are dealing with a video and the object is not visible in the first frame
                        # Return the datapoint without transformation
                        return None
                    transformed_masks.append(transformed_mask.squeeze())
                    transformed_bbox = masks_to_boxes(transformed_mask)
                    transformed_bboxes.append(transformed_bbox)

            # Set the new boxes and masks if all transformed masks and boxes are good.
            for i in range(len(img.objects)):
                img.objects[i].bbox = transformed_bboxes[i]
                img.objects[i].segment = transformed_masks[i]

            img.data = F.resized_crop(
                img.data,
                *crop_params,
                size=self.size,
                interpolation=InterpolationMode.BILINEAR,
            )
        return datapoint


class ResizeToMaxIfAbove:
    # Resize datapoint image if one of its sides is larger that max_size
    def __init__(
        self,
        max_size=None,
    ):
        self.max_size = max_size

    def __call__(self, datapoint: Datapoint, **kwargs):
        _, height, width = F.get_dimensions(datapoint.images[0].data)

        if height <= self.max_size and width <= self.max_size:
            # The original frames are small enough
            return datapoint
        elif height >= width:
            new_height = self.max_size
            new_width = int(round(self.max_size * width / height))
        else:
            new_height = int(round(self.max_size * height / width))
            new_width = self.max_size

        size = new_height, new_width

        for index in range(len(datapoint.images)):
            datapoint.images[index].data = F.resize(datapoint.images[index].data, size)

            for obj in datapoint.images[index].objects:
                obj.segment = F.resize(
                    obj.segment[None, None],
                    size,
                    interpolation=InterpolationMode.NEAREST,
                ).squeeze()

            h, w = size
            datapoint.images[index].size = (h, w)
        return datapoint


def get_bbox_xyxy_abs_coords_from_mask(mask):
    """Get the bounding box (XYXY format w/ absolute coordinates) of a binary mask."""
    assert mask.dim() == 2
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    row_inds = rows.nonzero().view(-1)
    col_inds = cols.nonzero().view(-1)
    if row_inds.numel() == 0:
        # mask is empty
        bbox = torch.zeros(1, 4, dtype=torch.float32)
        bbox_area = 0.0
    else:
        ymin, ymax = row_inds.min(), row_inds.max()
        xmin, xmax = col_inds.min(), col_inds.max()
        bbox = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32).view(1, 4)
        bbox_area = float((ymax - ymin) * (xmax - xmin))
    return bbox, bbox_area


class MotionBlur:
    def __init__(self, kernel_size=5, consistent_transform=True, p=0.5):
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.kernel_size = kernel_size
        self.consistent_transform = consistent_transform
        self.p = p

    def __call__(self, datapoint: Datapoint, **kwargs):
        if random.random() >= self.p:
            return datapoint
        if self.consistent_transform:
            # Generate a single motion blur kernel for all images
            kernel = self._generate_motion_blur_kernel()
        for img in datapoint.images:
            if not self.consistent_transform:
                # Generate a new motion blur kernel for each image
                kernel = self._generate_motion_blur_kernel()
            img.data = self._apply_motion_blur(img.data, kernel)

        return datapoint

    def _generate_motion_blur_kernel(self):
        kernel = torch.zeros((self.kernel_size, self.kernel_size))
        direction = random.choice(["horizontal", "vertical", "diagonal"])
        if direction == "horizontal":
            kernel[self.kernel_size // 2, :] = 1.0
        elif direction == "vertical":
            kernel[:, self.kernel_size // 2] = 1.0
        elif direction == "diagonal":
            for i in range(self.kernel_size):
                kernel[i, i] = 1.0
        kernel /= kernel.sum()
        return kernel

    def _apply_motion_blur(self, image, kernel):
        if isinstance(image, PILImage.Image):
            image = F.to_tensor(image)
        channels = image.shape[0]
        kernel = kernel.to(image.device).unsqueeze(0).unsqueeze(0)
        blurred_image = torch.nn.functional.conv2d(
            image.unsqueeze(0),
            kernel.repeat(channels, 1, 1, 1),
            padding=self.kernel_size // 2,
            groups=channels,
        )
        return F.to_pil_image(blurred_image.squeeze(0))


class LargeScaleJitter:
    def __init__(
        self,
        scale_range=(0.1, 2.0),
        aspect_ratio_range=(0.75, 1.33),
        crop_size=(640, 640),
        consistent_transform=True,
        p=0.5,
    ):
        """
        Args:rack
            scale_range (tuple): Range of scaling factors (min_scale, max_scale).
            aspect_ratio_range (tuple): Range of aspect ratios (min_aspect_ratio, max_aspect_ratio).
            crop_size (tuple): Target size of the cropped region (width, height).
            consistent_transform (bool): Whether to apply the same transformation across all frames.
            p (float): Probability of applying the transformation.
        """
        self.scale_range = scale_range
        self.aspect_ratio_range = aspect_ratio_range
        self.crop_size = crop_size
        self.consistent_transform = consistent_transform
        self.p = p

    def __call__(self, datapoint: Datapoint, **kwargs):
        if random.random() >= self.p:
            return datapoint

        # Sample a single scale factor and aspect ratio for all frames
        log_ratio = torch.log(torch.tensor(self.aspect_ratio_range))
        scale_factor = torch.empty(1).uniform_(*self.scale_range).item()
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        for idx, img in enumerate(datapoint.images):
            if not self.consistent_transform:
                # Sample a new scale factor and aspect ratio for each frame
                log_ratio = torch.log(torch.tensor(self.aspect_ratio_range))
                scale_factor = torch.empty(1).uniform_(*self.scale_range).item()
                aspect_ratio = torch.exp(
                    torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
                ).item()

            # Compute the dimensions of the jittered crop
            original_width, original_height = img.data.size
            target_area = original_width * original_height * scale_factor
            crop_width = int(round((target_area * aspect_ratio) ** 0.5))
            crop_height = int(round((target_area / aspect_ratio) ** 0.5))

            # Randomly select the top-left corner of the crop
            crop_x = random.randint(0, max(0, original_width - crop_width))
            crop_y = random.randint(0, max(0, original_height - crop_height))

            # Extract the cropped region
            datapoint = crop(datapoint, idx, (crop_x, crop_y, crop_width, crop_height))

            # Resize the cropped region to the target crop size
            datapoint = resize(datapoint, idx, self.crop_size)

        return datapoint
