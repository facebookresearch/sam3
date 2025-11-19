# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from pycocotools import mask as mask_util

from sam3.train.data.sam3_image_dataset import Datapoint
from torchvision.ops import masks_to_boxes


def sample_points_from_rle(rle, n_points, mode, box=None, normalize=True):
    """
    Sample random points from a mask provided in COCO RLE format. 'mode'
    'mode' is in ["centered", "random_mask", "random_box"]
      "centered": points are sampled farthest from the mask edges and each other
      "random_mask": points are sampled uniformly from the mask
      "random_box": points are sampled uniformly from the annotation's box
    'box' must be provided if 'mode' is "random_box".
    If 'normalize' is true, points are in [0,1], relative to mask h,w.
    """
    mask = np.ascontiguousarray(mask_util.decode(rle))
    points = sample_points_from_mask(mask, n_points, mode, box)

    if normalize:
        h, w = mask.shape
        norm = np.array([w, h, 1.0])[None, :]
        points = points / norm

    return points


def sample_points_from_mask(mask, n_points, mode, box=None):
    if mode == "centered":
        points = center_positive_sample(mask, n_points)
    elif mode == "random_mask":
        points = uniform_positive_sample(mask, n_points)
    elif mode == "random_box":
        assert box is not None, "'random_box' mode requires a provided box."
        points = uniform_sample_from_box(mask, box, n_points)
    else:
        raise ValueError(f"Unknown point sampling mode {mode}.")
    return points


def uniform_positive_sample(mask, n_points):
    """
    Samples positive points uniformly from the mask. Only integer pixel
    values are sampled.
    """
    # Sampling directly from the uncompressed RLE would be faster but is
    # likely unnecessary.
    mask_points = np.stack(np.nonzero(mask), axis=0).transpose(1, 0)
    assert len(mask_points) > 0, "Can't sample positive points from an empty mask."
    selected_idxs = np.random.randint(low=0, high=len(mask_points), size=n_points)
    selected_points = mask_points[selected_idxs]

    selected_points = selected_points[:, ::-1]  # (y, x) -> (x, y)
    labels = np.ones((len(selected_points), 1))
    selected_points = np.concatenate([selected_points, labels], axis=1)

    return selected_points


def center_positive_sample(mask, n_points):
    """
    Samples points farthest from mask edges (by distance transform)
    and subsequent points also farthest from each other. Each new point
    sampled is treated as an edge for future points. Edges of the image are
    treated as edges of the mask.
    """

    # Pad mask by one pixel on each end to assure distance transform
    # avoids edges
    padded_mask = np.pad(mask, 1)

    points = []
    for _ in range(n_points):
        assert np.max(mask) > 0, "Can't sample positive points from an empty mask."
        dist = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 0)
        point = np.unravel_index(dist.argmax(), dist.shape)
        # Mark selected point as background so next point avoids it
        padded_mask[point[0], point[1]] = 0
        points.append(point[::-1])  # (y, x) -> (x, y)

    points = np.stack(points, axis=0)
    points = points - 1  # Subtract left/top padding of 1
    labels = np.ones((len(points), 1))
    points = np.concatenate([points, labels], axis=1)

    return points


def uniform_sample_from_box(mask, box, n_points):
    """
    Sample points uniformly from the provided box. The points' labels
    are determined by the provided mask. Does not guarantee a positive
    point is sampled. The box is assumed unnormalized in XYXY format.
    Points are sampled at integer values.
    """

    # Since lower/right edges are exclusive, ceil can be applied to all edges
    int_box = np.ceil(box)

    x = np.random.randint(low=int_box[0], high=int_box[2], size=n_points)
    y = np.random.randint(low=int_box[1], high=int_box[3], size=n_points)
    labels = mask[y, x]
    points = np.stack([x, y, labels], axis=1)

    return points


def rescale_box_xyxy(box, factor, imsize=None):
    """
    Rescale a box providing in unnormalized XYXY format, fixing the center.
    If imsize is provided, clamp to the image.
    """
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = box[2] - box[0], box[3] - box[1]

    new_w, new_h = factor * w, factor * h

    new_x0, new_y0 = cx - new_w / 2, cy - new_h / 2
    new_x1, new_y1 = cx + new_w / 2, cy + new_h / 2

    if imsize is not None:
        new_x0 = max(min(new_x0, imsize[1]), 0)
        new_x1 = max(min(new_x1, imsize[1]), 0)
        new_y0 = max(min(new_y0, imsize[0]), 0)
        new_y1 = max(min(new_y1, imsize[0]), 0)

    return [new_x0, new_y0, new_x1, new_y1]


def noise_box(box, im_size, box_noise_std, box_noise_max, min_box_area):
    if box_noise_std <= 0.0:
        return box
    noise = box_noise_std * torch.randn(size=(4,))
    w, h = box[2] - box[0], box[3] - box[1]
    scale_factor = torch.tensor([w, h, w, h])
    noise = noise * scale_factor
    if box_noise_max is not None:
        noise = torch.clamp(noise, -box_noise_max, box_noise_max)
    input_box = box + noise
    # Clamp to maximum image size
    img_clamp = torch.tensor([im_size[1], im_size[0], im_size[1], im_size[0]])
    input_box = torch.maximum(input_box, torch.zeros_like(input_box))
    input_box = torch.minimum(input_box, img_clamp)
    if (input_box[2] - input_box[0]) * (input_box[3] - input_box[1]) <= min_box_area:
        return box

    return input_box


class RandomGeometricInputsAPI:
    """
    For geometric queries, replaces the input box or points with a random
    one sampled from the GT mask. Segments must be provided for objects
    that are targets of geometric queries, and must be binary masks. Existing
    point and box queries in the datapoint will be ignored and completely replaced.
    Will sample points and boxes in XYXY format in absolute pixel space.

    Geometry queries are currently determined by taking any query whose
    query text is a set value.

    Args:
      num_points (int or (int, int)): how many points to sample. If a tuple,
        sample a random number of points uniformly over the inclusive range.
      box_chance (float): fraction of time a box is sampled. A box will replace
        one sampled point.
      box_noise_std (float): if greater than 0, add noise to the sampled boxes
        with this std. Noise is relative to the length of the box side.
      box_noise_max (int): if not none, truncate any box noise larger than this
        in terms of absolute pixels.
      resample_box_from_mask (bool): if True, any sampled box will be determined
        by finding the extrema of the provided mask. If False, the bbox provided
        in the target object will be used.
      point_sample_mode (str): In ["centered", "random_mask", "random_box"],
        controlling how points are sampled:
          "centered": points are sampled farthest from the mask edges and each other
          "random_mask": points are sampled uniformly from the mask
          "random_box": points are sampled uniformly from the annotation's box
        Note that "centered" may be too slow for on-line generation.
      geometric_query_str (str): what string in query_text indicates a
        geometry query.
      minimum_box_area (float): sampled boxes with area this size or smaller after
        noising will use the original box instead. It is the input's responsibility
        to avoid original boxes that violate necessary area bounds.
      concat_points (bool): if True, any sampled points will be added to existing
        ones instead of replacing them.

    """

    def __init__(
        self,
        num_points,
        box_chance,
        box_noise_std=0.0,
        box_noise_max=None,
        minimum_box_area=0.0,
        resample_box_from_mask=False,
        point_sample_mode="random_mask",
        sample_box_scale_factor=1.0,
        geometric_query_str="geometric",
        concat_points=False,
    ):
        self.num_points = num_points
        if not isinstance(self.num_points, int):
            # Convert from inclusive range to exclusive range expected by torch
            self.num_points[1] += 1
            self.num_points = tuple(self.num_points)
        self.box_chance = box_chance
        self.box_noise_std = box_noise_std
        self.box_noise_max = box_noise_max
        self.minimum_box_area = minimum_box_area
        self.resample_box_from_mask = resample_box_from_mask
        self.point_sample_mode = point_sample_mode
        assert point_sample_mode in [
            "centered",
            "random_mask",
            "random_box",
        ], "Unknown point sample mode."
        self.geometric_query_str = geometric_query_str
        self.concat_points = concat_points
        self.sample_box_scale_factor = sample_box_scale_factor

    def _sample_num_points_and_if_box(self):
        if isinstance(self.num_points, tuple):
            n_points = torch.randint(
                low=self.num_points[0], high=self.num_points[1], size=(1,)
            ).item()
        else:
            n_points = self.num_points
        if self.box_chance > 0.0:
            use_box = torch.rand(size=(1,)).item() < self.box_chance
            n_points -= int(use_box)  # box stands in for one point
        else:
            use_box = False
        return n_points, use_box

    def _get_original_box(self, target_object):
        if not self.resample_box_from_mask:
            return target_object.bbox
        mask = target_object.segment
        return masks_to_boxes(mask[None, :, :])[0]

    def _get_target_object(self, datapoint, query):
        img = datapoint.images[query.image_id]
        targets = query.object_ids_output
        assert (
            len(targets) == 1
        ), "Geometric queries only support a single target object."
        target_idx = targets[0]
        return img.objects[target_idx]

    def __call__(self, datapoint, **kwargs):
        for query in datapoint.find_queries:
            if query.query_text != self.geometric_query_str:
                continue

            target_object = self._get_target_object(datapoint, query)
            n_points, use_box = self._sample_num_points_and_if_box()
            box = self._get_original_box(target_object)

            mask = target_object.segment
            if n_points > 0:
                # FIXME: The conversion to numpy and back to reuse code
                # is awkward, but this is all in the dataloader worker anyway
                # on CPU and so I don't think it should matter.
                if self.sample_box_scale_factor != 1.0:
                    sample_box = rescale_box_xyxy(
                        box.numpy(), self.sample_box_scale_factor, mask.shape
                    )
                else:
                    sample_box = box.numpy()
                input_points = sample_points_from_mask(
                    mask.numpy(),
                    n_points,
                    self.point_sample_mode,
                    sample_box,
                )
                input_points = torch.as_tensor(input_points)
                input_points = input_points[None, :, :]
                if self.concat_points and query.input_points is not None:
                    input_points = torch.cat([query.input_points, input_points], dim=1)
            else:
                input_points = query.input_points if self.concat_points else None

            if use_box:
                w, h = datapoint.images[query.image_id].size
                input_box = noise_box(
                    box,
                    (h, w),
                    box_noise_std=self.box_noise_std,
                    box_noise_max=self.box_noise_max,
                    min_box_area=self.minimum_box_area,
                )
                input_box = input_box[None, :]
            else:
                input_box = query.input_bbox if self.concat_points else None

            query.input_points = input_points
            query.input_bbox = input_box

        return datapoint


class RandomizeInputBbox:
    """
    Simplified version of the geometric transform that only deals with input boxes
    """

    def __init__(
        self,
        box_noise_std=0.0,
        box_noise_max=None,
        minimum_box_area=0.0,
    ):
        self.box_noise_std = box_noise_std
        self.box_noise_max = box_noise_max
        self.minimum_box_area = minimum_box_area

    def __call__(self, datapoint: Datapoint, **kwargs):
        for query in datapoint.find_queries:
            if query.input_bbox is None:
                continue

            img = datapoint.images[query.image_id].data
            if isinstance(img, PILImage.Image):
                w, h = img.size
            else:
                assert isinstance(img, torch.Tensor)
                h, w = img.shape[-2:]

            for box_id in range(query.input_bbox.shape[0]):
                query.input_bbox[box_id, :] = noise_box(
                    query.input_bbox[box_id, :].view(4),
                    (h, w),
                    box_noise_std=self.box_noise_std,
                    box_noise_max=self.box_noise_max,
                    min_box_area=self.minimum_box_area,
                ).view(1, 4)

        return datapoint
