# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

import contextlib
import logging
import os
import queue
import re
import time
from abc import ABC, abstractmethod
from io import BytesIO
from threading import Condition, get_ident, Lock, Thread
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from iopath.common.file_io import g_pathmgr
from numpy.typing import NDArray

from PIL import Image
from tqdm import tqdm

from .edt import edt_triton

IS_MAIN_PROCESS = os.getenv("IS_MAIN_PROCESS", "1") == "1"
RANK = int(os.getenv("RANK", "0"))


def sample_box_points(
    masks: torch.Tensor,
    noise: float = 0.1,  # SAM default
    noise_bound: int = 20,  # SAM default
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> Tuple[np.array, np.array]:
    """
    Sample a noised version of the top left and bottom right corners of a given `bbox`

    Inputs:
    - masks: [B, 1, H,W] boxes, dtype=torch.Tensor
    - noise: noise as a fraction of box width and height, dtype=float
    - noise_bound: maximum amount of noise (in pure pixesl), dtype=int

    Returns:
    - box_coords: [B, num_pt, 2], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.float
    - box_labels: [B, num_pt], label 2 is reserverd for top left and 3 for bottom right corners, dtype=torch.int32
    """
    device = masks.device
    box_coords = mask_to_box(masks)
    B, _, H, W = masks.shape
    box_labels = torch.tensor(
        [top_left_label, bottom_right_label], dtype=torch.int, device=device
    ).repeat(B)
    if noise > 0.0:
        if not isinstance(noise_bound, torch.Tensor):
            noise_bound = torch.tensor(noise_bound, device=device)
        bbox_w = box_coords[..., 2] - box_coords[..., 0]
        bbox_h = box_coords[..., 3] - box_coords[..., 1]
        max_dx = torch.min(bbox_w * noise, noise_bound)
        max_dy = torch.min(bbox_h * noise, noise_bound)
        box_noise = 2 * torch.rand(B, 1, 4, device=device) - 1
        box_noise = box_noise * torch.stack((max_dx, max_dy, max_dx, max_dy), dim=-1)

        box_coords = box_coords + box_noise
        img_bounds = (
            torch.tensor([W, H, W, H], device=device) - 1
        )  # uncentered pixel coords
        box_coords.clamp_(torch.zeros_like(img_bounds), img_bounds)  # In place clamping

    box_coords = box_coords.reshape(-1, 2, 2)  # always 2 points
    box_labels = box_labels.reshape(-1, 2)
    return box_coords, box_labels


def mask_to_box(masks: torch.Tensor):
    """
    compute bounding box given an input mask

    Inputs:
    - masks: [B, 1, H,W] boxes, dtype=torch.Tensor

    Returns:
    - box_coords: [B, 1, 4], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.Tensor
    """
    B, _, h, w = masks.shape
    device = masks.device
    mask_area = masks.sum(dim=(-1, -2))
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    bbox_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)
    bbox_coords = torch.where(
        mask_area[..., None] > 0, bbox_coords, torch.zeros_like(bbox_coords)
    )
    return bbox_coords


def sample_random_points_from_errors(gt_masks, pred_masks, num_pt=1):
    """
    Sample `num_pt` random points (along with their labels) independently from the error regions.

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - num_pt: int, number of points to sample independently for each of the B error maps

    Outputs:
    - points: [B, num_pt, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, num_pt], dtype=torch.int32, where 1 means positive clicks and 0 means
      negative clicks
    """
    if pred_masks is None:  # if pred_masks is not provided, treat it as empty
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape
    assert num_pt >= 0

    B, _, H_im, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks
    # whether the prediction completely match the ground-truth on each mask
    all_correct = torch.all((gt_masks == pred_masks).flatten(2), dim=2)
    all_correct = all_correct[..., None, None]

    # channel 0 is FP map, while channel 1 is FN map
    pts_noise = torch.rand(B, num_pt, H_im, W_im, 2, device=device)
    # sample a negative new click from FP region or a positive new click
    # from FN region, depend on where the maximum falls,
    # and in case the predictions are all correct (no FP or FN), we just
    # sample a negative click from the background region
    pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_masks)
    pts_noise[..., 1] *= fn_masks
    pts_idx = pts_noise.flatten(2).argmax(dim=2)
    labels = (pts_idx % 2).to(torch.int32)
    pts_idx = pts_idx // 2
    pts_x = pts_idx % W_im
    pts_y = pts_idx // W_im
    points = torch.stack([pts_x, pts_y], dim=2).to(torch.float)
    return points, labels


def sample_one_point_from_error_center(gt_masks, pred_masks, padding=True):
    """
    Sample 1 random point (along with its label) from the center of each error region,
    that is, the point with the largest distance to the boundary of each error region.
    This is the RITM sampling method from https://github.com/saic-vul/ritm_interactive_segmentation/blob/master/isegm/inference/clicker.py

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - padding: if True, pad with boundary of 1 px for distance transform

    Outputs:
    - points: [B, 1, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, 1], dtype=torch.int32, where 1 means positive clicks and 0 means negative clicks
    """
    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape

    B, _, H, W = gt_masks.shape

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = (~gt_masks & pred_masks).squeeze(1)
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = (gt_masks & ~pred_masks).squeeze(1)

    if padding:
        padded_fp_masks = torch.zeros(
            B, H + 2, W + 2, dtype=fp_masks.dtype, device=fp_masks.device
        )
        padded_fp_masks[:, 1 : H + 1, 1 : W + 1] = fp_masks
        padded_fn_masks = torch.zeros(
            B, H + 2, W + 2, dtype=fp_masks.dtype, device=fp_masks.device
        )
        padded_fn_masks[:, 1 : H + 1, 1 : W + 1] = fn_masks
    else:
        padded_fp_masks = fp_masks
        padded_fn_masks = fn_masks

    fn_mask_dt = edt_triton(padded_fn_masks)
    fp_mask_dt = edt_triton(padded_fp_masks)
    if padding:
        fn_mask_dt = fn_mask_dt[:, 1:-1, 1:-1]
        fp_mask_dt = fp_mask_dt[:, 1:-1, 1:-1]

    fn_max, fn_argmax = fn_mask_dt.reshape(B, -1).max(dim=-1)
    fp_max, fp_argmax = fp_mask_dt.reshape(B, -1).max(dim=-1)
    is_positive = fn_max > fp_max
    chosen = torch.where(is_positive, fn_argmax, fp_argmax)
    points_x = chosen % W
    points_y = chosen // W

    labels = is_positive.long()
    points = torch.stack([points_x, points_y], -1)
    return points.unsqueeze(1), labels.unsqueeze(1)


def sample_one_point_from_error_center_slow(gt_masks, pred_masks, padding=True):
    """
    Sample 1 random point (along with its label) from the center of each error region,
    that is, the point with the largest distance to the boundary of each error region.
    This is the RITM sampling method from https://github.com/saic-vul/ritm_interactive_segmentation/blob/master/isegm/inference/clicker.py

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - padding: if True, pad with boundary of 1 px for distance transform

    Outputs:
    - points: [B, 1, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, 1], dtype=torch.int32, where 1 means positive clicks and 0 means negative clicks
    """
    import cv2  # delay OpenCV import to avoid unnecessary dependency

    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape

    B, _, _, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks

    fp_masks = fp_masks.cpu().numpy()
    fn_masks = fn_masks.cpu().numpy()
    points = torch.zeros(B, 1, 2, dtype=torch.float)
    labels = torch.ones(B, 1, dtype=torch.int32)
    for b in range(B):
        fn_mask = fn_masks[b, 0]
        fp_mask = fp_masks[b, 0]
        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")
        # compute the distance of each point in FN/FP region to its boundary
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        # take the point in FN/FP region with the largest distance to its boundary
        fn_mask_dt_flat = fn_mask_dt.reshape(-1)
        fp_mask_dt_flat = fp_mask_dt.reshape(-1)
        fn_argmax = np.argmax(fn_mask_dt_flat)
        fp_argmax = np.argmax(fp_mask_dt_flat)
        is_positive = fn_mask_dt_flat[fn_argmax] > fp_mask_dt_flat[fp_argmax]
        pt_idx = fn_argmax if is_positive else fp_argmax
        points[b, 0, 0] = pt_idx % W_im  # x
        points[b, 0, 1] = pt_idx // W_im  # y
        labels[b, 0] = int(is_positive)

    points = points.to(device)
    labels = labels.to(device)
    return points, labels


def get_next_point(gt_masks, pred_masks, method):
    if method == "uniform":
        return sample_random_points_from_errors(gt_masks, pred_masks)
    elif method == "center":
        return sample_one_point_from_error_center(gt_masks, pred_masks)
    else:
        raise ValueError(f"unknown sampling method {method}")


def select_closest_cond_frames(
    frame_idx, cond_frame_outputs, max_cond_frame_num, keep_first_cond_frame=False
):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    that are temporally closest to the current frame at `frame_idx`. Here, we take
    - a) the closest conditioning frame before `frame_idx` (if any);
    - b) the closest conditioning frame after `frame_idx` (if any);
    - c) any other temporally closest conditioning frames until reaching a total
         of `max_cond_frame_num` conditioning frames.

    Outputs:
    - selected_outputs: selected items (keys & values) from `cond_frame_outputs`.
    - unselected_outputs: items (keys & values) not selected in `cond_frame_outputs`.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}
        if keep_first_cond_frame:
            idx_first = min(
                (t for t in cond_frame_outputs if t < frame_idx), default=None
            )
            if idx_first is None:
                # Maybe we are tracking in reverse
                idx_first = max(
                    (t for t in cond_frame_outputs if t > frame_idx), default=None
                )
            if idx_first is not None:
                selected_outputs[idx_first] = cond_frame_outputs[idx_first]
        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    Get 1D sine positional embedding as in the original Transformer paper.
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def get_best_gt_match_from_multimasks(pred_multimasks, gt_masks, pred_scores=None):
    """
    Get the mask with the best match to GT masks (based on IoU) from pred_multimasks.
    Optionally, use `pred_scores` to break ties in case all IoUs are zeros.
    """
    assert pred_multimasks.ndim == 4 and gt_masks.ndim == 4
    if pred_multimasks.size(1) == 1:
        return pred_multimasks  # only a single mask channel, nothing to select

    pred_multimasks_binary = pred_multimasks > 0
    area_i = torch.sum(pred_multimasks_binary & gt_masks, dim=(2, 3)).float()
    area_u = torch.sum(pred_multimasks_binary | gt_masks, dim=(2, 3)).float()
    ious = area_i / torch.clamp(area_u, min=1.0)

    # In case all IoUs are zeros (e.g. because the GT mask is empty), use pred_scores
    # to break ties and select the best mask
    if pred_scores is not None:
        has_nonzero_ious = torch.any(ious > 0).expand_as(ious)
        scores = torch.where(has_nonzero_ious, ious, pred_scores)
    else:
        scores = ious

    # Finally, take the best mask prediction (with the highest score)
    best_scores_inds = torch.argmax(scores, dim=-1)
    batch_inds = torch.arange(scores.size(0), device=scores.device)
    best_pred_mask = pred_multimasks[batch_inds, best_scores_inds].unsqueeze(1)
    return best_pred_mask


class AsyncVideoFrameLoader:
    """
    A list of video frames to be load asynchronously without blocking session start.
    """

    def __init__(self, img_paths, image_size, offload_video_to_cpu, img_mean, img_std):
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        # items in `self._images` will be loaded asynchronously
        self.images = [None] * len(img_paths)
        # catch and raise any exceptions in the async loading thread
        self.exception = None
        # video_height and video_width be filled when loading the first image
        self.video_height = None
        self.video_width = None

        # load the first frame to fill video_height and video_width and also
        # to cache it (since it's most likely where the user will click)
        self.__getitem__(0)

        # load the rest of frames asynchronously without blocking the session start
        def _load_frames():
            try:
                for n in tqdm(
                    range(len(self.images)),
                    desc=f"frame loading (JPEG) [rank={RANK}]",
                ):
                    self.__getitem__(n)
            except Exception as e:
                self.exception = e

        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

        img = self.images[index]
        if img is not None:
            return img

        img, video_height, video_width = _load_img_as_tensor(
            self.img_paths[index], self.image_size
        )
        self.video_height = video_height
        self.video_width = video_width
        # float16 precision should be sufficient for image tensor storage
        img = img.to(dtype=torch.float16)
        # normalize by mean and std
        img -= self.img_mean
        img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img.cuda()
        self.images[index] = img
        return img

    def __len__(self):
        return len(self.images)


class AsyncVideoFileLoader:
    """
    Loading frames from video files asynchronously without blocking session start.
    """

    def __init__(
        self,
        video_path_or_bytes,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        gpu_acceleration=True,
        gpu_device=None,
    ):
        from torchaudio.io import StreamReader
        from torchaudio.utils import ffmpeg_utils

        # Check whether GPU-accelerated decoding is available (and fall back to CPU decoding)
        if gpu_acceleration and "h264_cuvid" not in ffmpeg_utils.get_video_decoders():
            logging.warning(
                "Falling back to CPU video decoding as h264_cuvid is not available in ffmpeg. "
                "To use GPU-accelerated decoding, please compile and install ffmpeg with "
                "--enable-nvdec (see https://pytorch.org/audio/2.3.0/build.ffmpeg.html)."
            )
            gpu_acceleration = False
        # Check and possibly infer the output device (and also get its GPU id when applicable)
        assert gpu_device is None or gpu_device.type == "cuda"
        gpu_id = (gpu_device.index or 0) if gpu_device is not None else 0
        if offload_video_to_cpu:
            out_device = torch.device("cpu")
        else:
            out_device = torch.device("cuda") if gpu_device is None else gpu_device
        self.out_device = out_device
        self.gpu_acceleration = gpu_acceleration
        self.gpu_id = gpu_id
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        if not isinstance(img_mean, torch.Tensor):
            img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
        self.img_mean = img_mean
        if not isinstance(img_std, torch.Tensor):
            img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]
        self.img_std = img_std

        # see https://pytorch.org/audio/2.2.0/generated/torio.io.StreamingMediaDecoder.html#add-basic-video-stream
        if gpu_acceleration:
            self.img_mean = self.img_mean.to(f"cuda:{self.gpu_id}")
            self.img_std = self.img_std.to(f"cuda:{self.gpu_id}")
            stream_options = {
                "frames_per_chunk": -1,
                "buffer_chunk_size": -1,
                "decoder": "h264_cuvid",  # NVDEC for GPU-accelerated MP4 decoding
                "hw_accel": f"cuda:{self.gpu_id}",
                "decoder_option": {
                    "gpu": f"{self.gpu_id}",
                    "resize": f"{image_size}x{image_size}",
                },
            }
        else:
            self.img_mean = self.img_mean.cpu()
            self.img_std = self.img_std.cpu()
            stream_options = {
                "frames_per_chunk": -1,
                "buffer_chunk_size": -1,
                "decoder": "h264",
                "filter_desc": f"scale={image_size}:{image_size}:sws_flags=bicubic",
                "decoder_option": {"threads": "4"},  # 4 threads seems good enough
            }
        # An asynchronous reader to load frames in a background thread, and also
        # a synchronous reader to randomly seek to any single frame (in case the
        # async reader haven't reached this frame yet).
        if isinstance(video_path_or_bytes, str):
            # local file path, manifold path, or S3 path in this case
            f_async = g_pathmgr.open(video_path_or_bytes, "rb")
            self.async_reader = StreamReader(f_async)  # f_async needs to remain open
            # only open sync_reader if the file object supports `seek` method
            if hasattr(f_async, "seek"):
                f_sync = g_pathmgr.open(video_path_or_bytes, "rb")
                self.sync_reader = StreamReader(f_sync)  # f_sync needs to remain open
            else:
                logging.warning(
                    f"{video_path_or_bytes} doesn't support random file seeking, so "
                    f"accessing a random frame (other than the 1st frame) will be "
                    f"blocked until the entire video is loaded."
                )
                self.sync_reader = None
        elif isinstance(video_path_or_bytes, bytes):
            # raw video bytes in this case
            self.async_reader = StreamReader(BytesIO(video_path_or_bytes))
            self.sync_reader = StreamReader(BytesIO(video_path_or_bytes))
        else:
            raise RuntimeError(f"unsupported video type: {type(video_path_or_bytes)}")
        self.async_reader.add_video_stream(**stream_options)
        if self.sync_reader is not None:
            self.sync_reader.add_video_stream(**stream_options)
        video_info = self.async_reader.get_src_stream_info(0)
        if not (video_info.codec == "h264" and video_info.format.startswith("yuv")):
            raise RuntimeError("only MP4 videos in H.264 format is supported")
        if not video_info.num_frames > 0:
            # The video has invalid frame numbers in its metadata, which should be rare.
            # We can handle it in principle by falling back to synchronized loading,
            # but it's better to transcode all videos to a standard format instead.
            raise RuntimeError("video metadata does not have valid frame number info")
        self.num_frames = video_info.num_frames
        self.video_height = video_info.height
        self.video_width = video_info.width
        self.fps = video_info.frame_rate
        # items in `self._images` will be loaded asynchronously
        self.images = [None] * self.num_frames
        # catch and raise any exceptions in the async loading thread
        self.exception = None
        self._start_video_loading()

    def _start_video_loading(self):
        self.num_loaded_frames = 0
        desc = f"frame loading (MP4 w/ {'GPU' if self.gpu_acceleration else 'CPU'}) [rank={RANK}]"
        self.pbar = tqdm(desc=desc, total=self.num_frames)
        # load the first frame to cache it before the session is opened
        # (since it's most likely where the user will click)
        loaded_num, _ = self._load_chunk(self.async_reader, 0)
        self.num_loaded_frames += loaded_num
        self.pbar.update(n=loaded_num)
        self.all_frames_loaded = self.num_loaded_frames == self.num_frames

        # load the frames asynchronously without blocking the session start
        def _load_frames():
            try:
                finished = self.all_frames_loaded
                while not finished:
                    loaded_num, finished = self._load_chunk(
                        self.async_reader, self.num_loaded_frames
                    )
                    self.num_loaded_frames += loaded_num
                    self.pbar.update(n=loaded_num)

                # finished -- check whether we have loaded the total number of frames
                if self.num_loaded_frames < self.num_frames:
                    raise RuntimeError(
                        f"There are {self.num_frames} frames in the video, but only "
                        f"{self.num_loaded_frames} frames can be loaded successfully."
                    )
                else:
                    self.all_frames_loaded = True
                    self.pbar.close()
                    # all the video frames have been loaded, so we can release async_reader
                    # also remove pbar and thread (which shouldn't be a part of session saving)
                    self.async_reader = None
                    self.pbar = None
                    self.thread = None
            except Exception as e:
                self.exception = e

        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()

    def _load_chunk(
        self, reader, begin_frame_idx, fill_one_frame_only=False, overwrite=True
    ):
        loaded_num = 0
        finished = reader.fill_buffer()
        (frames,) = reader.pop_chunks()
        if frames is not None:
            loaded_num = frames.size(0)
            if fill_one_frame_only:
                # load one frame (mostly for the case of random seeking with sync_reader)
                frames = frames[:1]
            # Videos are encoded in H.264 YUV color space, so we must convert it to RGB
            # first (and we also subtract mean and divide by std in this conversion).
            frames = _yuv_to_rgb(frames, self.img_mean, self.img_std)
            # store the frame tensors
            for n, img in enumerate(frames):
                if self.offload_video_to_cpu:
                    img = img.cpu()
                elif img.device != self.out_device:
                    img = img.to(device=self.out_device, non_blocking=True)
                frame_idx = begin_frame_idx + n
                if frame_idx < self.num_frames and (
                    overwrite or self.images[frame_idx] is None
                ):
                    self.images[frame_idx] = img

        return loaded_num, finished

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

        img = self.images[index]
        if img is not None:
            if self.all_frames_loaded:
                # async_reader has finished loading all frames, so we also release
                # sync_reader (to close its underlying file object)
                self.sync_reader = None
            return img

        sync_reader = self.sync_reader
        if sync_reader is None:
            # We didn't open sync_reader because the file object doesn't support "seek".
            # method. In this case, there is nothing we can do other than just waiting
            # for async_reader to finish loading the video first (usually fast enough).
            async_thread = self.thread
            if async_thread is not None:
                # wait for the async loading to finish and try again
                async_thread.join()
            return self.__getitem__(index)

        # async_reader hasn't reached this frame yet, and we have an opened sync_reader,
        # so we use sync_reader to seek to the corresponding frame and load just one frame.
        if index < 0:
            index += self.num_frames
        assert index >= 0 and index < self.num_frames
        sync_reader.seek(timestamp=index / self.fps, mode="precise")
        # There could be potential racing between sync_reader and async_reader, and we use
        # `overwrite=False` to keep async_reader's results if it has just reached this frame.
        loaded_num, _ = self._load_chunk(
            sync_reader, index, fill_one_frame_only=True, overwrite=False
        )
        assert loaded_num == 1
        return self.images[index]

    def __len__(self):
        return len(self.images)

    def __getstate__(self):
        """
        Remove a few attributes during pickling, so that this async video loader can be
        saved and loaded as a part of the model session.
        """
        # wait for async video loading to finish before pickling
        async_thread = self.thread
        if async_thread is not None:
            async_thread.join()
        # release a few objects that cannot be pickled
        self.async_reader = None
        self.sync_reader = None
        self.pbar = None
        self.thread = None
        return self.__dict__.copy()


class TorchCodecDecoder:
    """
    A wrapper to support GPU device and num_threads in TorchCodec decoder,
    which are not supported by `torchcodec.decoders.SimpleVideoDecoder` yet.
    """

    def __init__(self, source, dimension_order="NCHW", device="cpu", num_threads=1):
        from torchcodec import _core as core

        self._source = source  # hold a reference to the source to prevent it from GC
        if isinstance(source, str):
            self._decoder = core.create_from_file(source, "exact")
        elif isinstance(source, bytes):
            self._decoder = core.create_from_bytes(source, "exact")
        else:
            raise TypeError(f"Unknown source type: {type(source)}.")
        assert dimension_order in ("NCHW", "NHWC")

        device_string = str(device)
        core.scan_all_streams_to_update_metadata(self._decoder)
        core.add_video_stream(
            self._decoder,
            dimension_order=dimension_order,
            device=device_string,
            num_threads=(1 if "cuda" in device_string else num_threads),
        )
        video_metadata = core.get_container_metadata(self._decoder)
        best_stream_index = video_metadata.best_video_stream_index
        assert best_stream_index is not None
        self.metadata = video_metadata.streams[best_stream_index]
        assert self.metadata.num_frames_from_content is not None
        self._num_frames = self.metadata.num_frames_from_content

    def __len__(self) -> int:
        return self._num_frames

    def __getitem__(self, key: int):
        from torchcodec import _core as core

        if key < 0:
            key += self._num_frames
        if key >= self._num_frames or key < 0:
            raise IndexError(
                f"Index {key} is out of bounds; length is {self._num_frames}"
            )
        frame_data, *_ = core.get_frame_at_index(
            self._decoder,
            frame_index=key,
        )
        return frame_data


class FIFOLock:
    """A lock that ensures FIFO ordering of lock acquisitions."""

    def __init__(self):
        self._lock = Lock()
        self._waiters = queue.Queue()
        self._condition = Condition()

    def acquire(self):
        ident = get_ident()
        with self._condition:
            self._waiters.put(ident)
            while self._waiters.queue[0] != ident or not self._lock.acquire(
                blocking=False
            ):
                self._condition.wait()
                # got the lock and it's our turn

    def release(self):
        with self._condition:
            self._lock.release()
            self._waiters.get()
            self._condition.notify_all()

    def __enter__(self):
        self.acquire()

    def __exit__(self, t, v, tb):
        self.release()


class AsyncVideoFileLoaderWithTorchCodec:
    """
    Loading frames from video files asynchronously without blocking session start.

    Unlike `AsyncVideoFileLoader`, this class uses PyTorch's offical TorchCodec library
    for video decoding, which is more efficient and supports more video formats.
    """

    def __init__(
        self,
        video_path_or_bytes,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        gpu_acceleration=True,
        gpu_device=torch.device("cuda:0"),
        use_rand_seek_in_loading=False,
        load_frames_in_rank0_only=None,  # if None, will be inferred automatically
    ):
        # Check and possibly infer the output device (and also get its GPU id when applicable)
        assert gpu_device is None or gpu_device.type == "cuda"
        gpu_id = (gpu_device.index or 0) if gpu_device is not None else 0
        if offload_video_to_cpu:
            out_device = torch.device("cpu")
        else:
            out_device = torch.device("cuda") if gpu_device is None else gpu_device
        self.out_device = out_device
        self.gpu_acceleration = gpu_acceleration
        self.gpu_id = gpu_id
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        if not isinstance(img_mean, torch.Tensor):
            img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
        self.img_mean = img_mean
        if not isinstance(img_std, torch.Tensor):
            img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]
        self.img_std = img_std

        if gpu_acceleration:
            self.img_mean = self.img_mean.to(f"cuda:{self.gpu_id}")
            self.img_std = self.img_std.to(f"cuda:{self.gpu_id}")
            decoder_option = {"device": f"cuda:{self.gpu_id}"}
        else:
            self.img_mean = self.img_mean.cpu()
            self.img_std = self.img_std.cpu()
            decoder_option = {"num_threads": 1}  # use a single thread to save memory

        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if load_frames_in_rank0_only is None:
            # if "load_frames_in_rank0_only" is not specified, we load frames
            # in rank 0 only when running distributedly and without GPU acceleration
            load_frames_in_rank0_only = (
                torch.distributed.is_initialized() and not gpu_acceleration
            )
        self.load_frames_in_rank0_only = load_frames_in_rank0_only
        if self.rank == 0 or not self.load_frames_in_rank0_only:
            self.async_reader = TorchCodecDecoder(video_path_or_bytes, **decoder_option)

            # `num_frames_from_content` is the true number of frames in the video content
            # from the scan operation (rather than from the metadata, which could be wrong)
            self.num_frames = self.async_reader.metadata.num_frames_from_content
            self.video_height = self.async_reader.metadata.height
            self.video_width = self.async_reader.metadata.width
        else:
            self.async_reader = None

        # broadcast video info from rank 0 to all other ranks
        if self.world_size > 1 and self.load_frames_in_rank0_only:
            logging.info(f"{self.rank=} video will be loaded on rank 0 only")
            buffer = torch.zeros(3, dtype=torch.int32, device="cuda")
            if self.rank == 0:
                buffer[0] = self.num_frames
                buffer[1] = self.video_height
                buffer[2] = self.video_width
            torch.distributed.broadcast(buffer, src=0)
            if self.rank != 0:
                self.num_frames, self.video_height, self.video_width = buffer.tolist()

        # items in `self._images` will be loaded asynchronously
        self.images_loaded = [False] * self.num_frames
        self.images = torch.zeros(
            self.num_frames,
            3,
            self.image_size,
            self.image_size,
            dtype=torch.float16,
            device=self.out_device,
        )
        # catch and raise any exceptions in the async loading thread
        self.exception = None
        self.use_rand_seek_in_loading = use_rand_seek_in_loading
        self.rand_seek_idx_queue = queue.Queue()
        # use a lock to avoid race condition between concurrent access to torchcodec
        # libs (which are not thread-safe); the lock is replaced with a nullcontext
        # when the video is fully loaded
        self.torchcodec_access_lock = FIFOLock()
        self._start_video_loading()

    def _load_one_frame(self, idx):
        if self.rank == 0 or not self.load_frames_in_rank0_only:
            frame_resized = self._transform_frame(self.async_reader[idx])

        # broadcast video frames from rank 0 to all other ranks
        if self.world_size > 1 and self.load_frames_in_rank0_only:
            shape = (3, self.image_size, self.image_size)
            if self.rank == 0:
                assert frame_resized.shape == shape
                assert frame_resized.dtype == torch.float16
                buffer = frame_resized.cuda().contiguous()
            else:
                buffer = torch.zeros(*shape, dtype=torch.float16, device="cuda")
            torch.distributed.broadcast(buffer, src=0)
            if self.rank != 0:
                frame_resized = buffer.to(device=self.out_device)

        return frame_resized

    @torch.inference_mode()
    def _start_video_loading(self):
        desc = f"frame loading (MP4 w/ {'GPU' if self.gpu_acceleration else 'CPU'}) [rank={RANK}]"
        pbar = tqdm(desc=desc, total=self.num_frames)
        self.num_loaded_frames = 0
        # load the first frame synchronously to cache it before the session is opened
        idx = self.num_loaded_frames
        self.images[idx] = self._load_one_frame(idx)
        self.images_loaded[idx] = True
        self.num_loaded_frames += 1
        pbar.update(n=1)
        self.all_frames_loaded = self.num_loaded_frames == self.num_frames

        # load the frames asynchronously without blocking the session start
        def _load_frames():
            finished = self.all_frames_loaded
            chunk_size = 16
            while not finished:
                # asynchronously load `chunk_size` frames each time we acquire the lock
                with self.torchcodec_access_lock, torch.inference_mode():
                    for _ in range(chunk_size):
                        try:
                            idx = self.num_loaded_frames
                            self.images[idx] = self._load_one_frame(idx)
                            self.images_loaded[idx] = True
                            self.num_loaded_frames += 1
                            pbar.update(n=1)
                            if self.num_loaded_frames >= self.num_frames:
                                finished = True
                                break
                        except Exception as e:
                            self.exception = e
                            raise

                    # also read the frame that is being randomly seeked to
                    while True:
                        try:
                            idx = self.rand_seek_idx_queue.get_nowait()
                            if not self.images_loaded[idx]:
                                self.images[idx] = self._load_one_frame(idx)
                                self.images_loaded[idx] = True
                        except queue.Empty:
                            break
                        except Exception as e:
                            self.exception = e
                            raise

            # finished -- check whether we have loaded the total number of frames
            if self.num_loaded_frames != self.num_frames:
                raise RuntimeError(
                    f"There are {self.num_frames} frames in the video, but only "
                    f"{self.num_loaded_frames} frames can be loaded successfully."
                )
            else:
                self.all_frames_loaded = True
                pbar.close()
                with self.torchcodec_access_lock:
                    import gc

                    # all frames have been loaded, so we can release the readers and free their memory
                    # also remove pbar and thread (which shouldn't be a part of session saving)
                    reader = self.async_reader
                    if reader is not None:
                        reader._source = None
                    self.async_reader = None
                    self.pbar = None
                    self.thread = None
                    self.rand_seek_idx_queue = None
                    gc.collect()
                # remove the lock (replace it with nullcontext) when the video is fully loaded
                self.torchcodec_access_lock = contextlib.nullcontext()

        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()

    def _transform_frame(self, frame):
        frame = frame.clone()  # make a copy to avoid modifying the original frame bytes
        frame = frame.float()  # convert to float32 before interpolation
        frame_resized = F.interpolate(
            frame[None, :], size=(self.image_size, self.image_size), mode="bicubic"
        )[0]
        # float16 precision should be sufficient for image tensor storage
        frame_resized = frame_resized.half()  # uint8 -> float16
        frame_resized /= 255
        frame_resized -= self.img_mean
        frame_resized /= self.img_std
        if self.offload_video_to_cpu:
            frame_resized = frame_resized.cpu()
        elif frame_resized.device != self.out_device:
            frame_resized = frame_resized.to(device=self.out_device, non_blocking=True)
        return frame_resized

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

        max_tries = 1200
        for _ in range(max_tries):
            # use a lock to avoid race condition between concurrent access to torchcodec
            # libs (which are not thread-safe); the lock is replaced with a nullcontext
            # when the video is fully loaded
            with self.torchcodec_access_lock:
                if self.images_loaded[index]:
                    return self.images[index]

                if self.use_rand_seek_in_loading:
                    # async loading hasn't reached this frame yet, so we load this frame individually
                    # (it will be loaded by in _load_frames thread and added to self.images[index])
                    self.rand_seek_idx_queue.put(index)

            time.sleep(0.1)

        raise RuntimeError(f"Failed to load frame {index} after {max_tries} tries")

    def __len__(self):
        return len(self.images)

    def __getstate__(self):
        """
        Remove a few attributes during pickling, so that this async video loader can be
        saved and loaded as a part of the model session.
        """
        # wait for async video loading to finish before pickling
        async_thread = self.thread
        if async_thread is not None:
            async_thread.join()
        # release a few objects that cannot be pickled
        reader = self.async_reader
        if reader is not None:
            reader._source = None
        self.async_reader = None
        self.pbar = None
        self.thread = None
        self.rand_seek_idx_queue = None
        self.torchcodec_access_lock = contextlib.nullcontext()
        return self.__dict__.copy()


def load_resource_as_video_frames(
    resource_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    use_torchcodec=False,
    use_cv2=False,
):
    """
    Load video frames from either a video or an image (as a single-frame video).
    Alternatively, if input is a list of PIL images, convert its format
    """
    if isinstance(resource_path, list):
        img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]
        assert all(isinstance(img_pil, Image.Image) for img_pil in resource_path)
        assert len(resource_path) is not None
        orig_height, orig_width = resource_path[0].size
        orig_height, orig_width = (
            orig_width,
            orig_height,
        )  # For some reason, this method returns these swapped
        images = []
        for img_pil in resource_path:
            img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
            assert img_np.dtype == np.uint8, "np.uint8 is expected for JPEG images"
            img_np = img_np / 255.0
            img = torch.from_numpy(img_np).permute(2, 0, 1)
            # float16 precision should be sufficient for image tensor storage
            img = img.to(dtype=torch.float16)
            # normalize by mean and std
            img -= img_mean
            img /= img_std
            images.append(img)
        images = torch.stack(images)
        if not offload_video_to_cpu:
            images = images.cuda()
        return images, orig_height, orig_width

    image_exts = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
    is_image = (
        isinstance(resource_path, str)
        and os.path.splitext(resource_path)[1] in image_exts
    )
    if is_image:
        return load_image_as_single_frame_video(
            image_path=resource_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
        )
    else:
        return load_video_frames(
            video_path=resource_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            use_torchcodec=use_torchcodec,
            use_cv2=use_cv2,
        )


def load_image_as_single_frame_video(
    image_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
):
    """Load an image as a single-frame video."""
    images, image_height, image_width = _load_img_as_tensor(image_path, image_size)
    images = images.unsqueeze(0).half()

    img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]
    if not offload_video_to_cpu:
        images = images.cuda()
        img_mean = img_mean.cuda()
        img_std = img_std.cuda()
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, image_height, image_width


class StreamingFrameLoaderInterface(ABC):
    @abstractmethod
    def get_image(self, index) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_total_num_frames(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_video_width(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_video_height(self) -> int:
        raise NotImplementedError


class FrameLoader(StreamingFrameLoaderInterface):
    def __init__(
        self,
        video_width: int,
        video_height: int,
        total_num_frames_override: Optional[int],
    ) -> None:
        self.frames: List[Optional[NDArray]] = []
        self.video_width: int = video_width
        self.video_height: int = video_height

        # Set in streaming decoding setup to tell the model to initialize state for this many video frames
        # This is because there are certain parts of the model that depends on the number of frames
        # And when in streaming decoding setup, we don't know the number of frames
        self.__total_num_frames_override = total_num_frames_override

    def get_image(self, index: int) -> NDArray:
        if index > len(self.frames):
            raise Exception(f"Frame index {index} is out of range")
        frame = self.frames[index]
        if frame is None:
            raise Exception(f"Frame {index} is not published yet")
        return frame

    def publish(self, frame_num: int, image_frame: NDArray) -> None:
        if (
            image_frame.shape[0] != self.video_height
            or image_frame.shape[1] != self.video_width
        ):
            raise Exception(
                f"FrameLoader.publish: Frame {frame_num} has incorrect shape {image_frame.shape}. Expected shape {self.video_height}x{self.video_width}"
            )
        if frame_num >= len(self.frames):
            self.frames += [None] * (frame_num + 1 - len(self.frames))
        self.frames[frame_num] = image_frame
        self.video_height = image_frame.shape[0]
        self.video_width = image_frame.shape[1]

    def get_total_num_frames(self) -> int:
        if self.__total_num_frames_override is None:
            return len(self.frames)
        else:
            return self.__total_num_frames_override

    def get_video_width(self) -> int:
        return self.video_width

    def get_video_height(self) -> int:
        return self.video_height


class StreamingVideoFrameLoader:
    def __init__(
        self,
        stream_loader: StreamingFrameLoaderInterface,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        compute_device,
    ):
        self.stream_loader = stream_loader

        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None].to(
            compute_device
        )
        self.img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None].to(
            compute_device
        )
        self.compute_device = compute_device

    def __getitem__(self, index):
        img_np = self.stream_loader.get_image(index)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        if not self.offload_video_to_cpu:
            img_tensor = img_tensor.to(self.compute_device, non_blocking=True)
        img_tensor = torchvision.transforms.functional.resize(
            img_tensor,
            (self.image_size, self.image_size),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        )
        if img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor / 255.0

        if not img_tensor.is_contiguous():
            img_tensor = img_tensor.contiguous()

        # normalize by mean and std
        img_tensor -= self.img_mean
        img_tensor /= self.img_std

        return img_tensor

    def __len__(self):
        return self.stream_loader.get_total_num_frames()

    def get_width(self) -> int:
        return self.stream_loader.get_video_width()

    def get_height(self) -> int:
        return self.stream_loader.get_video_height()


def load_video_frames(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    use_torchcodec=False,
    use_cv2=False,
):
    """
    Load the video frames from video_path. The frames are resized to image_size as in
    the model and are loaded to GPU if offload_video_to_cpu=False. This is used by the demo.
    """
    if isinstance(video_path, tuple):
        # Video is already loaded
        return video_path
    is_bytes = isinstance(video_path, bytes)
    is_str = isinstance(video_path, str)
    is_mp4_path = is_str and os.path.splitext(video_path)[-1] in [".mp4", ".MP4"]
    is_streaming = isinstance(video_path, StreamingFrameLoaderInterface)
    if is_str and video_path.startswith("<load-dummy-video"):
        # Check for pattern <load-dummy-video-N> where N is an integer
        match = re.match(r"<load-dummy-video-(\d+)>", video_path)
        if match:
            num_frames = int(match.group(1))
        else:
            # Default for original <load-dummy-video> path
            num_frames = 60
        return load_dummy_video(image_size, offload_video_to_cpu, num_frames=num_frames)
    elif is_bytes or is_mp4_path:
        return load_video_frames_from_video_file(
            video_path_or_bytes=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            use_torchcodec=use_torchcodec,
            use_cv2=use_cv2,
        )
    elif is_str and os.path.isdir(video_path):
        return load_video_frames_from_jpg_images(
            jpg_folder=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
        )
    elif is_streaming:
        lazy_images = StreamingVideoFrameLoader(
            stream_loader=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            compute_device="cuda",
        )
        return lazy_images, lazy_images.get_height(), lazy_images.get_width()
    else:
        raise NotImplementedError(
            "Only MP4 video and JPEG folder are supported at this moment"
        )


def load_dummy_video(image_size, offload_video_to_cpu, num_frames=60):
    """
    Load a dummy video with random frames for testing and compilation warmup purposes.
    """
    video_height, video_width = 480, 640  # dummy original video sizes
    images = torch.randn(num_frames, 3, image_size, image_size, dtype=torch.float16)
    if not offload_video_to_cpu:
        images = images.cuda()
    return images, video_height, video_width


def load_video_frames_from_jpg_images(
    jpg_folder,
    image_size,
    offload_video_to_cpu,
    img_mean,
    img_std,
    async_loading_frames,
):
    """
    Load the video frames from a directory of JPEG files ("<frame_index>.jpg" format)
    """
    frame_names = [p for p in os.listdir(jpg_folder) if p.endswith(".jpg")]
    try:
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    except ValueError:
        # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
        logging.warning(
            f'frame names are not in "<frame_index>.jpg" format: {frame_names[:5]=}, '
            f"falling back to lexicographic sort."
        )
        frame_names.sort()
    num_frames = len(frame_names)
    if num_frames == 0:
        raise RuntimeError(f"no images found in {jpg_folder}")
    img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]
    img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]

    if async_loading_frames:
        lazy_images = AsyncVideoFrameLoader(
            img_paths, image_size, offload_video_to_cpu, img_mean, img_std
        )
        return lazy_images, lazy_images.video_height, lazy_images.video_width

    # float16 precision should be sufficient for image tensor storage
    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float16)
    for n, img_path in enumerate(
        tqdm(img_paths, desc=f"frame loading (JPEG) [rank={RANK}]")
    ):
        images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
    if not offload_video_to_cpu:
        images = images.cuda()
        img_mean = img_mean.cuda()
        img_std = img_std.cuda()
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


def load_video_cv2(
    video_path: str,
    image_size: int,
    img_mean: tuple = (0.485, 0.456, 0.406),
    img_std: tuple = (0.229, 0.224, 0.225),
    gpu_acceleration: bool = False,
) -> torch.Tensor:
    """
    Load video from path, convert to normalized tensor with specified preprocessing

    Args:
        video_path: Path to video file
        image_size: Target size for square frames (height and width)
        img_mean: Normalization mean (RGB)
        img_std: Normalization standard deviation (RGB)

    Returns:
        torch.Tensor: Preprocessed video tensor in shape (T, C, H, W) with float16 dtype
    """
    import cv2  # delay OpenCV import to avoid unnecessary dependency

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB and resize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(
            frame_rgb, (image_size, image_size), interpolation=cv2.INTER_CUBIC
        )
        frames.append(frame_resized)

    cap.release()

    # Convert to tensor
    video_np = np.stack(frames, axis=0).astype(np.float32)  # (T, H, W, C)
    video_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2)  # (C, T, H, W)

    # Normalize
    mean = torch.tensor(img_mean).view(3, 1, 1, 1)
    std = torch.tensor(img_std).view(3, 1, 1, 1)
    video_tensor = (video_tensor / 255.0 - mean) / std

    video_tensor = video_tensor.half()  # Convert to float16

    if gpu_acceleration:
        video_tensor = video_tensor.to("cuda")

    video_tensor = video_tensor.permute(1, 0, 2, 3)  # [T, C, H, W]

    return video_tensor, original_height, original_width


def load_video_frames_from_video_file(
    video_path_or_bytes,
    image_size,
    offload_video_to_cpu,
    img_mean,
    img_std,
    async_loading_frames,
    gpu_acceleration=True,
    gpu_device=None,
    use_torchcodec=False,
    use_cv2=False,
):
    """Load the video frames from a video file."""
    if use_cv2:
        return load_video_cv2(
            video_path_or_bytes, image_size, img_mean, img_std, gpu_acceleration
        )
    if use_torchcodec:
        logging.info("Using torchcodec to load video file")
        lazy_images = AsyncVideoFileLoaderWithTorchCodec(
            video_path_or_bytes=video_path_or_bytes,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            gpu_acceleration=gpu_acceleration,
            gpu_device=gpu_device,
        )
    else:
        lazy_images = AsyncVideoFileLoader(
            video_path_or_bytes=video_path_or_bytes,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            gpu_acceleration=gpu_acceleration,
            gpu_device=gpu_device,
        )
    # The `AsyncVideoFileLoader` class always loads the videos asynchronously, so
    # we just wait for its loading thread to finish if async_loading_frames=False.
    if not async_loading_frames:
        async_thread = lazy_images.thread
        if async_thread is not None:
            async_thread.join()
    return lazy_images, lazy_images.video_height, lazy_images.video_width


def _load_img_as_tensor(img_path, image_size):
    # img_pil = Image.open(img_path)
    # img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    # if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
    #     img_np = img_np / 255.0
    # else:
    #     raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    # img = torch.from_numpy(img_np).permute(2, 0, 1)
    # video_width, video_height = img_pil.size  # the original video size
    # return img, video_height, video_width

    # Use the FA Playground pipeline to load the image as a tensor
    return fa_load_image(img_path, image_size=image_size, img_mean=None, img_std=None)


def fa_load_image(
    img_path,
    image_size=1008,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
    max_size=1280,
):
    """Load and resize an image as tensor following the FA Playground pipeline."""
    img = Image.open(img_path).convert("RGB")
    orig_width, orig_height = img.width, img.height
    scale = max_size / max(orig_width, orig_height)
    if scale < 1.0:
        # if the image is larger than max_size, downsize it first (with JPEG compression)
        new_width = max(round(orig_width * scale), 1)
        new_height = max(round(orig_height * scale), 1)
        img = img.resize((new_width, new_height))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        img = Image.open(buf)

    img = TF.resize(img, size=(image_size, image_size))
    img = TF.to_tensor(img)
    if img_mean is not None and img_std is not None:
        img = TF.normalize(img, mean=img_mean, std=img_std)
    return img, orig_height, orig_width


def _yuv_to_rgb(frames, img_mean, img_std):
    """Convert frames from YUV to RGB and also substract mean and divide by std."""
    # frames must be uint8 dtype and have at least 3 channels in YUV
    assert frames.dtype == torch.uint8 and frames.size(-2) >= 3
    frames = frames[..., :3, :, :].half()  # uint8 to float16 (remove any alpha channel)
    frames /= 255
    frames[..., 1:3, :, :] -= 0.5  # U, V in range -0.5 to +0.5

    # Using YUV => RGB formula from Wikipedia
    # https://en.wikipedia.org/wiki/Y%E2%80%B2UV#Y.27UV420p_.28and_Y.27V12_or_YV12.29_to_RGB888_conversion
    rgb = torch.zeros_like(frames)
    y = frames[..., 0, :, :]
    u = frames[..., 1, :, :]
    v = frames[..., 2, :, :]
    rgb[..., 0, :, :] = y + 1.13983 * v
    rgb[..., 1, :, :] = y - 0.39465 * u - 0.58060 * v
    rgb[..., 2, :, :] = y + 2.03211 * u
    rgb.clamp_(0, 1)
    rgb -= img_mean
    rgb /= img_std
    return rgb


def fill_holes_in_mask_scores(mask, max_area, fill_holes=True, remove_sprinkles=True):
    """
    A post processor to fill small holes in mask scores with area under `max_area`.
    Holes are those small connected components in either background or foreground.

    Note that it relies on the "cc_torch" package to find connected components. You can
    install it via the following command (`TORCH_CUDA_ARCH_LIST=8.0` is for A100 GPUs):
    ```
    pip uninstall -y cc_torch; TORCH_CUDA_ARCH_LIST=8.0 pip install git+https://github.com/ronghanghu/cc_torch
    ```
    """

    if max_area <= 0:
        return mask  # nothing to fill in this case

    if fill_holes:
        # We remove small connected components in background by changing them to foreground
        # with a small positive mask score (0.1).
        mask_bg = mask <= 0
        bg_area_thresh = max_area
        _, areas_bg = _get_connected_components_with_padding(mask_bg)
        small_components_bg = mask_bg & (areas_bg <= bg_area_thresh)
        mask = torch.where(small_components_bg, 0.1, mask)

    if remove_sprinkles:
        # We remove small connected components in foreground by changing them to background
        # with a small negative mask score (-0.1). Here we only remove connected components
        # whose areas are under both `max_area` and half of the entire mask's area. This
        # removes sprinkles while avoids filtering out tiny objects that we want to track.
        mask_fg = mask > 0
        fg_area_thresh = torch.sum(mask_fg, dim=(2, 3), keepdim=True, dtype=torch.int32)
        fg_area_thresh.floor_divide_(2).clamp_(max=max_area)
        _, areas_fg = _get_connected_components_with_padding(mask_fg)
        small_components_fg = mask_fg & (areas_fg <= fg_area_thresh)
        mask = torch.where(small_components_fg, -0.1, mask)
    return mask


def _get_connected_components_with_padding(mask, get_counts=True):
    """Get connected components from masks (possibly padding them to an even size)."""
    from cc_torch import get_connected_components

    mask = mask.to(torch.uint8)
    _, _, H, W = mask.shape
    # make sure both height and width are even (to be compatible with cc_torch)
    pad_h = H % 2
    pad_w = W % 2
    if pad_h == 0 and pad_w == 0:
        labels, counts = get_connected_components(mask, get_counts)
    else:
        # pad the mask to make its height and width even
        # padding format is (padding_left,padding_right,padding_top,padding_bottom)
        mask_pad = F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0)
        labels, counts = get_connected_components(mask_pad, get_counts)
        labels = labels[:, :, :H, :W]
        counts = counts[:, :, :H, :W]

    return labels, counts
