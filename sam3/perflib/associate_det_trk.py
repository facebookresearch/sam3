# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from collections import defaultdict

import torch
import torch.nn.functional as F
from sam3.perflib.masks_ops import mask_iou
from scipy.optimize import linear_sum_assignment


def associate_det_trk(
    det_masks,
    track_masks,
    iou_threshold=0.5,
    iou_threshold_trk=0.5,
    det_scores=None,
    new_det_thresh=0.0,
):
    """
    Optimized implementation of detection <-> track association that minimizes DtoH syncs.

    Args:
        det_masks: (N, H, W) tensor of predicted masks
        track_masks: (M, H, W) tensor of track masks

    Returns:
        new_det_indices: list of indices in det_masks considered 'new'
        unmatched_trk_indices: list of indices in track_masks considered 'unmatched'
    """
    with torch.autograd.profiler.record_function("perflib: associate_det_trk"):
        assert isinstance(det_masks, torch.Tensor), "det_masks should be a tensor"
        assert isinstance(track_masks, torch.Tensor), "track_masks should be a tensor"
        if det_masks.size(0) == 0 or track_masks.size(0) == 0:
            return list(range(det_masks.size(0))), [], {}, {}  # all detections are new

        if list(det_masks.shape[-2:]) != list(track_masks.shape[-2:]):
            # resize to the smaller size to save GPU memory
            if torch.numel(det_masks[-2:]) < torch.numel(track_masks[-2:]):
                track_masks = (
                    F.interpolate(
                        track_masks.unsqueeze(1).float(),
                        size=det_masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                    > 0
                )
            else:
                # resize detections to track size
                det_masks = (
                    F.interpolate(
                        det_masks.unsqueeze(1).float(),
                        size=track_masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                    > 0
                )

        det_masks = det_masks > 0
        track_masks = track_masks > 0

        iou = mask_iou(det_masks, track_masks)  # (N, M)
        igeit = iou >= iou_threshold
        igeit_any_dim_1 = igeit.any(dim=1)
        igeit_trk = iou >= iou_threshold_trk

        iou_list = iou.cpu().numpy().tolist()
        igeit_list = igeit.cpu().numpy().tolist()
        igeit_any_dim_1_list = igeit_any_dim_1.cpu().numpy().tolist()
        igeit_trk_list = igeit_trk.cpu().numpy().tolist()

        det_scores_list = (
            det_scores
            if det_scores is None
            else det_scores.cpu().float().numpy().tolist()
        )

        # Hungarian matching for tracks (one-to-one: each track matches at most one detection)
        # For detections: allow many tracks to match to the same detection (many-to-one)

        # If either is empty, return all detections as new
        if det_masks.size(0) == 0 or track_masks.size(0) == 0:
            return list(range(det_masks.size(0))), [], {}

        # Hungarian matching: maximize IoU for tracks
        cost_matrix = 1 - iou.cpu().numpy()  # Hungarian solves for minimum cost
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        def branchy_hungarian_better_uses_the_cpu(
            cost_matrix, row_ind, col_ind, iou_list, det_masks, track_masks
        ):
            matched_trk = set()
            matched_det = set()
            matched_det_scores = {}  # track index -> [det_score, det_score * iou] det score of matched detection mask
            for d, t in zip(row_ind, col_ind):
                matched_det_scores[t] = [
                    det_scores_list[d],
                    det_scores_list[d] * iou_list[d][t],
                ]
                if igeit_trk_list[d][t]:
                    matched_trk.add(t)
                    matched_det.add(d)

            # Tracks not matched by Hungarian assignment above threshold are unmatched
            unmatched_trk_indices = [
                t for t in range(track_masks.size(0)) if t not in matched_trk
            ]

            # For detections: allow many tracks to match to the same detection (many-to-one)
            # So, a detection is 'new' if it does not match any track above threshold
            assert track_masks.size(0) == igeit.size(
                1
            )  # Needed for loop optimizaiton below
            new_det_indices = []
            for d in range(det_masks.size(0)):
                if not igeit_any_dim_1_list[d]:
                    if det_scores is not None and det_scores[d] >= new_det_thresh:
                        new_det_indices.append(d)

            # for each detection, which tracks it matched to (above threshold)
            det_to_matched_trk = defaultdict(list)
            for d in range(det_masks.size(0)):
                for t in range(track_masks.size(0)):
                    if igeit_list[d][t]:
                        det_to_matched_trk[d].append(t)

            return (
                new_det_indices,
                unmatched_trk_indices,
                det_to_matched_trk,
                matched_det_scores,
            )

        return (branchy_hungarian_better_uses_the_cpu)(
            cost_matrix, row_ind, col_ind, iou_list, det_masks, track_masks
        )
