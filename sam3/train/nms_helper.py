# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
import warnings
from typing import Dict, List

import numpy as np

# Check if Numba is available
HAS_NUMBA = False
try:
    import numba as nb

    HAS_NUMBA = True
except ImportError:
    warnings.warn(
        "Numba not found. Using slower pure Python implementations.", UserWarning
    )


# -------------------- Helper Functions --------------------
def is_zero_box(bbox: list) -> bool:
    """Check if bounding box is invalid"""
    if bbox is None:
        return True
    return all(x <= 0 for x in bbox[:4]) or len(bbox) < 4


def convert_bbox_format(bbox: list) -> List[float]:
    """Convert bbox from (x,y,w,h) to (x1,y1,x2,y2)"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


# -------------------- Track-level NMS --------------------
def process_track_level_nms(video_groups: Dict, nms_threshold: float) -> Dict:
    """Apply track-level NMS to all videos"""
    for video_id, tracks in video_groups.items():
        track_detections = []

        # Process tracks
        for track_idx, track in enumerate(tracks):
            if not track["bboxes"]:
                continue

            converted_bboxes = []
            valid_frames = []
            for bbox in track["bboxes"]:
                if bbox and not is_zero_box(bbox):
                    converted_bboxes.append(convert_bbox_format(bbox))
                    valid_frames.append(True)
                else:
                    converted_bboxes.append([np.nan] * 4)
                    valid_frames.append(False)

            if any(valid_frames):
                track_detections.append(
                    {
                        "track_idx": track_idx,
                        "bboxes": np.array(converted_bboxes, dtype=np.float32),
                        "score": track["score"],
                    }
                )

        # Apply NMS
        if track_detections:
            scores = np.array([d["score"] for d in track_detections], dtype=np.float32)
            keep = apply_track_nms(track_detections, scores, nms_threshold)

            # Suppress non-kept tracks
            for idx, track in enumerate(track_detections):
                if idx not in keep:
                    tracks[track["track_idx"]]["bboxes"] = [None] * len(track["bboxes"])

    return video_groups


# -------------------- Frame-level NMS --------------------
def process_frame_level_nms(video_groups: Dict, nms_threshold: float) -> Dict:
    """Apply frame-level NMS to all videos"""
    for video_id, tracks in video_groups.items():
        if not tracks:
            continue

        num_frames = len(tracks[0]["bboxes"])

        for frame_idx in range(num_frames):
            frame_detections = []

            # Collect valid detections
            for track_idx, track in enumerate(tracks):
                bbox = track["bboxes"][frame_idx]
                if bbox and not is_zero_box(bbox):
                    frame_detections.append(
                        {
                            "track_idx": track_idx,
                            "bbox": np.array(
                                convert_bbox_format(bbox), dtype=np.float32
                            ),
                            "score": track["score"],
                        }
                    )

            # Apply NMS
            if frame_detections:
                bboxes = np.stack([d["bbox"] for d in frame_detections])
                scores = np.array(
                    [d["score"] for d in frame_detections], dtype=np.float32
                )
                keep = apply_frame_nms(bboxes, scores, nms_threshold)

                # Suppress non-kept detections
                for i, d in enumerate(frame_detections):
                    if i not in keep:
                        tracks[d["track_idx"]]["bboxes"][frame_idx] = None

    return video_groups


# Track-level NMS helpers ------------------------------------------------------
def compute_track_iou_matrix(
    bboxes_stacked: np.ndarray, valid_masks: np.ndarray, areas: np.ndarray
) -> np.ndarray:
    """IoU matrix computation for track-level NMS with fallback to pure Python"""
    num_tracks = bboxes_stacked.shape[0]
    iou_matrix = np.zeros((num_tracks, num_tracks), dtype=np.float32)
    if HAS_NUMBA:
        iou_matrix = _compute_track_iou_matrix_numba(bboxes_stacked, valid_masks, areas)
    else:
        # Pure Python implementation
        for i in range(num_tracks):
            for j in range(i + 1, num_tracks):
                valid_ij = valid_masks[i] & valid_masks[j]
                if not valid_ij.any():
                    continue
                bboxes_i = bboxes_stacked[i, valid_ij]
                bboxes_j = bboxes_stacked[j, valid_ij]
                area_i = areas[i, valid_ij]
                area_j = areas[j, valid_ij]
                inter_total = 0.0
                union_total = 0.0
                for k in range(bboxes_i.shape[0]):
                    x1 = max(bboxes_i[k, 0], bboxes_j[k, 0])
                    y1 = max(bboxes_i[k, 1], bboxes_j[k, 1])
                    x2 = min(bboxes_i[k, 2], bboxes_j[k, 2])
                    y2 = min(bboxes_i[k, 3], bboxes_j[k, 3])
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    union = area_i[k] + area_j[k] - inter
                    inter_total += inter
                    union_total += union
                if union_total > 0:
                    iou_matrix[i, j] = inter_total / union_total
                    iou_matrix[j, i] = iou_matrix[i, j]
    return iou_matrix


if HAS_NUMBA:

    @nb.jit(nopython=True, parallel=True)
    def _compute_track_iou_matrix_numba(bboxes_stacked, valid_masks, areas):
        """Numba-optimized IoU matrix computation for track-level NMS"""
        num_tracks = bboxes_stacked.shape[0]
        iou_matrix = np.zeros((num_tracks, num_tracks), dtype=np.float32)
        for i in nb.prange(num_tracks):
            for j in range(i + 1, num_tracks):
                valid_ij = valid_masks[i] & valid_masks[j]
                if not valid_ij.any():
                    continue
                bboxes_i = bboxes_stacked[i, valid_ij]
                bboxes_j = bboxes_stacked[j, valid_ij]
                area_i = areas[i, valid_ij]
                area_j = areas[j, valid_ij]
                inter_total = 0.0
                union_total = 0.0
                for k in range(bboxes_i.shape[0]):
                    x1 = max(bboxes_i[k, 0], bboxes_j[k, 0])
                    y1 = max(bboxes_i[k, 1], bboxes_j[k, 1])
                    x2 = min(bboxes_i[k, 2], bboxes_j[k, 2])
                    y2 = min(bboxes_i[k, 3], bboxes_j[k, 3])
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    union = area_i[k] + area_j[k] - inter
                    inter_total += inter
                    union_total += union
                if union_total > 0:
                    iou_matrix[i, j] = inter_total / union_total
                    iou_matrix[j, i] = iou_matrix[i, j]
        return iou_matrix


def apply_track_nms(
    track_detections: List[dict], scores: np.ndarray, nms_threshold: float
) -> List[int]:
    """Vectorized track-level NMS implementation"""
    if not track_detections:
        return []
    bboxes_stacked = np.stack([d["bboxes"] for d in track_detections], axis=0)
    valid_masks = ~np.isnan(bboxes_stacked).any(axis=2)
    areas = (bboxes_stacked[:, :, 2] - bboxes_stacked[:, :, 0]) * (
        bboxes_stacked[:, :, 3] - bboxes_stacked[:, :, 1]
    )
    areas[~valid_masks] = 0
    iou_matrix = compute_track_iou_matrix(bboxes_stacked, valid_masks, areas)
    keep = []
    order = np.argsort(-scores)
    suppress = np.zeros(len(track_detections), dtype=bool)
    for i in range(len(order)):
        if not suppress[order[i]]:
            keep.append(order[i])
            suppress[order[i:]] = suppress[order[i:]] | (
                iou_matrix[order[i], order[i:]] >= nms_threshold
            )
    return keep


# Frame-level NMS helpers ------------------------------------------------------
def compute_frame_ious(bbox: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    """IoU computation for frame-level NMS with fallback to pure Python"""
    if HAS_NUMBA:
        return _compute_frame_ious_numba(bbox, bboxes)
    else:
        # Pure Python implementation
        ious = np.zeros(len(bboxes), dtype=np.float32)
        for i in range(len(bboxes)):
            x1 = max(bbox[0], bboxes[i, 0])
            y1 = max(bbox[1], bboxes[i, 1])
            x2 = min(bbox[2], bboxes[i, 2])
            y2 = min(bbox[3], bboxes[i, 3])

            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            area2 = (bboxes[i, 2] - bboxes[i, 0]) * (bboxes[i, 3] - bboxes[i, 1])
            union = area1 + area2 - inter

            ious[i] = inter / union if union > 0 else 0.0
        return ious


if HAS_NUMBA:

    @nb.jit(nopython=True, parallel=True)
    def _compute_frame_ious_numba(bbox, bboxes):
        """Numba-optimized IoU computation"""
        ious = np.zeros(len(bboxes), dtype=np.float32)
        for i in nb.prange(len(bboxes)):
            x1 = max(bbox[0], bboxes[i, 0])
            y1 = max(bbox[1], bboxes[i, 1])
            x2 = min(bbox[2], bboxes[i, 2])
            y2 = min(bbox[3], bboxes[i, 3])

            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            area2 = (bboxes[i, 2] - bboxes[i, 0]) * (bboxes[i, 3] - bboxes[i, 1])
            union = area1 + area2 - inter

            ious[i] = inter / union if union > 0 else 0.0
        return ious


def apply_frame_nms(
    bboxes: np.ndarray, scores: np.ndarray, nms_threshold: float
) -> List[int]:
    """Frame-level NMS implementation with fallback to pure Python"""
    if HAS_NUMBA:
        return _apply_frame_nms_numba(bboxes, scores, nms_threshold)
    else:
        # Pure Python implementation
        order = np.argsort(-scores)
        keep = []
        suppress = np.zeros(len(bboxes), dtype=bool)

        for i in range(len(order)):
            if not suppress[order[i]]:
                keep.append(order[i])
                current_bbox = bboxes[order[i]]

                remaining_bboxes = bboxes[order[i + 1 :]]
                if len(remaining_bboxes) > 0:  # Check if there are any remaining boxes
                    ious = compute_frame_ious(current_bbox, remaining_bboxes)
                    suppress[order[i + 1 :]] = suppress[order[i + 1 :]] | (
                        ious >= nms_threshold
                    )

        return keep


if HAS_NUMBA:

    @nb.jit(nopython=True)
    def _apply_frame_nms_numba(bboxes, scores, nms_threshold):
        """Numba-optimized NMS implementation"""
        order = np.argsort(-scores)
        keep = []
        suppress = np.zeros(len(bboxes), dtype=nb.boolean)

        for i in range(len(order)):
            if not suppress[order[i]]:
                keep.append(order[i])
                current_bbox = bboxes[order[i]]

                if i + 1 < len(order):  # Check bounds
                    ious = _compute_frame_ious_numba(
                        current_bbox, bboxes[order[i + 1 :]]
                    )
                    suppress[order[i + 1 :]] = suppress[order[i + 1 :]] | (
                        ious >= nms_threshold
                    )

        return keep
