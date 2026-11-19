#!/usr/bin/env python3
"""Generate portable assets for temporal Ego-Exo cross-view localization.

Stage A (this script) is deterministic and suitable for a compute server:
1. Read each take/annotation.json.
2. Decode only source-view (aria*) COCO RLE masks.
3. Select the largest source mask by area ratio, while retaining per-view bests.
4. Build temporally contiguous sliding-window contact sheets for each target cam*.
5. Write portable metadata, rankings, window indexes, and a Codex prompt.

No target-view mask is inferred or fabricated by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pycocotools import mask as mask_utils

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FRAME_ID_RE = re.compile(r"(\d+)(?!.*\d)")


@dataclass(frozen=True)
class FrameImage:
    frame_id: int
    path: Path


@dataclass
class SourceMaskRecord:
    object_name: str
    view_name: str
    frame_id: int
    image_path: Optional[str]
    image_exists: bool
    mask_height: int
    mask_width: int
    mask_area_pixels: int
    mask_area_ratio: float
    decode_ok: bool
    error: Optional[str] = None
    rank_global: Optional[int] = None
    rank_within_view: Optional[int] = None


@dataclass(frozen=True)
class CaseSpec:
    take_dir: Path
    annotation_path: Path
    take_id: str
    object_name: str
    case_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate source-mask and target temporal-window assets."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("temporal_cross_view_assets")
    )
    parser.add_argument(
        "--annotation-name", default="annotation.json", help="Annotation filename."
    )
    parser.add_argument("--source-prefix", default="aria")
    parser.add_argument("--target-prefix", default="cam")
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=[16],
        help="Number of sampled target frames per window; e.g. 10 16 20.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=8,
        help="Stride measured in sampled target frames.",
    )
    parser.add_argument(
        "--target-sample-every",
        type=int,
        default=1,
        help="Use every Nth available target JPG before windowing.",
    )
    parser.add_argument(
        "--max-gap-factor",
        type=float,
        default=2.5,
        help="Split target sequences when a frame-id gap exceeds median gap times this factor.",
    )
    parser.add_argument("--sheet-columns", type=int, default=4)
    parser.add_argument("--cell-width", type=int, default=320)
    parser.add_argument("--cell-height", type=int, default=220)
    parser.add_argument("--cell-header-height", type=int, default=28)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--max-windows-per-cam",
        type=int,
        default=0,
        help="0 keeps every window. Positive values uniformly retain at most this many per cam.",
    )
    parser.add_argument(
        "--case-id", action="append", default=[], help="Process only matching case IDs."
    )
    parser.add_argument(
        "--object-name", action="append", default=[], help="Process only matching objects."
    )
    parser.add_argument(
        "--max-cases", type=int, default=0, help="0 means all discovered cases."
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop after the first failed case."
    )
    parser.add_argument(
        "--window-ratios",
        type=float,
        nargs="+",
        default=[],
        help=(
            "Window lengths as fractions of each target camera's sampled frames. "
            "Example: 0.20 0.25 0.30. When provided, this overrides "
            "--window-sizes."
        ),
    )

    parser.add_argument(
        "--window-stride-ratio",
        type=float,
        default=0.05,
        help=(
            "Sliding stride as a fraction of each target camera's sampled frames "
            "when --window-ratios is used. Example: 0.05."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def slugify(value: str, max_length: int = 80) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_.")
    return (value or "object")[:max_length]


def parse_frame_id(path_or_name: Any) -> int:
    stem = Path(str(path_or_name)).stem
    match = FRAME_ID_RE.search(stem)
    if not match:
        raise ValueError(f"Cannot parse trailing frame id from: {path_or_name}")
    return int(match.group(1))


def image_files(view_dir: Path) -> List[Path]:
    if not view_dir.is_dir():
        return []
    return sorted(
        (
            p
            for p in view_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda p: p.name,
    )


def build_frame_map(view_dir: Path) -> Tuple[Dict[int, Path], List[str]]:
    mapping: Dict[int, Path] = {}
    warnings: List[str] = []
    for path in image_files(view_dir):
        try:
            frame_id = parse_frame_id(path.name)
        except ValueError as exc:
            warnings.append(str(exc))
            continue
        if frame_id in mapping:
            warnings.append(
                f"Duplicate frame id {frame_id} in {view_dir}; keeping "
                f"{mapping[frame_id].name}, ignoring {path.name}."
            )
            continue
        mapping[frame_id] = path
    return mapping, warnings


def decode_coco_rle(rle_payload: Mapping[str, Any]) -> np.ndarray:
    if "size" not in rle_payload or "counts" not in rle_payload:
        raise ValueError("RLE payload must contain 'size' and 'counts'.")
    height, width = (int(v) for v in rle_payload["size"])
    counts = rle_payload["counts"]
    if isinstance(counts, str):
        rle: Dict[str, Any] = {
            "size": [height, width],
            "counts": counts.encode("utf-8"),
        }
    elif isinstance(counts, list):
        rle = mask_utils.frPyObjects(
            {"size": [height, width], "counts": counts}, height, width
        )
    else:
        rle = {"size": [height, width], "counts": counts}
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    mask = np.asarray(decoded, dtype=bool)
    if mask.shape != (height, width):
        raise ValueError(
            f"Decoded mask shape {mask.shape} does not match declared size "
            f"{(height, width)}."
        )
    return mask

def align_mask_to_image(
    mask: np.ndarray,
    image: Image.Image,
    aspect_tolerance: float = 1e-3,
) -> np.ndarray:
    """Resize an annotation mask to the actual JPG resolution when necessary.

    Mask area statistics must still be computed from the original annotation
    resolution. This aligned mask is only for saved visualization assets.
    """
    image_width, image_height = image.size
    mask_height, mask_width = mask.shape

    if (mask_width, mask_height) == (image_width, image_height):
        return mask.astype(bool)

    mask_aspect = mask_width / mask_height
    image_aspect = image_width / image_height

    if abs(mask_aspect - image_aspect) > aspect_tolerance:
        raise ValueError(
            "Image and mask aspect ratios differ: "
            f"image={image.size}, mask={(mask_width, mask_height)}. "
            "Refusing to stretch the mask."
        )

    resized = Image.fromarray(
        mask.astype(np.uint8) * 255,
        mode="L",
    ).resize(
        (image_width, image_height),
        resample=Image.Resampling.NEAREST,
    )

    aligned_mask = np.asarray(resized, dtype=np.uint8) > 0

    print(
        "[INFO] Aligned source mask for visualization: "
        f"{mask_width}x{mask_height} -> {image_width}x{image_height}",
        flush=True,
    )

    return aligned_mask

def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def save_mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    path: Path,
    alpha: float = 0.45,
) -> None:
    image_rgb = image.convert("RGB")
    if image_rgb.size != (mask.shape[1], mask.shape[0]):
        raise ValueError(
            f"Image size {image_rgb.size} and mask size "
            f"{(mask.shape[1], mask.shape[0])} differ."
        )
    image_arr = np.asarray(image_rgb, dtype=np.uint8).copy()
    tint = np.array([255, 0, 0], dtype=np.float32)
    pixels = image_arr[mask].astype(np.float32)
    image_arr[mask] = ((1.0 - alpha) * pixels + alpha * tint).astype(np.uint8)
    Image.fromarray(image_arr, mode="RGB").save(path, quality=95)


def discover_cases(
    data_root: Path,
    annotation_name: str,
    object_filters: Sequence[str],
) -> List[CaseSpec]:
    cases: List[CaseSpec] = []
    annotation_paths = sorted(data_root.rglob(annotation_name))
    wanted = set(object_filters)
    for annotation_path in annotation_paths:
        take_dir = annotation_path.parent
        annotation = load_json(annotation_path)
        masks = annotation.get("masks")
        if not isinstance(masks, dict):
            print(
                f"[WARN] No annotation['masks'] dict in {annotation_path}",
                file=sys.stderr,
            )
            continue
        for object_name in sorted(masks):
            if wanted and object_name not in wanted:
                continue
            take_id = take_dir.name
            case_id = f"{slugify(take_id)}__{slugify(object_name)}"
            cases.append(
                CaseSpec(
                    take_dir=take_dir,
                    annotation_path=annotation_path,
                    take_id=take_id,
                    object_name=object_name,
                    case_id=case_id,
                )
            )
    return cases


def rank_source_masks(
    annotation: Mapping[str, Any],
    case: CaseSpec,
    source_prefix: str,
) -> Tuple[List[SourceMaskRecord], Dict[Tuple[str, int], np.ndarray], List[str]]:
    object_masks = annotation["masks"][case.object_name]
    if not isinstance(object_masks, dict):
        raise ValueError(f"Masks for object {case.object_name!r} are not a dict.")

    records: List[SourceMaskRecord] = []
    decoded_masks: Dict[Tuple[str, int], np.ndarray] = {}
    warnings: List[str] = []

    for view_name in sorted(object_masks):
        if not view_name.startswith(source_prefix):
            continue
        frame_payloads = object_masks[view_name]
        if not isinstance(frame_payloads, dict):
            warnings.append(f"Source view {view_name} has non-dict frame payloads.")
            continue
        frame_map, frame_warnings = build_frame_map(case.take_dir / view_name)
        warnings.extend(frame_warnings)
        sortable_frames: List[Tuple[int, Any]] = []
        for raw_frame_id, rle_payload in frame_payloads.items():
            try:
                sortable_frames.append((int(raw_frame_id), rle_payload))
            except (TypeError, ValueError):
                warnings.append(
                    f"Ignoring non-integer frame key {raw_frame_id!r} in {view_name}."
                )
        for frame_id, rle_payload in sorted(sortable_frames):
            image_path = frame_map.get(frame_id)
            try:
                mask = decode_coco_rle(rle_payload)
                height, width = mask.shape
                area_pixels = int(mask.sum())
                area_ratio = (
                    float(area_pixels / (height * width)) if height and width else 0.0
                )
                decoded_masks[(view_name, frame_id)] = mask
                records.append(
                    SourceMaskRecord(
                        object_name=case.object_name,
                        view_name=view_name,
                        frame_id=frame_id,
                        image_path=(str(image_path) if image_path else None),
                        image_exists=image_path is not None,
                        mask_height=height,
                        mask_width=width,
                        mask_area_pixels=area_pixels,
                        mask_area_ratio=area_ratio,
                        decode_ok=True,
                    )
                )
            except Exception as exc:
                size = (
                    rle_payload.get("size", [0, 0])
                    if isinstance(rle_payload, dict)
                    else [0, 0]
                )
                records.append(
                    SourceMaskRecord(
                        object_name=case.object_name,
                        view_name=view_name,
                        frame_id=frame_id,
                        image_path=(str(image_path) if image_path else None),
                        image_exists=image_path is not None,
                        mask_height=int(size[0]) if len(size) > 0 else 0,
                        mask_width=int(size[1]) if len(size) > 1 else 0,
                        mask_area_pixels=0,
                        mask_area_ratio=0.0,
                        decode_ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

    ranked = sorted(
        records,
        key=lambda r: (
            r.decode_ok and r.image_exists,
            r.mask_area_ratio,
            r.mask_area_pixels,
            -r.frame_id,
        ),
        reverse=True,
    )
    for rank, record in enumerate(ranked, start=1):
        record.rank_global = rank

    by_view: Dict[str, List[SourceMaskRecord]] = {}
    for record in records:
        by_view.setdefault(record.view_name, []).append(record)
    for view_records in by_view.values():
        view_ranked = sorted(
            view_records,
            key=lambda r: (
                r.decode_ok and r.image_exists,
                r.mask_area_ratio,
                r.mask_area_pixels,
                -r.frame_id,
            ),
            reverse=True,
        )
        for rank, record in enumerate(view_ranked, start=1):
            record.rank_within_view = rank

    return ranked, decoded_masks, warnings


def usable_source(record: SourceMaskRecord) -> bool:
    return bool(
        record.decode_ok
        and record.image_exists
        and record.image_path
        and record.mask_area_pixels > 0
    )


def materialize_source_outputs(
    case_dir: Path,
    ranked: Sequence[SourceMaskRecord],
    decoded_masks: Mapping[Tuple[str, int], np.ndarray],
) -> Tuple[SourceMaskRecord, Dict[str, Dict[str, Any]]]:
    valid = [record for record in ranked if usable_source(record)]
    if not valid:
        raise RuntimeError(
            "No source frame has both a decodable non-empty mask and an image."
        )
    global_best = valid[0]

    per_view_best: Dict[str, SourceMaskRecord] = {}
    for record in valid:
        per_view_best.setdefault(record.view_name, record)

    summary: Dict[str, Dict[str, Any]] = {}
    for view_name, record in sorted(per_view_best.items()):
        out_dir = case_dir / "source_view_bests" / slugify(view_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(record.image_path).convert("RGB")
        native_mask = decoded_masks[(record.view_name, record.frame_id)]
        aligned_mask = align_mask_to_image(native_mask, image)

        image.save(out_dir / "best_frame.jpg", quality=95)

        # 与 JPG 对齐的 mask，供后续查看和处理。
        save_binary_mask(aligned_mask, out_dir / "best_mask.png")

        # 保留 annotation 原始分辨率的 mask。
        save_binary_mask(native_mask, out_dir / "best_mask_native.png")

        save_mask_overlay(
            image,
            aligned_mask,
            out_dir / "best_mask_overlay.png",
        )
        summary[view_name] = {
            "view_name": view_name,
            "frame_id": record.frame_id,
            "mask_area_pixels": record.mask_area_pixels,
            "mask_area_ratio": record.mask_area_ratio,
            "frame_path": str(
                Path("source_view_bests") / slugify(view_name) / "best_frame.jpg"
            ),
            "mask_path": str(
                Path("source_view_bests") / slugify(view_name) / "best_mask.png"
            ),
            "overlay_path": str(
                Path("source_view_bests")
                / slugify(view_name)
                / "best_mask_overlay.png"
            ),
        }

    best_image = Image.open(global_best.image_path).convert("RGB")
    native_best_mask = decoded_masks[
        (global_best.view_name, global_best.frame_id)
    ]
    aligned_best_mask = align_mask_to_image(
        native_best_mask,
        best_image,
    )

    best_image.save(
        case_dir / "source_best_frame.jpg",
        quality=95,
    )

    # 与 source JPG 相同分辨率。
    save_binary_mask(
        aligned_best_mask,
        case_dir / "source_best_mask.png",
    )

    # annotation 原始分辨率。
    save_binary_mask(
        native_best_mask,
        case_dir / "source_best_mask_native.png",
    )

    save_mask_overlay(
        best_image,
        aligned_best_mask,
        case_dir / "source_best_mask_overlay.png",
    )
    return global_best, summary


def write_source_rankings(
    case_dir: Path,
    ranked: Sequence[SourceMaskRecord],
    per_view_best: Mapping[str, Dict[str, Any]],
) -> None:
    payload = {
        "ranking_rule": (
            "valid image + decodable non-empty mask, then mask_area_ratio, "
            "then mask_area_pixels"
        ),
        "per_source_view_best": per_view_best,
        "frames": [asdict(record) for record in ranked],
    }
    write_json(case_dir / "source_mask_area_ranking.json", payload)

    fields = [
        "rank_global",
        "rank_within_view",
        "object_name",
        "view_name",
        "frame_id",
        "image_path",
        "image_exists",
        "mask_height",
        "mask_width",
        "mask_area_pixels",
        "mask_area_ratio",
        "decode_ok",
        "error",
    ]
    with (case_dir / "source_mask_area_ranking.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in ranked:
            writer.writerow(asdict(record))


def median_positive_gap(frame_ids: Sequence[int]) -> Optional[float]:
    gaps = [b - a for a, b in zip(frame_ids, frame_ids[1:]) if b > a]
    return float(statistics.median(gaps)) if gaps else None


def split_contiguous_runs(
    frames: Sequence[FrameImage], max_gap: Optional[float]
) -> List[List[FrameImage]]:
    if not frames:
        return []
    runs: List[List[FrameImage]] = [[frames[0]]]
    for previous, current in zip(frames, frames[1:]):
        if max_gap is not None and current.frame_id - previous.frame_id > max_gap:
            runs.append([current])
        else:
            runs[-1].append(current)
    return runs


def sliding_windows(
    items: Sequence[FrameImage], size: int, stride: int
) -> List[List[FrameImage]]:
    if size <= 0 or stride <= 0:
        raise ValueError("Window size and stride must be positive.")
    if len(items) < size:
        return []
    starts = list(range(0, len(items) - size + 1, stride))
    tail_start = len(items) - size
    if not starts or starts[-1] != tail_start:
        starts.append(tail_start)
    return [list(items[start : start + size]) for start in starts]


def uniformly_limit(items: Sequence[Any], limit: int) -> List[Any]:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]
    indexes = sorted(
        {round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)}
    )
    return [items[index] for index in indexes]


def create_contact_sheet(
    frames: Sequence[FrameImage],
    output_path: Path,
    columns: int,
    cell_width: int,
    cell_height: int,
    header_height: int,
    jpeg_quality: int,
) -> Dict[str, Any]:
    columns = max(1, columns)
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    cells: List[Dict[str, Any]] = []

    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        cell_x = column * cell_width
        cell_y = row * cell_height
        with Image.open(frame.path) as source:
            source_rgb = source.convert("RGB")
            original_width, original_height = source_rgb.size
            fitted = ImageOps.contain(
                source_rgb,
                (cell_width, cell_height - header_height),
                method=Image.Resampling.LANCZOS,
            )
        image_x = cell_x + (cell_width - fitted.width) // 2
        image_y = (
            cell_y
            + header_height
            + (cell_height - header_height - fitted.height) // 2
        )
        sheet.paste(fitted, (image_x, image_y))
        draw.rectangle(
            [
                cell_x,
                cell_y,
                cell_x + cell_width - 1,
                cell_y + cell_height - 1,
            ],
            outline="black",
            width=1,
        )
        draw.text(
            (cell_x + 6, cell_y + 7),
            f"idx={index:02d}  frame={frame.frame_id}",
            fill="black",
            font=font,
        )
        cells.append(
            {
                "cell_index": index,
                "frame_id": frame.frame_id,
                "original_image_size": [original_width, original_height],
                "cell_xyxy": [
                    cell_x,
                    cell_y,
                    cell_x + cell_width,
                    cell_y + cell_height,
                ],
                "image_xyxy": [
                    image_x,
                    image_y,
                    image_x + fitted.width,
                    image_y + fitted.height,
                ],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=jpeg_quality)
    return {
        "sheet_size": [sheet.width, sheet.height],
        "columns": columns,
        "rows": rows,
        "cell_width": cell_width,
        "cell_height": cell_height,
        "header_height": header_height,
        "cells": cells,
    }


def representative_frame_ids(
    frame_ids: Sequence[int], count: int = 5
) -> List[int]:
    if not frame_ids:
        return []
    count = min(count, len(frame_ids))
    if count == 1:
        return [frame_ids[len(frame_ids) // 2]]
    indexes = sorted(
        {
            round(i * (len(frame_ids) - 1) / (count - 1))
            for i in range(count)
        }
    )
    return [frame_ids[index] for index in indexes]


def generate_target_windows(
    case: CaseSpec,
    case_dir: Path,
    target_prefix: str,
    window_sizes: Sequence[int],
    window_stride: int,
    window_ratios: Sequence[float],
    window_stride_ratio: float,
    target_sample_every: int,
    max_gap_factor: float,
    max_windows_per_cam: int,
    sheet_columns: int,
    cell_width: int,
    cell_height: int,
    header_height: int,
    jpeg_quality: int,
) -> Tuple[Dict[str, Any], List[str]]:
    target_dirs = sorted(
        path
        for path in case.take_dir.iterdir()
        if path.is_dir() and path.name.startswith(target_prefix)
    )
    warnings: List[str] = []
    index: Dict[str, Any] = {
        "windowing": {
            "window_sizes": list(window_sizes),
            "window_stride_sampled_frames": window_stride,
            "target_sample_every_available_images": target_sample_every,
            "max_gap_factor": max_gap_factor,
            "max_windows_per_cam": max_windows_per_cam,
        },
        "cameras": {},
    }

    for target_dir in target_dirs:
        frame_map, frame_warnings = build_frame_map(target_dir)
        warnings.extend(frame_warnings)
        all_frames = [
            FrameImage(frame_id, path)
            for frame_id, path in sorted(frame_map.items())
        ]
        sampled_frames = all_frames[::target_sample_every]
        full_ids = [frame.frame_id for frame in all_frames]
        sampled_ids = [frame.frame_id for frame in sampled_frames]
        
        total_sampled_frames = len(sampled_frames)

        if total_sampled_frames < 2:
            warnings.append(
                f"{target_dir.name} has fewer than 2 sampled frames; skipping."
            )
            continue

        if window_ratios:
            size_to_requested_ratio: Dict[int, float] = {}

            for ratio in sorted(set(window_ratios)):
                window_size = max(
                    2,
                    int(round(total_sampled_frames * ratio)),
                )
                window_size = min(window_size, total_sampled_frames)
                size_to_requested_ratio[window_size] = float(ratio)

            window_specs = sorted(
                (window_size, requested_ratio)
                for window_size, requested_ratio
                in size_to_requested_ratio.items()
            )

            effective_window_stride = max(
                1,
                int(round(total_sampled_frames * window_stride_ratio)),
            )
        else:
            window_specs = [
                (
                    int(window_size),
                    float(window_size / total_sampled_frames),
                )
                for window_size in sorted(set(window_sizes))
                if 2 <= window_size <= total_sampled_frames
            ]

            effective_window_stride = window_stride
            
        full_step = median_positive_gap(full_ids)
        sampled_step = median_positive_gap(sampled_ids)
        max_gap = (
            sampled_step * max_gap_factor if sampled_step is not None else None
        )
        runs = split_contiguous_runs(sampled_frames, max_gap)

        candidate_windows: List[Tuple[int, List[FrameImage], int]] = []
        candidate_windows: List[
            Tuple[int, List[FrameImage], int, float]
        ] = []

        for run_index, run in enumerate(runs):
            for window_size, requested_ratio in window_specs:
                if window_size > len(run):
                    continue

                for window in sliding_windows(
                    run,
                    window_size,
                    effective_window_stride,
                ):
                    candidate_windows.append(
                        (
                            run_index,
                            window,
                            window_size,
                            requested_ratio,
                        )
                    )

        camera_windows: List[Dict[str, Any]] = []
        camera_out = case_dir / "target_temporal_windows" / target_dir.name
        camera_out.mkdir(parents=True, exist_ok=True)
        for window_index, (
            run_index,
            window,
            window_size,
            requested_ratio,
        ) in enumerate(candidate_windows):
            frame_ids = [frame.frame_id for frame in window]
            gaps = [b - a for a, b in zip(frame_ids, frame_ids[1:])]
            filename = (
                f"window_{window_index:04d}_frames_"
                f"{frame_ids[0]}_{frame_ids[-1]}.jpg"
            )
            relative_sheet_path = (
                Path("target_temporal_windows") / target_dir.name / filename
            )
            layout = create_contact_sheet(
                window,
                case_dir / relative_sheet_path,
                columns=sheet_columns,
                cell_width=cell_width,
                cell_height=cell_height,
                header_height=header_height,
                jpeg_quality=jpeg_quality,
            )
            camera_windows.append(
                {
                    "window_id": f"{target_dir.name}_window_{window_index:04d}",
                    "cam": target_dir.name,
                    "run_index": run_index,
                    "window_size_sampled_frames": window_size,
                    "start_frame": frame_ids[0],
                    "end_frame": frame_ids[-1],
                    "frame_span": frame_ids[-1] - frame_ids[0],
                    "frame_ids": frame_ids,
                    "representative_frame_ids": representative_frame_ids(
                        frame_ids
                    ),
                    "frame_gaps": gaps,
                    "max_frame_gap": max(gaps) if gaps else 0,
                    "mean_frame_gap": (
                        float(statistics.mean(gaps)) if gaps else 0.0
                    ),
                    "expected_sampled_frame_gap": sampled_step,
                    "continuity_ok": bool(
                        not gaps or max_gap is None or max(gaps) <= max_gap
                    ),
                    "contact_sheet": str(relative_sheet_path),
                    "sheet_layout": layout,
                }
            )

        index["cameras"][target_dir.name] = {
            "source_frame_count": len(all_frames),
            "sampled_frame_count": len(sampled_frames),
            "full_median_frame_gap": full_step,
            "sampled_median_frame_gap": sampled_step,
            "continuity_split_threshold": max_gap,
            "contiguous_run_lengths": [len(run) for run in runs],
            "window_count": len(camera_windows),
            "windows": camera_windows,
        }

    if not target_dirs:
        warnings.append(
            f"No target directories beginning with {target_prefix!r} "
            f"in {case.take_dir}."
        )
    return index, warnings


def build_prompt(
    case_metadata: Mapping[str, Any], temporal_index: Mapping[str, Any]
) -> str:
    cameras = temporal_index.get("cameras", {})
    window_counts = {
        name: data.get("window_count", 0) for name, data in cameras.items()
    }
    source = case_metadata["source_best"]
    expected_schema = {
        "case_id": case_metadata["case_id"],
        "target_object": case_metadata["target_object"],
        "source_best_view": source["view_name"],
        "source_best_frame": source["frame_id"],
        "source_mask_area_ratio": source["mask_area_ratio"],
        "status": "success | uncertain | failed",
        "best_cam": "camXX or null",
        "best_segment": {
            "window_id": "camXX_window_0000",
            "cam": "camXX",
            "start_frame": 0,
            "end_frame": 0,
            "representative_frames": [
                {
                    "frame_id": 0,
                    "target_region_xyxy": [0.0, 0.0, 1.0, 1.0],
                    "visibility": 0.0,
                    "occlusion": "none | low | medium | high",
                }
            ],
            "target_region_summary": {
                "coordinate_system": "normalized_xyxy_per_frame",
                "stable_region_xyxy": [0.0, 0.0, 1.0, 1.0],
                "position_drift": "low | medium | high",
                "scale_change": "low | medium | high",
                "region_explanation": (
                    "why this spatial range is stable and distinctive"
                ),
            },
            "scores": {
                "visibility_consistency": 0.0,
                "average_apparent_size": 0.0,
                "minimum_apparent_size": 0.0,
                "occlusion_consistency": 0.0,
                "identity_confidence": 0.0,
                "segmentation_suitability": 0.0,
                "temporal_stability": 0.0,
                "overall": 0.0,
            },
            "confidence": 0.0,
            "reason_selected": "",
            "recommended_sam_prompt": "",
        },
        "alternative_segments": [],
        "rejected_segments": [
            {
                "window_id": "camXX_window_0001",
                "cam": "camXX",
                "start_frame": 0,
                "end_frame": 0,
                "reason_rejected": "",
                "region_difference_from_best": "",
            }
        ],
        "uncertainty": "",
    }
    return f"""# Temporal cross-view analysis task

Analyze this downloaded case folder locally. Do not edit the source-mask assets and do not invent a target-view mask.

## Case
- case_id: `{case_metadata['case_id']}`
- target object: `{case_metadata['target_object']}`
- source best view/frame: `{source['view_name']}` / `{source['frame_id']}`
- source mask area ratio: `{source['mask_area_ratio']:.8f}`
- target window counts: `{json.dumps(window_counts, ensure_ascii=False)}`

## Files to use
1. `source_best_frame.jpg`
2. `source_best_mask.png`
3. `source_best_mask_overlay.png` — authoritative identity reference.
4. `metadata.json`
5. `temporal_window_index.json`
6. Every contact sheet referenced by `temporal_window_index.json`.

## Required decision procedure
1. Inspect the masked source identity first. Describe distinctive appearance, shape, material, color, nearby context, and likely confusers.
2. Evaluate complete temporal windows, not isolated attractive frames. A window with one clear frame but unstable/absent neighbors must not win.
3. Score each serious candidate on: target visibility consistency, average apparent size, minimum apparent size, occlusion consistency, identity confidence, segmentation suitability, and temporal stability.
4. Treat `continuity_ok=false`, major frame gaps, identity switching, repeated disappearance, severe truncation, or heavy occlusion as rejection evidence.
5. Compare spatial regions, not only camera names. For the best segment and representative frames, estimate the target's normalized bounding region `[x_min, y_min, x_max, y_max]` relative to each original frame. Explain position drift, scale change, and why the region is more stable than rejected candidates.
6. Rank the best three non-overlapping or meaningfully different candidate segments. Highly overlapping windows from the same event should not occupy all top positions; merge or keep only the strongest one.
7. Mark the case `uncertain` or `failed` instead of forcing a selection when the target is transparent, too small, confused with similar objects, absent, or not temporally stable.
8. Recommended SAM3 prompt must be concrete and identity-specific. Do not claim that an AI-estimated box is a ground-truth mask.

## Output
Write strict JSON to `temporal_analysis_result.json`. Use numbers in `[0, 1]` for every score and confidence. `representative_frames` should include 3–5 frames spanning the chosen segment, each with an estimated normalized region.

```json
{json.dumps(expected_schema, indent=2, ensure_ascii=False)}
```

After writing the JSON, run:

```bash
python render_temporal_analysis_results.py --case-dir . --analysis-json temporal_analysis_result.json
```

This produces annotated selected/rejected sheets, a selected-vs-rejected regional comparison, and a method feasibility report.
"""


def process_case(case: CaseSpec, args: argparse.Namespace) -> Dict[str, Any]:
    case_dir = args.output_root / case.case_id
    if case_dir.exists():
        if not args.overwrite:
            return {
                "case_id": case.case_id,
                "status": "skipped_existing",
                "path": str(case_dir),
            }
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    annotation = load_json(case.annotation_path)
    ranked, decoded_masks, source_warnings = rank_source_masks(
        annotation, case, args.source_prefix
    )
    global_best, per_view_best = materialize_source_outputs(
        case_dir, ranked, decoded_masks
    )
    write_source_rankings(case_dir, ranked, per_view_best)

    temporal_index, target_warnings = generate_target_windows(
        case=case,
        case_dir=case_dir,
        target_prefix=args.target_prefix,
        window_sizes=args.window_sizes,
        window_stride=args.window_stride,
        target_sample_every=args.target_sample_every,
        max_gap_factor=args.max_gap_factor,
        max_windows_per_cam=args.max_windows_per_cam,
        sheet_columns=args.sheet_columns,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        header_height=args.cell_header_height,
        jpeg_quality=args.jpeg_quality,
    )
    write_json(case_dir / "temporal_window_index.json", temporal_index)

    metadata = {
        "schema_version": 1,
        "case_id": case.case_id,
        "take_id": case.take_id,
        "target_object": case.object_name,
        "take_dir": str(case.take_dir),
        "annotation_path": str(case.annotation_path),
        "source_best": {
            "view_name": global_best.view_name,
            "frame_id": global_best.frame_id,
            "mask_area_pixels": global_best.mask_area_pixels,
            "mask_area_ratio": global_best.mask_area_ratio,
            "source_frame": "source_best_frame.jpg",
            "source_mask": "source_best_mask.png",
            "source_mask_overlay": "source_best_mask_overlay.png",
        },
        "per_source_view_best": per_view_best,
        "target_cameras": {
            camera: {
                "window_count": data["window_count"],
                "sampled_frame_count": data["sampled_frame_count"],
            }
            for camera, data in temporal_index["cameras"].items()
        },
        "warnings": source_warnings + target_warnings,
        "pipeline_notes": {
            "source_selection": "Ground-truth COCO RLE mask area ratio.",
            "target_selection": (
                "Deferred to local Codex/ChatGPT temporal analysis; "
                "no target mask generated here."
            ),
            "portable_paths": (
                "Paths inside temporal_window_index.json are relative "
                "to the case directory."
            ),
        },
    }
    write_json(case_dir / "metadata.json", metadata)
    (case_dir / "prompt_for_temporal_analysis.md").write_text(
        build_prompt(metadata, temporal_index), encoding="utf-8"
    )
    return {
        "case_id": case.case_id,
        "status": "generated",
        "path": str(case_dir),
        "target_object": case.object_name,
        "source_best_view": global_best.view_name,
        "source_best_frame": global_best.frame_id,
        "source_mask_area_ratio": global_best.mask_area_ratio,
        "target_window_count": sum(
            data["window_count"]
            for data in temporal_index["cameras"].values()
        ),
        "warnings": len(metadata["warnings"]),
    }


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_root.is_dir():
        raise SystemExit(
            f"Data root does not exist or is not a directory: {args.data_root}"
        )
    if any(size <= 1 for size in args.window_sizes):
        raise SystemExit("Every window size must be at least 2.")
    if args.window_stride <= 0 or args.target_sample_every <= 0:
        raise SystemExit(
            "Window stride and target sample interval must be positive."
        )
    if args.max_gap_factor <= 1.0:
        raise SystemExit("--max-gap-factor should be greater than 1.0.")


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(
        args.data_root, args.annotation_name, args.object_name
    )
    if args.case_id:
        wanted_case_ids = set(args.case_id)
        cases = [
            case for case in cases if case.case_id in wanted_case_ids
        ]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit("No matching take/object cases were discovered.")

    batch_results: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
        try:
            result = process_case(case, args)
        except Exception as exc:
            result = {
                "case_id": case.case_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(
                f"[ERROR] {case.case_id}: {result['error']}",
                file=sys.stderr,
            )
            if args.fail_fast:
                batch_results.append(result)
                break
        batch_results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    summary = {
        "schema_version": 1,
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "requested_case_count": len(cases),
        "generated_count": sum(
            result["status"] == "generated" for result in batch_results
        ),
        "failed_count": sum(
            result["status"] == "failed" for result in batch_results
        ),
        "skipped_count": sum(
            result["status"] == "skipped_existing"
            for result in batch_results
        ),
        "cases": batch_results,
    }
    write_json(args.output_root / "batch_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
