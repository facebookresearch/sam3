#!/usr/bin/env python3
"""Generate Ego-Exo temporal assets using camera-relative window ratios.

This wrapper keeps the stable source-mask logic from
`generate_temporal_cross_view_assets.py`, but generates target windows whose
length is a fraction of the complete sampled third-person camera sequence.

Example:
    python generate_temporal_cross_view_assets_ratio.py \
      --data-root /path/to/data \
      --output-root /path/to/output \
      --window-ratios 0.20 0.25 0.30 \
      --window-stride-ratio 0.05 \
      --max-cases 1 \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import generate_temporal_cross_view_assets as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate cross-view assets with target windows covering a "
            "camera-relative fraction of the complete sampled video."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("temporal_cross_view_assets_ratio"),
    )
    parser.add_argument("--annotation-name", default="annotation.json")
    parser.add_argument("--source-prefix", default="aria")
    parser.add_argument("--target-prefix", default="cam")
    parser.add_argument(
        "--window-ratios",
        type=float,
        nargs="+",
        default=[0.20, 0.25, 0.30],
        help=(
            "Fractions of each target camera's sampled frames used as window "
            "lengths. Default: 0.20 0.25 0.30."
        ),
    )
    parser.add_argument(
        "--window-stride-ratio",
        type=float,
        default=0.05,
        help=(
            "Sliding stride as a fraction of each target camera's sampled "
            "frames. Default: 0.05."
        ),
    )
    parser.add_argument(
        "--target-sample-every",
        type=int,
        default=1,
        help="Use every Nth available target image before windowing.",
    )
    parser.add_argument(
        "--max-gap-factor",
        type=float,
        default=2.5,
        help=(
            "Split a sampled sequence when a frame-id gap exceeds the median "
            "sampled gap multiplied by this factor."
        ),
    )
    parser.add_argument("--sheet-columns", type=int, default=6)
    parser.add_argument("--cell-width", type=int, default=320)
    parser.add_argument("--cell-height", type=int, default=220)
    parser.add_argument("--cell-header-height", type=int, default=28)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--max-windows-per-cam",
        type=int,
        default=0,
        help=(
            "0 keeps every candidate. Positive values uniformly retain at "
            "most this many windows per camera."
        ),
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Process only exact generated case IDs. May be repeated.",
    )
    parser.add_argument(
        "--object-name",
        action="append",
        default=[],
        help="Process only exact annotation object names. May be repeated.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="0 processes every discovered take-object case.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def ratio_window_specs(
    total_sampled_frames: int,
    ratios: Sequence[float],
) -> List[Tuple[int, float]]:
    """Return unique (window_size, requested_ratio) pairs.

    Very short videos can map multiple ratios to the same integer size. In that
    case the ratio whose requested value is closest to the actual integer ratio
    is retained.
    """
    by_size: Dict[int, float] = {}
    for ratio in sorted(set(float(value) for value in ratios)):
        size = max(2, min(total_sampled_frames, round(total_sampled_frames * ratio)))
        actual = size / total_sampled_frames
        previous = by_size.get(size)
        if previous is None or abs(actual - ratio) < abs(actual - previous):
            by_size[size] = ratio
    return sorted(by_size.items())


def generate_target_windows_ratio(
    case: base.CaseSpec,
    case_dir: Path,
    target_prefix: str,
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
        "schema_version": 2,
        "windowing": {
            "mode": "camera_relative_ratio",
            "window_ratios": list(window_ratios),
            "window_stride_ratio": window_stride_ratio,
            "target_sample_every_available_images": target_sample_every,
            "max_gap_factor": max_gap_factor,
            "max_windows_per_cam": max_windows_per_cam,
            "ratio_denominator": (
                "number of available sampled frames in the complete camera"
            ),
        },
        "cameras": {},
    }

    for target_dir in target_dirs:
        frame_map, frame_warnings = base.build_frame_map(target_dir)
        warnings.extend(frame_warnings)
        all_frames = [
            base.FrameImage(frame_id, path)
            for frame_id, path in sorted(frame_map.items())
        ]
        sampled_frames = all_frames[::target_sample_every]
        full_ids = [frame.frame_id for frame in all_frames]
        sampled_ids = [frame.frame_id for frame in sampled_frames]
        total_sampled = len(sampled_frames)

        if total_sampled < 2:
            warnings.append(
                f"{target_dir.name} has fewer than 2 sampled frames; skipping."
            )
            index["cameras"][target_dir.name] = {
                "source_frame_count": len(all_frames),
                "sampled_frame_count": total_sampled,
                "window_count": 0,
                "windows": [],
            }
            continue

        full_step = base.median_positive_gap(full_ids)
        sampled_step = base.median_positive_gap(sampled_ids)
        max_gap = (
            sampled_step * max_gap_factor if sampled_step is not None else None
        )
        runs = base.split_contiguous_runs(sampled_frames, max_gap)
        specs = ratio_window_specs(total_sampled, window_ratios)
        stride = max(1, round(total_sampled * window_stride_ratio))
        video_start = sampled_ids[0]
        video_end = sampled_ids[-1]
        video_span = video_end - video_start

        candidate_windows: List[
            Tuple[int, List[base.FrameImage], int, float]
        ] = []
        missing_specs: List[Dict[str, Any]] = []

        for window_size, requested_ratio in specs:
            eligible_run_count = 0
            for run_index, run in enumerate(runs):
                if len(run) < window_size:
                    continue
                eligible_run_count += 1
                for window in base.sliding_windows(run, window_size, stride):
                    candidate_windows.append(
                        (run_index, window, window_size, requested_ratio)
                    )
            if eligible_run_count == 0:
                missing_specs.append(
                    {
                        "requested_ratio": requested_ratio,
                        "window_size_sampled_frames": window_size,
                        "reason": (
                            "No contiguous run is long enough for this "
                            "whole-camera-relative window size."
                        ),
                    }
                )

        candidate_windows.sort(
            key=lambda item: (
                item[1][0].frame_id,
                item[2],
                item[1][-1].frame_id,
                item[3],
            )
        )
        candidate_windows = base.uniformly_limit(
            candidate_windows, max_windows_per_cam
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
            frame_span = frame_ids[-1] - frame_ids[0]
            actual_sampled_ratio = len(frame_ids) / total_sampled
            actual_span_ratio = (
                frame_span / video_span if video_span > 0 else 0.0
            )
            filename = (
                f"window_{window_index:04d}_ratio_{requested_ratio:.2f}_"
                f"frames_{frame_ids[0]}_{frame_ids[-1]}.jpg"
            )
            relative_sheet_path = (
                Path("target_temporal_windows") / target_dir.name / filename
            )
            layout = base.create_contact_sheet(
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
                    "requested_video_ratio": requested_ratio,
                    "actual_sampled_frame_ratio": actual_sampled_ratio,
                    "actual_frame_span_ratio": actual_span_ratio,
                    "ratio_error_sampled_frames": (
                        actual_sampled_ratio - requested_ratio
                    ),
                    "window_size_sampled_frames": window_size,
                    "start_frame": frame_ids[0],
                    "end_frame": frame_ids[-1],
                    "frame_span": frame_span,
                    "frame_ids": frame_ids,
                    "representative_frame_ids": base.representative_frame_ids(
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
            "sampled_frame_count": total_sampled,
            "video_start_frame": video_start,
            "video_end_frame": video_end,
            "video_frame_span": video_span,
            "full_median_frame_gap": full_step,
            "sampled_median_frame_gap": sampled_step,
            "continuity_split_threshold": max_gap,
            "contiguous_run_lengths": [len(run) for run in runs],
            "requested_window_ratios": list(window_ratios),
            "effective_window_specs": [
                {
                    "window_size_sampled_frames": size,
                    "requested_video_ratio": ratio,
                    "actual_sampled_frame_ratio": size / total_sampled,
                }
                for size, ratio in specs
            ],
            "effective_window_stride_sampled_frames": stride,
            "unavailable_window_specs": missing_specs,
            "window_count": len(camera_windows),
            "windows": camera_windows,
        }

    if not target_dirs:
        warnings.append(
            f"No target directories beginning with {target_prefix!r} "
            f"in {case.take_dir}."
        )
    return index, warnings


def build_ratio_prompt(
    case_metadata: Mapping[str, Any],
    temporal_index: Mapping[str, Any],
) -> str:
    cameras = temporal_index.get("cameras", {})
    window_counts = {
        name: data.get("window_count", 0) for name, data in cameras.items()
    }
    source = case_metadata["source_best"]
    expected_schema = {
        "schema_version": 2,
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
            "requested_video_ratio": 0.25,
            "actual_sampled_frame_ratio": 0.25,
            "actual_frame_span_ratio": 0.25,
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
                "region_explanation": "",
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
        "rejected_segments": [],
        "uncertainty": "",
    }
    return f"""# Whole-camera-relative temporal cross-view analysis

Analyze this case locally. The goal is to select the relatively best continuous
portion covering approximately 20%–30% of a complete third-person camera
sequence. Do not choose a shorter clip simply because one frame looks clear.

## Case
- case_id: `{case_metadata['case_id']}`
- target object: `{case_metadata['target_object']}`
- source best view/frame: `{source['view_name']}` / `{source['frame_id']}`
- source mask area ratio: `{source['mask_area_ratio']:.8f}`
- target window counts: `{json.dumps(window_counts, ensure_ascii=False)}`

## Required files
1. `source_best_mask_overlay.png` — authoritative source identity.
2. `source_best_frame.jpg`
3. `source_best_mask.png`
4. `metadata.json`
5. `temporal_window_index.json`
6. Every referenced temporal contact sheet.

## Selection rules
1. Only compare windows produced by the ratio pipeline. Their length is based on
   the complete sampled frame count of the corresponding camera.
2. Prefer windows whose `actual_sampled_frame_ratio` is between 0.20 and 0.30.
3. Evaluate the complete window rather than isolated attractive frames.
4. Prefer high minimum visibility, identity consistency, manageable occlusion,
   stable apparent size, stable spatial location, and suitability for SAM3
   initialization and propagation.
5. Short accidental clear moments must not outweigh poor surrounding frames.
6. Copy `requested_video_ratio`, `actual_sampled_frame_ratio`, and
   `actual_frame_span_ratio` exactly from the selected window index entry.
7. Rank meaningfully different alternatives; do not fill the ranking with
   heavily overlapping windows from the same event.
8. Use normalized per-frame `[x_min, y_min, x_max, y_max]` boxes only as
   AI-estimated localization evidence. They are not masks.
9. Return `uncertain` or `failed` when no 20%–30% window is reliably better than
   the alternatives.

## Output
Write strict JSON to `temporal_analysis_result.json`:

```json
{json.dumps(expected_schema, indent=2, ensure_ascii=False)}
```

After writing it, run the renderer with this case directory. The renderer can
use the same ratio-window contact sheets without modification.
"""


def process_case(case: base.CaseSpec, args: argparse.Namespace) -> Dict[str, Any]:
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

    annotation = base.load_json(case.annotation_path)
    ranked, decoded_masks, source_warnings = base.rank_source_masks(
        annotation, case, args.source_prefix
    )
    global_best, per_view_best = base.materialize_source_outputs(
        case_dir, ranked, decoded_masks
    )
    base.write_source_rankings(case_dir, ranked, per_view_best)

    temporal_index, target_warnings = generate_target_windows_ratio(
        case=case,
        case_dir=case_dir,
        target_prefix=args.target_prefix,
        window_ratios=args.window_ratios,
        window_stride_ratio=args.window_stride_ratio,
        target_sample_every=args.target_sample_every,
        max_gap_factor=args.max_gap_factor,
        max_windows_per_cam=args.max_windows_per_cam,
        sheet_columns=args.sheet_columns,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        header_height=args.cell_header_height,
        jpeg_quality=args.jpeg_quality,
    )
    base.write_json(case_dir / "temporal_window_index.json", temporal_index)

    metadata = {
        "schema_version": 2,
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
                "video_start_frame": data.get("video_start_frame"),
                "video_end_frame": data.get("video_end_frame"),
                "requested_window_ratios": data.get(
                    "requested_window_ratios", []
                ),
            }
            for camera, data in temporal_index["cameras"].items()
        },
        "warnings": source_warnings + target_warnings,
        "pipeline_notes": {
            "source_selection": "Ground-truth COCO RLE mask area ratio.",
            "target_window_definition": (
                "20%-30% camera-relative windows based on complete sampled "
                "target-camera frame count."
            ),
            "target_selection": (
                "Deferred to local Codex/ChatGPT analysis; no target mask is "
                "generated or fabricated."
            ),
        },
    }
    base.write_json(case_dir / "metadata.json", metadata)
    (case_dir / "prompt_for_temporal_analysis.md").write_text(
        build_ratio_prompt(metadata, temporal_index),
        encoding="utf-8",
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
    if not args.window_ratios:
        raise SystemExit("At least one --window-ratios value is required.")
    if any(ratio <= 0.0 or ratio >= 1.0 for ratio in args.window_ratios):
        raise SystemExit("Every --window-ratios value must be between 0 and 1.")
    if not 0.0 < args.window_stride_ratio < 1.0:
        raise SystemExit("--window-stride-ratio must be between 0 and 1.")
    if args.target_sample_every <= 0:
        raise SystemExit("--target-sample-every must be positive.")
    if args.max_gap_factor <= 1.0:
        raise SystemExit("--max-gap-factor must be greater than 1.0.")
    if args.sheet_columns <= 0:
        raise SystemExit("--sheet-columns must be positive.")
    if args.cell_width <= 0 or args.cell_height <= 0:
        raise SystemExit("Cell dimensions must be positive.")
    if args.max_windows_per_cam < 0:
        raise SystemExit("--max-windows-per-cam cannot be negative.")


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)

    cases = base.discover_cases(
        args.data_root,
        args.annotation_name,
        args.object_name,
    )
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case.case_id in wanted]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit("No matching take-object cases were discovered.")

    results: List[Dict[str, Any]] = []
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
                results.append(result)
                break
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    summary = {
        "schema_version": 2,
        "windowing_mode": "camera_relative_ratio",
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "window_ratios": list(args.window_ratios),
        "window_stride_ratio": args.window_stride_ratio,
        "requested_case_count": len(cases),
        "generated_count": sum(r["status"] == "generated" for r in results),
        "failed_count": sum(r["status"] == "failed" for r in results),
        "skipped_count": sum(
            r["status"] == "skipped_existing" for r in results
        ),
        "cases": results,
    }
    base.write_json(args.output_root / "batch_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
