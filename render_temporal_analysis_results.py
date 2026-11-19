#!/usr/bin/env python3
"""Render and validate local AI results for temporal cross-view assets.

The script consumes a case directory produced by
`generate_temporal_cross_view_assets.py` plus `temporal_analysis_result.json`.
It never creates a mask. Region boxes are explicitly labeled as AI-estimated
localization evidence, not ground truth.
"""

from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render selected/rejected temporal segment comparisons."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case-dir", type=Path)
    group.add_argument("--assets-root", type=Path)
    parser.add_argument(
        "--analysis-json",
        type=Path,
        help=(
            "Analysis JSON for --case-dir. Defaults to "
            "<case-dir>/temporal_analysis_result.json."
        ),
    )
    parser.add_argument(
        "--analysis-filename",
        default="temporal_analysis_result.json",
        help="Filename used in each case when --assets-root is provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for one case. Defaults to <case-dir>/analysis_outputs.",
    )
    parser.add_argument("--max-rejected", type=int, default=3)
    parser.add_argument("--max-alternatives", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def clamp01(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def normalized_bbox(
    value: Any,
) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (clamp01(component) for component in value)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def all_windows(index: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    for camera_data in index.get("cameras", {}).values():
        for window in camera_data.get("windows", []):
            yield window


def resolve_window(
    temporal_index: Mapping[str, Any], segment: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    window_id = segment.get("window_id")
    cam = segment.get("cam") or segment.get("best_cam")
    start = segment.get("start_frame")
    end = segment.get("end_frame")

    windows = list(all_windows(temporal_index))
    if window_id:
        for window in windows:
            if window.get("window_id") == window_id:
                return window
    candidates = []
    for window in windows:
        if cam and window.get("cam") != cam:
            continue
        if start is not None and int(window.get("start_frame")) != int(start):
            continue
        if end is not None and int(window.get("end_frame")) != int(end):
            continue
        candidates.append(window)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return min(
            candidates,
            key=lambda window: abs(
                int(window["end_frame"]) - int(window["start_frame"])
            ),
        )
    return None


def representative_regions(
    segment: Mapping[str, Any]
) -> Dict[int, Tuple[float, float, float, float]]:
    regions: Dict[int, Tuple[float, float, float, float]] = {}
    for item in segment.get("representative_frames", []):
        if isinstance(item, int):
            continue
        if not isinstance(item, Mapping) or "frame_id" not in item:
            continue
        bbox = normalized_bbox(
            item.get("target_region_xyxy")
            or item.get("region_xyxy")
            or item.get("bbox")
        )
        if bbox:
            regions[int(item["frame_id"])] = bbox
    return regions


def stable_region(
    segment: Mapping[str, Any]
) -> Optional[Tuple[float, float, float, float]]:
    summary = segment.get("target_region_summary", {})
    if not isinstance(summary, Mapping):
        return None
    return normalized_bbox(summary.get("stable_region_xyxy"))


def map_bbox_to_sheet(
    bbox: Tuple[float, float, float, float], image_xyxy: Sequence[int]
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    left, top, right, bottom = (int(value) for value in image_xyxy)
    width = right - left
    height = bottom - top
    return (
        round(left + x1 * width),
        round(top + y1 * height),
        round(left + x2 * width),
        round(top + y2 * height),
    )


def draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    xyxy: Tuple[int, int, int, int],
    width: int = 3,
    dash: int = 10,
) -> None:
    x1, y1, x2, y2 = xyxy
    for start in range(x1, x2, dash * 2):
        draw.line(
            (start, y1, min(start + dash, x2), y1),
            fill="white",
            width=width,
        )
        draw.line(
            (start, y2, min(start + dash, x2), y2),
            fill="white",
            width=width,
        )
    for start in range(y1, y2, dash * 2):
        draw.line(
            (x1, start, x1, min(start + dash, y2)),
            fill="white",
            width=width,
        )
        draw.line(
            (x2, start, x2, min(start + dash, y2)),
            fill="white",
            width=width,
        )


def annotate_sheet(
    case_dir: Path,
    window: Mapping[str, Any],
    segment: Mapping[str, Any],
    output_path: Path,
    label: str,
) -> Dict[str, Any]:
    sheet_path = case_dir / str(window["contact_sheet"])
    image = Image.open(sheet_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    per_frame = representative_regions(segment)
    fallback_region = stable_region(segment)
    drawn_frames: List[int] = []

    for cell in window.get("sheet_layout", {}).get("cells", []):
        frame_id = int(cell["frame_id"])
        bbox = per_frame.get(frame_id)
        is_fallback = False
        if bbox is None and fallback_region is not None and not per_frame:
            bbox = fallback_region
            is_fallback = True
        if bbox is None:
            continue
        mapped = map_bbox_to_sheet(bbox, cell["image_xyxy"])
        if is_fallback:
            draw_dashed_rectangle(draw, mapped)
        else:
            draw.rectangle(mapped, outline="white", width=4)
        text_y = max(int(cell["image_xyxy"][1]), mapped[1] - 14)
        draw.rectangle(
            (mapped[0], text_y, mapped[0] + 105, text_y + 14),
            fill="black",
        )
        draw.text(
            (mapped[0] + 2, text_y + 2),
            f"AI region f={frame_id}",
            fill="white",
            font=font,
        )
        drawn_frames.append(frame_id)

    banner_height = 42
    canvas = Image.new(
        "RGB", (image.width, image.height + banner_height), "white"
    )
    canvas.paste(image, (0, banner_height))
    banner = ImageDraw.Draw(canvas)
    banner.text((8, 7), label, fill="black", font=font)
    banner.text(
        (8, 23),
        "Boxes are AI-estimated localization regions, not ground-truth masks.",
        fill="black",
        font=font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    return {
        "source_sheet": str(window["contact_sheet"]),
        "output": str(output_path.name),
        "drawn_region_frames": drawn_frames,
    }


def safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def segment_title(prefix: str, segment: Mapping[str, Any]) -> str:
    cam = segment.get("cam", "")
    start = segment.get("start_frame", "?")
    end = segment.get("end_frame", "?")
    return f"{prefix}: {cam} frames {start}-{end}"


def add_caption_panel(
    image: Image.Image, title: str, reason: str
) -> Image.Image:
    font = ImageFont.load_default()
    wrapped = textwrap.wrap(reason or "No reason supplied.", width=105)[:5]
    height = 28 + 16 * len(wrapped)
    canvas = Image.new(
        "RGB", (image.width, image.height + height), "white"
    )
    canvas.paste(image, (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill="black", font=font)
    for row, line in enumerate(wrapped):
        draw.text((8, 22 + row * 14), line, fill="black", font=font)
    return canvas


def stack_comparison(
    panels: Sequence[Image.Image], output_path: Path
) -> None:
    if not panels:
        return
    max_width = max(panel.width for panel in panels)
    resized: List[Image.Image] = []
    for panel in panels:
        if panel.width == max_width:
            resized.append(panel)
            continue
        new_height = round(panel.height * max_width / panel.width)
        resized.append(
            panel.resize(
                (max_width, new_height), Image.Resampling.LANCZOS
            )
        )
    gap = 16
    total_height = sum(panel.height for panel in resized) + gap * (
        len(resized) - 1
    )
    canvas = Image.new("RGB", (max_width, total_height), "white")
    y = 0
    for panel in resized:
        canvas.paste(panel, (0, y))
        y += panel.height + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def score_lines(segment: Mapping[str, Any]) -> List[str]:
    scores = segment.get("scores", {})
    if not isinstance(scores, Mapping):
        return []
    ordered = [
        "visibility_consistency",
        "average_apparent_size",
        "minimum_apparent_size",
        "occlusion_consistency",
        "identity_confidence",
        "segmentation_suitability",
        "temporal_stability",
        "overall",
    ]
    lines = []
    for key in ordered:
        if key in scores:
            try:
                lines.append(f"- {key}: {float(scores[key]):.3f}")
            except (TypeError, ValueError):
                lines.append(f"- {key}: {scores[key]}")
    return lines


def create_report(
    case_dir: Path,
    metadata: Mapping[str, Any],
    analysis: Mapping[str, Any],
    rendered: Mapping[str, Any],
    output_path: Path,
) -> None:
    best = analysis.get("best_segment") or {}
    region_summary = (
        best.get("target_region_summary", {})
        if isinstance(best, Mapping)
        else {}
    )
    status = analysis.get("status", "unknown")
    target_windows = metadata.get("target_cameras", {})
    total_windows = sum(
        int(value.get("window_count", 0))
        for value in target_windows.values()
    )
    lines = [
        "# Method feasibility report",
        "",
        f"- Case: `{metadata.get('case_id', case_dir.name)}`",
        (
            f"- Target object: `"
            f"{metadata.get('target_object', analysis.get('target_object', 'unknown'))}`"
        ),
        f"- Analysis status: **{status}**",
        (
            f"- Source reference: `"
            f"{metadata.get('source_best', {}).get('view_name')}` frame `"
            f"{metadata.get('source_best', {}).get('frame_id')}`"
        ),
        (
            f"- Source mask area ratio: `"
            f"{metadata.get('source_best', {}).get('mask_area_ratio')}`"
        ),
        (
            f"- Target cameras/windows inspected: `{len(target_windows)}` "
            f"cameras / `{total_windows}` windows"
        ),
        "",
        "## Selected temporal segment",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- Camera: `{analysis.get('best_cam') or best.get('cam')}`",
                (
                    f"- Frame range: `"
                    f"{best.get('start_frame')}-{best.get('end_frame')}`"
                ),
                f"- Window ID: `{best.get('window_id', 'not supplied')}`",
                f"- Confidence: `{best.get('confidence', 'not supplied')}`",
                f"- Selection reason: {safe_text(best.get('reason_selected'))}",
                (
                    f"- Recommended SAM3 prompt: `"
                    f"{safe_text(best.get('recommended_sam_prompt'))}`"
                ),
                "",
                "### Temporal scores",
                "",
                *score_lines(best),
                "",
                "### Regional evidence",
                "",
                (
                    f"- Stable normalized region: `"
                    f"{region_summary.get('stable_region_xyxy', 'not supplied')}`"
                ),
                (
                    f"- Position drift: `"
                    f"{region_summary.get('position_drift', 'not supplied')}`"
                ),
                (
                    f"- Scale change: `"
                    f"{region_summary.get('scale_change', 'not supplied')}`"
                ),
                (
                    f"- Explanation: "
                    f"{safe_text(region_summary.get('region_explanation'))}"
                ),
            ]
        )
    else:
        lines.append("No best segment was selected.")

    lines.extend(["", "## Alternatives and rejected ranges", ""])
    for index, segment in enumerate(
        analysis.get("alternative_segments", []), start=1
    ):
        lines.append(
            f"- Alternative {index}: `{segment.get('cam', '')}` frames "
            f"`{segment.get('start_frame')}-{segment.get('end_frame')}` — "
            f"{safe_text(segment.get('reason_selected') or segment.get('reason'))}"
        )
    for index, segment in enumerate(
        analysis.get("rejected_segments", []), start=1
    ):
        lines.append(
            f"- Rejected {index}: `{segment.get('cam', '')}` frames "
            f"`{segment.get('start_frame')}-{segment.get('end_frame')}` — "
            f"{safe_text(segment.get('reason_rejected'))} "
            f"Regional difference: "
            f"{safe_text(segment.get('region_difference_from_best'))}"
        )

    lines.extend(
        [
            "",
            "## Uncertainty and feasibility conclusion",
            "",
            (
                safe_text(analysis.get("uncertainty"))
                or "No uncertainty statement supplied."
            ),
            "",
            (
                "The source-frame choice is grounded in decoded annotation "
                "mask area. The target temporal decision is an AI-assisted "
                "visual assessment over contact sheets, not a target mask "
                "annotation. It is feasible for candidate selection only when "
                "identity, continuity, and regional stability are sufficiently "
                "clear. Cases with small, transparent, heavily occluded, or "
                "confusable targets should remain uncertain/failed and should "
                "not be forced into SAM3."
            ),
            "",
            "## Generated artifacts",
            "",
            (
                f"```json\n"
                f"{json.dumps(rendered, indent=2, ensure_ascii=False)}\n"
                f"```"
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_analysis(
    metadata: Mapping[str, Any], analysis: Mapping[str, Any]
) -> List[str]:
    warnings: List[str] = []
    if (
        analysis.get("case_id")
        and analysis.get("case_id") != metadata.get("case_id")
    ):
        warnings.append(
            f"Analysis case_id {analysis.get('case_id')!r} does not match "
            f"metadata {metadata.get('case_id')!r}."
        )
    status = analysis.get("status", "success")
    if status not in {"success", "uncertain", "failed"}:
        warnings.append(f"Unexpected status {status!r}.")
    best = analysis.get("best_segment")
    if status == "success" and not isinstance(best, Mapping):
        warnings.append("status=success but best_segment is missing.")
    if isinstance(best, Mapping):
        scores = best.get("scores", {})
        if isinstance(scores, Mapping):
            for key, value in scores.items():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    warnings.append(
                        f"Score {key!r} is not numeric: {value!r}"
                    )
                    continue
                if not 0.0 <= number <= 1.0:
                    warnings.append(
                        f"Score {key!r} is outside [0, 1]: {number}"
                    )
    return warnings


def process_case(
    case_dir: Path,
    analysis_path: Path,
    output_dir: Path,
    overwrite: bool,
    max_rejected: int,
    max_alternatives: int,
) -> Dict[str, Any]:
    metadata_path = case_dir / "metadata.json"
    index_path = case_dir / "temporal_window_index.json"
    if not metadata_path.exists() or not index_path.exists():
        raise FileNotFoundError(f"Missing metadata/index in {case_dir}")
    if not analysis_path.exists():
        raise FileNotFoundError(f"Missing analysis JSON: {analysis_path}")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_json(metadata_path)
    temporal_index = load_json(index_path)
    analysis = load_json(analysis_path)
    warnings = validate_analysis(metadata, analysis)
    rendered: Dict[str, Any] = {
        "selected": None,
        "alternatives": [],
        "rejected": [],
        "warnings": warnings,
    }
    comparison_panels: List[Image.Image] = []

    best = analysis.get("best_segment")
    if isinstance(best, Mapping):
        segment = dict(best)
        segment.setdefault("cam", analysis.get("best_cam"))
        window = resolve_window(temporal_index, segment)
        if window:
            selected_path = (
                output_dir / "selected_best_segment_contact_sheet.jpg"
            )
            rendered["selected"] = annotate_sheet(
                case_dir,
                window,
                segment,
                selected_path,
                segment_title("SELECTED", segment),
            )
            comparison_panels.append(
                add_caption_panel(
                    Image.open(selected_path).convert("RGB"),
                    segment_title("SELECTED", segment),
                    safe_text(segment.get("reason_selected")),
                )
            )
        else:
            warnings.append(
                "Could not resolve best_segment to a generated window."
            )

    alternatives = analysis.get("alternative_segments", [])
    for index, raw_segment in enumerate(
        alternatives[:max_alternatives], start=1
    ):
        if not isinstance(raw_segment, Mapping):
            continue
        segment = dict(raw_segment)
        window = resolve_window(temporal_index, segment)
        if not window:
            warnings.append(f"Could not resolve alternative segment {index}.")
            continue
        path = output_dir / f"alternative_segment_{index:02d}.jpg"
        artifact = annotate_sheet(
            case_dir,
            window,
            segment,
            path,
            segment_title(f"ALTERNATIVE {index}", segment),
        )
        rendered["alternatives"].append(artifact)

    rejected = analysis.get("rejected_segments", [])
    for index, raw_segment in enumerate(
        rejected[:max_rejected], start=1
    ):
        if not isinstance(raw_segment, Mapping):
            continue
        segment = dict(raw_segment)
        window = resolve_window(temporal_index, segment)
        if not window:
            warnings.append(f"Could not resolve rejected segment {index}.")
            continue
        path = output_dir / f"rejected_segment_{index:02d}.jpg"
        artifact = annotate_sheet(
            case_dir,
            window,
            segment,
            path,
            segment_title(f"REJECTED {index}", segment),
        )
        rendered["rejected"].append(artifact)
        comparison_panels.append(
            add_caption_panel(
                Image.open(path).convert("RGB"),
                segment_title(f"REJECTED {index}", segment),
                (
                    safe_text(segment.get("reason_rejected"))
                    + " "
                    + safe_text(segment.get("region_difference_from_best"))
                ),
            )
        )

    if comparison_panels:
        stack_comparison(
            comparison_panels,
            output_dir / "selected_vs_rejected_region_comparison.jpg",
        )
    create_report(
        case_dir,
        metadata,
        analysis,
        rendered,
        output_dir / "method_feasibility_report.md",
    )
    write_json(output_dir / "render_summary.json", rendered)
    return {
        "case_id": metadata.get("case_id", case_dir.name),
        "status": "rendered",
        "output_dir": str(output_dir),
        "warnings": warnings,
    }


def main() -> int:
    args = parse_args()
    results: List[Dict[str, Any]] = []
    if args.case_dir:
        case_dir = args.case_dir.resolve()
        analysis_path = (
            args.analysis_json.resolve()
            if args.analysis_json
            else case_dir / args.analysis_filename
        )
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir
            else case_dir / "analysis_outputs"
        )
        results.append(
            process_case(
                case_dir,
                analysis_path,
                output_dir,
                args.overwrite,
                args.max_rejected,
                args.max_alternatives,
            )
        )
    else:
        if args.analysis_json or args.output_dir:
            raise SystemExit(
                "--analysis-json/--output-dir are only valid with --case-dir."
            )
        assets_root = args.assets_root.resolve()
        for case_dir in sorted(
            path for path in assets_root.iterdir() if path.is_dir()
        ):
            analysis_path = case_dir / args.analysis_filename
            if not analysis_path.exists():
                results.append(
                    {
                        "case_id": case_dir.name,
                        "status": "skipped_no_analysis",
                        "analysis_path": str(analysis_path),
                    }
                )
                continue
            try:
                results.append(
                    process_case(
                        case_dir,
                        analysis_path,
                        case_dir / "analysis_outputs",
                        args.overwrite,
                        args.max_rejected,
                        args.max_alternatives,
                    )
                )
            except Exception as exc:
                results.append(
                    {
                        "case_id": case_dir.name,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        write_json(
            assets_root / "render_batch_summary.json", {"cases": results}
        )

    print(json.dumps({"cases": results}, indent=2, ensure_ascii=False))
    return 1 if any(
        result["status"] == "failed" for result in results
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
