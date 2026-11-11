"""
SAM 3 — Batch Video Tracking + Evaluation (Hugging Face transformers version)
==========================================

For each folder under --input-dir:
    b1.MP4  +  b1pf.xml (GT)
    ↓
    SAM3 tracking → b1pf_pred.xml  (CVAT track format, matches GT schema)
    ↓
    frame_compare.py → results/b1_metrics.txt

Directory structure expected:
    hard/
        007/
            b1.MP4   b1pf.xml
            b2.MP4   b2pf.xml
            results/
        064/
            ...

Output:
    ./result_sam3/007/b1pf_pred.xml               ← prediction XML (flushed every 100 frames)
    ./result_sam3/007/results/b1_metrics.txt      ← per-video metrics
    ./result_sam3/007/results/summary.txt         ← folder-level summary
    ./result_sam3/results_summary.txt             ← global summary across all folders
    ./result_sam3/batch_checkpoint.json           ← resume checkpoint (auto-updated)

Progress log example:
    [2/12] frame   450/2089  21.6%   8.3 fps  ETA 3m 02s  tracks=3  dets=2
"""

import argparse
import json
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.dom import minidom

import cv2
import numpy as np
import torch

# ── frame_compare imports ────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from frame_compare import parse_xml, evaluate, calculate_ap

# ── Hugging Face SAM3 Video imports ──────────────────────────────────────────
from transformers import Sam3VideoModel, Sam3VideoProcessor

# ── Defaults ─────────────────────────────────────────────────────────────────
INPUT_DIR       = "/home/ptekchan/vision-stream-worker/worker-go/tmp/hard"
CHECKPOINT_PATH = "facebook/sam3"   # HF model ID (or local path)
CONCEPTS        = ["person"]
MAX_SIDE        = 720
LABELS_TO_EVAL  = {"person"}
XML_FLUSH_EVERY = 100       # atomically re-write pred XML every N frames
LOG_INTERVAL    = 10        # seconds between progress lines

PALETTE = [
    (0, 114, 189), (217, 83, 25),  (32, 178, 170),
    (126, 47, 142),(119, 172, 48), (77, 190, 238),
    (162, 20, 47), (0, 128, 0),    (128, 0, 128), (255, 165, 0),
]


# ─────────────────────────────────────────────────────────────────────────────
# BatchCheckpoint
# ─────────────────────────────────────────────────────────────────────────────

class BatchCheckpoint:
    def __init__(self, input_dir: Path):
        self._path = Path.cwd() / "result_sam3" / "batch_checkpoint.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                print(f"[CHECKPOINT] Loaded {self._path}")
                videos  = data.get("videos", {})
                done    = sum(1 for v in videos.values() if v.get("status") == "done")
                skipped = sum(1 for v in videos.values() if v.get("status") == "skipped")
                failed  = sum(1 for v in videos.values() if v.get("status") == "failed")
                print(f"[CHECKPOINT]   done={done}  skipped={skipped}  failed={failed}")
                return data
            except (json.JSONDecodeError, OSError) as e:
                print(f"[CHECKPOINT] Warning: could not read checkpoint ({e}). Starting fresh.")
        return {"videos": {}}

    def _flush(self):
        """Atomic write — never leaves a half-written file."""
        def _cvt(obj):
            if isinstance(obj, dict):   return {k: _cvt(v) for k, v in obj.items()}
            if isinstance(obj, list):   return [_cvt(v) for v in obj]
            if hasattr(obj, "item"):    return obj.item()      # numpy scalar
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj

        tmp = str(self._path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_cvt(self._data), f, indent=2)
        os.replace(tmp, self._path)

    def is_complete(self, video_path: Path) -> bool:
        entry = self._data["videos"].get(str(video_path), {})
        return entry.get("status") in ("done", "skipped")

    def get_status(self, video_path: Path) -> str | None:
        entry = self._data["videos"].get(str(video_path))
        return entry.get("status") if entry else None

    def get_metrics(self, video_path: Path):
        return self._data["videos"].get(str(video_path), {}).get("metrics")

    def mark_done(self, video_path: Path, pred_xml: str, metrics):
        self._data["videos"][str(video_path)] = {
            "status": "done", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pred_xml": pred_xml, "metrics": metrics,
        }
        self._flush()

    def mark_skipped(self, video_path: Path, pred_xml: str, metrics):
        self._data["videos"][str(video_path)] = {
            "status": "skipped", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pred_xml": pred_xml, "metrics": metrics,
        }
        self._flush()

    def mark_failed(self, video_path: Path, pred_xml: str):
        self._data["videos"][str(video_path)] = {
            "status": "failed", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pred_xml": pred_xml, "metrics": None,
        }
        self._flush()

    def reset(self):
        self._data = {"videos": {}}
        self._flush()
        print("[CHECKPOINT] State cleared — full re-run.")

    def reset_failed(self):
        before = len(self._data["videos"])
        self._data["videos"] = {
            k: v for k, v in self._data["videos"].items()
            if v.get("status") != "failed"
        }
        removed = before - len(self._data["videos"])
        self._flush()
        print(f"[CHECKPOINT] Cleared {removed} failed entr{'y' if removed==1 else 'ies'} — will retry them.")


# ─────────────────────────────────────────────────────────────────────────────
# Geometry / formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_colour(idx):
    return PALETTE[int(idx) % len(PALETTE)]

def scale_box(box_xyxy, scale):
    """Convert inference-resolution XYXY box back to original-resolution."""
    if scale == 1.0:
        return box_xyxy
    x1, y1, x2, y2 = box_xyxy
    return [x1 / scale, y1 / scale, x2 / scale, y2 / scale]

def fmt_eta(seconds):
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:  return f"{h}h {m:02d}m {s:02d}s"
    if m:  return f"{m}m {s:02d}s"
    return f"{s}s"


# ─────────────────────────────────────────────────────────────────────────────
# Box validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_person_box(box, frame_w, frame_h,
                        max_aspect=3.0, min_side=5, edge_margin=5):
    """
    Validate a person bounding box with edge awareness.

    Boxes touching the frame border (within edge_margin px) get relaxed aspect
    checking — a half-visible person at the edge has an unusual w/h ratio by
    definition and must not be dropped when the mask pixel count is healthy.
    The mask_px >= min_pixels check is handled separately by the caller.
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    if w < min_side or h < min_side:
        return False

    # Is the box clipped by the frame border?
    at_edge = (x1 <= edge_margin or y1 <= edge_margin or
               x2 >= frame_w - edge_margin or y2 >= frame_h - edge_margin)

    if at_edge:
        # Relaxed: only require non-trivial size — aspect can be anything
        return True

    # Interior box: enforce aspect ratio
    aspect = max(w / h, h / w)
    return aspect < max_aspect


def masks_to_bboxes(masks_np, out_h, out_w, threshold=0.5, min_mask_ratio=0.001):
    """
    Derive XYXY bounding boxes from SAM3 binary masks.

    SAM3 commonly outputs masks at a smaller internal resolution (e.g. 256×256)
    than the inference frame.  Coords are rescaled to (out_h, out_w).

    min_mask_ratio:
        Discard masks whose active-pixel count is below this fraction of the
        output frame area.  Prevents stray border pixels from producing giant
        boxes on uncertain frames.  Default 0.001 ≈ 920 px for 720×1280.
    """
    bboxes = []
    if masks_np is None:
        return bboxes

    mask_h, mask_w = masks_np.shape[-2], masks_np.shape[-1]
    min_pixels = int(out_h * out_w * min_mask_ratio)

    for mask in masks_np:
        binary   = mask > threshold
        n_active = int(binary.sum())

        if n_active < min_pixels:
            bboxes.append([0, 0, 0, 0])
            continue

        ys, xs = np.where(binary)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        # Rescale from mask-space → inference-frame-space
        sx = out_w / mask_w
        sy = out_h / mask_h
        x1 = max(0, int(x1 * sx))
        y1 = max(0, int(y1 * sy))
        x2 = min(out_w - 1, int(x2 * sx))
        y2 = min(out_h - 1, int(y2 * sy))

        bboxes.append([x1, y1, x2, y2])

    return bboxes


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_detections(frame, boxes_xyxy, track_ids, scores, masks=None):
    h_frame, w_frame = frame.shape[:2]
    for i, (box, tid, score) in enumerate(zip(boxes_xyxy, track_ids, scores)):
        x1, y1, x2, y2 = map(int, box)
        colour = get_colour(tid)
        if masks is not None and i < len(masks):
            m = cv2.resize(masks[i].astype(np.uint8),
                           (w_frame, h_frame),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
            overlay = frame.copy()
            overlay[m] = colour
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        lbl = f"ID:{tid} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(frame, lbl, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# PredXmlBuilder  (CVAT track format)
# ─────────────────────────────────────────────────────────────────────────────

class PredXmlBuilder:
    def __init__(self, out_path, label="person", flush_every=XML_FLUSH_EVERY):
        self._tracks      = defaultdict(list)
        self._out_path    = out_path
        self._label       = label
        self._flush_every = flush_every
        self._frames_seen = 0
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    def add_frame(self, frame_idx, boxes_xyxy_orig, track_ids, scores):
        for box, tid, score in zip(boxes_xyxy_orig, track_ids, scores):
            x1, y1, x2, y2 = box
            self._tracks[int(tid)].append(
                (frame_idx, float(x1), float(y1), float(x2), float(y2), float(score))
            )
        self._frames_seen += 1
        if self._frames_seen % self._flush_every == 0:
            self._flush()

    def _build_xml(self):
        root = ET.Element("annotations")
        ET.SubElement(root, "version").text = "1.1"
        for tid in sorted(self._tracks):
            entries  = sorted(self._tracks[tid], key=lambda e: e[0])
            track_el = ET.SubElement(root, "track")
            track_el.set("id",    str(tid))
            track_el.set("label", self._label)
            for (fidx, x1, y1, x2, y2, score) in entries:
                box_el = ET.SubElement(track_el, "box")
                box_el.set("frame",    str(fidx))
                box_el.set("xtl",      f"{x1:.2f}")
                box_el.set("ytl",      f"{y1:.2f}")
                box_el.set("xbr",      f"{x2:.2f}")
                box_el.set("ybr",      f"{y2:.2f}")
                box_el.set("outside",  "0")
                box_el.set("occluded", "0")
                gid = ET.SubElement(box_el, "attribute")
                gid.set("name", "Global User Id")
                gid.text = str(tid)
                sc = ET.SubElement(box_el, "attribute")
                sc.set("name", "score")
                sc.text = f"{score:.4f}"
        raw = ET.tostring(root, encoding="unicode")
        return minidom.parseString(raw).toprettyxml(indent="  ")

    def _flush(self):
        tmp = self._out_path + ".tmp"
        with open(tmp, "w") as f:
            f.write(self._build_xml())
        os.replace(tmp, self._out_path)

    def finalize(self):
        self._flush()
        print(f"    [XML] Prediction written → {self._out_path}"
              f"  ({self._frames_seen} frames, {len(self._tracks)} tracks)")


# ─────────────────────────────────────────────────────────────────────────────
# Frame extraction  (writes resized JPEGs for reuse across runs)
# ─────────────────────────────────────────────────────────────────────────────

_JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 85]

def _write_jpeg(path, frame):
    cv2.imwrite(path, frame, _JPEG_PARAMS)

def extract_frames(video_path, jpeg_dir, max_side=MAX_SIDE, num_threads=8):
    os.makedirs(jpeg_dir, exist_ok=True)
    existing = sorted(f for f in os.listdir(jpeg_dir) if f.endswith(".jpg"))

    cap    = cv2.VideoCapture(str(video_path))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    longest = max(orig_w, orig_h)
    scale   = min(1.0, max_side / longest)
    new_w   = int(orig_w * scale)
    new_h   = int(orig_h * scale)

    if existing:
        cap.release()
        sample  = cv2.imread(os.path.join(jpeg_dir, existing[0]))
        scale_w = sample.shape[1] / orig_w
        scale_h = sample.shape[0] / orig_h
        scale   = min(scale_w, scale_h)
        print(f"    [FRAMES] Already extracted ({len(existing)} frames) — reusing.")
        return jpeg_dir, scale, orig_w, orig_h, len(existing)

    print(f"    [FRAMES] Extracting ~{total} frames  "
          f"({orig_w}x{orig_h} → {new_w}x{new_h}, scale={scale:.3f}) ...")
    t0   = time.time()
    idx  = 0
    pool = ThreadPoolExecutor(max_workers=num_threads)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pool.submit(_write_jpeg, os.path.join(jpeg_dir, f"{idx:05d}.jpg"), frame)
        idx += 1
        if idx % 500 == 0:
            elapsed = time.time() - t0
            fps = idx / elapsed if elapsed else 0
            eta = (total - idx) / fps if fps else 0
            print(f"    [FRAMES]   {idx}/{total}  {fps:.0f} fps  ETA {fmt_eta(eta)}")

    pool.shutdown(wait=True)
    cap.release()
    print(f"    [FRAMES] Done — {idx} frames in {time.time()-t0:.1f}s")
    return jpeg_dir, scale, orig_w, orig_h, idx


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(gt_xml, pred_xml, results_dir, video_stem):
    gt_data = parse_xml(gt_xml)
    pd_data = parse_xml(pred_xml)

    if gt_data is None or pd_data is None:
        print("    [EVAL] Could not load XML pair — skipping.")
        return None

    labels = set()
    for f in gt_data:
        for o in gt_data[f]: labels.add(o["label"])
    for f in pd_data:
        for o in pd_data[f]: labels.add(o["label"])
    labels = [l for l in labels if l in LABELS_TO_EVAL]

    if not labels:
        print("    [EVAL] No matching labels found.")
        return None

    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, f"{video_stem}_metrics.txt")

    header = (f"{'CATEGORY':<10} | {'mAP':<8} | {'MOTA':<8} | {'IDF1':<8} | "
              f"{'STABILITY':<10} | {'GT':<6} | {'MISS':<6} | {'FP':<6} | {'ID_SW':<6}")
    sep   = "-" * 100
    lines = ["=" * 100, header, sep]

    summary = {}
    maps, motas, idf1s, stabs = [], [], [], []

    for label in labels:
        ap = calculate_ap(gt_data, pd_data, label)
        if ap is None:
            continue
        m   = evaluate(gt_data, pd_data, label)
        row = (f"{label.capitalize():<10} | {ap:<8.2%} | {m['mota']:<8.2%} | "
               f"{m['idf1']:<8.2%} | {m['stability']:<10.2%} | {m['gt']:<6} | "
               f"{m['miss']:<6} | {m['fp']:<6} | {m['id_sw']:<6}")
        lines.append(row)
        maps.append(ap); motas.append(m["mota"])
        idf1s.append(m["idf1"]); stabs.append(m["stability"])
        summary[label] = {"ap": ap, **m}

    if len(labels) > 1:
        lines += [sep,
                  f"{'AVERAGE':<10} | {np.mean(maps):<8.2%} | {np.mean(motas):<8.2%} | "
                  f"{np.mean(idf1s):<8.2%} | {np.mean(stabs):<10.2%}"]
    lines += ["=" * 100, f"\nGT XML  : {gt_xml}", f"Pred XML: {pred_xml}"]

    report = "\n".join(lines)
    print("\n" + report)
    with open(metrics_path, "w") as f:
        f.write(report + "\n")
    print(f"    [EVAL] Metrics → {metrics_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Core tracking  —  HF SAM3 Video session-based API
#
# API flow (from HF docs):
#   1. processor.init_video_session(video=frames, ...)   → inference_session
#   2. processor.add_text_prompt(inference_session, text)
#   3. for out in model.propagate_in_video_iterator(inference_session):
#          processed = processor.postprocess_outputs(inference_session, out)
#          # processed keys: object_ids, scores, boxes (XYXY abs), masks, prompt_to_obj_ids
# ─────────────────────────────────────────────────────────────────────────────

def _load_jpeg_frames(jpeg_paths):
    """Load all JPEG files as a list of numpy BGR arrays (cv2 format)."""
    frames = []
    for p in jpeg_paths:
        img = cv2.imread(p)
        if img is not None:
            frames.append(img)
    return frames


def process_video(processor, model, device,
                  video_path, gt_xml, pred_xml, results_dir,
                  concepts, max_side, batch_tag=""):
    video_stem = video_path.stem
    _out_base  = Path.cwd() / "result_sam3" / video_path.parent.name / "results" / video_stem
    frames_dir = str(_out_base / "_vis")
    jpeg_dir   = str(_out_base / "_jpeg_input")
    os.makedirs(frames_dir, exist_ok=True)

    # ── 1. Extract / reuse resized JPEG frames ────────────────────────────────
    infer_path, scale, orig_w, orig_h, total_frames = extract_frames(
        video_path, jpeg_dir, max_side
    )
    jpeg_paths = sorted(
        os.path.join(infer_path, f)
        for f in os.listdir(infer_path) if f.endswith(".jpg")
    )
    if not jpeg_paths:
        print("    [SKIP] No frames extracted.")
        return None

    # ── 2. Load frames into memory for the HF session ─────────────────────────
    print(f"    [HF-SAM3] Loading {total_frames} frames into session ...")
    video_frames = _load_jpeg_frames(jpeg_paths)

    xml_builder = PredXmlBuilder(pred_xml, label="person")
    io_pool     = ThreadPoolExecutor(max_workers=4)
    prog_start  = time.time()
    last_log    = prog_start

    try:
        # ── 3. Initialise the HF SAM3 Video session ───────────────────────────
        inference_session = processor.init_video_session(
            video=video_frames,
            inference_device=device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=torch.bfloat16,
        )

        # ── 4. Register all concept prompts ───────────────────────────────────
        processor.add_text_prompt(inference_session, concepts)

        print(f"    [HF-SAM3] Propagating over {total_frames} frames ...")

        # ── 5. Iterate frame-by-frame via the propagation iterator ────────────
        for model_outputs in model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=total_frames,
        ):
            fidx = model_outputs.frame_idx

            processed = processor.postprocess_outputs(
                inference_session, model_outputs
            )

            obj_ids = processed["object_ids"]   # torch.Tensor  [N]
            scores  = processed["scores"]        # torch.Tensor  [N]
            boxes   = processed["boxes"]         # torch.Tensor  [N, 4]  XYXY abs
            masks   = processed.get("masks")     # torch.Tensor  [N, H, W] or None

            # ── tensor → numpy ────────────────────────────────────────────────
            obj_ids_np = obj_ids.cpu().numpy() if len(obj_ids) else np.array([], dtype=np.int64)
            scores_np  = scores.cpu().numpy()  if len(scores)  else np.array([], dtype=np.float32)
            boxes_np   = boxes.cpu().numpy()   if len(boxes)   else np.zeros((0, 4), dtype=np.float32)
            masks_np   = masks.cpu().numpy()   if (masks is not None and len(masks)) else None

            # ── derive boxes: prefer masks (tighter), fall back to model boxes ─
            h_inf, w_inf = video_frames[fidx].shape[:2]
            min_pixels   = int(h_inf * w_inf * 0.001)

            if masks_np is not None and len(masks_np):
                boxes_xyxy_inf = masks_to_bboxes(masks_np, out_h=h_inf, out_w=w_inf)
            else:
                boxes_xyxy_inf = [list(b) for b in boxes_np]

            # ── two-stage filter ──────────────────────────────────────────────
            # Stage 1: junk mask  (too few active pixels → [0,0,0,0] or tiny)
            # Stage 2: bad aspect (interior box with implausible w/h ratio)
            # Edge-clipped boxes pass stage 2 unconditionally — a half-visible
            # person at the frame border has an unusual ratio by definition.
            valid_idx = []
            for i, b in enumerate(boxes_xyxy_inf):
                active_px = (
                    int((masks_np[i] > 0.5).sum())
                    if masks_np is not None and i < len(masks_np)
                    else min_pixels   # no mask → trust the box
                )

                if active_px < min_pixels:
                    print(f"    [FILTER] frame={fidx} "
                          f"id={obj_ids_np[i] if i < len(obj_ids_np) else '?'} "
                          f"box={[int(v) for v in b]}  mask_px={active_px}"
                          f"  → junk mask")
                    continue

                if not is_valid_person_box(b, frame_w=w_inf, frame_h=h_inf):
                    print(f"    [FILTER] frame={fidx} "
                          f"id={obj_ids_np[i] if i < len(obj_ids_np) else '?'} "
                          f"box={[int(v) for v in b]}  mask_px={active_px}"
                          f"  → bad aspect")
                    continue

                valid_idx.append(i)

            # Keep only validated detections — all arrays stay in sync
            boxes_xyxy_inf  = [boxes_xyxy_inf[i] for i in valid_idx]
            boxes_xyxy_orig = [scale_box(b, scale) for b in boxes_xyxy_inf]
            obj_ids_filt    = obj_ids_np[valid_idx] if len(valid_idx) else np.array([], dtype=np.int64)
            scores_filt     = scores_np[valid_idx]  if len(valid_idx) else np.array([], dtype=np.float32)
            masks_filt      = masks_np[valid_idx]   if (masks_np is not None and len(valid_idx)) else None

            # ── accumulate detections for XML output ──────────────────────────
            xml_builder.add_frame(fidx, boxes_xyxy_orig, obj_ids_filt, scores_filt)

            # ── progress logging ──────────────────────────────────────────────
            now     = time.time()
            elapsed = now - prog_start
            is_last = (fidx + 1 >= total_frames)
            if now - last_log >= LOG_INTERVAL or is_last:
                fps   = (fidx + 1) / elapsed if elapsed else 0
                pct   = 100.0 * (fidx + 1) / total_frames
                eta   = (total_frames - fidx - 1) / fps if fps else 0
                n_trk = len(xml_builder._tracks)
                print(f"    {batch_tag} frame {fidx+1:>5}/{total_frames}"
                      f"  {pct:5.1f}%  {fps:5.1f} fps  ETA {fmt_eta(eta)}"
                      f"  tracks={n_trk}  dets={len(boxes_xyxy_inf)}")
                last_log = now

            # ── visualise frame (async I/O) ───────────────────────────────────
            def _write(
                _fidx=fidx,
                _src=jpeg_paths[fidx],
                _boxes=list(boxes_xyxy_inf),
                _tids=obj_ids_filt.tolist(),
                _scores=scores_filt.tolist(),
                _masks=masks_filt,
                _out_dir=frames_dir,
            ):
                try:
                    img = cv2.imread(_src)
                    if img is None:
                        return
                    img = draw_detections(img, _boxes, _tids, _scores, _masks)
                    cv2.imwrite(os.path.join(_out_dir, f"frame_{_fidx:06d}.jpg"), img)
                except Exception as e:
                    print(f"    [WARNING] Frame {_fidx} vis write failed: {e}")

            io_pool.submit(_write)

        io_pool.shutdown(wait=True)
        xml_builder.finalize()

        if os.path.exists(gt_xml):
            return run_evaluation(gt_xml, pred_xml, results_dir, video_stem)
        else:
            print(f"    [EVAL] No GT XML at {gt_xml} — skipping evaluation.")
            return None

    except Exception:
        traceback.print_exc()
        try:
            xml_builder.finalize()
            print("    [XML] Partial prediction XML saved despite error.")
        except Exception:
            pass
        io_pool.shutdown(wait=False)
        return None

    finally:
        try:
            inference_session.reset_state()
        except Exception:
            pass
        torch.cuda.empty_cache()
        time.sleep(1)


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_video_gt_pairs(input_dir):
    pairs     = []
    input_dir = Path(input_dir)
    for folder in sorted(input_dir.iterdir()):
        if not folder.is_dir() or folder.name == "results":
            continue
        mp4s = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() == ".mp4"
        )
        for mp4 in mp4s:
            stem        = mp4.stem
            gt_xml      = folder / f"{stem}pf.xml"
            out_folder  = Path.cwd() / "result_sam3" / folder.name
            results_dir = out_folder / "results"
            pred_xml    = out_folder / f"{stem}pf_pred.xml"
            pairs.append((mp4, str(gt_xml), str(pred_xml), str(results_dir)))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(input_dir, checkpoint_path, concepts, max_side,
              ignore_checkpoint=False, retry_failed=False):
    pairs = find_video_gt_pairs(input_dir)
    if not pairs:
        print(f"No MP4 files found under {input_dir}")
        return

    total = len(pairs)

    ckpt = BatchCheckpoint(Path(input_dir))
    if ignore_checkpoint:
        ckpt.reset()
    elif retry_failed:
        ckpt.reset_failed()

    print(f"\nFound {total} video(s):")
    to_run  = []
    to_skip = []
    for i, (mp4, gt, pred, _) in enumerate(pairs):
        gt_flag   = "GT found" if os.path.exists(gt) else "no GT"
        ck_status = ckpt.get_status(mp4)
        if ckpt.is_complete(mp4):
            status_str = f"[{ck_status}]"
            to_skip.append(i)
        else:
            status_str = "[pending]"
            to_run.append(i)
        print(f"  [{i+1:>2}/{total}] {Path(mp4).relative_to(input_dir)}"
              f"  [{gt_flag}]  {status_str}")

    print(f"\n  Will process : {len(to_run)}  |  Already done / skipped : {len(to_skip)}")

    if not to_run:
        print("\nAll videos already processed.  "
              "Use --ignore-checkpoint for a full re-run or "
              "--retry-failed to retry failures.\n")
        all_results = {
            str(pairs[i][0]): ckpt.get_metrics(pairs[i][0]) or "skipped"
            for i in range(total)
        }
        _write_summaries(input_dir, pairs, all_results)
        return

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading HF SAM3 Video from '{checkpoint_path}' ...")
    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = Sam3VideoProcessor.from_pretrained(checkpoint_path)
    model     = Sam3VideoModel.from_pretrained(
        checkpoint_path, torch_dtype=torch.bfloat16
    ).to(device).eval()
    print(f"Model ready on {device}.\n" + "=" * 70)

    all_results = {}
    batch_t0    = time.time()
    done_count  = 0

    for i, (video_path, gt_xml, pred_xml, results_dir) in enumerate(pairs):
        batch_tag = f"[{i+1}/{total}]"
        stem      = video_path.stem

        if ckpt.is_complete(video_path):
            ck_status = ckpt.get_status(video_path)
            print(f"\n{batch_tag} ── {video_path.name}  [checkpoint: {ck_status}]  →  skipping")
            all_results[str(video_path)] = ckpt.get_metrics(video_path) or "skipped"
            continue

        print(f"\n{batch_tag} ── {video_path}")

        if os.path.exists(pred_xml) and os.path.getsize(pred_xml) > 500:
            print(f"  Pred XML already on disk (not in checkpoint): {pred_xml}")
            metrics_file = os.path.join(results_dir, f"{stem}_metrics.txt")
            if not os.path.exists(metrics_file) and os.path.exists(gt_xml):
                print("  Re-running evaluation only ...")
                summary = run_evaluation(gt_xml, pred_xml, results_dir, stem)
            else:
                summary = ckpt.get_metrics(video_path)
            ckpt.mark_skipped(video_path, pred_xml, summary)
            all_results[str(video_path)] = summary or "skipped"
            continue

        vid_t0  = time.time()
        summary = process_video(
            processor, model, device,
            video_path, gt_xml, pred_xml, results_dir,
            concepts, max_side, batch_tag=batch_tag,
        )
        elapsed = time.time() - vid_t0

        done_count += 1
        avg_t      = (time.time() - batch_t0) / done_count
        remaining  = len(to_run) - done_count
        eta_tot    = avg_t * remaining

        if summary is not None:
            status = "OK"
            ckpt.mark_done(video_path, pred_xml, summary)
        else:
            status = "FAILED/NO-EVAL"
            ckpt.mark_failed(video_path, pred_xml)

        all_results[str(video_path)] = summary
        print(f"\n{batch_tag} {status}  video took {fmt_eta(elapsed)}"
              f"  |  batch ETA {fmt_eta(eta_tot)}"
              f"  ({done_count}/{len(to_run)} processed this session)\n"
              + "-" * 70)

    _write_summaries(input_dir, pairs, all_results)

    torch.cuda.empty_cache()
    total_elapsed = time.time() - batch_t0
    print("\n" + "=" * 70)
    print(f"BATCH COMPLETE  ({fmt_eta(total_elapsed)} total)")
    print("=" * 70)
    ok   = sum(1 for v in all_results.values() if isinstance(v, dict))
    skip = sum(1 for v in all_results.values() if v == "skipped")
    fail = sum(1 for v in all_results.values() if v is None)
    print(f"  OK (with eval) : {ok}")
    print(f"  Skipped        : {skip}")
    print(f"  Failed/no-eval : {fail}")
    if fail:
        print("\n  Failed videos:")
        for vp, st in all_results.items():
            if st is None:
                print(f"    {vp}")
    print(f"\n  Checkpoint     : {Path.cwd() / 'result_sam3' / 'batch_checkpoint.json'}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Summary writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_summaries(input_dir, pairs, all_results):
    input_dir  = Path(input_dir)
    folder_map = defaultdict(list)
    for row in pairs:
        folder_map[str(row[0].parent)].append(row)

    global_lines = [
        "GLOBAL SUMMARY", "=" * 110,
        f"{'VIDEO':<35} | {'mAP':<8} | {'MOTA':<8} | {'IDF1':<8} | "
        f"{'STABILITY':<10} | {'GT':<6} | {'MISS':<6} | {'FP':<6} | {'ID_SW':<6}",
        "-" * 110,
    ]

    for folder_str, vids in folder_map.items():
        folder      = Path(folder_str)
        results_dir = str(vids[0][3])
        os.makedirs(results_dir, exist_ok=True)

        folder_lines = [
            f"FOLDER: {folder}", "=" * 110,
            f"{'VIDEO':<20} | {'mAP':<8} | {'MOTA':<8} | {'IDF1':<8} | "
            f"{'STABILITY':<10} | {'GT':<6} | {'MISS':<6} | {'FP':<6} | {'ID_SW':<6}",
            "-" * 110,
        ]

        for (video_path, _, _, _) in vids:
            stem    = video_path.stem
            summary = all_results.get(str(video_path))
            tag     = f"{stem} ({folder.name})"

            if isinstance(summary, dict) and summary:
                m     = next(iter(summary.values()))
                row   = (f"{stem:<20} | {m['ap']:<8.2%} | {m['mota']:<8.2%} | "
                         f"{m['idf1']:<8.2%} | {m['stability']:<10.2%} | {m['gt']:<6} | "
                         f"{m['miss']:<6} | {m['fp']:<6} | {m['id_sw']:<6}")
                g_row = (f"{tag:<35} | {m['ap']:<8.2%} | {m['mota']:<8.2%} | "
                         f"{m['idf1']:<8.2%} | {m['stability']:<10.2%} | {m['gt']:<6} | "
                         f"{m['miss']:<6} | {m['fp']:<6} | {m['id_sw']:<6}")
            elif summary == "skipped":
                row   = f"{stem:<20} | SKIPPED"
                g_row = f"{tag:<35} | SKIPPED"
            else:
                row   = f"{stem:<20} | FAILED / NO EVAL"
                g_row = f"{tag:<35} | FAILED / NO EVAL"

            folder_lines.append(row)
            global_lines.append(g_row)

        folder_lines.append("=" * 110)
        folder_summary = os.path.join(results_dir, "summary.txt")
        with open(folder_summary, "w") as f:
            f.write("\n".join(folder_lines) + "\n")
        print(f"  Folder summary → {folder_summary}")

    global_lines.append("=" * 110)
    global_summary = str(Path.cwd() / "result_sam3" / "results_summary.txt")
    with open(global_summary, "w") as f:
        f.write("\n".join(global_lines) + "\n")
    print(f"  Global summary → {global_summary}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Batch SAM3 tracker + evaluator (Hugging Face)")
    p.add_argument("--input-dir",  default=INPUT_DIR,
                   help="Root dir with sub-folders (007/, 064/, …) containing MP4 + GT XML")
    p.add_argument("--checkpoint", default=CHECKPOINT_PATH,
                   help="HF model ID or local path, e.g. 'facebook/sam3'")
    p.add_argument("--concepts",   nargs="+", default=CONCEPTS)
    p.add_argument("--max-side",   type=int, default=MAX_SIDE)

    ck_group = p.add_mutually_exclusive_group()
    ck_group.add_argument("--ignore-checkpoint", action="store_true",
                          help="Ignore saved checkpoint and re-run everything")
    ck_group.add_argument("--retry-failed", action="store_true",
                          help="Re-run only videos marked 'failed'")

    args = p.parse_args()

    run_batch(
        input_dir         = args.input_dir,
        checkpoint_path   = args.checkpoint,
        concepts          = args.concepts,
        max_side          = args.max_side,
        ignore_checkpoint = args.ignore_checkpoint,
        retry_failed      = args.retry_failed,
    )