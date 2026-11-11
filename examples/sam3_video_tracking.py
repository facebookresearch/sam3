"""
SAM 3 — Video Tracking → Output Frames + XML annotations
=========================================================

Fixes applied:
- Closure bug: all loop variables captured by default args in _write()
- Wrong output key: out_scores → out_probs
- Normalized boxes: out_boxes_xywh are [0,1], denormalized to pixel coords before drawing
- NCCL / Distributed stability fixes (IPv4, async error handling, CUDA sync, barrier)
"""

import argparse
import os
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.distributed as dist

# ─────────────────────────────────────────────
# Distributed / NCCL Stability Environment
# ─────────────────────────────────────────────

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_TIMEOUT"] = "600"
os.environ.setdefault("GLOO_SOCKET_TIMEOUT", "600")

from sam3.model_builder import build_sam3_video_predictor

VIDEO_PATH      = "/home/ptekchan/vision-stream-worker/worker-go/tmp/hard/007/b1.MP4"
FRAMES_DIR      = "output_frames"
CHECKPOINT_PATH = "/home/ptekchan/sam3/examples/model/sam3.pt"
CONCEPTS        = ["person"]
GPUS            = [0, 1, 2, 3]
MAX_SIDE        = 720

PALETTE = [
    (0, 114, 189), (217, 83, 25), (32, 178, 170),
    (126, 47, 142), (119, 172, 48), (77, 190, 238),
    (162, 20, 47), (0, 128, 0), (128, 0, 128), (255, 165, 0),
]


def get_colour(idx):
    return PALETTE[int(idx) % len(PALETTE)]


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def scale_box(box_xyxy, scale):
    if scale == 1.0:
        return box_xyxy
    x1, y1, x2, y2 = box_xyxy
    return [x1 / scale, y1 / scale, x2 / scale, y2 / scale]


# ─────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────

def draw_detections(frame, boxes_xyxy, track_ids, scores, masks=None):
    h_frame, w_frame = frame.shape[:2]

    for i, (box, tid, score) in enumerate(zip(boxes_xyxy, track_ids, scores)):
        x1, y1, x2, y2 = map(int, box)
        colour = get_colour(tid)

        if masks is not None and i < len(masks):
            m = cv2.resize(
                masks[i].astype(np.uint8),
                (w_frame, h_frame),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            overlay = frame.copy()
            overlay[m] = colour
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"ID:{tid} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


# ─────────────────────────────────────────────
# XML Helpers
# ─────────────────────────────────────────────

def make_frame_el(frame_idx, filename, orig_w, orig_h, boxes_orig, track_ids, scores):
    el = ET.Element("frame")
    el.set("index", str(frame_idx))
    el.set("filename", filename)
    el.set("width", str(orig_w))
    el.set("height", str(orig_h))

    for box, tid, score in zip(boxes_orig, track_ids, scores):
        x1, y1, x2, y2 = [round(float(v), 2) for v in box]
        obj = ET.SubElement(el, "object")
        obj.set("track_id", str(tid))
        obj.set("score", f"{float(score):.4f}")
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(x1)
        ET.SubElement(bb, "ymin").text = str(y1)
        ET.SubElement(bb, "xmax").text = str(x2)
        ET.SubElement(bb, "ymax").text = str(y2)
    return el


def pretty_xml(element):
    raw = ET.tostring(element, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ")


# ─────────────────────────────────────────────
# Frame Extraction
# ─────────────────────────────────────────────

def extract_and_resize_frames(video_path, jpeg_dir, max_side):
    os.makedirs(jpeg_dir, exist_ok=True)
    existing = sorted(f for f in os.listdir(jpeg_dir) if f.endswith(".jpg"))
    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if existing:
        cap.release()
        sample = cv2.imread(os.path.join(jpeg_dir, existing[0]))
        scale = sample.shape[1] / orig_w
        return jpeg_dir, scale, orig_w, orig_h

    longest = max(orig_w, orig_h)
    if longest <= max_side:
        scale = 1.0
        new_w, new_h = orig_w, orig_h
    else:
        scale = max_side / longest
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(jpeg_dir, f"{idx:05d}.jpg"), frame)
        idx += 1
    cap.release()
    return jpeg_dir, scale, orig_w, orig_h


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run(video_path, frames_dir, checkpoint_path, concepts, gpus, max_side):
    os.makedirs(frames_dir, exist_ok=True)
    jpeg_dir = os.path.join(frames_dir, "_jpeg_input")
    infer_path, scale, orig_w, orig_h = extract_and_resize_frames(video_path, jpeg_dir, max_side)
    jpeg_paths = sorted(
        os.path.join(infer_path, f) for f in os.listdir(infer_path) if f.endswith(".jpg")
    )

    predictor = build_sam3_video_predictor(checkpoint_path=checkpoint_path, gpus_to_use=gpus)

    io_pool = ThreadPoolExecutor(max_workers=4)
    session_id = None
    master_root = ET.Element("annotations")

    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            response = predictor.handle_request(dict(
                type="start_session",
                resource_path=infer_path,
                offload_video_to_cpu=True,
                async_loading_frames=True,
            ))
        session_id = response["session_id"]

        for concept in concepts:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.handle_request(dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=concept,
                ))

        with torch.autocast("cuda", dtype=torch.bfloat16):
            for response in predictor.handle_stream_request(dict(
                type="propagate_in_video",
                session_id=session_id,
            )):
                fidx = response["frame_index"]
                out = response.get("outputs", {})

                # ── Correct key is out_probs, not out_scores ──────────────────
                boxes_xywh_norm = out.get("out_boxes_xywh", [])   # normalized [0,1]
                scores          = out.get("out_probs", [])         # FIX: was "out_scores"
                track_ids       = out.get("out_obj_ids", [])
                raw_masks       = out.get("out_binary_masks", None)

                # Move tensors to CPU numpy
                if hasattr(boxes_xywh_norm, "cpu"):
                    boxes_xywh_norm = boxes_xywh_norm.cpu().numpy()
                else:
                    boxes_xywh_norm = np.array(boxes_xywh_norm) if len(boxes_xywh_norm) else np.zeros((0, 4))

                if hasattr(scores, "cpu"):
                    scores = scores.cpu().numpy()
                else:
                    scores = np.array(scores)

                if hasattr(track_ids, "cpu"):
                    track_ids = track_ids.cpu().numpy()
                else:
                    track_ids = np.array(track_ids)

                if raw_masks is not None:
                    masks_np = raw_masks.detach().cpu().numpy() if hasattr(raw_masks, "detach") else np.array(raw_masks)
                    del raw_masks
                else:
                    masks_np = None

                # ── Denormalize boxes from [0,1] to inference-frame pixel coords ──
                if len(boxes_xywh_norm):
                    sample_img = cv2.imread(jpeg_paths[fidx])
                    h_inf, w_inf = sample_img.shape[:2]
                    del sample_img

                    bx = boxes_xywh_norm.copy()
                    bx[:, 0] *= w_inf   # x
                    bx[:, 1] *= h_inf   # y
                    bx[:, 2] *= w_inf   # w
                    bx[:, 3] *= h_inf   # h
                    boxes_xyxy_inf  = [xywh_to_xyxy(b) for b in bx]
                    boxes_xyxy_orig = [scale_box(b, scale) for b in boxes_xyxy_inf]
                else:
                    boxes_xyxy_inf  = []
                    boxes_xyxy_orig = []

                # ── FIX: capture all loop-local data by value as default args ──
                # Without this, the ThreadPool closure holds a reference to the
                # loop variable; by the time the thread runs, `del` has already
                # removed it from the enclosing scope → NameError / free-var error.
                def _write(
                    _fidx=fidx,
                    _src_path=jpeg_paths[fidx],
                    _boxes=list(boxes_xyxy_inf),
                    _tids=list(track_ids),
                    _scores=list(scores),
                    _masks=masks_np,
                    _out_dir=frames_dir,
                ):
                    try:
                        img = cv2.imread(_src_path)
                        if img is None:
                            return
                        img = draw_detections(img, _boxes, _tids, _scores, _masks)
                        cv2.imwrite(os.path.join(_out_dir, f"frame_{_fidx:06d}.jpg"), img)
                    except Exception as e:
                        print(f"[WARNING] Frame {_fidx} write failed: {e}")

                io_pool.submit(_write)

                if fidx % 50 == 0:
                    torch.cuda.synchronize()

                del boxes_xywh_norm, boxes_xyxy_inf, boxes_xyxy_orig, track_ids, scores, masks_np
                torch.cuda.empty_cache()

        io_pool.shutdown(wait=True)

    finally:
        torch.cuda.synchronize()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        time.sleep(2)
        if session_id is not None:
            try:
                predictor.handle_request(dict(type="close_session", session_id=session_id))
            except Exception:
                pass
        predictor.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video",       default=VIDEO_PATH)
    p.add_argument("--frames-dir",  default=FRAMES_DIR)
    p.add_argument("--checkpoint",  default=CHECKPOINT_PATH)
    p.add_argument("--concepts",    nargs="+", default=CONCEPTS)
    p.add_argument("--gpus",        nargs="+", type=int, default=GPUS)
    p.add_argument("--max-side",    type=int, default=MAX_SIDE)
    args = p.parse_args()

    run(args.video, args.frames_dir, args.checkpoint, args.concepts, args.gpus, args.max_side)