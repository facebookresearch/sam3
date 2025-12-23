#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
SAM3 Web Command Center

A Flask-based web interface for real-time object detection and tracking
using SAM3. Features include:
- Live camera feed with segmentation overlay
- Multi-prompt detection configuration
- Object count limits with show/hide functionality
- Claude Vision API integration for detailed object analysis
- Command center style interface with verbose logging

Usage:
    python app.py --prompt "person, car" --camera 0

Then open http://localhost:5000 in your browser.
"""

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional, Dict, List, Any

import cv2
import numpy as np
import torch
from PIL import Image
from flask import Flask, Response, render_template, request, jsonify

# Add parent directory to path for sam3 imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sam3.utils.device import get_device, get_device_str, setup_device_optimizations, empty_cache

app = Flask(__name__)

# Global state
class CommandCenter:
    """Global state manager for the command center."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.paused = False

        # Detection settings
        self.prompts = ["object"]
        self.confidence_threshold = 0.3
        self.max_objects_per_prompt = {}  # prompt -> max count (None = unlimited)
        self.show_all_matches = {}  # prompt -> bool (show all even if over limit)

        # Current detection state
        self.current_detections = []  # List of detection dicts
        self.frame_count = 0
        self.fps = 0.0
        self.device_str = "cpu"

        # Verbose log
        self.log_entries = deque(maxlen=100)

        # Claude analysis results
        self.analysis_queue = []  # Objects waiting for analysis
        self.analysis_results = deque(maxlen=20)  # Recent analysis results
        self.analyzing = False

        # Frame for streaming
        self.current_frame = None
        self.current_frame_jpeg = None

        # Camera and model
        self.camera = None
        self.processor = None
        self.state = None

        # Tracking state
        self.enable_tracking = True
        self.skip_frames = 3
        self.last_masks = None
        self.last_boxes = None
        self.last_scores = None
        self.last_labels = None
        self.prev_gray = None

    def log(self, message: str, level: str = "INFO"):
        """Add a log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message
        }
        with self.lock:
            self.log_entries.append(entry)

    def get_logs(self, limit: int = 50) -> List[Dict]:
        """Get recent log entries."""
        with self.lock:
            return list(self.log_entries)[-limit:]

    def add_detection(self, detection: Dict):
        """Add a detection to the current list."""
        with self.lock:
            self.current_detections.append(detection)

    def clear_detections(self):
        """Clear all current detections."""
        with self.lock:
            self.current_detections = []

    def get_filtered_detections(self) -> List[Dict]:
        """Get detections filtered by max count settings."""
        with self.lock:
            detections = self.current_detections.copy()

        # Group by prompt
        by_prompt = {}
        for det in detections:
            prompt = det.get("label", "unknown")
            if prompt not in by_prompt:
                by_prompt[prompt] = []
            by_prompt[prompt].append(det)

        # Apply filters
        filtered = []
        hidden_counts = {}

        for prompt, dets in by_prompt.items():
            max_count = self.max_objects_per_prompt.get(prompt)
            show_all = self.show_all_matches.get(prompt, False)

            if max_count is not None and not show_all:
                # Sort by confidence and take top N
                dets_sorted = sorted(dets, key=lambda d: d.get("confidence", 0), reverse=True)
                filtered.extend(dets_sorted[:max_count])
                hidden = len(dets_sorted) - max_count
                if hidden > 0:
                    hidden_counts[prompt] = hidden
            else:
                filtered.extend(dets)

        return filtered, hidden_counts

    def queue_analysis(self, detection_id: int, image_data: str):
        """Queue an object for Claude analysis."""
        with self.lock:
            self.analysis_queue.append({
                "id": detection_id,
                "image_data": image_data,
                "timestamp": datetime.now().isoformat()
            })

    def add_analysis_result(self, detection_id: int, result: str):
        """Add a Claude analysis result."""
        with self.lock:
            self.analysis_results.append({
                "id": detection_id,
                "result": result,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })


# Global command center instance
cc = CommandCenter()


# Color palette (BGR for OpenCV)
COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Purple
    (255, 128, 0),  # Orange
]


def load_model(checkpoint_path: Optional[str] = None):
    """Load the SAM3 model."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    cc.log("Loading SAM3 model...")
    cc.device_str = get_device_str()

    # Setup device-specific optimizations (MPS memory, CUDA TF32, etc.)
    setup_device_optimizations()
    cc.log(f"Device optimizations enabled for {cc.device_str}")

    model = build_sam3_image_model(
        device=cc.device_str,
        checkpoint_path=checkpoint_path,
        load_from_HF=checkpoint_path is None,
        eval_mode=True,
        enable_segmentation=True,
    )

    cc.processor = Sam3Processor(
        model=model,
        resolution=1008,
        device=cc.device_str,
        confidence_threshold=cc.confidence_threshold,
    )

    cc.log(f"Model loaded on {cc.device_str}", "SUCCESS")


def process_frame(frame: np.ndarray) -> np.ndarray:
    """Process a frame through SAM3 and overlay results."""
    global cc

    cc.frame_count += 1
    is_keyframe = cc.frame_count % cc.skip_frames == 0

    if is_keyframe and not cc.paused:
        # Full inference
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        cc.state = cc.processor.set_image(pil_image, cc.state)

        # Clear current detections
        cc.clear_detections()

        all_masks = []
        all_boxes = []
        all_scores = []
        all_labels = []

        for prompt in cc.prompts:
            if "geometric_prompt" in cc.state:
                del cc.state["geometric_prompt"]

            cc.state = cc.processor.set_text_prompt(prompt.strip(), cc.state)

            masks = cc.state.get("masks")
            boxes = cc.state.get("boxes")
            scores = cc.state.get("scores")

            if masks is not None and masks.numel() > 0:
                for i in range(len(masks)):
                    detection = {
                        "id": len(all_masks),
                        "label": prompt.strip(),
                        "confidence": float(scores[i].cpu()) if scores is not None and i < len(scores) else 0.0,
                        "box": boxes[i].cpu().numpy().tolist() if boxes is not None and i < len(boxes) else None,
                    }
                    cc.add_detection(detection)

                    all_masks.append(masks[i:i+1])
                    if boxes is not None and i < len(boxes):
                        all_boxes.append(boxes[i:i+1])
                    if scores is not None and i < len(scores):
                        all_scores.append(scores[i:i+1])
                    all_labels.append(prompt.strip())

        # Store for tracking
        if all_masks:
            cc.last_masks = torch.cat(all_masks, dim=0)
            cc.last_boxes = torch.cat(all_boxes, dim=0) if all_boxes else None
            cc.last_scores = torch.cat(all_scores, dim=0) if all_scores else None
            cc.last_labels = all_labels
            cc.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            cc.last_masks = None
            cc.last_boxes = None
            cc.last_scores = None
            cc.last_labels = None

        if all_labels:
            cc.log(f"Detected: {', '.join(all_labels)}")

    elif cc.enable_tracking and cc.last_masks is not None and not cc.paused:
        # Track with optical flow
        tracked = track_frame(frame)
        if tracked is not None:
            cc.last_masks = tracked

    # Overlay masks on frame
    display = frame.copy()
    if cc.last_masks is not None:
        display = overlay_masks(display, cc.last_masks, cc.last_boxes, cc.last_scores, cc.last_labels)

    return display


def track_frame(frame: np.ndarray) -> Optional[torch.Tensor]:
    """Track masks using optical flow."""
    if cc.last_masks is None or cc.prev_gray is None:
        return None

    try:
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            cc.prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        h, w = curr_gray.shape
        flow_map_x = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(np.float32)
        flow_map_y = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(np.float32)
        flow_map_x += flow[:, :, 0]
        flow_map_y += flow[:, :, 1]

        tracked_masks = []
        for mask in cc.last_masks:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy().squeeze()
            else:
                mask_np = mask.squeeze()

            if mask_np.shape != (h, w):
                mask_np = cv2.resize(mask_np.astype(np.float32), (w, h))

            warped = cv2.remap(
                mask_np.astype(np.float32),
                flow_map_x, flow_map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            warped = (warped > 0.5).astype(np.float32)
            tracked_masks.append(torch.from_numpy(warped).unsqueeze(0).to(cc.device_str))

        cc.prev_gray = curr_gray

        if tracked_masks:
            return torch.stack(tracked_masks)

    except Exception as e:
        cc.log(f"Tracking error: {e}", "ERROR")

    return None


def overlay_masks(frame: np.ndarray, masks: torch.Tensor, boxes=None, scores=None, labels=None, alpha=0.5) -> np.ndarray:
    """Overlay masks on frame."""
    if masks is None or masks.numel() == 0:
        return frame

    overlay = frame.copy()
    h, w = frame.shape[:2]
    masks_np = masks.squeeze(1).cpu().numpy()

    scores_np = scores.cpu().numpy() if scores is not None else None

    for i, mask in enumerate(masks_np):
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5

        color = COLORS[i % len(COLORS)]
        mask_region = mask.astype(bool)
        overlay[mask_region] = (
            overlay[mask_region] * (1 - alpha) + np.array(color) * alpha
        ).astype(np.uint8)

        # Draw contour
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

        # Draw label
        if len(contours) > 0:
            largest = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest)

            label = labels[i] if labels and i < len(labels) else "object"
            conf = scores_np[i] if scores_np is not None and i < len(scores_np) else 0.0
            text = f"{label} {conf:.0%}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)

            cv2.rectangle(overlay, (x, y - th - 4), (x + tw + 4, y), color, -1)
            cv2.putText(overlay, text, (x + 2, y - 2), font, 0.5, (255, 255, 255), 1)

    return overlay


def generate_frames():
    """Generator for video streaming."""
    global cc

    while cc.running:
        if cc.camera is None or not cc.camera.isOpened():
            time.sleep(0.1)
            continue

        ret, frame = cc.camera.read()
        if not ret:
            time.sleep(0.1)
            continue

        start = time.time()

        # Process frame
        display = process_frame(frame)

        # Calculate FPS
        elapsed = time.time() - start
        cc.fps = 1.0 / elapsed if elapsed > 0 else 0

        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cc.current_frame = display
        cc.current_frame_jpeg = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cc.current_frame_jpeg + b'\r\n')


def analyze_with_claude(image_data: str, label: str) -> str:
    """Send image to Claude for analysis."""
    try:
        import anthropic

        client = anthropic.Anthropic()

        # Remove data URL prefix if present
        if image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"This is a cropped image of a detected '{label}'. Please provide a brief, detailed description of what you see. Focus on: appearance, distinctive features, actions/pose, and any notable details. Keep it concise (2-3 sentences)."
                        }
                    ],
                }
            ],
        )

        return message.content[0].text

    except Exception as e:
        return f"Analysis error: {str(e)}"


def analysis_worker():
    """Background worker for Claude analysis."""
    global cc

    while cc.running:
        if cc.analysis_queue:
            with cc.lock:
                if cc.analysis_queue:
                    item = cc.analysis_queue.pop(0)
                    cc.analyzing = True
                else:
                    item = None

            if item:
                cc.log(f"Analyzing object #{item['id']}...", "INFO")

                # Find the detection to get its label
                detections = cc.current_detections
                label = "object"
                for det in detections:
                    if det.get("id") == item["id"]:
                        label = det.get("label", "object")
                        break

                result = analyze_with_claude(item["image_data"], label)
                cc.add_analysis_result(item["id"], result)
                cc.log(f"Analysis complete for #{item['id']}", "SUCCESS")
                cc.analyzing = False
        else:
            time.sleep(0.5)


# Flask routes

@app.route('/')
def index():
    """Main command center page."""
    return render_template('index.html',
                          prompts=cc.prompts,
                          threshold=cc.confidence_threshold,
                          skip_frames=cc.skip_frames,
                          tracking=cc.enable_tracking)


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    """Get current status."""
    filtered, hidden = cc.get_filtered_detections()
    return jsonify({
        "running": cc.running,
        "paused": cc.paused,
        "fps": round(cc.fps, 1),
        "frame_count": cc.frame_count,
        "device": cc.device_str,
        "detections": filtered,
        "hidden_counts": hidden,
        "prompts": cc.prompts,
        "max_objects": cc.max_objects_per_prompt,
        "show_all": cc.show_all_matches,
        "analyzing": cc.analyzing,
        "analysis_queue_size": len(cc.analysis_queue),
    })


@app.route('/api/logs')
def api_logs():
    """Get recent logs."""
    return jsonify({"logs": cc.get_logs()})


@app.route('/api/analysis_results')
def api_analysis_results():
    """Get analysis results."""
    with cc.lock:
        results = list(cc.analysis_results)
    return jsonify({"results": results})


@app.route('/api/set_prompts', methods=['POST'])
def api_set_prompts():
    """Set detection prompts."""
    data = request.json
    prompts_str = data.get("prompts", "object")
    cc.prompts = [p.strip() for p in prompts_str.split(",") if p.strip()]
    cc.state = None  # Reset detection state
    cc.last_masks = None
    cc.last_boxes = None
    cc.last_scores = None
    cc.last_labels = None
    cc.log(f"Prompts updated: {', '.join(cc.prompts)}")
    return jsonify({"success": True, "prompts": cc.prompts})


@app.route('/api/set_limit', methods=['POST'])
def api_set_limit():
    """Set max objects limit for a prompt."""
    data = request.json
    prompt = data.get("prompt")
    limit = data.get("limit")  # None for unlimited

    if limit is not None:
        cc.max_objects_per_prompt[prompt] = int(limit)
    elif prompt in cc.max_objects_per_prompt:
        del cc.max_objects_per_prompt[prompt]

    cc.log(f"Limit for '{prompt}': {limit if limit else 'unlimited'}")
    return jsonify({"success": True})


@app.route('/api/toggle_show_all', methods=['POST'])
def api_toggle_show_all():
    """Toggle show all matches for a prompt."""
    data = request.json
    prompt = data.get("prompt")
    cc.show_all_matches[prompt] = not cc.show_all_matches.get(prompt, False)
    cc.log(f"Show all for '{prompt}': {cc.show_all_matches[prompt]}")
    return jsonify({"success": True, "show_all": cc.show_all_matches[prompt]})


@app.route('/api/toggle_pause', methods=['POST'])
def api_toggle_pause():
    """Toggle pause state."""
    cc.paused = not cc.paused
    cc.log("Paused" if cc.paused else "Resumed")
    return jsonify({"success": True, "paused": cc.paused})


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset detection state."""
    cc.state = None
    cc.last_masks = None
    cc.last_boxes = None
    cc.last_scores = None
    cc.last_labels = None
    cc.clear_detections()
    cc.log("Detection state reset")
    return jsonify({"success": True})


@app.route('/api/set_threshold', methods=['POST'])
def api_set_threshold():
    """Set confidence threshold."""
    data = request.json
    cc.confidence_threshold = float(data.get("threshold", 0.3))
    if cc.processor:
        cc.processor.confidence_threshold = cc.confidence_threshold
    cc.log(f"Threshold set to {cc.confidence_threshold:.2f}")
    return jsonify({"success": True})


@app.route('/api/set_skip_frames', methods=['POST'])
def api_set_skip_frames():
    """Set skip frames value."""
    data = request.json
    cc.skip_frames = max(1, int(data.get("skip_frames", 3)))
    cc.log(f"Skip frames set to {cc.skip_frames}")
    return jsonify({"success": True})


@app.route('/api/toggle_tracking', methods=['POST'])
def api_toggle_tracking():
    """Toggle tracking."""
    cc.enable_tracking = not cc.enable_tracking
    cc.log(f"Tracking {'enabled' if cc.enable_tracking else 'disabled'}")
    return jsonify({"success": True, "tracking": cc.enable_tracking})


@app.route('/api/analyze_object', methods=['POST'])
def api_analyze_object():
    """Queue an object for Claude analysis."""
    data = request.json
    detection_id = data.get("detection_id")
    box = data.get("box")

    if cc.current_frame is None:
        return jsonify({"success": False, "error": "No frame available"})

    try:
        # Crop the object from current frame
        frame = cc.current_frame.copy()

        if box:
            x1, y1, x2, y2 = [int(v) for v in box]
            # Add padding
            h, w = frame.shape[:2]
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame

        # Encode to base64
        _, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        image_data = base64.b64encode(buffer).decode('utf-8')

        cc.queue_analysis(detection_id, image_data)
        cc.log(f"Queued object #{detection_id} for analysis")

        return jsonify({"success": True})

    except Exception as e:
        cc.log(f"Failed to queue analysis: {e}", "ERROR")
        return jsonify({"success": False, "error": str(e)})


def main():
    global cc

    parser = argparse.ArgumentParser(description="SAM3 Web Command Center")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device ID")
    parser.add_argument("--device", "-d", type=str, default=None, help="Device (cuda, mps, cpu)")
    parser.add_argument("--prompt", type=str, default="object", help="Initial prompts (comma-separated)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--skip-frames", type=int, default=3, help="Process every N frames")
    parser.add_argument("--no-tracking", action="store_true", help="Disable optical flow tracking")

    args = parser.parse_args()

    # Configure command center
    cc.prompts = [p.strip() for p in args.prompt.split(",") if p.strip()]
    cc.confidence_threshold = args.threshold
    cc.skip_frames = args.skip_frames
    cc.enable_tracking = not args.no_tracking

    if args.device:
        cc.device_str = args.device

    # Load model
    load_model(args.checkpoint)

    # Open camera
    cc.log(f"Opening camera {args.camera}...")
    cc.camera = cv2.VideoCapture(args.camera)

    if not cc.camera.isOpened():
        cc.log(f"Failed to open camera {args.camera}", "ERROR")
        return

    width = int(cc.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cc.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cc.log(f"Camera opened: {width}x{height}", "SUCCESS")

    cc.running = True

    # Start analysis worker
    analysis_thread = threading.Thread(target=analysis_worker, daemon=True)
    analysis_thread.start()

    print(f"\n{'='*50}")
    print(f"SAM3 Web Command Center")
    print(f"{'='*50}")
    print(f"Open http://localhost:{args.port} in your browser")
    print(f"{'='*50}\n")

    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False)
    finally:
        cc.running = False
        if cc.camera:
            cc.camera.release()


if __name__ == "__main__":
    main()
