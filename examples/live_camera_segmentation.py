#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Live Camera Segmentation with SAM3

This script captures video from a device camera and runs real-time segmentation
using SAM3. It supports text-based detection or interactive point/box prompts.

Usage:
    # Detect objects using text prompt
    python live_camera_segmentation.py --prompt "person"

    # Use specific camera device
    python live_camera_segmentation.py --camera 0 --prompt "cat"

    # Specify device (cuda, mps, or cpu)
    python live_camera_segmentation.py --device mps --prompt "dog"

    # Interactive mode - click to add box prompts
    python live_camera_segmentation.py --interactive

    # Skip frames with tracking (masks follow objects between full inference frames)
    python live_camera_segmentation.py --prompt "person" --skip-frames 5 --track

Controls:
    - 'q' or ESC: Quit
    - 'r': Reset/clear all segments
    - 's': Save current frame
    - 'p': Pause/resume
    - Left click + drag: Draw box prompt (in interactive mode)
    - 't': Enter new text prompt
"""

import argparse
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.utils.device import get_device, get_device_str


class LiveCameraSegmenter:
    """Real-time camera segmentation using SAM3."""

    # Color palette for different object masks (BGR format for OpenCV)
    COLORS = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Purple
        (255, 128, 0),  # Orange
        (0, 128, 255),  # Light blue
        (128, 255, 0),  # Lime
    ]

    def __init__(
        self,
        camera_id: int = 0,
        device: Optional[str] = None,
        text_prompt: str = "object",
        confidence_threshold: float = 0.3,
        checkpoint_path: Optional[str] = None,
        interactive: bool = False,
        process_every_n_frames: int = 1,
        use_half_precision: bool = False,
        enable_tracking: bool = False,
    ):
        """
        Initialize the live camera segmenter.

        Args:
            camera_id: Camera device ID (default 0 for primary camera)
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
            text_prompt: Text description of objects to detect
            confidence_threshold: Confidence threshold for detections
            checkpoint_path: Optional path to model checkpoint
            interactive: Enable interactive box-based prompting
            process_every_n_frames: Only process every N frames (higher = faster but less smooth)
            use_half_precision: Use float16 for faster inference (may reduce accuracy)
            enable_tracking: Enable mask tracking between skipped frames
        """
        self.camera_id = camera_id
        self.device_str = device if device else get_device_str()
        self.device = torch.device(self.device_str)
        self.text_prompt = text_prompt
        self.confidence_threshold = confidence_threshold
        self.interactive = interactive
        self.process_every_n_frames = process_every_n_frames
        self.use_half_precision = use_half_precision
        self.enable_tracking = enable_tracking
        self.frame_count = 0

        # State
        self.paused = False
        self.state = None
        self.fps_history = deque(maxlen=30)

        # Tracking state
        self.tracker = None
        self.tracker_state = None
        self.last_masks = None
        self.last_boxes = None
        self.last_scores = None  # Store confidence scores
        self.video_height = None
        self.video_width = None

        # For interactive box drawing
        self.drawing = False
        self.box_start = None
        self.box_end = None

        print(f"Initializing SAM3 on device: {self.device}")
        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: Optional[str] = None):
        """Load the SAM3 model and processor."""
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        print("Loading SAM3 model...")
        model = build_sam3_image_model(
            device=self.device_str,
            checkpoint_path=checkpoint_path,
            load_from_HF=checkpoint_path is None,
            eval_mode=True,
            enable_segmentation=True,
        )

        # Convert to half precision for faster inference (CUDA only - MPS doesn't support it)
        if self.use_half_precision:
            if self.device_str == "mps":
                print("Warning: Half precision not supported on MPS due to Metal limitations, using float32")
                self.use_half_precision = False
            else:
                print("Converting model to half precision (float16)...")
                model = model.half()

        self.processor = Sam3Processor(
            model=model,
            resolution=1008,  # Fixed resolution due to precomputed positional encodings
            device=self.device_str,
            confidence_threshold=self.confidence_threshold,
        )
        print("Model loaded successfully!")

        # For tracking between keyframes, we use optical flow instead of the full SAM3 tracker
        # This provides lightweight motion-based tracking without device compatibility issues
        if self.enable_tracking:
            print("Tracking mode enabled - using optical flow for inter-frame tracking")
            self.prev_gray = None  # Store previous frame for optical flow

    def _load_tracker(self, checkpoint_path: Optional[str] = None):
        """Load the SAM3 tracker for mask propagation between frames."""
        from sam3.model_builder import build_tracker

        print("Loading SAM3 tracker for inter-frame tracking...")

        # Build tracker with backbone for processing new frames
        self.tracker = build_tracker(
            apply_temporal_disambiguation=True,
            with_backbone=True,
        )
        self.tracker = self.tracker.to(self.device)
        self.tracker.eval()

        # Try to load tracker weights from the same source as the main model
        # The tracker shares weights with the main SAM3 model
        import os
        tracker_ckpt_path = None

        # Use provided checkpoint path first
        if checkpoint_path and os.path.exists(checkpoint_path):
            tracker_ckpt_path = checkpoint_path
        else:
            # Check common locations for the checkpoint
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                os.path.join(script_dir, "sam3.pt"),  # Same folder as script (examples/)
                "sam3.pt",
                "./sam3.pt",
                "../sam3.pt",
                "examples/sam3.pt",
                os.path.expanduser("~/.cache/huggingface/hub/models--facebook--sam3/sam3.pt"),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    tracker_ckpt_path = path
                    break

        if tracker_ckpt_path is None:
            print("Warning: Could not find sam3.pt checkpoint for tracker.")
            print("Please ensure sam3.pt is in the current directory or provide --checkpoint path.")
            print("Tracking will be disabled.")
            self.tracker = None
            return

        print(f"Loading tracker weights from: {tracker_ckpt_path}")
        tracker_state_dict = torch.load(tracker_ckpt_path, map_location=self.device, weights_only=False)

        # Filter and load tracker-compatible weights
        tracker_keys = set(k for k in self.tracker.state_dict().keys())
        filtered_state_dict = {k: v for k, v in tracker_state_dict.items() if k in tracker_keys}
        self.tracker.load_state_dict(filtered_state_dict, strict=False)

        print("Tracker loaded successfully!")

    def _init_tracker_state(self, height: int, width: int):
        """Initialize tracking state for a video stream."""
        self.video_height = height
        self.video_width = width
        # Reset masks and optical flow state
        self.last_masks = None
        self.last_boxes = None
        self.last_scores = None
        self.prev_gray = None

    def _track_frame(self, frame: np.ndarray, frame_idx: int) -> Optional[torch.Tensor]:
        """
        Use optical flow to track masks to a new frame.

        This provides lightweight motion-based tracking between keyframes
        without needing the full SAM3 tracker model.

        Returns the tracked masks or None if tracking isn't available.
        """
        if self.last_masks is None or len(self.last_masks) == 0:
            return None

        if self.prev_gray is None:
            return self.last_masks

        try:
            # Convert current frame to grayscale
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            # Create coordinate grids for remapping
            h, w = curr_gray.shape
            flow_map_x = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(np.float32)
            flow_map_y = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(np.float32)

            # Add flow to get new positions
            flow_map_x += flow[:, :, 0]
            flow_map_y += flow[:, :, 1]

            # Warp each mask using the flow
            tracked_masks = []
            for mask in self.last_masks:
                # Convert mask to numpy for warping
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy().squeeze()
                else:
                    mask_np = mask.squeeze()

                # Ensure mask is the right size
                if mask_np.shape != (h, w):
                    mask_np = cv2.resize(mask_np.astype(np.float32), (w, h))

                # Warp mask using optical flow
                warped_mask = cv2.remap(
                    mask_np.astype(np.float32),
                    flow_map_x, flow_map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

                # Threshold to get binary mask
                warped_mask = (warped_mask > 0.5).astype(np.float32)

                # Convert back to tensor
                tracked_masks.append(
                    torch.from_numpy(warped_mask).unsqueeze(0).to(self.device)
                )

            # Update prev_gray for next iteration
            self.prev_gray = curr_gray

            if tracked_masks:
                return torch.stack(tracked_masks)

        except Exception as e:
            print(f"Optical flow tracking error: {e}")

        return self.last_masks

    def _add_mask_to_tracker(self, masks: torch.Tensor, frame: np.ndarray, frame_idx: int):
        """Store frame for optical flow tracking."""
        # Store grayscale frame for optical flow computation
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Masks are already stored in self.last_masks by the caller

    def _process_frame(self, frame: np.ndarray) -> dict:
        """Process a frame through SAM3."""
        # Convert BGR to RGB PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Set the image
        self.state = self.processor.set_image(pil_image, self.state)

        # Run text-based detection
        if not self.interactive:
            self.state = self.processor.set_text_prompt(self.text_prompt, self.state)

        return self.state

    def _add_box_prompt(self, box: Tuple[int, int, int, int], frame_size: Tuple[int, int]):
        """Add a box prompt in interactive mode."""
        if self.state is None:
            return

        h, w = frame_size
        x1, y1, x2, y2 = box

        # Convert to center format and normalize to [0, 1]
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = abs(x2 - x1) / w
        bh = abs(y2 - y1) / h

        normalized_box = [cx, cy, bw, bh]
        self.state = self.processor.add_geometric_prompt(
            box=normalized_box,
            label=True,  # Positive box
            state=self.state,
        )

    def _overlay_masks(
        self,
        frame: np.ndarray,
        masks: torch.Tensor,
        boxes: torch.Tensor = None,
        scores: torch.Tensor = None,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Overlay segmentation masks on the frame with labels and confidence scores."""
        if masks is None or masks.numel() == 0:
            return frame

        overlay = frame.copy()
        h, w = frame.shape[:2]

        # masks shape: [N, 1, H, W]
        masks_np = masks.squeeze(1).cpu().numpy()

        # Get scores if available
        scores_np = None
        if scores is not None:
            scores_np = scores.cpu().numpy()

        # Get boxes if available
        boxes_np = None
        if boxes is not None:
            boxes_np = boxes.cpu().numpy()

        for i, mask in enumerate(masks_np):
            # Resize mask to frame size if needed
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5

            # Get color for this mask
            color = self.COLORS[i % len(self.COLORS)]

            # Create colored overlay
            mask_region = mask.astype(bool)
            overlay[mask_region] = (
                overlay[mask_region] * (1 - alpha) +
                np.array(color) * alpha
            ).astype(np.uint8)

            # Draw contour
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 2)

            # Draw label with confidence score
            # Find the top-center of the mask for label placement
            if len(contours) > 0:
                # Get bounding rect of largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, cw, ch = cv2.boundingRect(largest_contour)

                # Get confidence score
                conf = scores_np[i] if scores_np is not None and i < len(scores_np) else 0.0

                # Create label text
                label = f"{self.text_prompt} #{i+1}"
                conf_text = f"{conf:.0%}"

                # Draw label background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1

                # Get text sizes
                (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale, thickness)

                # Position at top of bounding box
                label_x = x + cw // 2 - label_w // 2
                label_y = max(y - 5, label_h + 5)

                # Draw label background
                cv2.rectangle(overlay,
                    (label_x - 2, label_y - label_h - 2),
                    (label_x + label_w + 2, label_y + 2),
                    color, -1)

                # Draw label text
                cv2.putText(overlay, label,
                    (label_x, label_y),
                    font, font_scale, (255, 255, 255), thickness)

                # Draw confidence below label
                conf_x = x + cw // 2 - conf_w // 2
                conf_y = label_y + conf_h + 8

                cv2.rectangle(overlay,
                    (conf_x - 2, conf_y - conf_h - 2),
                    (conf_x + conf_w + 2, conf_y + 2),
                    (0, 0, 0), -1)
                cv2.putText(overlay, conf_text,
                    (conf_x, conf_y),
                    font, font_scale, (0, 255, 0), thickness)

        return overlay

    def _draw_boxes(self, frame: np.ndarray, boxes: torch.Tensor, scores: torch.Tensor = None) -> np.ndarray:
        """Draw bounding boxes on the frame with labels."""
        if boxes is None or boxes.numel() == 0:
            return frame

        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy() if scores is not None else None

        for i, box in enumerate(boxes_np):
            x1, y1, x2, y2 = box.astype(int)
            color = self.COLORS[i % len(self.COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        return frame

    def _draw_object_panel(self, frame: np.ndarray, masks: torch.Tensor,
                           boxes: torch.Tensor, scores: torch.Tensor) -> np.ndarray:
        """Draw an info panel on the right side showing detected objects."""
        h, w = frame.shape[:2]

        # Panel dimensions
        panel_width = 200
        panel_x = w - panel_width - 10

        # Count objects
        num_objects = len(masks) if masks is not None else 0

        # Calculate panel height based on number of objects
        header_height = 40
        object_height = 50
        panel_height = header_height + max(num_objects, 1) * object_height + 20

        # Draw semi-transparent panel background
        overlay = frame.copy()
        cv2.rectangle(overlay,
            (panel_x, 10),
            (w - 10, min(10 + panel_height, h - 10)),
            (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Draw panel header
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "DETECTED OBJECTS",
            (panel_x + 10, 35),
            font, 0.5, (255, 255, 255), 1)
        cv2.line(frame, (panel_x + 5, 45), (w - 15, 45), (100, 100, 100), 1)

        if num_objects == 0:
            cv2.putText(frame, "No objects found",
                (panel_x + 10, 75),
                font, 0.4, (150, 150, 150), 1)
            return frame

        # Draw each object
        masks_np = masks.squeeze(1).cpu().numpy() if masks is not None else []
        scores_np = scores.cpu().numpy() if scores is not None else []
        boxes_np = boxes.cpu().numpy() if boxes is not None else []

        for i in range(num_objects):
            y_offset = header_height + 15 + i * object_height

            if 10 + y_offset + 40 > h - 10:
                # Panel would exceed frame height
                cv2.putText(frame, f"... +{num_objects - i} more",
                    (panel_x + 10, 10 + y_offset),
                    font, 0.4, (150, 150, 150), 1)
                break

            color = self.COLORS[i % len(self.COLORS)]

            # Color indicator
            cv2.rectangle(frame,
                (panel_x + 10, 10 + y_offset),
                (panel_x + 25, 10 + y_offset + 15),
                color, -1)

            # Object label
            label = f"{self.text_prompt} #{i+1}"
            cv2.putText(frame, label,
                (panel_x + 35, 10 + y_offset + 12),
                font, 0.4, (255, 255, 255), 1)

            # Confidence score
            if i < len(scores_np):
                conf = scores_np[i]
                conf_color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.4 else (0, 0, 255)
                cv2.putText(frame, f"Conf: {conf:.0%}",
                    (panel_x + 35, 10 + y_offset + 28),
                    font, 0.35, conf_color, 1)

            # Bounding box size
            if i < len(boxes_np):
                box = boxes_np[i]
                bw = int(box[2] - box[0])
                bh = int(box[3] - box[1])
                cv2.putText(frame, f"Size: {bw}x{bh}",
                    (panel_x + 100, 10 + y_offset + 28),
                    font, 0.35, (150, 150, 150), 1)

        return frame

    def _draw_info(self, frame: np.ndarray, fps: float, num_objects: int) -> np.ndarray:
        """Draw information overlay on the frame."""
        h, w = frame.shape[:2]

        # Semi-transparent background for text
        overlay = frame.copy()
        info_height = 165 if self.enable_tracking else 140
        cv2.rectangle(overlay, (10, 10), (350, info_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Objects: {num_objects}", (20, 60), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Device: {self.device_str}", (20, 85), font, 0.6, (255, 255, 255), 2)

        mode = "Interactive" if self.interactive else f"Prompt: {self.text_prompt}"
        cv2.putText(frame, f"Mode: {mode}", (20, 110), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {self.confidence_threshold:.2f}", (20, 135), font, 0.6, (255, 255, 255), 2)

        if self.enable_tracking:
            skip_info = f"Skip: {self.process_every_n_frames} (tracking ON)"
            cv2.putText(frame, skip_info, (20, 160), font, 0.6, (0, 255, 0), 2)

        # Draw controls hint at bottom
        hint = "Q: Quit | R: Reset | S: Save | P: Pause | T: New prompt"
        cv2.putText(frame, hint, (10, h - 10), font, 0.4, (200, 200, 200), 1)

        return frame

    def _draw_current_box(self, frame: np.ndarray) -> np.ndarray:
        """Draw the box currently being drawn."""
        if self.drawing and self.box_start and self.box_end:
            cv2.rectangle(
                frame,
                self.box_start,
                self.box_end,
                (0, 255, 0),
                2
            )
        return frame

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for interactive mode."""
        if not self.interactive:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.box_start = (x, y)
            self.box_end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.box_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.box_end = (x, y)

                # Add the box prompt if it's a valid box
                x1, y1 = self.box_start
                x2, y2 = self.box_end
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    frame_size = param  # Passed as param
                    self._add_box_prompt((x1, y1, x2, y2), frame_size)

                self.box_start = None
                self.box_end = None

    def run(self):
        """Run the live camera segmentation loop."""
        # Open camera
        print(f"Opening camera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return

        # Get camera properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {frame_width}x{frame_height}")

        # Initialize tracker state if tracking is enabled
        if self.enable_tracking:
            print("Initializing tracker state...")
            self._init_tracker_state(frame_height, frame_width)

        # Create window
        window_name = "SAM3 Live Segmentation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback, (frame_height, frame_width))

        print("\nStarting live segmentation...")
        print("Controls:")
        print("  Q/ESC: Quit")
        print("  R: Reset segments")
        print("  S: Save frame")
        print("  P: Pause/resume")
        print("  T: Enter new text prompt")
        if self.interactive:
            print("  Left click + drag: Draw box prompt")

        frame_count = 0
        try:
            while True:
                start_time = time.time()

                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                display_frame = frame.copy()
                self.frame_count += 1

                if not self.paused:
                    is_keyframe = self.frame_count % self.process_every_n_frames == 0

                    if is_keyframe:
                        # Full inference frame - run text detection
                        self._process_frame(frame)

                        # Store masks, boxes, and scores for tracking
                        if self.state is not None:
                            self.last_masks = self.state.get("masks")
                            self.last_boxes = self.state.get("boxes")
                            self.last_scores = self.state.get("scores")

                            # Add masks to tracker for memory-based propagation
                            if self.enable_tracking and self.last_masks is not None:
                                self._add_mask_to_tracker(self.last_masks, frame, self.frame_count)

                    elif self.enable_tracking and self.last_masks is not None:
                        # Intermediate frame - use tracker to propagate masks
                        tracked_masks = self._track_frame(frame, self.frame_count)
                        if tracked_masks is not None:
                            self.last_masks = tracked_masks
                            # Update state with tracked masks
                            if self.state is not None:
                                self.state["masks"] = tracked_masks
                    # else: Just reuse last masks (no tracking)

                # Overlay results - use last_masks if tracking is enabled
                masks_to_display = None
                boxes_to_display = None
                scores_to_display = None

                if self.enable_tracking:
                    masks_to_display = self.last_masks
                    boxes_to_display = self.last_boxes
                    scores_to_display = self.last_scores
                elif self.state is not None:
                    masks_to_display = self.state.get("masks")
                    boxes_to_display = self.state.get("boxes")
                    scores_to_display = self.state.get("scores")

                if masks_to_display is not None:
                    display_frame = self._overlay_masks(
                        display_frame, masks_to_display,
                        boxes=boxes_to_display, scores=scores_to_display
                    )
                if boxes_to_display is not None:
                    display_frame = self._draw_boxes(display_frame, boxes_to_display, scores_to_display)

                # Draw object info panel on the right
                display_frame = self._draw_object_panel(
                    display_frame, masks_to_display, boxes_to_display, scores_to_display
                )

                # Draw current box being drawn
                if self.interactive:
                    display_frame = self._draw_current_box(display_frame)

                # Calculate FPS
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                self.fps_history.append(fps)
                avg_fps = sum(self.fps_history) / len(self.fps_history)

                # Draw info overlay
                num_objects = 0
                if masks_to_display is not None:
                    num_objects = len(masks_to_display)
                display_frame = self._draw_info(display_frame, avg_fps, num_objects)

                # Show frame
                cv2.imshow(window_name, display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q or ESC
                    print("Quitting...")
                    break

                elif key == ord('r'):  # Reset
                    print("Resetting segments...")
                    if self.state is not None:
                        self.processor.reset_all_prompts(self.state)
                    self.state = None
                    self.last_masks = None
                    self.last_boxes = None
                    self.last_scores = None
                    # Reset tracker state
                    if self.enable_tracking:
                        self._init_tracker_state(frame_height, frame_width)

                elif key == ord('s'):  # Save
                    filename = f"sam3_capture_{frame_count}.png"
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved frame to {filename}")

                elif key == ord('p'):  # Pause
                    self.paused = not self.paused
                    print("Paused" if self.paused else "Resumed")

                elif key == ord('t'):  # New text prompt
                    self.paused = True
                    new_prompt = input("Enter new text prompt: ").strip()
                    if new_prompt:
                        self.text_prompt = new_prompt
                        if self.state is not None:
                            self.processor.reset_all_prompts(self.state)
                        self.state = None
                        self.last_masks = None
                        self.last_boxes = None
                        self.last_scores = None
                        # Reset tracker for new prompt
                        if self.enable_tracking:
                            self._init_tracker_state(frame_height, frame_width)
                        print(f"Text prompt set to: {self.text_prompt}")
                    self.paused = False

                frame_count += 1

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Live Camera Segmentation with SAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to run on (default: auto-detect)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="object",
        help="Text prompt for detection (default: 'object')",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: download from HuggingFace)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enable interactive box-based prompting",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Process every N frames (higher = faster, default: 1)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use half precision (float16) for faster inference",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable mask tracking between skipped frames (smoother results when using --skip-frames)",
    )

    args = parser.parse_args()

    # Print device info
    device = args.device or get_device_str()
    print(f"SAM3 Live Camera Segmentation")
    print(f"=" * 40)
    print(f"Device: {device}")
    print(f"Camera: {args.camera}")
    print(f"Text prompt: {args.prompt}")
    print(f"Threshold: {args.threshold}")
    print(f"Interactive: {args.interactive}")
    print(f"Skip frames: {args.skip_frames}")
    print(f"Half precision: {args.half}")
    print(f"Tracking: {args.track}")
    print(f"=" * 40)

    # Create and run segmenter
    segmenter = LiveCameraSegmenter(
        camera_id=args.camera,
        device=args.device,
        text_prompt=args.prompt,
        confidence_threshold=args.threshold,
        checkpoint_path=args.checkpoint,
        interactive=args.interactive,
        process_every_n_frames=args.skip_frames,
        use_half_precision=args.half,
        enable_tracking=args.track,
    )
    segmenter.run()


if __name__ == "__main__":
    main()
