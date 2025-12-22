#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Live Camera Segmentation with SAM3

This script captures video from a device camera and runs real-time segmentation
using SAM3. It supports automatic object detection or interactive point prompts.

Usage:
    # Auto-detect and segment all objects
    python live_camera_segmentation.py

    # Use specific camera device
    python live_camera_segmentation.py --camera 0

    # Specify device (cuda, mps, or cpu)
    python live_camera_segmentation.py --device mps

    # Interactive mode - click to add points
    python live_camera_segmentation.py --interactive

Controls:
    - 'q' or ESC: Quit
    - 'r': Reset/clear all segments
    - 's': Save current frame
    - 'p': Pause/resume
    - Left click: Add positive point (in interactive mode)
    - Right click: Add negative point (in interactive mode)
    - 'd': Toggle detection mode (auto-detect objects)
"""

import argparse
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

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
        image_size: int = 1008,
        detection_threshold: float = 0.5,
        checkpoint_path: Optional[str] = None,
        interactive: bool = False,
    ):
        """
        Initialize the live camera segmenter.

        Args:
            camera_id: Camera device ID (default 0 for primary camera)
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
            image_size: Image size for SAM3 processing
            detection_threshold: Confidence threshold for detections
            checkpoint_path: Optional path to model checkpoint
            interactive: Enable interactive point-based prompting
        """
        self.camera_id = camera_id
        self.device = torch.device(device) if device else get_device()
        self.image_size = image_size
        self.detection_threshold = detection_threshold
        self.interactive = interactive

        # State
        self.paused = False
        self.detection_mode = True
        self.points: List[Tuple[int, int]] = []
        self.labels: List[int] = []  # 1 for positive, 0 for negative
        self.current_masks: Optional[np.ndarray] = None
        self.current_scores: Optional[np.ndarray] = None
        self.fps_history = deque(maxlen=30)

        print(f"Initializing SAM3 on device: {self.device}")
        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: Optional[str] = None):
        """Load the SAM3 model."""
        from sam3.model_builder import build_sam3_image_model

        print("Loading SAM3 model...")
        self.model = build_sam3_image_model(
            device=str(self.device),
            checkpoint_path=checkpoint_path,
            load_from_HF=checkpoint_path is None,
            eval_mode=True,
            enable_segmentation=True,
        )
        print("Model loaded successfully!")

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a camera frame for SAM3."""
        # Resize to model input size
        frame_resized = cv2.resize(frame, (self.image_size, self.image_size))

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Normalize and convert to tensor
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW

        # Normalize with ImageNet stats (SAM3 uses 0.5, 0.5, 0.5)
        mean = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
        std = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
        frame_tensor = (frame_tensor - mean) / std

        # Add batch dimension and move to device
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)

        return frame_tensor

    def _run_detection(self, frame_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run object detection on a frame."""
        with torch.inference_mode():
            # Run the model in detection mode
            outputs = self.model(
                frame_tensor,
                multimask_output=True,
            )

            # Extract masks and scores
            if "pred_masks" in outputs:
                masks = outputs["pred_masks"]
                scores = outputs.get("pred_scores", torch.ones(masks.shape[0]))
            else:
                # Handle different output formats
                masks = outputs.get("masks", torch.zeros(1, 1, self.image_size, self.image_size))
                scores = outputs.get("scores", torch.ones(1))

            # Filter by threshold
            if scores.numel() > 0:
                keep = scores > self.detection_threshold
                masks = masks[keep] if keep.any() else masks[:0]
                scores = scores[keep] if keep.any() else scores[:0]

            # Convert to numpy
            masks_np = masks.cpu().numpy() if masks.numel() > 0 else np.array([])
            scores_np = scores.cpu().numpy() if scores.numel() > 0 else np.array([])

            # Get boxes if available
            boxes_np = np.array([])
            if "pred_boxes" in outputs:
                boxes = outputs["pred_boxes"]
                if keep.any():
                    boxes = boxes[keep]
                boxes_np = boxes.cpu().numpy()

        return masks_np, scores_np, boxes_np

    def _run_point_prompt(
        self,
        frame_tensor: torch.Tensor,
        points: List[Tuple[int, int]],
        labels: List[int],
        orig_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run segmentation with point prompts."""
        if not points:
            return np.array([]), np.array([])

        # Scale points to model input size
        h, w = orig_size
        scale_x = self.image_size / w
        scale_y = self.image_size / h

        scaled_points = [
            (int(p[0] * scale_x), int(p[1] * scale_y))
            for p in points
        ]

        # Convert to tensors
        points_tensor = torch.tensor(scaled_points, dtype=torch.float32).unsqueeze(0)
        labels_tensor = torch.tensor(labels, dtype=torch.int64).unsqueeze(0)

        points_tensor = points_tensor.to(self.device)
        labels_tensor = labels_tensor.to(self.device)

        with torch.inference_mode():
            # Run with point prompts
            outputs = self.model(
                frame_tensor,
                point_coords=points_tensor,
                point_labels=labels_tensor,
                multimask_output=True,
            )

            masks = outputs.get("masks", outputs.get("pred_masks", torch.zeros(1, 1, self.image_size, self.image_size)))
            scores = outputs.get("iou_predictions", outputs.get("pred_scores", torch.ones(1)))

            masks_np = masks.cpu().numpy()
            scores_np = scores.cpu().numpy()

        return masks_np, scores_np

    def _overlay_masks(
        self,
        frame: np.ndarray,
        masks: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Overlay segmentation masks on the frame."""
        if len(masks) == 0:
            return frame

        overlay = frame.copy()
        h, w = frame.shape[:2]

        for i, mask in enumerate(masks):
            # Resize mask to frame size if needed
            if mask.shape[-2:] != (h, w):
                if mask.ndim == 3:
                    mask = mask[0]  # Remove channel dim if present
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

        return overlay

    def _draw_points(self, frame: np.ndarray) -> np.ndarray:
        """Draw interaction points on the frame."""
        for point, label in zip(self.points, self.labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Green for positive, red for negative
            cv2.circle(frame, point, 5, color, -1)
            cv2.circle(frame, point, 7, (255, 255, 255), 2)
        return frame

    def _draw_info(self, frame: np.ndarray, fps: float, num_objects: int) -> np.ndarray:
        """Draw information overlay on the frame."""
        h, w = frame.shape[:2]

        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Objects: {num_objects}", (20, 60), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Device: {self.device}", (20, 85), font, 0.6, (255, 255, 255), 2)

        mode = "Interactive" if self.interactive else ("Detection" if self.detection_mode else "Paused")
        cv2.putText(frame, f"Mode: {mode}", (20, 110), font, 0.6, (255, 255, 255), 2)

        # Draw controls hint at bottom
        hint = "Q: Quit | R: Reset | S: Save | P: Pause | D: Toggle Detection"
        cv2.putText(frame, hint, (10, h - 10), font, 0.4, (200, 200, 200), 1)

        return frame

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for interactive mode."""
        if not self.interactive:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click - positive point
            self.points.append((x, y))
            self.labels.append(1)
            print(f"Added positive point at ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click - negative point
            self.points.append((x, y))
            self.labels.append(0)
            print(f"Added negative point at ({x}, {y})")

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

        # Create window
        window_name = "SAM3 Live Segmentation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        print("\nStarting live segmentation...")
        print("Controls:")
        print("  Q/ESC: Quit")
        print("  R: Reset segments")
        print("  S: Save frame")
        print("  P: Pause/resume")
        print("  D: Toggle detection mode")
        if self.interactive:
            print("  Left click: Add positive point")
            print("  Right click: Add negative point")

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

                if not self.paused:
                    # Preprocess frame
                    frame_tensor = self._preprocess_frame(frame)

                    # Run segmentation
                    if self.interactive and self.points:
                        # Point-based segmentation
                        masks, scores = self._run_point_prompt(
                            frame_tensor,
                            self.points,
                            self.labels,
                            (frame_height, frame_width),
                        )
                        boxes = np.array([])
                    elif self.detection_mode:
                        # Auto detection
                        masks, scores, boxes = self._run_detection(frame_tensor)
                    else:
                        masks, scores, boxes = np.array([]), np.array([]), np.array([])

                    self.current_masks = masks
                    self.current_scores = scores

                # Overlay masks
                if self.current_masks is not None and len(self.current_masks) > 0:
                    display_frame = self._overlay_masks(display_frame, self.current_masks)

                # Draw points in interactive mode
                if self.interactive:
                    display_frame = self._draw_points(display_frame)

                # Calculate FPS
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                self.fps_history.append(fps)
                avg_fps = sum(self.fps_history) / len(self.fps_history)

                # Draw info overlay
                num_objects = len(self.current_masks) if self.current_masks is not None else 0
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
                    self.points.clear()
                    self.labels.clear()
                    self.current_masks = None
                    self.current_scores = None

                elif key == ord('s'):  # Save
                    filename = f"sam3_capture_{frame_count}.png"
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved frame to {filename}")

                elif key == ord('p'):  # Pause
                    self.paused = not self.paused
                    print("Paused" if self.paused else "Resumed")

                elif key == ord('d'):  # Toggle detection
                    self.detection_mode = not self.detection_mode
                    print(f"Detection mode: {'ON' if self.detection_mode else 'OFF'}")

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
        "--image-size",
        type=int,
        default=1008,
        help="Image size for SAM3 processing (default: 1008)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
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
        help="Enable interactive point-based prompting",
    )

    args = parser.parse_args()

    # Print device info
    device = args.device or get_device_str()
    print(f"SAM3 Live Camera Segmentation")
    print(f"=" * 40)
    print(f"Device: {device}")
    print(f"Camera: {args.camera}")
    print(f"Image size: {args.image_size}")
    print(f"Interactive: {args.interactive}")
    print(f"=" * 40)

    # Create and run segmenter
    segmenter = LiveCameraSegmenter(
        camera_id=args.camera,
        device=args.device,
        image_size=args.image_size,
        detection_threshold=args.threshold,
        checkpoint_path=args.checkpoint,
        interactive=args.interactive,
    )
    segmenter.run()


if __name__ == "__main__":
    main()
