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
        """
        self.camera_id = camera_id
        self.device_str = device if device else get_device_str()
        self.device = torch.device(self.device_str)
        self.text_prompt = text_prompt
        self.confidence_threshold = confidence_threshold
        self.interactive = interactive
        self.process_every_n_frames = process_every_n_frames
        self.use_half_precision = use_half_precision
        self.frame_count = 0

        # State
        self.paused = False
        self.state = None
        self.fps_history = deque(maxlen=30)

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
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Overlay segmentation masks on the frame."""
        if masks is None or masks.numel() == 0:
            return frame

        overlay = frame.copy()
        h, w = frame.shape[:2]

        # masks shape: [N, 1, H, W]
        masks_np = masks.squeeze(1).cpu().numpy()

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

        return overlay

    def _draw_boxes(self, frame: np.ndarray, boxes: torch.Tensor) -> np.ndarray:
        """Draw bounding boxes on the frame."""
        if boxes is None or boxes.numel() == 0:
            return frame

        boxes_np = boxes.cpu().numpy()

        for i, box in enumerate(boxes_np):
            x1, y1, x2, y2 = box.astype(int)
            color = self.COLORS[i % len(self.COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        return frame

    def _draw_info(self, frame: np.ndarray, fps: float, num_objects: int) -> np.ndarray:
        """Draw information overlay on the frame."""
        h, w = frame.shape[:2]

        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Objects: {num_objects}", (20, 60), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Device: {self.device_str}", (20, 85), font, 0.6, (255, 255, 255), 2)

        mode = "Interactive" if self.interactive else f"Prompt: {self.text_prompt}"
        cv2.putText(frame, f"Mode: {mode}", (20, 110), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {self.confidence_threshold:.2f}", (20, 135), font, 0.6, (255, 255, 255), 2)

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
                    # Only process every N frames for performance
                    if self.frame_count % self.process_every_n_frames == 0:
                        self._process_frame(frame)

                # Overlay results
                if self.state is not None:
                    masks = self.state.get("masks")
                    boxes = self.state.get("boxes")

                    if masks is not None:
                        display_frame = self._overlay_masks(display_frame, masks)
                    if boxes is not None:
                        display_frame = self._draw_boxes(display_frame, boxes)

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
                if self.state is not None and self.state.get("masks") is not None:
                    num_objects = len(self.state["masks"])
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
    )
    segmenter.run()


if __name__ == "__main__":
    main()
