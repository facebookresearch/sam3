# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
SAM3 Video Segmentation Service
"""

import gc
import os
import shutil
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from api.utils.video_utils import (
    apply_mask_to_frame,
    combine_masks,
    create_video_from_frames,
    create_video_with_alpha,
    extract_frames_from_video,
    load_video_frames,
)


class Sam3VideoService:
    """Service for video segmentation using SAM3"""
    
    _instance = None
    _predictor = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one model instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the service"""
        if not hasattr(self, '_initialized'):
            self._initialized = False
            self._predictor = None
            self._model_loaded = False
    
    def load_model(self) -> bool:
        """
        Load SAM3 video predictor model.
        
        Returns:
            True if model loaded successfully
        """
        if self._model_loaded:
            return True
        
        try:
            from sam3.model_builder import build_sam3_video_predictor
            
            # Use available GPUs
            if torch.cuda.is_available():
                gpus_to_use = list(range(torch.cuda.device_count()))
            else:
                raise RuntimeError("CUDA is required for SAM3")
            
            # Use local weights from container (downloaded from GCS during build)
            checkpoint_path = "/app/weights/sam3.pt"
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Model weights not found at {checkpoint_path}. "
                    "Weights should be downloaded from GCS during container build."
                )
            
            print(f"Loading SAM3 model from: {checkpoint_path}")
            print(f"Using GPUs: {gpus_to_use}")
            self._predictor = build_sam3_video_predictor(
                checkpoint_path=checkpoint_path,
                gpus_to_use=gpus_to_use
            )
            self._model_loaded = True
            print("SAM3 model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading SAM3 model: {e}")
            self._model_loaded = False
            raise
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model_loaded
    
    def get_gpu_info(self) -> dict:
        """Get GPU information"""
        info = {
            "gpu_available": torch.cuda.is_available(),
            "cuda_version": None,
            "gpu_name": None,
            "gpu_memory_allocated": None,
            "gpu_memory_total": None,
        }
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        
        return info
    
    def segment_video(
        self,
        video_path: str,
        prompt: str = "dog",
        background_mode: str = "transparent",
        output_dir: Optional[str] = None,
        include_overlay: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> Dict:
        """
        Segment objects matching the prompt from a video.
        
        Args:
            video_path: Path to input video file
            prompt: Text prompt describing what to segment (e.g., "dog")
            background_mode: How to handle background
            output_dir: Directory to save output files
            include_overlay: Whether to create overlay video
            progress_callback: Callback function for progress updates
        
        Returns:
            Dictionary with results and output paths
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Create output directory
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="sam3_output_")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create temp directory for extracted frames
        temp_frames_dir = tempfile.mkdtemp(prefix="sam3_frames_")
        
        try:
            # Extract frames from video
            if progress_callback:
                progress_callback("Extracting video frames...", 0.0, 0, 0)
            
            frame_paths, fps, (width, height) = extract_frames_from_video(
                video_path, temp_frames_dir
            )
            total_frames = len(frame_paths)
            
            if progress_callback:
                progress_callback("Starting SAM3 session...", 0.05, 0, total_frames)
            
            # Start SAM3 session
            response = self._predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=temp_frames_dir,
                )
            )
            session_id = response["session_id"]
            
            try:
                # Add text prompt on first frame
                if progress_callback:
                    progress_callback(f"Adding prompt: '{prompt}'...", 0.1, 0, total_frames)
                
                response = self._predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=0,
                        text=prompt,
                    )
                )
                
                initial_output = response["outputs"]
                objects_detected = len(initial_output.get("out_obj_ids", []))
                
                if objects_detected == 0:
                    return {
                        "success": False,
                        "message": f"No objects matching '{prompt}' detected in the video",
                        "output_video_path": None,
                        "overlay_video_path": None,
                        "total_frames": total_frames,
                        "objects_detected": 0,
                        "processing_time_seconds": time.time() - start_time,
                    }
                
                # Propagate through video
                if progress_callback:
                    progress_callback("Propagating segmentation...", 0.15, 0, total_frames)
                
                outputs_per_frame = {}
                for response in self._predictor.handle_stream_request(
                    request=dict(
                        type="propagate_in_video",
                        session_id=session_id,
                    )
                ):
                    frame_idx = response["frame_index"]
                    outputs_per_frame[frame_idx] = response["outputs"]
                    
                    if progress_callback:
                        progress = 0.15 + (0.6 * (frame_idx + 1) / total_frames)
                        progress_callback(
                            f"Processing frame {frame_idx + 1}/{total_frames}",
                            progress,
                            frame_idx + 1,
                            total_frames
                        )
                
                # Process frames with masks
                if progress_callback:
                    progress_callback("Applying masks to frames...", 0.75, 0, total_frames)
                
                processed_frames = []
                overlay_frames = []
                
                # Load original video frames
                video_frames, _ = load_video_frames(video_path)
                
                for frame_idx in sorted(outputs_per_frame.keys()):
                    frame = video_frames[frame_idx]
                    output = outputs_per_frame[frame_idx]
                    
                    # Get all masks for this frame
                    masks = output.get("out_binary_masks", [])
                    
                    if len(masks) > 0:
                        # Combine all detected object masks
                        combined_mask = combine_masks(list(masks))
                        
                        # Apply mask to remove background
                        processed_frame = apply_mask_to_frame(
                            frame, combined_mask, background_mode
                        )
                        processed_frames.append(processed_frame)
                        
                        # Create overlay version if requested
                        if include_overlay:
                            overlay = self._create_overlay_frame(
                                frame, masks, output.get("out_obj_ids", [])
                            )
                            overlay_frames.append(overlay)
                    else:
                        # No mask for this frame, use transparent/background
                        if background_mode == "transparent":
                            alpha = np.zeros((*frame.shape[:2], 1), dtype=np.uint8)
                            processed_frames.append(np.dstack([frame, alpha]))
                        else:
                            bg_frame = apply_mask_to_frame(
                                frame,
                                np.zeros(frame.shape[:2], dtype=np.uint8),
                                background_mode
                            )
                            processed_frames.append(bg_frame)
                        
                        if include_overlay:
                            overlay_frames.append(frame)
                    
                    if progress_callback:
                        progress = 0.75 + (0.2 * (frame_idx + 1) / total_frames)
                        progress_callback(
                            f"Applying mask to frame {frame_idx + 1}/{total_frames}",
                            progress,
                            frame_idx + 1,
                            total_frames
                        )
                
                # Create output video
                if progress_callback:
                    progress_callback("Creating output video...", 0.95, total_frames, total_frames)
                
                has_alpha = background_mode == "transparent"
                output_ext = ".webm" if has_alpha else ".mp4"
                output_video_path = os.path.join(output_dir, f"segmented{output_ext}")
                
                if has_alpha:
                    create_video_with_alpha(processed_frames, output_video_path, fps)
                else:
                    create_video_from_frames(processed_frames, output_video_path, fps)
                
                # Create overlay video if requested
                overlay_video_path = None
                if include_overlay and overlay_frames:
                    overlay_video_path = os.path.join(output_dir, "overlay.mp4")
                    create_video_from_frames(overlay_frames, overlay_video_path, fps)
                
                if progress_callback:
                    progress_callback("Complete!", 1.0, total_frames, total_frames)
                
                return {
                    "success": True,
                    "message": f"Successfully segmented {objects_detected} object(s) matching '{prompt}'",
                    "output_video_path": output_video_path,
                    "overlay_video_path": overlay_video_path,
                    "total_frames": total_frames,
                    "objects_detected": objects_detected,
                    "processing_time_seconds": time.time() - start_time,
                }
                
            finally:
                # Close session
                self._predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=session_id,
                    )
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        finally:
            # Cleanup temp frames directory
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
    
    def _create_overlay_frame(
        self,
        frame: np.ndarray,
        masks: List[np.ndarray],
        obj_ids: List[int]
    ) -> np.ndarray:
        """
        Create a frame with colored mask overlays.
        
        Args:
            frame: Original frame
            masks: List of binary masks
            obj_ids: List of object IDs
        
        Returns:
            Frame with overlays
        """
        # Generate distinct colors
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]
        
        overlay = frame.copy().astype(np.float32)
        
        for idx, (mask, obj_id) in enumerate(zip(masks, obj_ids)):
            color = colors[obj_id % len(colors)]
            mask_bool = mask.astype(bool)
            
            # Blend color with frame
            for c in range(3):
                overlay[..., c][mask_bool] = (
                    0.6 * overlay[..., c][mask_bool] +
                    0.4 * color[c]
                )
            
            # Draw contour
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay.astype(np.uint8), contours, -1, color, 2)
        
        return overlay.astype(np.uint8)
    
    def shutdown(self):
        """Shutdown the predictor and free resources"""
        if self._predictor is not None:
            try:
                self._predictor.shutdown()
            except Exception as e:
                print(f"Error shutting down predictor: {e}")
            
            self._predictor = None
            self._model_loaded = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Global service instance
sam3_service = Sam3VideoService()

