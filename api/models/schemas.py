# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Pydantic models for SAM3 Video Segmentation API
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BackgroundMode(str, Enum):
    """Background removal mode options"""
    TRANSPARENT = "transparent"  # RGBA with transparent background
    BLACK = "black"  # Black background
    WHITE = "white"  # White background
    BLUR = "blur"  # Blurred background


class OutputFormat(str, Enum):
    """Output video format options"""
    MP4 = "mp4"
    WEBM = "webm"
    MOV = "mov"


class VideoSegmentationRequest(BaseModel):
    """Request model for video segmentation"""
    prompt: str = Field(
        default="dog",
        description="Text prompt describing what to segment (e.g., 'dog', 'cat', 'person')"
    )
    background_mode: BackgroundMode = Field(
        default=BackgroundMode.TRANSPARENT,
        description="How to handle the background"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.MP4,
        description="Output video format"
    )
    include_overlay: bool = Field(
        default=False,
        description="Include original video with mask overlay"
    )


class VideoSegmentationResponse(BaseModel):
    """Response model for video segmentation"""
    success: bool
    message: str
    output_video_path: Optional[str] = None
    overlay_video_path: Optional[str] = None
    total_frames: int = 0
    objects_detected: int = 0
    processing_time_seconds: float = 0.0


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    gpu_available: bool
    model_loaded: bool
    cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None


class SegmentationProgressResponse(BaseModel):
    """Response model for segmentation progress"""
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    current_frame: int = 0
    total_frames: int = 0
    message: Optional[str] = None

