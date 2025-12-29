# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
SAM3 Video Segmentation FastAPI Backend

This API provides endpoints for video segmentation using Meta's SAM3 model.
Upload a video and get back a video with only the specified object (e.g., dog)
visible, with the background removed.
"""

import asyncio
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.models.schemas import (
    BackgroundMode,
    HealthCheckResponse,
    OutputFormat,
    SegmentationProgressResponse,
    VideoSegmentationRequest,
    VideoSegmentationResponse,
)
from api.services.sam3_service import sam3_service


# Store for tracking async tasks
task_store: dict = {}

# Output directory for processed videos
OUTPUT_DIR = Path("outputs")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Loads the model on startup and cleans up on shutdown.
    """
    # Startup
    print("=" * 60)
    print("SAM3 Video Segmentation API Starting...")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load the SAM3 model
    try:
        sam3_service.load_model()
        print("SAM3 model loaded successfully!")
    except Exception as e:
        print(f"Warning: Failed to load SAM3 model on startup: {e}")
        print("Model will be loaded on first request.")
    
    yield
    
    # Shutdown
    print("Shutting down SAM3 service...")
    sam3_service.shutdown()
    print("Cleanup complete.")


# Create FastAPI app
app = FastAPI(
    title="SAM3 Video Segmentation API",
    description="""
    ## Video Background Removal with SAM3
    
    This API uses Meta's Segment Anything Model 3 (SAM3) to segment objects from videos
    and remove the background.
    
    ### Features:
    - **Text-based segmentation**: Simply describe what you want to keep (e.g., "dog", "cat", "person")
    - **Multiple background modes**: Transparent, black, white, or blurred background
    - **Multiple output formats**: MP4, WebM (with alpha), MOV
    
    ### Example Usage:
    1. Upload a video with a dog
    2. Set prompt to "dog"
    3. Get back a video with only the dog visible
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving output videos
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "SAM3 Video Segmentation API",
        "version": "1.0.0",
        "description": "Remove backgrounds from videos using SAM3",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Check API health and model status.
    
    Returns GPU availability, model loading status, and CUDA information.
    """
    gpu_info = sam3_service.get_gpu_info()
    
    return HealthCheckResponse(
        status="healthy" if sam3_service.is_loaded else "model_not_loaded",
        gpu_available=gpu_info["gpu_available"],
        model_loaded=sam3_service.is_loaded,
        cuda_version=gpu_info.get("cuda_version"),
        gpu_name=gpu_info.get("gpu_name"),
    )


@app.post("/segment/dog", response_model=VideoSegmentationResponse, tags=["Segmentation"])
async def segment_dog_from_video(
    video: UploadFile = File(..., description="Video file to process"),
    background_mode: BackgroundMode = Form(
        default=BackgroundMode.TRANSPARENT,
        description="Background removal mode"
    ),
    output_format: OutputFormat = Form(
        default=OutputFormat.MP4,
        description="Output video format"
    ),
    include_overlay: bool = Form(
        default=False,
        description="Include overlay visualization"
    ),
):
    """
    Segment dogs from a video and remove the background.
    
    This endpoint specifically extracts dogs from the video, removing all
    other elements. The background can be made transparent, black, white,
    or blurred.
    
    **Parameters:**
    - **video**: The input video file (MP4, MOV, AVI, etc.)
    - **background_mode**: How to handle the background
        - `transparent`: RGBA with transparent background (requires WebM format)
        - `black`: Solid black background
        - `white`: Solid white background  
        - `blur`: Blurred version of the original background
    - **output_format**: Output video format (mp4, webm, mov)
    - **include_overlay**: Include a visualization video with colored mask overlay
    
    **Returns:**
    - Processed video with only the dog(s) visible
    - Optional overlay video for visualization
    """
    return await _process_video_segmentation(
        video=video,
        prompt="dog",
        background_mode=background_mode,
        output_format=output_format,
        include_overlay=include_overlay,
    )


@app.post("/segment", response_model=VideoSegmentationResponse, tags=["Segmentation"])
async def segment_from_video(
    video: UploadFile = File(..., description="Video file to process"),
    prompt: str = Form(
        default="dog",
        description="Text prompt describing what to segment"
    ),
    background_mode: BackgroundMode = Form(
        default=BackgroundMode.TRANSPARENT,
        description="Background removal mode"
    ),
    output_format: OutputFormat = Form(
        default=OutputFormat.MP4,
        description="Output video format"
    ),
    include_overlay: bool = Form(
        default=False,
        description="Include overlay visualization"
    ),
):
    """
    Segment objects from a video using a custom text prompt.
    
    Use this endpoint to extract any object you can describe with text.
    For example: "dog", "cat", "person", "car", "bird", etc.
    
    **Parameters:**
    - **video**: The input video file
    - **prompt**: Text describing what to segment (e.g., "dog", "person with red shirt")
    - **background_mode**: How to handle the background
    - **output_format**: Output video format
    - **include_overlay**: Include visualization video
    
    **Returns:**
    - Processed video with only the specified object(s) visible
    """
    return await _process_video_segmentation(
        video=video,
        prompt=prompt,
        background_mode=background_mode,
        output_format=output_format,
        include_overlay=include_overlay,
    )


async def _process_video_segmentation(
    video: UploadFile,
    prompt: str,
    background_mode: BackgroundMode,
    output_format: OutputFormat,
    include_overlay: bool,
) -> VideoSegmentationResponse:
    """
    Internal function to process video segmentation.
    """
    # Validate file type
    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    file_ext = Path(video.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Ensure model is loaded
    if not sam3_service.is_loaded:
        try:
            sam3_service.load_model()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load SAM3 model: {str(e)}"
            )
    
    # Create unique task ID and directories
    task_id = str(uuid.uuid4())
    task_output_dir = OUTPUT_DIR / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded video
    input_video_path = task_output_dir / f"input{file_ext}"
    try:
        with open(input_video_path, "wb") as f:
            content = await video.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded video: {str(e)}"
        )
    
    try:
        # Process video
        result = sam3_service.segment_video(
            video_path=str(input_video_path),
            prompt=prompt,
            background_mode=background_mode.value,
            output_dir=str(task_output_dir),
            include_overlay=include_overlay,
        )
        
        # Build response
        output_video_path = None
        overlay_video_path = None
        
        if result["output_video_path"]:
            output_video_path = f"/outputs/{task_id}/{Path(result['output_video_path']).name}"
        
        if result["overlay_video_path"]:
            overlay_video_path = f"/outputs/{task_id}/{Path(result['overlay_video_path']).name}"
        
        return VideoSegmentationResponse(
            success=result["success"],
            message=result["message"],
            output_video_path=output_video_path,
            overlay_video_path=overlay_video_path,
            total_frames=result["total_frames"],
            objects_detected=result["objects_detected"],
            processing_time_seconds=result["processing_time_seconds"],
        )
        
    except Exception as e:
        # Cleanup on error
        shutil.rmtree(task_output_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )


@app.get("/download/{task_id}/{filename}", tags=["Download"])
async def download_video(task_id: str, filename: str):
    """
    Download a processed video file.
    
    **Parameters:**
    - **task_id**: The task ID from the segmentation response
    - **filename**: The video filename to download
    
    **Returns:**
    - The video file for download
    """
    file_path = OUTPUT_DIR / task_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type
    media_type = "video/mp4"
    if filename.endswith(".webm"):
        media_type = "video/webm"
    elif filename.endswith(".mov"):
        media_type = "video/quicktime"
    
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename,
    )


@app.delete("/cleanup/{task_id}", tags=["Cleanup"])
async def cleanup_task(task_id: str):
    """
    Clean up files for a completed task.
    
    Call this endpoint after downloading the processed video to free up
    server storage.
    
    **Parameters:**
    - **task_id**: The task ID to clean up
    """
    task_dir = OUTPUT_DIR / task_id
    
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    
    try:
        shutil.rmtree(task_dir)
        return {"success": True, "message": f"Task {task_id} cleaned up successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup task: {str(e)}"
        )


@app.get("/tasks", tags=["Tasks"])
async def list_tasks():
    """
    List all task directories and their files.
    
    Useful for debugging and monitoring storage usage.
    """
    tasks = []
    
    if OUTPUT_DIR.exists():
        for task_dir in OUTPUT_DIR.iterdir():
            if task_dir.is_dir():
                files = list(task_dir.iterdir())
                tasks.append({
                    "task_id": task_dir.name,
                    "files": [f.name for f in files],
                    "size_mb": sum(f.stat().st_size for f in files) / (1024 * 1024),
                })
    
    return {
        "total_tasks": len(tasks),
        "tasks": tasks,
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,  # SAM3 requires single worker due to GPU memory
    )

