"""
SAM 3 FastAPI Service - Simple & Clean Version

Thread-safe service with request queue for GPU concurrency control.
Supports both local video paths and video file uploads.
"""

import asyncio
import logging
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Queue, Full
from threading import Thread, Event
from typing import Optional, List

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Temporary directory for uploaded videos
UPLOAD_DIR = tempfile.mkdtemp(prefix="sam3_uploads_")
logger.info(f"📁 Upload directory: {UPLOAD_DIR}")


# ==================== Models ====================

class StartSessionRequest(BaseModel):
    resource_path: str
    session_id: Optional[str] = None


class StartSessionResponse(BaseModel):
    session_id: str
    resource_path: str
    uploaded: bool  # True if video was uploaded, False if using local path


class AddPromptRequest(BaseModel):
    session_id: str
    frame_index: int
    text: Optional[str] = None
    points: Optional[List[List[float]]] = None
    bounding_boxes: Optional[List[List[float]]] = None


class PropagateRequest(BaseModel):
    session_id: str
    direction: str = "both"


# ==================== Worker Thread ====================

class SAM3Worker:
    """Single worker thread for processing all SAM3 requests"""

    def __init__(self, queue_size: int = 100):
        self.queue = Queue(maxsize=queue_size)
        self.ready = Event()
        self.shutdown_flag = Event()
        self.video_predictor = None
        self.image_model = None
        self.image_processor = None

        # Track uploaded files per session for cleanup
        self.session_temp_files = {}  # session_id -> temp_file_path

    def start(self):
        """Start worker thread and load models"""
        Thread(target=self._worker_loop, daemon=True).start()

        if not self.ready.wait(timeout=120):
            raise RuntimeError("Model loading timeout")

        logger.info("✅ Worker ready")

    def _worker_loop(self):
        """Main worker loop - processes requests sequentially"""
        try:
            self._load_models()
            self.ready.set()

            while not self.shutdown_flag.is_set():
                try:
                    item = self.queue.get(timeout=1.0)
                    self._process(item)
                except:
                    pass

        except Exception as e:
            logger.error(f"Worker error: {e}")
            self.ready.set()

    def _load_models(self):
        """Load SAM3 models"""
        logger.info("🚀 Loading models...")

        from sam3.model_builder import build_sam3_video_predictor, build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.video_predictor = build_sam3_video_predictor(gpus_to_use=[0])
        self.image_model = build_sam3_image_model(device="cuda", eval_mode=True)
        self.image_processor = Sam3Processor(self.image_model)

    def _process(self, item):
        """Process a single request"""
        request_id, request_type, data, future = item

        try:
            if request_type == "video":
                result = self.video_predictor.handle_request(data)

                # Track temp file for cleanup if this is a start_session with uploaded file
                if data.get("type") == "start_session" and data.get("_temp_file"):
                    session_id = result["session_id"]
                    self.session_temp_files[session_id] = data["_temp_file"]
                    logger.info(f"📌 Tracked temp file for session {session_id}")

                # Cleanup temp file if closing session
                elif data.get("type") == "close_session":
                    session_id = data["session_id"]
                    self._cleanup_session(session_id)

            elif request_type == "video_stream":
                results = []
                for frame in self.video_predictor.handle_stream_request(data):
                    results.append(frame)
                result = {"frames": results}

            elif request_type == "image":
                result = self._segment_image(data)

            else:
                raise ValueError(f"Unknown type: {request_type}")

            asyncio.get_event_loop().call_soon_threadsafe(future.set_result, result)

        except Exception as e:
            logger.error(f"Process error: {e}")
            asyncio.get_event_loop().call_soon_threadsafe(future.set_exception, e)

    def _segment_image(self, data):
        """Process image segmentation"""
        image, prompt = data["image"], data["prompt"]

        state = self.image_processor.set_image(image)
        output = self.image_processor.set_text_prompt(state=state, prompt=prompt)

        return {
            "image_width": image.size[0],
            "image_height": image.size[1],
            "boxes": output["boxes"].cpu().tolist(),
            "scores": output["scores"].cpu().tolist(),
            "num_objects": len(output["scores"]),
            "prompt": prompt
        }

    def _cleanup_session(self, session_id):
        """Clean up temporary files for a session"""
        temp_file = self.session_temp_files.pop(session_id, None)
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.info(f"🗑️  Deleted temp file for session {session_id}: {temp_file}")
            except Exception as e:
                logger.error(f"Failed to delete temp file {temp_file}: {e}")

    async def submit(self, request_type, data, timeout=30.0):
        """Submit request and wait for result"""
        request_id = str(uuid.uuid4())
        future = asyncio.Future()

        try:
            self.queue.put_nowait((request_id, request_type, data, future))
        except Full:
            raise HTTPException(503, "Queue full, try again later")

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            raise HTTPException(504, f"Timeout after {timeout}s")

    def stats(self):
        """Get worker stats"""
        return {
            "queue_size": self.queue.qsize(),
            "ready": self.ready.is_set(),
            "active_sessions": len(self.video_predictor._ALL_INFERENCE_STATES) if self.video_predictor else 0,
            "gpu_memory_mb": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        }


# ==================== FastAPI App ====================

worker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global worker

    # Startup
    logger.info("Starting SAM3 service...")
    worker = SAM3Worker(queue_size=100)
    worker.start()
    logger.info("SAM3 service started")

    yield

    # Shutdown
    logger.info("Shutting down SAM3 service...")
    if worker:
        worker.shutdown_flag.set()

    # Clean up upload directory
    if os.path.exists(UPLOAD_DIR):
        try:
            shutil.rmtree(UPLOAD_DIR)
            logger.info(f"🗑️  Cleaned up upload directory: {UPLOAD_DIR}")
        except Exception as e:
            logger.error(f"Failed to clean upload directory: {e}")


app = FastAPI(title="SAM 3 Service", version="1.0.0", lifespan=lifespan)


# ==================== Endpoints ====================

@app.get("/")
async def root():
    """API info"""
    return {
        "service": "SAM 3 API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check"""
    if not worker or not worker.ready.is_set():
        raise HTTPException(503, "Service not ready")

    return worker.stats()


@app.post("/api/v1/image/segment")
async def segment_image(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    """Segment objects in an image"""
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = await worker.submit("image", {"image": image, "prompt": prompt})
    return result


@app.post("/api/v1/video/session/start")
async def start_session(req: StartSessionRequest):
    """
    Start video session from a local server path.

    Use this endpoint when the video is already on the server.
    For uploading videos, use POST /api/v1/video/session/upload instead.
    """
    result = await worker.submit("video", {
        "type": "start_session",
        "resource_path": req.resource_path,
        "session_id": req.session_id
    }, timeout=60.0)

    return StartSessionResponse(
        session_id=result["session_id"],
        resource_path=req.resource_path,
        uploaded=False
    )


@app.post("/api/v1/video/session/upload", response_model=StartSessionResponse)
async def upload_and_start_session(
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV, etc.)"),
    session_id: Optional[str] = Form(None, description="Optional custom session ID")
):
    """
    Upload a video file and start a tracking session.

    The video is saved temporarily on the server and will be automatically
    deleted when the session is closed.

    Supported formats: MP4, AVI, MOV, MKV, etc.
    """
    # Validate file extension
    filename = file.filename or "video.mp4"
    file_ext = Path(filename).suffix.lower()
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

    if file_ext not in allowed_extensions:
        raise HTTPException(
            400,
            f"Unsupported video format: {file_ext}. "
            f"Allowed: {', '.join(allowed_extensions)}"
        )

    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join(UPLOAD_DIR, unique_filename)

    # Save uploaded file
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"📤 Uploaded video: {filename} ({file_size_mb:.1f} MB) -> {temp_path}")

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(500, f"Failed to save video: {str(e)}")

    # Start session with the uploaded video
    try:
        result = await worker.submit("video", {
            "type": "start_session",
            "resource_path": temp_path,
            "session_id": session_id,
            "_temp_file": temp_path  # Mark as temporary for cleanup
        }, timeout=60.0)

        return StartSessionResponse(
            session_id=result["session_id"],
            resource_path=temp_path,
            uploaded=True
        )

    except Exception as e:
        # Clean up file if session start failed
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


@app.post("/api/v1/video/prompt/add")
async def add_prompt(req: AddPromptRequest):
    """Add prompt to video frame"""
    result = await worker.submit("video", {
        "type": "add_prompt",
        "session_id": req.session_id,
        "frame_index": req.frame_index,
        "text": req.text,
        "points": req.points,
        "bounding_boxes": req.bounding_boxes
    })

    return result


@app.post("/api/v1/video/propagate")
async def propagate(req: PropagateRequest):
    """Propagate tracking across video"""
    result = await worker.submit("video_stream", {
        "type": "propagate_in_video",
        "session_id": req.session_id,
        "propagation_direction": req.direction
    }, timeout=300.0)

    return result


@app.delete("/api/v1/video/session/{session_id}")
async def close_session(session_id: str):
    """Close video session"""
    result = await worker.submit("video", {
        "type": "close_session",
        "session_id": session_id
    })

    return result


# ==================== Run ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
