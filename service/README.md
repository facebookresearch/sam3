# SAM 3 FastAPI Service

Simple REST API for SAM 3 image and video segmentation with thread-safe concurrency control.

## Quick Start

```bash
# 1. Install dependencies
pip install -r service/requirements.txt

# 2. Start service
python service/fastapi_server.py

# 3. Open browser
http://localhost:8000/docs
```

## Architecture

```
FastAPI (handles connections)
    ↓
Request Queue (thread-safe)
    ↓
Worker Thread (processes sequentially)
    ↓
SAM3 Models (GPU-safe)
```

**Why?** SAM3 uses GPU which isn't thread-safe. The queue ensures all requests are processed one at a time while FastAPI still accepts connections concurrently.

## Usage

### Python Client

```python
from service.client_example import SAM3Client

client = SAM3Client()

# Segment image
result = client.segment_image("image.jpg", "a person")
print(f"Found {result['num_objects']} objects")

# Video tracking - Option 1: Server local path
session = client.start_video_session("/path/to/video.mp4")
sid = session["session_id"]

# Video tracking - Option 2: Upload video file
session = client.upload_video_and_start_session("my_video.mp4")
sid = session["session_id"]

# Continue with tracking
client.add_prompt(sid, frame_index=0, text="a car")
results = client.propagate_video(sid)
client.close_session(sid)  # Cleanup (deletes uploaded file)
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Segment image
curl -X POST http://localhost:8000/api/v1/image/segment \
  -F "file=@image.jpg" \
  -F "prompt=a car"
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service status |
| POST | `/api/v1/image/segment` | Segment image |
| POST | `/api/v1/video/session/start` | Start video session (local path) |
| POST | `/api/v1/video/session/upload` | Upload video and start session |
| POST | `/api/v1/video/prompt/add` | Add prompt to frame |
| POST | `/api/v1/video/propagate` | Track across video |
| DELETE | `/api/v1/video/session/{id}` | Close session |

Full API docs at: http://localhost:8000/docs

### Video Input Methods

**Method 1: Server Local Path** (for videos already on server)
```bash
curl -X POST http://localhost:8000/api/v1/video/session/start \
  -H "Content-Type: application/json" \
  -d '{"resource_path": "/path/to/video.mp4"}'
```

**Method 2: Upload Video File** (for user-uploaded videos)
```bash
curl -X POST http://localhost:8000/api/v1/video/session/upload \
  -F "file=@my_video.mp4"
```

Uploaded videos are stored temporarily and automatically deleted when the session closes.

## Concurrency

### How it works

```
Request 1 arrives → Accepted → Queued → Processing
Request 2 arrives → Accepted → Queued → Waiting
Request 3 arrives → Accepted → Queued → Waiting
                                    ↓
                            Worker processes one at a time
```

- ✅ All requests accepted immediately (non-blocking)
- ✅ GPU operations run sequentially (thread-safe)
- ✅ Queue full → Returns 503
- ✅ Timeout → Returns 504

### Configuration

Edit `fastapi_server.py`:

```python
# Line 170: Queue size
worker = SAM3Worker(queue_size=100)  # Increase if needed

# Line 210, 221, 248: Timeouts
timeout=30.0   # Image timeout
timeout=60.0   # Session start timeout
timeout=300.0  # Video propagate timeout
```

## Monitoring

```bash
# Check health
curl http://localhost:8000/health

# Response
{
  "queue_size": 2,
  "ready": true,
  "active_sessions": 1,
  "gpu_memory_mb": 4536.5
}
```

## File Management

### Uploaded Videos

- Uploaded videos are stored in a temporary directory
- Files are automatically deleted when:
  - The session is closed normally
  - The service shuts down
- Each uploaded file gets a unique name (UUID)

**Important**: Always close sessions to free up disk space!

```python
# Good practice - always use try/finally
try:
    session = client.upload_video_and_start_session("video.mp4")
    # ... do tracking ...
finally:
    client.close_session(session["session_id"])
```

### Supported Video Formats

- MP4 (recommended)
- AVI
- MOV
- MKV
- WEBM
- FLV

## Troubleshooting

### Service not ready
Wait 30-60s for models to load on first startup.

### Queue full (503 error)
Too many concurrent requests. Wait or increase `queue_size`.

### Out of memory
Close unused video sessions:
```python
client.close_session(session_id)
```

### Unsupported video format (400 error)
Upload failed. Make sure your video is in a supported format (MP4, AVI, MOV, etc.).

### File upload failed (500 error)
Check disk space and file permissions. The service needs write access to the temp directory.

### WSL file paths
```python
# ❌ Don't use Windows paths for local files
"/mnt/c/Users/user/video.mp4"  # Slow!

# ✅ Use WSL filesystem for local files
"/home/user/videos/video.mp4"  # Fast!

# ✅ Or just upload the file
client.upload_video_and_start_session("C:/Users/user/video.mp4")  # Works!
```

## Code Overview

The entire service is **~270 lines** of clean Python:

- **Lines 1-43**: Request/response models
- **Lines 45-157**: Worker thread with queue
- **Lines 159-268**: FastAPI endpoints

Simple and readable!

## Testing

```bash
# Run test script
./service/test_service.sh

# Or use Python client
python -c "from service.client_example import SAM3Client; \
           c = SAM3Client(); \
           print(c.health())"
```

## WSL Tips

```bash
# Run in WSL filesystem (not /mnt/c)
cd ~/workspace/sam3
python service/fastapi_server.py

# Access from Windows browser
http://localhost:8000
```

That's it! Simple and clean. 🎉
