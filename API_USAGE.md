# SAM3 Video API - Deployment Guide

A GPU-accelerated video segmentation API powered by Meta's SAM3 (Segment Anything Model 3), deployed on Google Cloud Run.

## 🌐 Service URL

```
https://sam3-api-service-g6gkfu4ava-ez.a.run.app
```

Replace with your actual deployed service URL.

---

## 📋 Available Endpoints

### 1. Health Check
### 2. Root Information
### 3. Video Segmentation (General)
### 4. Video Segmentation (Dog-specific)
### 5. Download Processed Video
### 6. List All Tasks
### 7. Cleanup Task

---

## 🔍 Endpoint Details & Examples

### 1. **GET /health** - Health Check

Check API status, GPU availability, and model loading status.

**cURL Example:**
```bash
curl -X GET "https://sam3-api-service-g6gkfu4ava-ez.a.run.app/health"
```

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "model_loaded": true,
  "cuda_version": "12.1",
  "gpu_name": "NVIDIA L4"
}
```

---

### 2. **GET /** - Root Information

Get basic API information and available endpoints.

**cURL Example:**
```bash
curl -X GET "https://sam3-api-service-g6gkfu4ava-ez.a.run.app/"
```

**Response:**
```json
{
  "name": "SAM3 Video Segmentation API",
  "version": "1.0.0",
  "description": "Remove backgrounds from videos using SAM3",
  "docs": "/docs",
  "health": "/health"
}
```

---

### 3. **POST /segment** - Video Segmentation (General)

Segment any object from a video using a text prompt.

**Parameters:**
- `video` (required): Video file (MP4, MOV, AVI, MKV, WebM, M4V)
- `prompt` (optional, default: "dog"): Text description of object to segment
- `background_mode` (optional, default: "black"): Background style
  - `black` - Solid black background
  - `white` - Solid white background
  - `blur` - Blurred background
  - `transparent` - Alpha channel (requires WebM format)
- `output_format` (optional, default: "mp4"): Output video format (mp4, webm, mov)
- `include_overlay` (optional, default: false): Include visualization video
- `upload_to_gcs` (optional, default: true): Upload to GCS bucket
- `gcs_bucket` (optional, default: "nannie_sam3"): GCS bucket name

**cURL Example 1 - Basic (Segment a dog with black background):**
```bash
curl -X POST "https://sam3-api-service-g6gkfu4ava-ez.a.run.app/segment" \
  -F "video=@input.mp4" \
  -F "prompt=dog" \
  -F "background_mode=black"
```

**cURL Example 2 - Segment a cat with white background:**
```bash
curl -X POST "https://sam3-api-service-g6gkfu4ava-ez.a.run.app/segment" \
  -F "video=@my_cat_video.mp4" \
  -F "prompt=cat" \
  -F "background_mode=white" \
  -F "output_format=mp4"
```

**cURL Example 3 - Segment a person with blurred background:**
```bash
curl -X POST "https://sam3-api-service-g6gkfu4ava-ez.a.run.app/segment" \
  -F "video=@person_video.mp4" \
  -F "prompt=person" \
  -F "background_mode=blur" \
  -F "include_overlay=true"
```

**cURL Example 4 - Disable GCS upload:**
```bash
curl -X POST "https://sam3-api-service-g6gkfu4ava-ez.a.run.app/segment" \
  -F "video=@input.mp4" \
  -F "prompt=dog" \
  -F "upload_to_gcs=false"
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully segmented 1 object(s) matching 'dog'. GCS URLs: gs://nannie_sam3/outputs/abc-123/segmented.mp4",
  "output_video_path": "/outputs/abc-123/segmented.mp4",
  "overlay_video_path": null,
  "total_frames": 120,
  "objects_detected": 1,
  "processing_time_seconds": 45.3
}
```

---

### 4. **POST /segment/dog** - Dog Segmentation (Shortcut)

Quick endpoint specifically for dog segmentation.

**Parameters:** Same as `/segment` but `prompt` is fixed to "dog"

**cURL Example:**
```bash
curl -X POST "https://sam3-api-service-g6gkfu4ava-ez.a.run.app/segment/dog" \
  -F "video=@dog_video.mp4" \
  -F "background_mode=black"
```

---

### 5. **GET /download/{task_id}/{filename}** - Download Video

Download a processed video file.

**cURL Example:**
```bash
# Download the segmented video
curl -O "https://sam3-api-service-g6gkfu4ava-ez.a.run.app/download/abc-123/segmented.mp4"

# Or save with custom name
curl -o my_output.mp4 "https://sam3-api-service-g6gkfu4ava-ez.a.run.app/download/abc-123/segmented.mp4"
```

---

### 6. **GET /outputs/{task_id}/{filename}** - Direct File Access

Access output videos directly (served as static files).

**cURL Example:**
```bash
# View in browser or download
curl -O "https://sam3-api-service-g6gkfu4ava-ez.a.run.app/outputs/abc-123/segmented.mp4"
```

---

### 7. **GET /tasks** - List All Tasks

List all processing tasks and their files.

**cURL Example:**
```bash
curl -X GET "https://sam3-api-service-g6gkfu4ava-ez.a.run.app/tasks"
```

**Response:**
```json
{
  "total_tasks": 2,
  "tasks": [
    {
      "task_id": "abc-123",
      "files": ["input.mp4", "segmented.mp4", "overlay.mp4"],
      "size_mb": 15.4
    },
    {
      "task_id": "xyz-789",
      "files": ["input.mp4", "segmented.mp4"],
      "size_mb": 8.2
    }
  ]
}
```

---

### 8. **DELETE /cleanup/{task_id}** - Cleanup Task

Delete files for a completed task to free up storage.

**cURL Example:**
```bash
curl -X DELETE "https://sam3-api-service-g6gkfu4ava-ez.a.run.app/cleanup/abc-123"
```

**Response:**
```json
{
  "success": true,
  "message": "Task abc-123 cleaned up successfully"
}
```

---

## 🐍 Python Examples

### Basic Example
```python
import requests

SERVICE_URL = "https://sam3-api-service-g6gkfu4ava-ez.a.run.app"

# Upload and process video
with open('input.mp4', 'rb') as video_file:
    files = {'video': ('input.mp4', video_file, 'video/mp4')}
    data = {
        'prompt': 'dog',
        'background_mode': 'black',
        'output_format': 'mp4'
    }
    
    response = requests.post(f"{SERVICE_URL}/segment", files=files, data=data)
    result = response.json()
    
    print(f"Success: {result['success']}")
    print(f"Output: {result['output_video_path']}")

# Download the result
if result['success']:
    output_url = f"{SERVICE_URL}{result['output_video_path']}"
    video_data = requests.get(output_url).content
    
    with open('output.mp4', 'wb') as f:
        f.write(video_data)
    
    print("Downloaded: output.mp4")
```

### Complete Workflow Example
```python
import requests
import time

SERVICE_URL = "https://sam3-api-service-g6gkfu4ava-ez.a.run.app"

# 1. Check health
health = requests.get(f"{SERVICE_URL}/health").json()
print(f"GPU Available: {health['gpu_available']}")
print(f"Model Loaded: {health['model_loaded']}")

# 2. Process video
with open('my_video.mp4', 'rb') as video_file:
    response = requests.post(
        f"{SERVICE_URL}/segment",
        files={'video': video_file},
        data={
            'prompt': 'cat',
            'background_mode': 'blur',
            'include_overlay': 'true'
        },
        timeout=600
    )

result = response.json()
print(f"Processing time: {result['processing_time_seconds']:.2f}s")
print(f"Objects detected: {result['objects_detected']}")

# 3. Download results
if result['output_video_path']:
    output_url = f"{SERVICE_URL}{result['output_video_path']}"
    output = requests.get(output_url).content
    with open('segmented.mp4', 'wb') as f:
        f.write(output)
    print("Saved: segmented.mp4")

# 4. Download overlay if included
if result['overlay_video_path']:
    overlay_url = f"{SERVICE_URL}{result['overlay_video_path']}"
    overlay = requests.get(overlay_url).content
    with open('overlay.mp4', 'wb') as f:
        f.write(overlay)
    print("Saved: overlay.mp4")

# 5. Check GCS for backups
print(f"Check GCS: {result['message']}")
```

---

## 🎬 Common Use Cases

### Use Case 1: Remove Background from Dog Video
```bash
curl -X POST "https://sam3-api-service-g6gkfu4ava-ez.a.run.app/segment" \
  -F "video=@dog_playing.mp4" \
  -F "prompt=dog" \
  -F "background_mode=black" \
  -o response.json

# Extract output path from response
TASK_ID=$(cat response.json | jq -r '.output_video_path' | cut -d'/' -f3)

# Download result
curl -O "https://sam3-api-service-g6gkfu4ava-ez.a.run.app/outputs/${TASK_ID}/segmented.mp4"
```

### Use Case 2: Create Transparent Background Video
```bash
curl -X POST "https://sam3-api-service-g6gkfu4ava-ez.a.run.app/segment" \
  -F "video=@person.mp4" \
  -F "prompt=person" \
  -F "background_mode=transparent" \
  -F "output_format=webm"
```

### Use Case 3: Batch Processing Multiple Videos
```bash
#!/bin/bash
for video in videos/*.mp4; do
  echo "Processing $video..."
  curl -X POST "https://sam3-api-service-g6gkfu4ava-ez.a.run.app/segment" \
    -F "video=@$video" \
    -F "prompt=dog" \
    -F "background_mode=black"
  sleep 2
done
```

---

## 📦 GCS Bucket Storage

All processed videos are automatically uploaded to GCS:

**Location:** `gs://nannie_sam3/outputs/<task_id>/`

**List outputs:**
```bash
gsutil ls gs://nannie_sam3/outputs/
```

**Download from GCS:**
```bash
gsutil cp gs://nannie_sam3/outputs/abc-123/segmented.mp4 ./
```

**Download all outputs:**
```bash
gsutil -m cp -r gs://nannie_sam3/outputs/* ./local_outputs/
```

---

## 🔧 Testing with Test Script

Use the provided test script:

```bash
# Health check only
python test_deployment.py https://sam3-api-service-g6gkfu4ava-ez.a.run.app --health-only

# Full test with video
python test_deployment.py https://sam3-api-service-g6gkfu4ava-ez.a.run.app input.mp4

# Custom prompt
python test_deployment.py https://sam3-api-service-g6gkfu4ava-ez.a.run.app input.mp4 --prompt "cat"

# With overlay
python test_deployment.py https://sam3-api-service-g6gkfu4ava-ez.a.run.app input.mp4 --overlay

# Custom background
python test_deployment.py https://sam3-api-service-g6gkfu4ava-ez.a.run.app input.mp4 --background blur
```

---

## 🎯 Interactive API Documentation

Visit the auto-generated Swagger UI documentation:

```
https://sam3-api-service-g6gkfu4ava-ez.a.run.app/docs
```

Test all endpoints directly in your browser!

---

## ⚙️ Configuration

### Background Modes
- **black**: Solid black background (fast, works with all formats)
- **white**: Solid white background (fast, works with all formats)
- **blur**: Blurred version of original background (moderate speed)
- **transparent**: Alpha channel transparency (requires WebM/MOV format)

### Supported Video Formats
**Input:** MP4, MOV, AVI, MKV, WebM, M4V  
**Output:** MP4, WebM, MOV

### Segmentation Prompts
Use natural language to describe what to segment:
- "dog"
- "cat"
- "person"
- "car"
- "bird"
- "person with red shirt"
- Any object description SAM3 can understand

---

## 🚨 Troubleshooting

### 500 Internal Server Error
- Check if model is loaded: `curl <service-url>/health`
- Verify video format is supported
- Check Cloud Run logs: `gcloud run services logs read sam3-api-service --region=europe-west4`

### Timeout Errors
- Video processing can take time (30s - 2min depending on length)
- Increase timeout in your client: `timeout=600` (10 minutes)
- Check GPU availability in health endpoint

### No Objects Detected
- Try a clearer, more specific prompt
- Ensure the object is visible in the video
- Try with `include_overlay=true` to see what was detected

---

## 📊 Performance

- **GPU**: NVIDIA L4 (24GB VRAM)
- **Processing Speed**: ~2-5 FPS depending on resolution
- **Max Video Length**: ~15 minutes (adjust timeout as needed)
- **Concurrent Requests**: 1 (GPU exclusive access)

---

## 🔐 Security Notes

- Service is currently configured with `--allow-unauthenticated`
- To add authentication, update Cloud Run settings
- Consider adding API key authentication for production use

---

## 📝 Additional Resources

- **Cloud Run Console**: https://console.cloud.google.com/run
- **GCS Console**: https://console.cloud.google.com/storage/browser/nannie_sam3
- **Logs**: `gcloud run services logs read sam3-api-service --region=europe-west4 --limit=50`

---

## 🎉 Quick Start

```bash
# 1. Check service is running
curl https://sam3-api-service-g6gkfu4ava-ez.a.run.app/health

# 2. Process a video
curl -X POST "https://sam3-api-service-g6gkfu4ava-ez.a.run.app/segment" \
  -F "video=@input.mp4" \
  -F "prompt=dog" \
  -F "background_mode=black" \
  -o response.json

# 3. Download result
cat response.json | jq -r '.output_video_path'
# Copy the path and download:
curl -O "https://sam3-api-service-g6gkfu4ava-ez.a.run.app<output_video_path>"

# 4. Clean up (optional)
curl -X DELETE "https://sam3-api-service-g6gkfu4ava-ez.a.run.app/cleanup/<task_id>"
```

---

**Made with ❤️ using Meta's SAM3 and Google Cloud Run**
