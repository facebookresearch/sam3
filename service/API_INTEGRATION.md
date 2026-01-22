# SAM3 API Integration Guide for Frontend

Complete guide for integrating SAM3 segmentation API into your frontend application.

## 📋 Quick Start

**Base URL**: `http://your-server:8000`

**Documentation**: `http://your-server:8000/docs` (Interactive Swagger UI)

## 🎯 Use Cases

1. **Image Segmentation**: Detect and segment objects in images
2. **Video Tracking**: Track objects across video frames
3. **Real-time Processing**: Queue-based processing for concurrent requests

## 🔌 API Endpoints

### Health Check

```http
GET /health
```

**Response**:
```json
{
  "queue_size": 2,
  "ready": true,
  "active_sessions": 1,
  "gpu_memory_mb": 4536.5
}
```

---

## 📷 Image Segmentation

### Segment Objects in Image

```http
POST /api/v1/image/segment
Content-Type: multipart/form-data
```

**Request**:
- `file` (file): Image file (JPEG, PNG, etc.)
- `prompt` (string): Text description of objects to find (e.g., "a person", "cars")

**Response**:
```json
{
  "image_width": 1920,
  "image_height": 1080,
  "boxes": [[100, 200, 300, 400], ...],
  "scores": [0.95, 0.87, ...],
  "num_objects": 3,
  "prompt": "a person"
}
```

**JavaScript Example**:
```javascript
async function segmentImage(imageFile, prompt) {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('prompt', prompt);

  const response = await fetch('http://localhost:8000/api/v1/image/segment', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }

  return await response.json();
}

// Usage
const file = document.getElementById('imageInput').files[0];
const result = await segmentImage(file, 'a car');
console.log(`Found ${result.num_objects} objects`);
```

**TypeScript Example**:
```typescript
interface SegmentationResult {
  image_width: number;
  image_height: number;
  boxes: number[][];
  scores: number[];
  num_objects: number;
  prompt: string;
}

async function segmentImage(
  imageFile: File,
  prompt: string
): Promise<SegmentationResult> {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('prompt', prompt);

  const response = await fetch('http://localhost:8000/api/v1/image/segment', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(`Segmentation failed: ${response.statusText}`);
  }

  return await response.json();
}
```

---

## 🎬 Video Tracking

### Method 1: Upload Video File (Recommended for Frontend)

#### Step 1: Upload Video and Start Session

```http
POST /api/v1/video/session/upload
Content-Type: multipart/form-data
```

**Request**:
- `file` (file): Video file (MP4, AVI, MOV, MKV, WEBM, FLV)
- `session_id` (optional string): Custom session ID

**Response**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "resource_path": "/tmp/sam3_uploads_abc/550e8400.mp4",
  "uploaded": true
}
```

**JavaScript Example**:
```javascript
async function uploadAndStartSession(videoFile) {
  const formData = new FormData();
  formData.append('file', videoFile);

  const response = await fetch('http://localhost:8000/api/v1/video/session/upload', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.statusText}`);
  }

  return await response.json();
}

// Usage
const videoFile = document.getElementById('videoInput').files[0];
const session = await uploadAndStartSession(videoFile);
console.log(`Session ID: ${session.session_id}`);
```

#### Step 2: Add Prompt to Frame

```http
POST /api/v1/video/prompt/add
Content-Type: application/json
```

**Request Body**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "frame_index": 0,
  "text": "a person walking",
  "points": [[100, 200], [300, 400]],  // Optional
  "bounding_boxes": [[10, 20, 100, 200]]  // Optional
}
```

**Response**:
```json
{
  "frame_index": 0,
  "outputs": [
    {
      "box": [100, 200, 300, 400],
      "score": 0.95,
      "mask": "..."
    }
  ]
}
```

**JavaScript Example**:
```javascript
async function addPrompt(sessionId, frameIndex, text) {
  const response = await fetch('http://localhost:8000/api/v1/video/prompt/add', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      session_id: sessionId,
      frame_index: frameIndex,
      text: text
    })
  });

  if (!response.ok) {
    throw new Error(`Add prompt failed: ${response.statusText}`);
  }

  return await response.json();
}

// Usage
await addPrompt(session.session_id, 0, 'a car');
```

#### Step 3: Propagate Tracking

```http
POST /api/v1/video/propagate
Content-Type: application/json
```

**Request Body**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "direction": "both"  // "both", "forward", or "backward"
}
```

**Response**:
```json
{
  "frames": [
    {
      "frame_index": 0,
      "outputs": [...]
    },
    {
      "frame_index": 1,
      "outputs": [...]
    }
  ]
}
```

**JavaScript Example**:
```javascript
async function propagateVideo(sessionId, direction = 'both') {
  const response = await fetch('http://localhost:8000/api/v1/video/propagate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      session_id: sessionId,
      direction: direction
    })
  });

  if (!response.ok) {
    throw new Error(`Propagation failed: ${response.statusText}`);
  }

  return await response.json();
}

// Usage
const results = await propagateVideo(session.session_id);
console.log(`Tracked ${results.frames.length} frames`);
```

#### Step 4: Close Session (Important!)

```http
DELETE /api/v1/video/session/{session_id}
```

**Response**:
```json
{
  "is_success": true
}
```

**JavaScript Example**:
```javascript
async function closeSession(sessionId) {
  const response = await fetch(
    `http://localhost:8000/api/v1/video/session/${sessionId}`,
    { method: 'DELETE' }
  );

  if (!response.ok) {
    throw new Error(`Close session failed: ${response.statusText}`);
  }

  return await response.json();
}

// Always close in finally block
try {
  const session = await uploadAndStartSession(videoFile);
  await addPrompt(session.session_id, 0, 'a person');
  const results = await propagateVideo(session.session_id);
  // Process results...
} finally {
  await closeSession(session.session_id);
}
```

### Method 2: Use Server Path (For Pre-uploaded Videos)

```http
POST /api/v1/video/session/start
Content-Type: application/json
```

**Request Body**:
```json
{
  "resource_path": "/path/to/video.mp4",
  "session_id": "optional-custom-id"
}
```

**Response**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "resource_path": "/path/to/video.mp4",
  "uploaded": false
}
```

---

## 🎨 Complete Frontend Example

### React Component

```typescript
import React, { useState } from 'react';

interface VideoTrackingProps {
  apiBaseUrl: string;
}

interface SessionInfo {
  session_id: string;
  resource_path: string;
  uploaded: boolean;
}

export const VideoTracking: React.FC<VideoTrackingProps> = ({ apiBaseUrl }) => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleVideoUpload = async (file: File) => {
    setLoading(true);
    setError(null);

    try {
      // 1. Upload video
      const formData = new FormData();
      formData.append('file', file);

      const uploadResponse = await fetch(`${apiBaseUrl}/api/v1/video/session/upload`, {
        method: 'POST',
        body: formData
      });

      if (!uploadResponse.ok) {
        throw new Error('Upload failed');
      }

      const session: SessionInfo = await uploadResponse.json();
      setSessionId(session.session_id);

      // 2. Add prompt
      await fetch(`${apiBaseUrl}/api/v1/video/prompt/add`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: session.session_id,
          frame_index: 0,
          text: 'a person'
        })
      });

      // 3. Propagate tracking
      const propagateResponse = await fetch(`${apiBaseUrl}/api/v1/video/propagate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: session.session_id,
          direction: 'both'
        })
      });

      const trackingResults = await propagateResponse.json();
      setResults(trackingResults);

      // 4. Close session
      await fetch(`${apiBaseUrl}/api/v1/video/session/${session.session_id}`, {
        method: 'DELETE'
      });

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input
        type="file"
        accept="video/*"
        onChange={(e) => e.target.files?.[0] && handleVideoUpload(e.target.files[0])}
        disabled={loading}
      />
      {loading && <p>Processing...</p>}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}
      {results && <p>Tracked {results.frames.length} frames</p>}
    </div>
  );
};
```

### Vue.js Component

```vue
<template>
  <div>
    <input
      type="file"
      accept="video/*"
      @change="handleVideoUpload"
      :disabled="loading"
    />
    <p v-if="loading">Processing...</p>
    <p v-if="error" style="color: red">Error: {{ error }}</p>
    <p v-if="results">Tracked {{ results.frames.length }} frames</p>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

const apiBaseUrl = 'http://localhost:8000';
const sessionId = ref<string | null>(null);
const results = ref<any>(null);
const loading = ref(false);
const error = ref<string | null>(null);

async function handleVideoUpload(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;

  loading.value = true;
  error.value = null;

  try {
    // 1. Upload
    const formData = new FormData();
    formData.append('file', file);

    const uploadRes = await fetch(`${apiBaseUrl}/api/v1/video/session/upload`, {
      method: 'POST',
      body: formData
    });

    const session = await uploadRes.json();
    sessionId.value = session.session_id;

    // 2. Add prompt
    await fetch(`${apiBaseUrl}/api/v1/video/prompt/add`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: session.session_id,
        frame_index: 0,
        text: 'a person'
      })
    });

    // 3. Propagate
    const propagateRes = await fetch(`${apiBaseUrl}/api/v1/video/propagate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: session.session_id,
        direction: 'both'
      })
    });

    results.value = await propagateRes.json();

    // 4. Close
    await fetch(`${apiBaseUrl}/api/v1/video/session/${session.session_id}`, {
      method: 'DELETE'
    });

  } catch (err: any) {
    error.value = err.message;
  } finally {
    loading.value = false;
  }
}
</script>
```

---

## ⚠️ Error Handling

### HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Process response |
| 400 | Bad request | Check input format |
| 503 | Queue full | Retry after delay |
| 504 | Timeout | Increase timeout or reduce video size |
| 500 | Server error | Check server logs |

### Error Response Format

```json
{
  "detail": "Error message here"
}
```

### JavaScript Error Handling Example

```javascript
async function safeApiCall(url, options) {
  try {
    const response = await fetch(url, options);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return await response.json();

  } catch (error) {
    if (error.name === 'TypeError') {
      throw new Error('Network error: Cannot connect to server');
    }
    throw error;
  }
}

// Usage with retry
async function segmentImageWithRetry(file, prompt, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await segmentImage(file, prompt);
    } catch (error) {
      if (error.message.includes('503') && i < maxRetries - 1) {
        // Queue full, wait and retry
        await new Promise(resolve => setTimeout(resolve, 2000));
        continue;
      }
      throw error;
    }
  }
}
```

---

## 🚀 Best Practices

### 1. Always Close Sessions

```javascript
let sessionId = null;
try {
  const session = await uploadAndStartSession(videoFile);
  sessionId = session.session_id;
  // ... process video
} finally {
  if (sessionId) {
    await closeSession(sessionId);
  }
}
```

### 2. Show Upload Progress

```javascript
async function uploadWithProgress(videoFile, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        const percentComplete = (e.loaded / e.total) * 100;
        onProgress(percentComplete);
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status === 200) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        reject(new Error(`Upload failed: ${xhr.status}`));
      }
    });

    xhr.addEventListener('error', () => reject(new Error('Network error')));

    xhr.open('POST', 'http://localhost:8000/api/v1/video/session/upload');

    const formData = new FormData();
    formData.append('file', videoFile);
    xhr.send(formData);
  });
}

// Usage
await uploadWithProgress(videoFile, (percent) => {
  console.log(`Upload: ${percent.toFixed(1)}%`);
});
```

### 3. Handle Large Videos

```javascript
// Check file size before upload
const MAX_VIDEO_SIZE_MB = 100;

function validateVideoFile(file) {
  const sizeMB = file.size / (1024 * 1024);

  if (sizeMB > MAX_VIDEO_SIZE_MB) {
    throw new Error(`Video too large: ${sizeMB.toFixed(1)}MB (max: ${MAX_VIDEO_SIZE_MB}MB)`);
  }

  const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv'];
  if (!allowedTypes.includes(file.type)) {
    throw new Error(`Unsupported video format: ${file.type}`);
  }

  return true;
}
```

### 4. Check Service Health Before Upload

```javascript
async function checkServiceHealth() {
  const response = await fetch('http://localhost:8000/health');
  const health = await response.json();

  if (!health.ready) {
    throw new Error('Service not ready. Please wait...');
  }

  if (health.queue_size > 50) {
    throw new Error('Service busy. Please try again later.');
  }

  return health;
}

// Before processing
await checkServiceHealth();
const session = await uploadAndStartSession(videoFile);
```

### 5. Implement Timeout

```javascript
function fetchWithTimeout(url, options, timeout = 30000) {
  return Promise.race([
    fetch(url, options),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Request timeout')), timeout)
    )
  ]);
}

// Usage
const response = await fetchWithTimeout(
  'http://localhost:8000/api/v1/video/propagate',
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId })
  },
  60000  // 60 second timeout for video processing
);
```

---

## 📊 Data Format Details

### Bounding Box Format

Boxes are in `[x1, y1, x2, y2]` format (top-left and bottom-right corners):

```javascript
// Box: [100, 200, 300, 400]
// means:
// - Top-left corner: (100, 200)
// - Bottom-right corner: (300, 400)
// - Width: 300 - 100 = 200
// - Height: 400 - 200 = 200

function drawBox(ctx, box) {
  const [x1, y1, x2, y2] = box;
  const width = x2 - x1;
  const height = y2 - y1;

  ctx.strokeStyle = 'red';
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, width, height);
}
```

### Point Format

Points are in `[x, y]` format:

```javascript
const points = [[100, 200], [300, 400]];  // Two points
const point_labels = [1, 0];  // 1 = positive, 0 = negative
```

---

## 🔗 CORS Configuration

If your frontend runs on a different domain, you may need CORS enabled:

```javascript
// Example: Frontend on http://localhost:3000
// Backend on http://localhost:8000

// The backend needs to allow CORS
// (This should be configured on the server side)

// Your fetch requests should work without additional headers
const response = await fetch('http://localhost:8000/api/v1/image/segment', {
  method: 'POST',
  body: formData
  // No need for 'mode: cors' if server allows it
});
```

If you encounter CORS issues, contact the backend team to add your origin to the allowed list.

---

## 📞 Support

- **API Documentation**: http://your-server:8000/docs
- **Health Check**: http://your-server:8000/health
- **GitHub Issues**: [Report bugs here]

---

## 🎯 Quick Reference

```javascript
// Image Segmentation (Simple)
const formData = new FormData();
formData.append('file', imageFile);
formData.append('prompt', 'a car');
const result = await fetch('/api/v1/image/segment', {
  method: 'POST',
  body: formData
}).then(r => r.json());

// Video Tracking (Complete Flow)
const formData = new FormData();
formData.append('file', videoFile);
const session = await fetch('/api/v1/video/session/upload', {
  method: 'POST',
  body: formData
}).then(r => r.json());

await fetch('/api/v1/video/prompt/add', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: session.session_id,
    frame_index: 0,
    text: 'a person'
  })
});

const results = await fetch('/api/v1/video/propagate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ session_id: session.session_id })
}).then(r => r.json());

await fetch(`/api/v1/video/session/${session.session_id}`, {
  method: 'DELETE'
});
```

Happy coding! 🚀
