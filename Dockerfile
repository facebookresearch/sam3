FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget curl libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgl1-mesa-glx \
    ffmpeg \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install torch and torchvision with CUDA support
RUN pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# Copy requirements and install dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the SAM3 package and install it
COPY sam3 ./sam3
COPY pyproject.toml .
COPY MANIFEST.in .
COPY LICENSE .
COPY README.md .
RUN pip install -e .

# Copy SAM3 model weights from Cloud Build
COPY weights /app/weights

# Copy API application code
COPY api ./api
COPY run_api.py .
COPY health_check.py .

# Create outputs directory
RUN mkdir -p /app/outputs

# Environment configuration
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PROJECT_ID=nannieai-website-stealth
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV OMP_NUM_THREADS=4
ENV CUDA_LAUNCH_BLOCKING=0

EXPOSE 8080

# Launch the FastAPI/uvicorn app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
