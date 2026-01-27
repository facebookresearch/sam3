FROM python:3.12-slim

ARG TORCH_VERSION=2.7.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG SAM3_EXTRAS=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch early to leverage Docker layer caching.
RUN python -m pip install --upgrade pip \
    && pip install torch==${TORCH_VERSION} torchvision torchaudio --index-url ${TORCH_INDEX_URL}

COPY . /app

RUN if [ -n "${SAM3_EXTRAS}" ]; then \
        pip install -e ".[${SAM3_EXTRAS}]"; \
    else \
        pip install -e .; \
    fi

CMD ["bash"]
