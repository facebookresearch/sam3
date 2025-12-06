# ================== Stage 1: 构建依赖 ==================
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS builder

# 配置 apt 使用国内源（阿里云镜像）
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true

# 安装构建 Python 包所需的系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 使用虚拟环境，便于在最终镜像中整体复制
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 配置虚拟环境的 pip 使用清华源
RUN /opt/venv/bin/pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    /opt/venv/bin/pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 升级 pip / setuptools，避免版本过老带来的构建问题
RUN pip install --upgrade pip "setuptools>=62.3.0,<75.9" wheel

# 复制整个项目结构（pip install -e . 需要完整的项目结构）
COPY . /app/

# 按照 GUIDE.md 的方式安装依赖：
# 1. 先安装项目本身（这会自动安装 pyproject.toml 中的主依赖）
RUN pip install --no-cache-dir -e /app/

# 2. 安装可选依赖（包含 pycocotools, decord, einops 等）
# 安装 notebooks 和 train 的可选依赖（包含代码中实际需要的包）
RUN pip install --no-cache-dir \
    pycocotools \
    decord \
    einops \
    hydra-core \
    omegaconf \
    scikit-image \
    scikit-learn \
    scipy \
    psutil

# 3. 安装 FastAPI 应用依赖（从 app/requirements.txt）
RUN pip install --no-cache-dir -r /app/app/requirements.txt


# ================== Stage 2: 运行镜像 ==================
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# 配置 apt 使用国内源（阿里云镜像）
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true

# 安装运行时系统依赖（包含 OpenCV 所需的库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi-dev \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 基础环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PROJECT_ROOT=/app

# FastAPI 服务配置（可通过环境变量覆盖）
ENV HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO

# 模型路径配置（volume 挂载）
ENV SAM3_MODEL_PATH=/data/checkpoints/sam3.pt \
    BPE_VOCAB_PATH=/app/assets/bpe_simple_vocab_16e6.txt.gz

# Hugging Face 模型路径配置（通过 volume 挂载）
ENV TRANSFORMERS_CACHE=/data/hf_models \
    HF_HOME=/data/hf_models \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# 推理配置（可通过环境变量覆盖）
ENV DEFAULT_DETECTION_THRESHOLD=0.5 \
    DEFAULT_EPSILON=1.0 \
    DEFAULT_RESIZE_SIZE=1008 \
    MAX_SIZE=1008

# 从 builder 拷贝已经安装好的虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 确保运行时镜像的 pip 也配置使用清华源（保持一致性）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 复制整个项目结构（确保所有模块都能被正确导入）
COPY . /app/

# 创建挂载点目录（模型权重和 HF 模型通过 volume 挂载）
RUN mkdir -p /data/checkpoints \
    /data/hf_models

# 暴露端口
EXPOSE 8000

# 启动命令（使用 uvicorn 运行 FastAPI 应用）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]

