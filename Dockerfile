FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY app/requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置Python路径
ENV PYTHONPATH=/app

# 环境变量配置
ENV SAM3_MODEL_PATH=/app/models/sam3.pt
ENV BPE_VOCAB_PATH=/app/assets/bpe_simple_vocab_16e6.txt.gz
ENV HOST=0.0.0.0
ENV PORT=8000
ENV LOG_LEVEL=INFO

# 创建模型目录
RUN mkdir -p /app/models

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "-m", "app.main"]