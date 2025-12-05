# SAM3 API 部署说明

## 环境变量配置

### 必需的环境变量

- `SAM3_MODEL_PATH`: SAM3模型文件(.pt)的绝对路径
- `BPE_VOCAB_PATH`: BPE词汇表文件的绝对路径

### 可选的环境变量

- `HOST`: 服务监听地址 (默认: 0.0.0.0)
- `PORT`: 服务监听端口 (默认: 8000)
- `DEFAULT_DETECTION_THRESHOLD`: 默认检测阈值 (默认: 0.5)
- `DEFAULT_EPSILON`: 默认epsilon参数 (默认: 1.0)
- `DEFAULT_RESIZE_SIZE`: 默认图像尺寸 (默认: 1008)
- `MAX_SIZE`: 最大图像尺寸 (默认: 1008)
- `LOG_LEVEL`: 日志级别 (默认: INFO)

## Docker部署

### 1. 构建镜像

```bash
docker build -t sam3-api .
```

### 2. 运行容器

```bash
docker run -d \
  -p 8000:8000 \
  -e SAM3_MODEL_PATH=/app/models/sam3.pt \
  -v /path/to/your/sam3.pt:/app/models/sam3.pt:ro \
  sam3-api
```

### 3. 使用docker-compose

```bash
# 修改docker-compose.yml中的模型路径
# 然后运行：
docker-compose up -d
```

## 本地开发

### 1. 设置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑.env文件，设置正确的路径
vim .env
```

### 2. 运行应用

```bash
# 加载环境变量并运行
source .env
python -m app.main
```

## 健康检查

服务启动后，可以通过以下端点检查状态：

- 健康检查: `GET /health`
- 基础信息: `GET /`

## API使用

### 推理接口

```bash
curl -X POST "http://localhost:8000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "text_prompt": ["cat", "dog"],
    "image_base64": "base64_encoded_image",
    "detection_threshold": 0.5,
    "epsilon": 1.0
  }'
```