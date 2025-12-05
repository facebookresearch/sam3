import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# 添加SAM3模块路径
sys.path.append('/home/duan/workspace/python/sam3')

# 导入自定义模块
from .config import Config
from .model_manager import ModelManager
from .inference_service import InferenceService
from .utils import decode_base64_image, generate_request_id

# 请求响应模型
class InferenceRequest(BaseModel):
    text_prompt: List[str]
    image_base64: str
    detection_threshold: Optional[float] = None
    epsilon: Optional[float] = None

class InferenceResponse(BaseModel):
    status: str
    results: List[Dict[str, Any]]
    count: int
    image_shape: Dict[str, int]
    inference_time: float
    request_id: str

# 全局模型管理器
model_manager = ModelManager()
inference_service = InferenceService(model_manager)

# 配置日志
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    success = model_manager.load_model()
    if not success:
        logger.error("Failed to load model during startup")
        raise RuntimeError("Model loading failed")
    
    yield
    
    # 关闭时清理资源
    model_manager.clear()

app = FastAPI(
    title="SAM3 API",
    description="FastAPI service for SAM3 model",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """根路径"""
    return {"message": "SAM3 API is running"}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "SAM3 API",
        "version": "1.0.0",
        "model_loaded": model_manager.is_loaded
    }

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """SAM3推理接口"""
    request_id = generate_request_id()
    
    try:
        # 检查模型是否已加载
        if not model_manager.is_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # 解码图像
        image = decode_base64_image(request.image_base64)
        
        # 执行推理
        result = inference_service.run_inference(
            image=image,
            text_prompts=request.text_prompt,
            detection_threshold=request.detection_threshold,
            epsilon=request.epsilon
        )
        
        return InferenceResponse(
            status="success",
            results=result['results'],
            count=result['count'],
            image_shape=result['image_shape'],
            inference_time=result['inference_time'],
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error for request {request_id}: {str(e)}")
        return InferenceResponse(
            status="error",
            results=[],
            count=0,
            image_shape={"height": 0, "width": 0},
            inference_time=0.0,
            request_id=request_id
        )

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=Config.HOST, 
        port=Config.PORT,
        log_level=Config.LOG_LEVEL.lower()
    )