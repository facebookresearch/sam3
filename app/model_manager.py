import logging
import torch
import sam3
from sam3 import build_sam3_image_model
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)
from app.config import Config

logger = logging.getLogger(__name__)

class ModelManager:
    """SAM3模型管理器"""
    
    def __init__(self):
        self.model = None
        self.transform = None
        self.device = None
        self.is_loaded = False
    
    def load_model(self) -> bool:
        """加载SAM3模型和相关组件"""
        try:
            logger.info("Loading SAM3 model...")
            
            # 设备选择
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # 获取路径配置
            checkpoint_path = Config.get_model_checkpoint_path()
            bpe_path = Config.get_bpe_path()
            
            # 加载模型
            if checkpoint_path:
                logger.info(f"Loading checkpoint from: {checkpoint_path}")
                self.model = build_sam3_image_model(
                    bpe_path=bpe_path, 
                    checkpoint_path=checkpoint_path
                ).to(self.device)
            else:
                logger.info("Loading model with default weights")
                self.model = build_sam3_image_model(bpe_path=bpe_path).to(self.device)
            
            self.model.eval()
            
            # 初始化Transform
            self.transform = ComposeAPI(
                transforms=[
                    RandomResizeAPI(
                        sizes=Config.DEFAULT_RESIZE_SIZE,
                        max_size=Config.MAX_SIZE,
                        square=True,
                        consistent_transform=False,
                    ),
                    ToTensorAPI(),
                    NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            
            self.is_loaded = True
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.is_loaded = False
            return False
    
    def get_model(self):
        """获取模型实例"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        return self.model
    
    def get_transform(self):
        """获取Transform实例"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        return self.transform
    
    def get_device(self):
        """获取设备"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        return self.device
    
    def clear(self):
        """清理模型资源"""
        self.model = None
        self.transform = None
        self.device = None
        self.is_loaded = False