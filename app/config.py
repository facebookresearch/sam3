import os
from typing import Optional

class Config:
    """应用配置管理"""
    
    # SAM3模型配置
    SAM3_MODEL_PATH: str = os.getenv(
        "SAM3_MODEL_PATH", 
        "/home/duan/workspace/python/sam3/checkpoints/sam3.pt"
    )
    
    # BPE词汇表路径
    BPE_VOCAB_PATH: str = os.getenv(
        "BPE_VOCAB_PATH",
        "/home/duan/workspace/python/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    )
    
    # 推理配置
    DEFAULT_DETECTION_THRESHOLD: float = float(os.getenv("DEFAULT_DETECTION_THRESHOLD", "0.5"))
    DEFAULT_EPSILON: float = float(os.getenv("DEFAULT_EPSILON", "1.0"))
    DEFAULT_RESIZE_SIZE: int = int(os.getenv("DEFAULT_RESIZE_SIZE", "1008"))
    MAX_SIZE: int = int(os.getenv("MAX_SIZE", "1008"))
    
    # 服务配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_model_checkpoint_path(cls) -> Optional[str]:
        """获取模型检查点路径"""
        if os.path.exists(cls.SAM3_MODEL_PATH):
            return cls.SAM3_MODEL_PATH
        
        # 回退到默认路径查找
        import sam3
        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        fallback_paths = [
            os.path.join(sam3_root, "checkpoints", "sam3.pt"),
            os.path.join(sam3_root, "examples", "checkpoint", "sam3.pt")
        ]
        
        for path in fallback_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    @classmethod
    def get_bpe_path(cls) -> str:
        """获取BPE词汇表路径"""
        if os.path.exists(cls.BPE_VOCAB_PATH):
            return cls.BPE_VOCAB_PATH
        
        # 回退到默认路径
        import sam3
        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        default_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
        return default_path