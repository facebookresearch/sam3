import base64
import io
import logging
from PIL import Image
from fastapi import HTTPException

logger = logging.getLogger(__name__)

def decode_base64_image(base64_str: str) -> Image.Image:
    """解码base64图像"""
    try:
        # 处理data URL格式
        if base64_str.startswith('data:image/'):
            base64_str = base64_str.split(',')[1]
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return image
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

def generate_request_id() -> str:
    """生成请求ID"""
    import random
    import string
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))