import time
import logging
from typing import List, Dict, Any
import torch
import numpy as np
from PIL import Image

from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata,
    FindQueryLoaded,
    Image as SAMImage,
    Datapoint,
)
from sam3.eval.postprocessors import PostProcessImage
from .convert import convert_sam3_masks_to_json
from .config import Config

logger = logging.getLogger(__name__)

class InferenceService:
    """推理服务类"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    def create_empty_datapoint(self) -> Datapoint:
        """创建空的数据点"""
        return Datapoint(find_queries=[], images=[])
    
    def set_image(self, datapoint: Datapoint, pil_image: Image.Image):
        """添加图像到数据点"""
        w, h = pil_image.size
        datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]
    
    def add_text_prompt(self, datapoint: Datapoint, text_query: str, query_id: int):
        """添加文本查询到数据点"""
        w, h = datapoint.images[0].size
        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=query_id,
                    original_image_id=query_id,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            )
        )
    
    def run_inference(
        self,
        image: Image.Image,
        text_prompts: List[str],
        detection_threshold: float = None,
        epsilon: float = None
    ) -> Dict[str, Any]:
        """执行推理"""
        start_time = time.time()
        
        if not self.model_manager.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # 使用配置默认值
        if detection_threshold is None:
            detection_threshold = Config.DEFAULT_DETECTION_THRESHOLD
        if epsilon is None:
            epsilon = Config.DEFAULT_EPSILON
        
        model = self.model_manager.get_model()
        transform = self.model_manager.get_transform()
        device = self.model_manager.get_device()
        
        width, height = image.size
        
        # 创建postprocessor
        postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=detection_threshold,
            to_cpu=False,
        )
        
        # 构造datapoint
        datapoint = self.create_empty_datapoint()
        self.set_image(datapoint, image)
        
        # 添加文本提示
        query_id = 1
        prompt_to_id = {}
        for text_prompt in text_prompts:
            self.add_text_prompt(datapoint, text_prompt, query_id)
            prompt_to_id[text_prompt] = query_id
            query_id += 1
        
        # 应用transform
        datapoint = transform(datapoint)
        
        # 创建batch
        batch = collate([datapoint], dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, device, non_blocking=True)
        
        # 推理
        with torch.inference_mode():
            output = model(batch)
        
        # 后处理
        processed_results = postprocessor.process_results(
            output, batch.find_metadatas
        )
        
        # 转换结果
        all_results = []
        for prompt, query_id in prompt_to_id.items():
            result = processed_results.get(query_id)
            if result is not None:
                masks = result.get("masks", [])
                boxes = result.get("boxes", [])
                scores = result.get("scores", [])
                
                if len(masks) > 0:
                    confidence_list = scores.detach().cpu().tolist()
                    
                    convert_result = convert_sam3_masks_to_json(
                        masks=masks,
                        input_boxes=boxes,
                        label=prompt,
                        confidence=confidence_list,
                        original_image_path=None,
                        w=width,
                        h=height,
                        epsilon=epsilon
                    )
                    
                    all_results.extend(convert_result.get('results', []))
        
        inference_time = time.time() - start_time
        
        return {
            'results': all_results,
            'count': len(all_results),
            'image_shape': {"height": height, "width": width},
            'inference_time': round(inference_time, 3)
        }