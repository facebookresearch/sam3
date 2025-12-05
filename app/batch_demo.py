"""
SAM3 批量图像推理示例
完全参考官方示例 sam3_image_batched_inference.ipynb
支持多张图片，每张图片多个文本提示词的批量推理
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata,
    FindQueryLoaded,
    Image as SAMImage,
    Datapoint,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)
from sam3.eval.postprocessors import PostProcessImage
from sam3.visualization_utils import save_results
from app.convert import convert_sam3_masks_to_json


# ============ ID 生成器 ============

@dataclass
class IdGenerator:
    """简单的递增 ID 生成器，保证在当前进程内唯一。"""
    current: int = 1

    def next(self) -> int:
        value = self.current
        self.current += 1
        return value


# ============ 辅助函数（完全按照官方示例） ============

def create_empty_datapoint():
    """A datapoint is a single image on which we can apply several queries at once."""
    return Datapoint(find_queries=[], images=[])


def set_image(datapoint, pil_image):
    """Add the image to be processed to the datapoint"""
    w, h = pil_image.size
    # 注意：官方示例使用 [h, w] 而不是 [w, h]
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def add_text_prompt(datapoint, text_query, id_gen: IdGenerator):
    """Add a text query to the datapoint"""
    # in this function, we require that the image is already set.
    # that's because we'll get its size to figure out what dimension to resize masks and boxes
    # In practice you're free to set any size you want, just edit the rest of the function
    assert len(datapoint.images) == 1, "please set the image first"

    w, h = datapoint.images[0].size
    qid = id_gen.next()
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],  # unused for inference
            is_exhaustive=True,  # unused for inference
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=qid,
                original_image_id=qid,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        )
    )
    return qid


# ============ 批量推理函数 ============

def sam3_batch_text_inference(
    image_paths: List[str],
    text_prompts: List[str],
    device: Optional[str] = None,
    detection_threshold: float = 0.5,
    checkpoint_path: Optional[str] = None,
    model_size: int = 1008,
) -> List[Dict[str, Any]]:
    """
    在 SAM3 上做 batch text-prompt segmentation：
        - 支持多张图片
        - 每张图片跑同一组 text_prompts

    Args:
        image_paths: 每张图片的路径列表
        text_prompts: 要在每张图上跑的文本 prompt 列表，如 ["person", "car", "truck"]
        device: "cuda" / "cpu"，默认自动选择
        detection_threshold: 后处理的 detection 阈值
        checkpoint_path: 模型检查点路径，如果为 None 则使用默认路径
        model_size: 模型输入尺寸，默认 1008

    Returns:
        results_per_image: 长度 = len(image_paths)
            results_per_image[i] = {
                'query_ids': [id1, id2, ...],  # 每个 prompt 对应的 query_id
                'prompt_to_id': {'prompt1': id1, 'prompt2': id2, ...},  # prompt 到 id 的映射
                'results': processed_results  # 完整的处理结果字典 {query_id: result}
            }
    """
    # -------- 设备选择 --------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------- SAM3 模型与组件（完全按照官方示例） --------
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")

    # 如果没有指定检查点路径，尝试使用默认路径
    if checkpoint_path is None:
        default_checkpoint = os.path.join(sam3_root, "checkpoints", "sam3.pt")
        if os.path.exists(default_checkpoint):
            checkpoint_path = default_checkpoint
        else:
            # 尝试 examples/checkpoint 目录
            alt_checkpoint = os.path.join(sam3_root, "examples", "checkpoint", "sam3.pt")
            if os.path.exists(alt_checkpoint):
                checkpoint_path = alt_checkpoint
            else:
                print("Warning: No checkpoint found, model will use default weights")

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        model = build_sam3_image_model(
            bpe_path=bpe_path, checkpoint_path=checkpoint_path
        ).to(device)
    else:
        print("Loading model with default weights")
        model = build_sam3_image_model(bpe_path=bpe_path).to(device)
    
    model.eval()

    # 完全按照官方示例的 transform 配置
    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=model_size,
                max_size=model_size,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # 完全按照官方示例的 postprocessor 配置
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,  # if this number is positive, the processor will return topk
        iou_type="segm",  # we want masks
        use_original_sizes_box=True,  # our boxes should be resized to the image size
        use_original_sizes_mask=True,  # our masks should be resized to the image size
        convert_mask_to_rle=False,  # the postprocessor supports efficient conversion to RLE format
        detection_threshold=detection_threshold,  # Only return confident detections
        to_cpu=False,
    )

    # -------- 构造所有 Datapoint（完全按照官方示例的方式） --------
    id_gen = IdGenerator()  # 创建 ID 生成器实例
    datapoints: List[Datapoint] = []
    query_ids_per_image: List[Dict[str, Any]] = []

    print(f"Processing {len(image_paths)} image(s) with {len(text_prompts)} prompt(s) each...")
    for path in image_paths:
        if not os.path.exists(path):
            print(f"Warning: Image not found: {path}, skipping...")
            continue
        
        # 按照官方示例的方式创建 datapoint
        image = Image.open(path).convert("RGB")
        datapoint = create_empty_datapoint()
        set_image(datapoint, image)
        
        # 为每个 prompt 添加查询，记录 query_id
        prompt_to_id = {}
        query_ids = []
        for text_prompt in text_prompts:
            query_id = add_text_prompt(datapoint, text_prompt, id_gen)
            prompt_to_id[text_prompt] = query_id
            query_ids.append(query_id)
        
        # 应用 transform（完全按照官方示例）
        datapoint = transform(datapoint)
        
        datapoints.append(datapoint)
        query_ids_per_image.append({
            'query_ids': query_ids,
            'prompt_to_id': prompt_to_id,
        })

    if not datapoints:
        print("No valid datapoints to process!")
        return []

    # -------- collate 成 batch 并搬到设备（完全按照官方示例） --------
    print("Collating batch...")
    # 注意：官方示例使用 dict_key="dummy"
    batch = collate(datapoints, dict_key="dummy")["dummy"]
    batch = copy_data_to_device(batch, device, non_blocking=True)

    # -------- 前向推理 --------
    print("Running inference...")
    start_time = time.time()
    with torch.inference_mode():
        output = model(batch)
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f} seconds")

    # -------- 后处理（完全按照官方示例） --------
    print("Post-processing results...")
    processed_results: Dict[int, Any] = postprocessor.process_results(
        output, batch.find_metadatas
    )
    # processed_results: key = query_id, value = 该 query 的检测/分割结果

    # -------- 组织结果（按照图片和 prompt 分组） --------
    results_per_image: List[Dict[str, Any]] = []

    for img_idx, query_info in enumerate(query_ids_per_image):
        prompt_to_id = query_info['prompt_to_id']
        per_image_results: Dict[str, Any] = {}
        
        # 为每个 prompt 获取结果
        for prompt_str, query_id in prompt_to_id.items():
            result = processed_results.get(query_id, None)
            if result is not None:
                per_image_results[prompt_str] = result
            else:
                print(f"Warning: No result found for image {img_idx}, prompt '{prompt_str}' (query_id={query_id})")
        
        results_per_image.append({
            'results': per_image_results,
            'query_ids': query_info['query_ids'],
            'prompt_to_id': prompt_to_id,
            'all_processed_results': processed_results,  # 保存完整结果以便后续使用
        })

    return results_per_image


# ============ 批量保存结果 ============

def save_batch_results(
    image_paths: List[str],
    results_per_image: List[Dict[str, Any]],
    output_dir: str,
    prefix: str = "result",
    dpi: int = 300,
) -> List[str]:
    """
    批量保存推理结果的可视化图像。
    
    Args:
        image_paths: 输入图像路径列表
        results_per_image: sam3_batch_text_inference 返回的结果列表
        output_dir: 输出目录
        prefix: 输出文件名前缀
        dpi: 图像分辨率
        
    Returns:
        saved_paths: 保存的文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for img_idx, (img_path, result_info) in enumerate(zip(image_paths, results_per_image)):
        image = Image.open(img_path).convert("RGB")
        per_image_results = result_info['results']
        
        for prompt, result in per_image_results.items():
            if result is None:
                continue
            
            # 生成输出文件名
            img_name = Path(img_path).stem
            safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt)
            output_filename = f"{prefix}_{img_idx}_{img_name}_{safe_prompt}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存结果
            save_results(image, result, output_path, dpi=dpi)
            saved_paths.append(output_path)
            print(f"Saved: {output_path}")

    return saved_paths


# ============ 主程序示例 ============

if __name__ == "__main__":
    # 示例配置
    image_paths = [
        "/home/sdgs007/workspace/sam3/assets/images/frame_1.png",
        # 可以添加更多图片路径
    ]
    
    text_prompts = ["car", "road"]
    
    # 可选：指定检查点路径
    checkpoint_path = "/home/sdgs007/workspace/sam3/examples/checkpoint/sam3.pt"
    
    # 运行批量推理
    print("=" * 60)
    print("SAM3 Batch Text Inference Demo")
    print("(Completely following official sam3_image_batched_inference.ipynb)")
    print("=" * 60)
    
    results = sam3_batch_text_inference(
        image_paths=image_paths,
        text_prompts=text_prompts,
        detection_threshold=0.1,
        checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else None,
    )
    
    # 打印结果统计
    print("\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)
    for img_idx, result_info in enumerate(results):
        print(f"\nImage {img_idx} ({image_paths[img_idx]}):")
        per_image_results = result_info['results']
        for prompt, result in per_image_results.items():
            if result is not None:
                num_objects = len(result.get("scores", []))
                print("scores",result.get("scores", []))
                confs = [round(x, 4) for x in result.get("scores", []).detach().cpu().tolist()]
                print("confs",confs)
                print(f"  Prompt '{prompt}': {num_objects} object(s) detected")
                image = Image.open(image_paths[0]).convert("RGB")
                width, height = image.size
                # 每个置信度对应一个mask，所有的置信度需要传入
                masks = result.get("masks", [])
                boxes = result.get("boxes", [])
                convert_result = convert_sam3_masks_to_json(masks, boxes,label=prompt,confidence=confs,original_image_path=image_paths[img_idx],w=width,h=height,epsilon=0.5)
                print("convert_result-------------------------------",convert_result)
            else:
                print(f"  Prompt '{prompt}': No results")
    
    # 保存结果
    output_dir = "/home/sdgs007/workspace/sam3/assets/images/results"
    print(f"\nSaving results to: {output_dir}")
    saved_paths = save_batch_results(
        image_paths=image_paths,
        results_per_image=results,
        output_dir=output_dir,
        prefix="batch_demo",
        dpi=200,
    )
    
    print(f"\nTotal {len(saved_paths)} result image(s) saved.")
