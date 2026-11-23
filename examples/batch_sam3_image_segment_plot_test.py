#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单张图像 SAM3 分割：根据文本提示生成 metal / fixture 二值 mask，并保存叠加可视化。

仅支持下列参数（其余均已移除）：
  --input            输入 RGB 图像路径
  --output-dir       输出目录（默认：与输入图像同一目录）
  --checkpoint       SAM3 权重 sam3.pt
  --metal-prompt     钣金文本提示
  --fixture-prompt   定位块文本提示
  --confidence-threshold  置信度阈值

输出（写入 output-dir，文件名基于输入 stem）：
  <stem>_metal_mask.png
  <stem>_fixture_mask.png
  <stem>_overlay.png    （mask 叠加在原图上）

可选：YOLO 等检测框（像素坐标 xyxy，与图像同尺寸）可与文本一起送入 SAM3：
  --metal-box X1 Y1 X2 Y2 / --fixture-box X1 Y1 X2 Y2
  内部会转为归一化 [center_x, center_y, width, height]，在 set_text_prompt 之后调用
  Sam3Processor.add_geometric_prompt（见 sam3_image_processor.py）。

示例：
  cd /home/wls/Test/GDRNPP_6D_evaluation/sam3
  python examples/batch_sam3_image_segment_plot_test.py \\
    --input /path/to/frame_0_rgb.png \\
    --checkpoint /home/wls/Test/GDRNPP_6D_evaluation/base_model/sam3/sam3.pt \\
    --metal-prompt "silver sheet metal in robot gripper" \\
    --fixture-prompt "small black rectangular block"

  # 带 YOLO 框（像素）：先文本 grounding，再追加正样本框约束区域
  python examples/batch_sam3_image_segment_plot_test.py \\
    -i frame.png --checkpoint .../sam3.pt \\
    --metal-prompt "..." --metal-box 120 80 400 320
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# 将 sam3 仓库根目录加入 Python 路径，以便导入 sam3.*
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import sam3.perflib.fused as sam3_fused
from sam3.model import vitdet as sam3_vitdet
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def _patch_sam3_addmm_act_fp32() -> None:
    """
    修补 SAM3 融合算子：推理时用 FP32 线性层替代 mixed-precision，
    否则 vitdet 里 bf16 与 float32 混用会在部分 GPU 上报错。
    """

    def addmm_act_fp32(activation, linear, mat1):
        if torch.is_grad_enabled():
            raise ValueError("Expected grad to be disabled.")
        x = F.linear(mat1, linear.weight, linear.bias)
        if activation in (F.relu, torch.nn.ReLU):
            return F.relu(x)
        if activation in (F.gelu, torch.nn.GELU):
            return F.gelu(x)
        raise ValueError(f"Unexpected activation {activation}")

    sam3_fused.addmm_act = addmm_act_fp32
    sam3_vitdet.addmm_act = addmm_act_fp32


def _norm_cxcywh_from_xyxy_pixel(
    img_w: int, img_h: int, x1: float, y1: float, x2: float, y2: float
) -> list[float]:
    """
    SAM3 add_geometric_prompt 要求框为 [cx, cy, w, h]，相对整图归一化到 [0, 1]。
    输入为像素坐标 xyxy（左上、右下，可与 YOLO 输出一致）。
    """
    x1, x2 = sorted((float(x1), float(x2)))
    y1, y2 = sorted((float(y1), float(y2)))
    x1 = max(0.0, min(x1, float(img_w - 1)))
    x2 = max(x1 + 1e-4, min(x2, float(img_w)))
    y1 = max(0.0, min(y1, float(img_h - 1)))
    y2 = max(y1 + 1e-4, min(y2, float(img_h)))
    cx = ((x1 + x2) * 0.5) / img_w
    cy = ((y1 + y2) * 0.5) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return [cx, cy, bw, bh]


def _masks_union_uint8(masks: torch.Tensor | None, height: int, width: int) -> np.ndarray:
    """
    将 SAM3 输出的多实例 mask（N×H×W）沿实例维取并集，再转为单通道 8bit 灰度图。
    无预测时返回与图像同尺寸的纯黑图。
    """
    if masks is None or masks.numel() == 0:
        return np.zeros((height, width), dtype=np.uint8)
    m = masks.detach().cpu()
    # 模型可能输出 float logits，先二值化
    if m.dtype != torch.bool:
        m = m > 0.5
    m = m.numpy()
    if m.ndim == 4 and m.shape[1] == 1:
        m = m.squeeze(1)
    if m.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    # 多个实例：任意一处为前景则该像素为前景
    u = np.any(m, axis=0)
    return (u.astype(np.uint8) * 255).reshape(height, width)


def _run_one_prompt(
    processor: Sam3Processor,
    image: Image.Image,
    prompt: str,
    box_xyxy_pixel: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """
    对当前图像执行一次「文本提示 → 开放词汇分割」推理；可选追加像素框作为几何正样本。
    流程：编码图像 → 注入英文 prompt →（可选）add_geometric_prompt → 合并 masks。
    """
    h, w = image.height, image.width
    state = processor.set_image(image)
    # set_text_prompt 内部会按 confidence_threshold 过滤低分实例
    state = processor.set_text_prompt(prompt, state)
    if box_xyxy_pixel is not None:
        bx = _norm_cxcywh_from_xyxy_pixel(w, h, *box_xyxy_pixel)
        state = processor.add_geometric_prompt(bx, True, state)
    mu = _masks_union_uint8(state.get("masks"), h, w)
    processor.reset_all_prompts(state)
    return mu


def _overlay(rgb: np.ndarray, metal_u8: np.ndarray, fixture_u8: np.ndarray) -> np.ndarray:
    """
    在原图 RGB 上做半透明叠色，便于肉眼看分割是否贴合。
    钣金 mask：偏绿；定位块 mask：偏红。两类别重叠像素以后绘制的 fixture（红色）为准。
    """
    out = rgb.astype(np.float32)
    m = metal_u8 > 127
    f = fixture_u8 > 127
    out[m] = out[m] * 0.55 + np.array([40.0, 255.0, 80.0], dtype=np.float32) * 0.45
    out[f] = out[f] * 0.55 + np.array([255.0, 80.0, 80.0], dtype=np.float32) * 0.45
    return np.clip(out, 0, 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    """解析命令行：仅暴露输入图、输出目录、权重与两类文本提示及置信度阈值。"""
    p = argparse.ArgumentParser(description="SAM3 单图分割：metal + fixture mask + overlay")
    p.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="输入 RGB 图像路径（png/jpg/webp）",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="输出目录；默认与输入图像所在目录相同",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="/home/wls/Test/GDRNPP_6D_evaluation/base_model/sam3/sam3.pt",
        help="sam3.pt 路径",
    )
    p.add_argument(
        "--metal-prompt",
        type=str,
        default="the only metal part holding in robot hand",
        help="钣金英文文本提示",
    )
    p.add_argument(
        "--fixture-prompt",
        type=str,
        default="small dark black matte rectangular positioning block",
        help="定位块英文文本提示",
    )
    p.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.15,
        help="Sam3Processor 置信度阈值",
    )
    p.add_argument(
        "--metal-box",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="可选：钣金 YOLO 框（像素 xyxy），与 --metal-prompt 联用",
    )
    p.add_argument(
        "--fixture-box",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="可选：定位块 YOLO 框（像素 xyxy），与 --fixture-prompt 联用",
    )
    return p.parse_args()


def main() -> None:
    """主流程：校验路径 → 加载 SAM3 → 分别推理钣金/定位块 → 写 mask 与叠加图。"""
    args = parse_args()
    inp = Path(args.input).expanduser().resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"输入图像不存在: {inp}")

    # 未指定输出目录时，与输入图像同一文件夹，便于对照原图
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _patch_sam3_addmm_act_fp32()

    # 构建图像 backbone + 文本 grounding，权重来自本地 sam3.pt
    model = build_sam3_image_model(
        checkpoint_path=str(ckpt),
        device=device,
        load_from_HF=False,
    )
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=float(args.confidence_threshold),
    )

    image = Image.open(inp).convert("RGB")
    stem = inp.stem

    metal_box = tuple(args.metal_box) if args.metal_box is not None else None
    fixture_box = tuple(args.fixture_box) if args.fixture_box is not None else None

    # 两类物体各跑一次文本分割（互不共用 prompt，故 reset 后各推理一次）；可选传入 YOLO 像素框
    metal_u8 = _run_one_prompt(processor, image, args.metal_prompt, box_xyxy_pixel=metal_box)
    fixture_u8 = _run_one_prompt(
        processor, image, args.fixture_prompt, box_xyxy_pixel=fixture_box
    )

    path_metal = out_dir / f"{stem}_metal_mask.png"
    path_fix = out_dir / f"{stem}_fixture_mask.png"
    path_vis = out_dir / f"{stem}_overlay.png"

    # 二值 mask 单通道保存；叠加图需 BGR 写入（OpenCV 约定）
    cv2.imwrite(str(path_metal), metal_u8)
    cv2.imwrite(str(path_fix), fixture_u8)

    rgb = np.array(image)
    vis = _overlay(rgb, metal_u8, fixture_u8)
    cv2.imwrite(str(path_vis), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"[OK] device={device} conf={args.confidence_threshold}")
    print(f"     metal_mask : {path_metal}")
    print(f"     fixture_mask: {path_fix}")
    print(f"     overlay    : {path_vis}")


if __name__ == "__main__":
    main()
