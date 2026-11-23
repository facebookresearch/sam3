#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用微调后的 SAM3 分割一个文件夹里的图片，并保存叠加结果。

默认读 Factory_photo，权重是把头微调 checkpoint 合并进 sam3.pt 之后的文件。
提示词和训练标注的类别名一致：Tank body、Tank handle。
每张图只编码一次，再分别用这两个短语做 grounding，和实时脚本同一条推理路径。

输出（默认 output/factory_photo/）：
  <原名>_overlay.jpg          原图上的轮廓、类别和分数
  <原名>_Tank_body.png        该类实例的并集 mask
  <原名>_Tank_handle.png

示例：
  conda activate sam3
  python examples/segment_factory_photos.py

  python examples/segment_factory_photos.py \\
    --image-dir /home/ygz/Projects/sam3/Factory_photo \\
    --checkpoint /home/ygz/Projects/sam3/outputs/tank_head_ft/sam3_tank_head.pt \\
    --prompt "Tank body,Tank handle" \\
    --confidence 0.4
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import sam3.perflib.fused as sam3_fused
from sam3.model import vitdet as sam3_vitdet
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_PROMPTS = ("Tank body", "Tank handle")
CLASS_COLORS = {
    "Tank body": (
        (40, 180, 255),
        (0, 140, 255),
        (90, 210, 255),
        (20, 100, 230),
    ),
    "Tank handle": (
        (80, 220, 90),
        (40, 180, 40),
        (140, 255, 140),
        (0, 160, 70),
    ),
}
FALLBACK_COLORS = (
    (255, 140, 40),
    (180, 80, 255),
    (60, 220, 220),
    (80, 80, 255),
)


def _patch_sam3_addmm_act_fp32() -> None:
    """推理时避开 fused addmm 的 bf16/fp32 混用。"""

    def addmm_act_fp32(activation, linear, mat1):
        if torch.is_grad_enabled():
            raise ValueError("Expected grad to be disabled.")
        x = F.linear(mat1, linear.weight, linear.bias)
        if activation in (torch.nn.functional.relu, torch.nn.ReLU):
            return F.relu(x)
        if activation in (torch.nn.functional.gelu, torch.nn.GELU):
            return F.gelu(x)
        raise ValueError(f"Unexpected activation {activation}")

    sam3_fused.addmm_act = addmm_act_fp32
    sam3_vitdet.addmm_act = addmm_act_fp32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="用微调 SAM3 分割文件夹里的图片")
    p.add_argument(
        "--image-dir",
        type=str,
        default=str(_REPO_ROOT / "Factory_photo"),
        help="待测图片目录",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(_REPO_ROOT / "outputs" / "tank_head_ft" / "sam3_tank_head.pt"),
        help="微调后、已合并成 sam3.pt 格式的权重",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=",".join(DEFAULT_PROMPTS),
        help="文本提示，多个类别用英文逗号分隔。应与训练类别名一致",
    )
    p.add_argument("--confidence", type=float, default=0.4, help="实例置信度阈值")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(_REPO_ROOT / "output" / "factory_photo"),
        help="叠加图和 mask 的保存目录",
    )
    return p.parse_args()


def split_prompts(text: str) -> list[str]:
    prompts = []
    for chunk in text.replace("，", ",").split(","):
        item = chunk.strip()
        if item and item not in prompts:
            prompts.append(item)
    return prompts


def list_images(image_dir: Path) -> list[Path]:
    files = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: p.name)
    return files


class Segmenter:
    def __init__(self, checkpoint: str, device: str, confidence: float):
        _patch_sam3_addmm_act_fp32()
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"找不到权重：{checkpoint}")
        print(f"加载 SAM3：{checkpoint}  device={device}", flush=True)
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint,
            load_from_HF=False,
            device=device,
            enable_inst_interactivity=False,
        )
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=confidence)
        self._text_cache: dict[str, dict] = {}

    @torch.inference_mode()
    def segment(self, rgb: np.ndarray, prompts: list[str]) -> list[tuple[np.ndarray, float, str]]:
        """编码图像一次，再对每个短语做 grounding。"""
        state = self.processor.set_image(Image.fromarray(np.ascontiguousarray(rgb), mode="RGB"))
        found: list[tuple[np.ndarray, float, str]] = []
        for prompt in prompts:
            if prompt not in self._text_cache:
                self._text_cache[prompt] = self.model.backbone.forward_text(
                    [prompt], device=self.processor.device
                )
            state["backbone_out"].update(
                {
                    key: value.clone() if torch.is_tensor(value) else value
                    for key, value in self._text_cache[prompt].items()
                }
            )
            state["geometric_prompt"] = self.model._get_dummy_prompt()
            state = self.processor._forward_grounding(state)
            found.extend(_unpack(state, prompt))
        return found


def _unpack(state, prompt: str) -> list[tuple[np.ndarray, float, str]]:
    masks = state["masks"]
    scores = state["scores"]
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().float().cpu().numpy()
    if masks.size == 0:
        return []
    if masks.ndim == 4:
        masks = masks[:, 0]
    out = []
    for i in range(masks.shape[0]):
        mask = masks[i] > 0.5 if masks.dtype != np.bool_ else masks[i]
        out.append((mask.astype(bool), float(scores[i]), prompt))
    return out


def _color(prompt: str, index_in_class: int) -> tuple[int, int, int]:
    palette = CLASS_COLORS.get(prompt, FALLBACK_COLORS)
    return palette[index_in_class % len(palette)]


def _centroid(mask: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size < 30:
        return None
    return int(xs.mean()), int(ys.mean())


def paint(bgr: np.ndarray, instances: list[tuple[np.ndarray, float, str]], title: str) -> np.ndarray:
    vis = bgr.copy()
    seen: dict[str, int] = {}
    for mask, score, prompt in instances:
        if mask.shape[:2] != vis.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (vis.shape[1], vis.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        slot = seen.get(prompt, 0)
        seen[prompt] = slot + 1
        color = _color(prompt, slot)
        tint = np.array(color, dtype=np.float32)
        vis[mask] = (0.55 * vis[mask].astype(np.float32) + 0.45 * tint).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        center = _centroid(mask)
        if center is not None:
            label = f"{prompt} {score:.2f}"
            cv2.putText(vis, label, (center[0] + 4, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, label, (center[0] + 4, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(vis, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def union_mask(instances: list[tuple[np.ndarray, float, str]], prompt: str, shape: tuple[int, int]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    for mask, _, name in instances:
        if name != prompt:
            continue
        if mask.shape[:2] != shape:
            mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        canvas[mask] = 255
    return canvas


def safe_name(prompt: str) -> str:
    return prompt.replace(" ", "_")


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        raise SystemExit(f"图片目录不存在：{image_dir}")
    prompts = split_prompts(args.prompt)
    if not prompts:
        raise SystemExit("至少要有一个文本提示")
    paths = list_images(image_dir)
    if not paths:
        raise SystemExit(f"{image_dir} 里没有 jpg/png/webp 图片")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"图片 {len(paths)} 张  prompts={prompts}  confidence={args.confidence}", flush=True)
    print(f"输出 {out_dir}", flush=True)

    segmenter = Segmenter(args.checkpoint, args.device, args.confidence)
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"读图失败，跳过 {path.name}", flush=True)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        instances = segmenter.segment(rgb, prompts)
        counts = {name: 0 for name in prompts}
        for _, _, name in instances:
            counts[name] = counts.get(name, 0) + 1
        summary = "  ".join(f"{name} {counts[name]}" for name in prompts)
        overlay = paint(bgr, instances, f"{path.name}  {summary}")
        overlay_path = out_dir / f"{path.stem}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        h, w = bgr.shape[:2]
        for prompt in prompts:
            mask_path = out_dir / f"{path.stem}_{safe_name(prompt)}.png"
            cv2.imwrite(str(mask_path), union_mask(instances, prompt, (h, w)))
        scores = ", ".join(f"{name} {score:.2f}" for _, score, name in instances) or "无实例"
        print(f"{path.name}  {summary}  [{scores}]", flush=True)
        print(f"  -> {overlay_path}", flush=True)

    print(f"完成，共 {len(paths)} 张 -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
