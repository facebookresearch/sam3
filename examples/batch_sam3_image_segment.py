#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对文件夹内的 RGB 图像逐帧运行 SAM3 图像模型，分别用文本提示分割两类目标（默认：方形冲压钣金件 / 黑色定位块），
保存二值 mask（metal=钣金，fixture=定位块）；可选将分割结果叠加可视化（风格与 video notebook 中的 render_masklet_frame 一致）。

两类物体与图中标注对应关系简述：
- metal（钣金）：大面积银色反光冲压件，带孔洞、加强筋与矩形镂空，可被夹爪抓取或置于绿色台面与橙色治具上。
- fixture（定位块）：小而哑光的黑/深褐长方定位块，常装在细长银色立柱顶端或贴在钣金边角附近，与高光钣金对比明显。

示例（单集）：
  cd /home/wls/Test/GDRNPP_6D_evaluation/sam3
  python examples/batch_sam3_image_segment.py \\
    --rgb-dir .../EpisodeName/rgb \\
    --checkpoint /home/wls/Test/GDRNPP_6D_evaluation/base_model/sam3/sam3.pt \\
    --visualize

示例（整目录下所有 …/EpisodeName/rgb）：
  python examples/batch_sam3_image_segment.py \\
    --rgb-parent-dir .../1_head_rgbdepth_image_eefpose_data \\
    --output-root .../1_sam3_image_mask \\
    --checkpoint .../sam3.pt

启动时默认成对审计（rgb/depth 张数、metal/fixture 与 rgb 对齐）；不合格则删 1_head 与 1_sam3 下同名的 episode 目录。
仅审计：加 --audit-only；关闭审计：加 --skip-audit。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# 保证可从仓库根目录导入 sam3（若在 examples 下直接运行）
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import sam3.perflib.fused as sam3_fused
from sam3.model_builder import build_sam3_image_model
from sam3.model import vitdet as sam3_vitdet
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import render_masklet_frame


def _patch_sam3_addmm_act_fp32() -> None:
    """
    sam3.perflib.fused.addmm_act 将 mat1/权重强制转为 bfloat16，fc1 输出为 bf16，
    而同一 Mlp 的 fc2 仍为 float32，会触发 Linear 上 bf16/float32 混用错误。
    推理时改为标准 FP32 Linear + 激活（与禁用融合内核等价）。

    注意：vitdet 使用「from fused import addmm_act」，仅改 fused 模块属性无效，
    必须同时替换 sam3.model.vitdet 模块内的绑定。
    """

    def addmm_act_fp32(activation, linear, mat1):
        if torch.is_grad_enabled():
            raise ValueError("Expected grad to be disabled.")
        x = F.linear(mat1, linear.weight, linear.bias)
        if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
            return F.relu(x)
        if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
            return F.gelu(x)
        raise ValueError(f"Unexpected activation {activation}")

    sam3_fused.addmm_act = addmm_act_fp32
    sam3_vitdet.addmm_act = addmm_act_fp32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM3 图像批量分割：钣金 + 黑色定位块")
    p.add_argument(
        "--rgb-dir",
        type=str,
        default=None,
        help="包含 RGB 图像的目录（如 .../EpisodeName/rgb）；与 --rgb-parent-dir 二选一",
    )
    p.add_argument(
        "--rgb-parent-dir",
        type=str,
        default=None,
        help="父目录：其下每个子目录若含 rgb/ 则视为一集并依次处理（批量抽 mask）",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="/home/wls/Test/GDRNPP_6D_evaluation/base_data/1_sam3_image_mask",
        help="输出根目录；其下会创建与 episode 同名的子文件夹",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="/home/wls/Test/GDRNPP_6D_evaluation/base_model/sam3/sam3.pt",
        help="sam3.pt 权重路径",
    )

# default="the square sheet metal part holding in the robot hand",
# default="the only black objects",


    p.add_argument(
        "--metal-prompt",
        type=str,
        default=(
            "the large reflective silver stamped sheet metal plate with circular holes, embossed ridges, "
            "and a rectangular cutout; the main shiny metal workpiece held by the robot gripper"
        ),
        help="钣金件（方形冲压件）英文文本提示；侧重银色反光、孔洞与冲压结构",
    )
    p.add_argument(
        "--fixture-prompt",
        type=str,
        default=(
            "the small dark matte black or brown rectangular positioning blocks on thin vertical silver rods, "
            "non-reflective fixture blocks near sheet metal corners"
        ),
        help="黑色定位块英文文本提示；侧重小块、哑光、立柱安装或与钣金对比",
    )
    p.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Sam3Processor 置信度阈值",
    )
    p.add_argument(
        "--visualize",
        action="store_true",
        help="开启则将叠加可视化保存到 plot_image 子文件夹",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda 或 cpu",
    )
    p.add_argument(
        "--force-regenerate",
        action="store_true",
        help="忽略已有 metal/fixture 输出，强制对该集重新推理",
    )
    p.add_argument(
        "--skip-audit",
        action="store_true",
        help="跳过启动前的输入/输出成对校验与无效目录清理",
    )
    p.add_argument(
        "--audit-only",
        action="store_true",
        help="仅执行校验：删除不一致的 episode（输入与输出子文件夹同时删），不加载模型、不推理",
    )
    return p.parse_args()


def _episode_name_from_rgb_dir(rgb_dir: Path) -> str:
    """若路径以 .../xxx/rgb 结尾，则 episode 为父目录名；否则用 rgb 目录名。"""
    if rgb_dir.name.lower() == "rgb" and rgb_dir.parent.name:
        return rgb_dir.parent.name
    return rgb_dir.name


def _list_rgb_images(rgb_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = [p for p in rgb_dir.iterdir() if p.suffix.lower() in exts]
    if not files:
        return []

    def sort_key(p: Path):
        stem = p.stem
        nums = re.findall(r"\d+", stem)
        return (int(nums[-1]) if nums else stem, stem)

    files.sort(key=sort_key)
    return files


_MASK_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _count_image_files_in_dir(d: Path) -> int:
    """统计目录下 rgb / depth / mask 等图像文件数量（不含子目录）。"""
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() in _MASK_IMG_EXTS)


def _count_mask_files(mask_dir: Path) -> int:
    """统计 metal / fixture 目录下已保存的 mask 图像数量。"""
    return _count_image_files_in_dir(mask_dir)


def _sam3_episode_already_complete(
    metal_dir: Path, fixture_dir: Path, n_rgb: int
) -> bool:
    """metal 与 fixture 中图像数均等于 rgb 张数则认为本集已生成完毕，可跳过。"""
    if n_rgb <= 0:
        return False
    nm = _count_mask_files(metal_dir)
    nf = _count_mask_files(fixture_dir)
    return nm == nf == n_rgb


def _masks_to_union_bool(masks: torch.Tensor) -> np.ndarray:
    """masks: (N,H,W) 或 (N,1,H,W) bool，合并为单通道 bool(H,W)。"""
    if masks is None or masks.numel() == 0:
        return None
    m = masks.cpu().numpy()
    if m.size == 0:
        return None
    if m.ndim == 4 and m.shape[1] == 1:
        m = m.squeeze(1)
    return np.any(m, axis=0)


def _union_to_uint8(union: np.ndarray | None, h: int, w: int) -> np.ndarray:
    if union is None:
        return np.zeros((h, w), dtype=np.uint8)
    return (union.astype(np.uint8) * 255).reshape(h, w)


def _boxes_xyxy_to_xywh_norm(boxes: torch.Tensor, img_w: int, img_h: int) -> np.ndarray:
    """(N,4) xyxy 像素坐标 -> (N,4) 归一化 xywh（左上角 + 宽高）。"""
    if boxes is None or boxes.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    b = boxes.detach().cpu().numpy().astype(np.float64)
    out = []
    for x0, y0, x1, y1 in b:
        w = max(x1 - x0, 0)
        h = max(y1 - y0, 0)
        out.append([x0 / img_w, y0 / img_h, w / img_w, h / img_h])
    return np.asarray(out, dtype=np.float32)


def _build_masklet_dict_for_vis(
    metal_union: np.ndarray | None,
    fixture_union: np.ndarray | None,
    scores_tensor_metal: torch.Tensor,
    scores_tensor_fixture: torch.Tensor,
    boxes_metal: torch.Tensor,
    boxes_fixture: torch.Tensor,
    img_h: int,
    img_w: int,
) -> Dict:
    """
    构造 render_masklet_frame 所需的 outputs：
    最多两个实例：id=0 钣金，id=1 定位块（颜色区分）。
    """
    out_binary_masks: List[np.ndarray] = []
    out_probs: List[float] = []
    out_obj_ids: List[int] = []
    out_boxes_xywh: List[List[float]] = []

    boxes_xywh_m = _boxes_xyxy_to_xywh_norm(boxes_metal, img_w, img_h)
    boxes_xywh_f = _boxes_xyxy_to_xywh_norm(boxes_fixture, img_w, img_h)

    if metal_union is not None and metal_union.any():
        out_binary_masks.append(metal_union.astype(np.float32))
        sm = scores_tensor_metal.float().mean().item() if scores_tensor_metal.numel() else 1.0
        out_probs.append(float(sm))
        out_obj_ids.append(0)
        out_boxes_xywh.append(boxes_xywh_m[0].tolist() if len(boxes_xywh_m) > 0 else [0.0, 0.0, 1.0, 1.0])

    if fixture_union is not None and fixture_union.any():
        out_binary_masks.append(fixture_union.astype(np.float32))
        sf = scores_tensor_fixture.float().mean().item() if scores_tensor_fixture.numel() else 1.0
        out_probs.append(float(sf))
        out_obj_ids.append(1)
        out_boxes_xywh.append(boxes_xywh_f[0].tolist() if len(boxes_xywh_f) > 0 else [0.0, 0.0, 1.0, 1.0])

    if not out_binary_masks:
        return {
            "out_binary_masks": [],
            "out_probs": [],
            "out_obj_ids": [],
            "out_boxes_xywh": [],
        }

    return {
        "out_binary_masks": out_binary_masks,
        "out_probs": out_probs,
        "out_obj_ids": out_obj_ids,
        "out_boxes_xywh": out_boxes_xywh,
    }


def run_one_image(
    processor: Sam3Processor,
    image_path: Path,
    metal_prompt: str,
    fixture_prompt: str,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    返回：
      metal_uint8 (H,W), fixture_uint8 (H,W), 用于可视化的 masklet dict 所需中间量（在函数内已合并）
    """
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    # --- 钣金 ---
    state_m = processor.set_image(image)
    state_m = processor.set_text_prompt(state=state_m, prompt=metal_prompt)
    metal_union = _masks_to_union_bool(state_m.get("masks"))
    metal_scores = state_m.get("scores", torch.tensor([]))
    metal_boxes = state_m.get("boxes", torch.tensor([]).reshape(0, 4))

    processor.reset_all_prompts(state_m)

    # --- 定位块 ---
    state_f = processor.set_image(image)
    state_f = processor.set_text_prompt(state=state_f, prompt=fixture_prompt)
    fixture_union = _masks_to_union_bool(state_f.get("masks"))
    fixture_scores = state_f.get("scores", torch.tensor([]))
    fixture_boxes = state_f.get("boxes", torch.tensor([]).reshape(0, 4))

    processor.reset_all_prompts(state_f)

    metal_u8 = _union_to_uint8(metal_union, h, w)
    fixture_u8 = _union_to_uint8(fixture_union, h, w)

    vis_payload = (metal_union, fixture_union, metal_scores, fixture_scores, metal_boxes, fixture_boxes, h, w)
    return metal_u8, fixture_u8, vis_payload


def run_one_dataset(args: argparse.Namespace, processor: Sam3Processor, rgb_dir: Path) -> None:
    rgb_dir = rgb_dir.resolve()
    if not rgb_dir.is_dir():
        raise NotADirectoryError(rgb_dir)

    paths = _list_rgb_images(rgb_dir)
    if not paths:
        print(f"[WARN] 跳过：rgb 目录内无 png/jpg/webp → {rgb_dir}")
        return

    episode = _episode_name_from_rgb_dir(rgb_dir)
    out_episode = Path(args.output_root).resolve() / episode
    metal_dir = out_episode / "metal"
    fixture_dir = out_episode / "fixture"
    plot_dir = out_episode / "plot_image"
    n_rgb = len(paths)
    if not args.force_regenerate and _sam3_episode_already_complete(metal_dir, fixture_dir, n_rgb):
        print(
            f"[INFO] 跳过（已有完整分割：metal/fixture 各 {n_rgb} 张）→ {out_episode}"
        )
        return

    metal_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] rgb_dir={rgb_dir}")
    print(f"[INFO] episode={episode}")
    print(f"[INFO] output={out_episode}")
    print(f"[INFO] visualize={args.visualize}")

    print(f"[INFO] 共 {len(paths)} 张图像")

    for ipath in paths:
        metal_u8, fixture_u8, vis_payload = run_one_image(
            processor, ipath, args.metal_prompt, args.fixture_prompt
        )
        metal_union, fixture_union, ms, fs, mb, fb, h, w = vis_payload

        cv2.imwrite(str(metal_dir / ipath.name), metal_u8)
        cv2.imwrite(str(fixture_dir / ipath.name), fixture_u8)

        if args.visualize:
            img_rgb = np.array(Image.open(ipath).convert("RGB"))
            outputs = _build_masklet_dict_for_vis(
                metal_union, fixture_union, ms, fs, mb, fb, h, w
            )
            overlay = render_masklet_frame(img_rgb, outputs, frame_idx=None, alpha=0.5)
            cv2.imwrite(
                str(plot_dir / ipath.name),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

    print("[INFO] 本集完成。")


def _audit_one_episode(episode_name: str, head_parent: Path, mask_root: Path) -> bool:
    """
    校验单集：输入侧 rgb/depth 张数一致；输出侧存在 metal/ 与 fixture/，且与 rgb 张数一致。
    不通过则打印 [AUDIT][ERROR] 并删除 head_parent/episode 与 mask_root/episode（存在的一侧或两侧）。

    返回 True 表示本集有效；False 表示已按无效对清理。
    """
    ep_in = head_parent / episode_name
    ep_out = mask_root / episode_name
    has_in = ep_in.is_dir()
    has_out = ep_out.is_dir()
    if not has_in and not has_out:
        return True

    rgb_d = ep_in / "rgb"
    depth_d = ep_in / "depth"
    n_rgb = _count_image_files_in_dir(rgb_d) if has_in else 0
    n_dep = _count_image_files_in_dir(depth_d) if has_in and depth_d.is_dir() else 0
    metal_d = ep_out / "metal"
    fix_d = ep_out / "fixture"
    nm = _count_image_files_in_dir(metal_d) if has_out and metal_d.is_dir() else 0
    nf = _count_image_files_in_dir(fix_d) if has_out and fix_d.is_dir() else 0

    ok = (
        has_in
        and has_out
        and n_rgb > 0
        and n_dep == n_rgb
        and metal_d.is_dir()
        and fix_d.is_dir()
        and nm == nf == n_rgb
    )
    if ok:
        return True

    reasons: List[str] = []
    if not has_in and has_out:
        reasons.append("仅存在 mask 输出，无输入 episode")
    elif has_in and not has_out:
        reasons.append("仅有输入，无 1_sam3 下该集输出")
    else:
        if n_rgb <= 0:
            reasons.append("rgb 无有效图像或缺少 rgb 目录")
        if has_in and (not depth_d.is_dir() or n_dep != n_rgb):
            reasons.append(f"depth 与 rgb 不一致或缺少 depth (rgb={n_rgb}, depth={n_dep})")
        if has_out:
            if not metal_d.is_dir():
                reasons.append("缺少 metal 目录")
            if not fix_d.is_dir():
                reasons.append("缺少 fixture 目录")
            if metal_d.is_dir() and fix_d.is_dir() and (nm != nf or n_rgb != nm or n_rgb != nf):
                reasons.append(f"metal({nm})/fixture({nf}) 与 rgb({n_rgb}) 未对齐")
    detail = "；".join(reasons) if reasons else "未知原因"
    print(f"[AUDIT][ERROR] {episode_name}: {detail} → 删除输入/输出对应子文件夹（若存在）")
    try:
        if has_in:
            shutil.rmtree(ep_in)
            print(f"[AUDIT] 已删输入 {ep_in}")
    except OSError as exc:
        print(f"[AUDIT] 删输入失败 {ep_in}: {exc}")
    try:
        if has_out:
            shutil.rmtree(ep_out)
            print(f"[AUDIT] 已删输出 {ep_out}")
    except OSError as exc:
        print(f"[AUDIT] 删输出失败 {ep_out}: {exc}")
    return False


def _audit_all_episodes(head_parent: Path, mask_root: Path) -> None:
    """对 head_parent 与 mask_root 下出现过的所有 episode 名成对检查并清理无效项。"""
    names: set = set()
    if head_parent.is_dir():
        for c in head_parent.iterdir():
            if c.is_dir():
                names.add(c.name)
    if mask_root.is_dir():
        for c in mask_root.iterdir():
            if c.is_dir():
                names.add(c.name)
    for name in sorted(names):
        _audit_one_episode(name, head_parent, mask_root)
    print(f"[AUDIT] 全目录检查结束，共遍历 {len(names)} 个 episode 名称。")


def _discover_rgb_dirs(parent: Path) -> List[Path]:
    """parent/EpisodeName/rgb → 返回所有存在的 rgb 目录路径（排序）。"""
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    out: List[Path] = []
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        rd = child / "rgb"
        if rd.is_dir():
            out.append(rd)
    if not out:
        raise FileNotFoundError(f"在 {parent} 下未发现任何 */rgb 子目录")
    return out


def main() -> None:
    args = parse_args()
    if args.rgb_parent_dir and args.rgb_dir:
        raise SystemExit("请只指定 --rgb-parent-dir 或 --rgb-dir 其一")
    mask_root = Path(args.output_root).resolve()

    if not args.skip_audit:
        if args.rgb_parent_dir:
            hp = Path(args.rgb_parent_dir).resolve()
            print(f"[AUDIT] 成对检查并清理: head={hp}  mask_root={mask_root}")
            _audit_all_episodes(hp, mask_root)
        else:
            default_single = (
                Path("/home/wls/Test/GDRNPP_6D_evaluation/base_data/1_head_rgbdepth_image_eefpose_data")
                / "A10-A15-I-L-01-TQ_19_01-P4_391-dex_hand-20260415172514-49-c4ef5a-v003"
                / "rgb"
            )
            rd = Path(args.rgb_dir).resolve() if args.rgb_dir else default_single
            hp = rd.parent.parent if rd.name.lower() == "rgb" else rd.parent
            en = _episode_name_from_rgb_dir(rd)
            print(f"[AUDIT] 单集成对检查: episode={en} head_parent={hp} mask_root={mask_root}")
            _audit_one_episode(en, hp, mask_root)
        if args.audit_only:
            print("[INFO] --audit-only：仅校验结束，不加载模型。")
            return

    if args.rgb_parent_dir:
        try:
            rgb_dirs = _discover_rgb_dirs(Path(args.rgb_parent_dir).resolve())
        except FileNotFoundError as exc:
            print(f"[WARN] 无可处理的 rgb 子目录：{exc}")
            return
        print(f"[INFO] rgb-parent-dir 下共 {len(rgb_dirs)} 集 rgb，将依次处理。")
    else:
        default_single = (
            Path("/home/wls/Test/GDRNPP_6D_evaluation/base_data/1_head_rgbdepth_image_eefpose_data")
            / "A10-A15-I-L-01-TQ_19_01-P4_391-dex_hand-20260415172514-49-c4ef5a-v003"
            / "rgb"
        )
        rd = Path(args.rgb_dir).resolve() if args.rgb_dir else default_single
        rgb_dirs = [rd]

    _patch_sam3_addmm_act_fp32()

    device = args.device
    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"未找到权重文件: {ckpt}")
    model = build_sam3_image_model(
        checkpoint_path=str(ckpt),
        device=device,
        load_from_HF=False,
    )
    processor = Sam3Processor(model, device=device, confidence_threshold=args.confidence_threshold)

    failed: List[str] = []
    for idx, rgb_dir in enumerate(rgb_dirs, start=1):
        print(f"\n[INFO] ========== [{idx}/{len(rgb_dirs)}] {rgb_dir} ==========")
        try:
            run_one_dataset(args, processor, rgb_dir)
        except Exception as exc:
            failed.append(str(rgb_dir))
            print(f"[ERROR] 本集失败，已跳过并继续下一集: {exc}")
            traceback.print_exc()

    print("\n[INFO] 全部完成。")
    if failed:
        print(f"[WARN] 共 {len(failed)} 集失败，请向上搜索 [ERROR] / Traceback。")


if __name__ == "__main__":
    main()
