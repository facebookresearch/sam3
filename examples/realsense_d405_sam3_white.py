#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intel RealSense D405 实时画面 + SAM3 自动分割。

两种选目标的方式：
  1. 文字：在运行这个脚本的终端里输入短语并回车（英文更稳，例如 bottle、a red cup）。
     之后每一帧都会按这句话分割画面里所有匹配实例。
  2. 点选 / 框选：在预览窗口里单击物体，或按住左键拖一个框。
     选中后默认跟着物体中心继续分割（按 f 可关掉跟随）。

画面左右并排：左边是相机原图，右边是自动处理后送给分割的图。
白色壶放在白色泡沫上、灯也是白光时，整幅加减亮度分不开它们。
程序会自动找出又亮、又几乎没有颜色的区域，只在这些区域里：
  - 拉开局部明暗（泡沫颗粒、壶的轮廓和阴影）
  - 稍微放大残留色差（泡沫偏暖、塑料偏冷）
  - 若深度能测到壶比泡沫更近，按远近再补一层明暗
彩色区域保持不动。滑条 strength 控制力度，0 等于关闭。

按键（焦点在预览窗口时）：
  q        退出
  c / Esc  清除当前文字或点选
  s        保存当前画面、叠加图和 mask
  f        开关“点选后跟随”
  space    暂停 / 继续分割（预览仍刷新）
  [  ]     降低 / 提高文字分割的置信度阈值
  -  =     减弱 / 加强白区增强
  0        关闭白区增强

示例：
  conda activate sam3
  python examples/realsense_d405_sam3_white.py \\
    --checkpoint /home/ygz/下载/base_model/sam3/sam3.pt \\
    --prompt "white kettle"
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import sam3.perflib.fused as sam3_fused
from sam3.model import vitdet as sam3_vitdet
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

WINDOW = "D405 SAM3  |  click or drag  |  type prompt in terminal  |  q=quit"
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
COLORS_BGR = (
    (40, 180, 255),
    (80, 220, 90),
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
    p = argparse.ArgumentParser(description="D405 实时画面，用文字或鼠标点选做 SAM3 分割")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="/home/ygz/下载/base_model/sam3/sam3.pt",
        help="sam3.pt 权重路径",
    )
    p.add_argument("--prompt", type=str, default="", help="启动时使用的文字提示")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--serial", type=str, default="", help="多相机时指定序列号")
    p.add_argument("--confidence", type=float, default=0.5, help="文字分割置信度阈值")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(_REPO_ROOT / "output" / "d405_sam3_white"),
        help="按 s 时的保存目录",
    )
    p.add_argument("--no-follow", action="store_true", help="点选后不根据 mask 中心继续跟")
    return p.parse_args()


def start_d405(width: int, height: int, fps: int, serial: str):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    candidates = [(width, height, fps), (640, 480, 30), (848, 480, 30), (1280, 720, 30)]
    seen = set()
    last_error = None
    for w, h, f in candidates:
        if (w, h, f) in seen:
            continue
        seen.add((w, h, f))
        for with_depth in (True, False):
            cfg = rs.config()
            if serial:
                cfg.enable_device(serial)
            if with_depth:
                cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, f)
            cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
            try:
                profile = pipeline.start(cfg)
            except RuntimeError as exc:
                last_error = exc
                print(f"跳过 {w}x{h}@{f} depth={with_depth}: {exc}", flush=True)
                continue
            for sensor in profile.get_device().sensors:
                if sensor.supports(rs.option.frames_queue_size):
                    sensor.set_option(rs.option.frames_queue_size, 1)
            align = rs.align(rs.stream.color) if with_depth else None
            depth_scale = 0.001
            if with_depth:
                depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            print(f"D405 {w}x{h}@{f}  depth={'on' if with_depth else 'off'}", flush=True)
            return pipeline, profile, align, depth_scale
    raise RuntimeError(f"无法打开 D405：{last_error}")


def latest_frames(pipeline, align, depth_scale: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    frames = pipeline.wait_for_frames(1000)
    if align is not None:
        frames = align.process(frames)
    color = frames.get_color_frame()
    if not color:
        return None, None
    bgr = np.asanyarray(color.get_data())
    depth_m = None
    if align is not None:
        depth = frames.get_depth_frame()
        if depth:
            depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * depth_scale
    return bgr, depth_m


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
            enable_inst_interactivity=True,
        )
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=confidence)
        self._text_cache_key = None
        self._text_cache = None

    def set_confidence(self, value: float) -> float:
        self.processor.confidence_threshold = float(np.clip(value, 0.05, 0.95))
        return self.processor.confidence_threshold

    @torch.inference_mode()
    def segment_text(self, rgb: np.ndarray, prompt: str) -> list[tuple[np.ndarray, float]]:
        state = self.processor.set_image(_as_pil(rgb))
        if prompt != self._text_cache_key:
            self._text_cache = self.model.backbone.forward_text([prompt], device=self.processor.device)
            self._text_cache_key = prompt
        state["backbone_out"].update(
            {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in self._text_cache.items()
            }
        )
        state["geometric_prompt"] = self.model._get_dummy_prompt()
        state = self.processor._forward_grounding(state)
        return _unpack_text_masks(state)

    @torch.inference_mode()
    def segment_point(self, rgb: np.ndarray, x: int, y: int) -> list[tuple[np.ndarray, float]]:
        state = self.processor.set_image(_as_pil(rgb))
        masks, scores, _ = self.model.predict_inst(
            state,
            point_coords=np.array([[x, y]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        return _best_mask(masks, scores)

    @torch.inference_mode()
    def segment_box(self, rgb: np.ndarray, box_xyxy: tuple[int, int, int, int]) -> list[tuple[np.ndarray, float]]:
        x0, y0, x1, y1 = box_xyxy
        state = self.processor.set_image(_as_pil(rgb))
        masks, scores, _ = self.model.predict_inst(
            state,
            point_coords=None,
            point_labels=None,
            box=np.array([[x0, y0, x1, y1]], dtype=np.float32),
            multimask_output=False,
        )
        return _best_mask(masks, scores)


def _as_pil(rgb: np.ndarray) -> Image.Image:
    """set_image 对 ndarray 按 CHW 读高宽，这里改走 PIL，避免 HWC 被读成错误尺寸。"""
    return Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")


def _unpack_text_masks(state) -> list[tuple[np.ndarray, float]]:
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
        out.append((mask, float(scores[i])))
    return out


def _best_mask(masks, scores) -> list[tuple[np.ndarray, float]]:
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().float().cpu().numpy()
    if masks is None or len(masks) == 0:
        return []
    idx = int(np.argmax(scores))
    mask = masks[idx]
    if mask.dtype != np.bool_:
        mask = mask > 0.0
    return [(mask.astype(bool), float(scores[idx]))]


def mask_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size < 30:
        return None
    return int(xs.mean()), int(ys.mean())


def enhance_white_on_white(
    bgr: np.ndarray,
    strength: int,
    depth_m: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """自动找出发白区域，只在那里拉开壶和泡沫的差别。

    又亮、色度又低的像素视为白壶 / 白泡沫 / 白灯。彩色像素不改。
    白区内部做局部直方图均衡、放大高频明暗，并轻轻放大 a/b 色差。
    有深度时，比周围更近的表面变亮，深度跳变处压暗一条边。
    """
    info = {"ratio": 0.0, "flatness": 0.0, "depth": False}
    strength = int(np.clip(strength, 0, 100))
    if strength == 0:
        return bgr, info

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lf = l.astype(np.float32)
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    chroma = np.hypot(af - 128.0, bf - 128.0)
    white = (lf >= 130.0) & (chroma <= 32.0)
    ratio = float(white.mean())
    info["ratio"] = ratio
    if ratio < 0.02:
        return bgr, info

    local = cv2.GaussianBlur(lf, (0, 0), 4.0)
    contrast = np.abs(lf - local)
    mean_contrast = float(contrast[white].mean())
    flatness = float(np.clip(1.0 - mean_contrast / 12.0, 0.0, 1.0))
    info["flatness"] = flatness
    # 白区占比越高、内部越平，增强越强。滑条是上限。
    auto = float(np.clip((ratio - 0.05) / 0.25, 0.0, 1.0))
    amount = (strength / 100.0) * (0.35 + 0.65 * auto) * (0.45 + 0.55 * flatness)
    if amount < 0.04:
        return bgr, info

    soft = cv2.GaussianBlur(white.astype(np.float32), (0, 0), 2.5)
    blend = np.clip(soft * (strength / 100.0) * (0.4 + 0.6 * auto), 0.0, 1.0)

    clip = 1.2 + 7.0 * amount
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
    l_eq = clahe.apply(l).astype(np.float32)
    detail = lf - cv2.GaussianBlur(lf, (0, 0), 1.1)
    l_enh = l_eq + detail * (3.0 * amount)

    if depth_m is not None and depth_m.shape[:2] == lf.shape[:2]:
        relief, edge = _depth_cues(depth_m)
        if relief is not None:
            info["depth"] = True
            # 近处大约每 3cm 提亮 24 级；深度台阶处压一条暗边，方便轮廓从泡沫里出来。
            l_enh = l_enh + np.clip(relief, -0.04, 0.04) * (800.0 * amount)
            edge_w = np.clip((edge - 0.003) / 0.012, 0.0, 1.0)
            l_enh = l_enh - edge_w * (40.0 * amount)

    l_out = lf * (1.0 - blend) + np.clip(l_enh, 0.0, 255.0) * blend
    chroma_gain = 1.0 + 1.6 * blend
    a_out = 128.0 + (af - 128.0) * chroma_gain
    b_out = 128.0 + (bf - 128.0) * chroma_gain
    out = cv2.merge(
        [
            np.clip(l_out, 0, 255).astype(np.uint8),
            np.clip(a_out, 0, 255).astype(np.uint8),
            np.clip(b_out, 0, 255).astype(np.uint8),
        ]
    )
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR), info


def _depth_cues(depth_m: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """relief > 0 表示比周围更近。D405 有效距离大约 7cm–80cm。"""
    d = depth_m.astype(np.float32)
    valid = (d > 0.07) & (d < 0.80)
    if float(valid.mean()) < 0.05:
        return None, None
    weight = valid.astype(np.float32)
    filled = np.where(valid, d, 0.0).astype(np.float32)
    blur_d = cv2.GaussianBlur(filled, (0, 0), 11)
    blur_w = cv2.GaussianBlur(weight, (0, 0), 11)
    local = blur_d / np.maximum(blur_w, 1e-3)
    relief = np.zeros_like(d)
    relief[valid] = (local - d)[valid]
    dense = np.where(valid, d, local).astype(np.float32)
    gx = cv2.Sobel(dense, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dense, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    edge[~valid] = 0
    return relief, edge


def to_image_xy(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    """左右两幅并排时，把窗口坐标折回单幅画面。"""
    if x >= width:
        x -= width
    x = int(np.clip(x, 0, width - 1))
    y = int(np.clip(y, 0, height - 1))
    return x, y


def paint(bgr: np.ndarray, instances: list[tuple[np.ndarray, float]], point, box, rubber) -> np.ndarray:
    vis = bgr.copy()
    for i, (mask, _score) in enumerate(instances):
        if mask.shape[:2] != vis.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        color = np.array(COLORS_BGR[i % len(COLORS_BGR)], dtype=np.float32)
        vis[mask] = (0.55 * vis[mask].astype(np.float32) + 0.45 * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, COLORS_BGR[i % len(COLORS_BGR)], 2)
    if box is not None:
        x0, y0, x1, y1 = box
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
    if rubber is not None:
        x0, y0, x1, y1 = rubber
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 255), 1)
    if point is not None:
        cv2.drawMarker(vis, point, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
    return vis


def draw_status(
    bgr: np.ndarray,
    lines: list[str],
    font: ImageFont.ImageFont | None,
    *,
    at_bottom: bool = False,
) -> np.ndarray:
    if font is None:
        y = bgr.shape[0] - 8 - 26 * (len(lines) - 1) if at_bottom else 28
        for line in lines:
            cv2.putText(bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y += 26
        return bgr
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    y = image.height - 8 - 26 * len(lines) if at_bottom else 8
    for line in lines:
        draw.text((12, y), line, font=font, fill=(0, 0, 0))
        draw.text((10, y - 1), line, font=font, fill=(255, 255, 255))
        y += 26
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def save_outputs(
    out_dir: Path,
    original: np.ndarray,
    adjusted: np.ndarray,
    vis: np.ndarray,
    instances: list[tuple[np.ndarray, float]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(str(out_dir / f"{stamp}_rgb.jpg"), original)
    cv2.imwrite(str(out_dir / f"{stamp}_balanced.jpg"), adjusted)
    cv2.imwrite(str(out_dir / f"{stamp}_overlay.jpg"), vis)
    if not instances:
        print(f"已保存 {stamp}（无 mask）", flush=True)
        return
    union = np.zeros(original.shape[:2], dtype=np.uint8)
    for i, (mask, _score) in enumerate(instances):
        m = mask.astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / f"{stamp}_mask_{i:02d}.png"), m)
        union[mask] = 255
    cv2.imwrite(str(out_dir / f"{stamp}_mask_union.png"), union)
    print(f"已保存 {stamp}，{len(instances)} 个实例 -> {out_dir}", flush=True)


def stdin_prompt_loop(prompt_q: queue.Queue, stop: threading.Event) -> None:
    print("文字提示：在本终端输入后回车。例如 bottle、cup、a black block。输入 clear 可清除。", flush=True)
    while not stop.is_set():
        line = sys.stdin.readline()
        if line == "":
            return
        text = line.strip()
        if text:
            prompt_q.put(text)


def main() -> None:
    args = parse_args()
    try:
        import pyrealsense2  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "当前环境没有 pyrealsense2。请在 sam3 环境里执行：pip install pyrealsense2"
        ) from exc

    font = ImageFont.truetype(FONT_PATH, 22) if Path(FONT_PATH).is_file() else None
    segmenter = Segmenter(args.checkpoint, args.device, args.confidence)

    pipeline, profile, align, depth_scale = start_d405(args.width, args.height, args.fps, args.serial)
    print("device:", pyrealsense_camera_name(profile.get_device()), flush=True)

    prompt_q: queue.Queue[str] = queue.Queue()
    stop_stdin = threading.Event()
    threading.Thread(target=stdin_prompt_loop, args=(prompt_q, stop_stdin), daemon=True).start()

    mode = "text" if args.prompt.strip() else "idle"
    prompt = args.prompt.strip()
    point = None
    box = None
    follow = not args.no_follow
    paused = False
    instances: list[tuple[np.ndarray, float]] = []
    infer_ms = 0.0
    mouse = {"down": None, "cur": None, "dragging": False}
    rubber = None
    view = {"w": args.width, "h": args.height}
    tone = {"strength": 80}

    def on_mouse(event, x, y, _flags, _param):
        nonlocal mode, prompt, point, box, instances
        ix, iy = to_image_xy(x, y, view["w"], view["h"])
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["down"] = (ix, iy)
            mouse["cur"] = (ix, iy)
            mouse["dragging"] = True
        elif event == cv2.EVENT_MOUSEMOVE and mouse["dragging"]:
            mouse["cur"] = (ix, iy)
        elif event == cv2.EVENT_LBUTTONUP and mouse["down"] is not None:
            mouse["dragging"] = False
            x0, y0 = mouse["down"]
            mouse["down"] = None
            if abs(ix - x0) < 8 and abs(iy - y0) < 8:
                mode, prompt, point, box = "point", "", (ix, iy), None
            else:
                box_xyxy = (min(x0, ix), min(y0, iy), max(x0, ix), max(y0, iy))
                mode, prompt, point, box = "box", "", None, box_xyxy
            instances = []

    def on_strength(pos: int) -> None:
        tone["strength"] = int(pos)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("strength", WINDOW, tone["strength"], 100, on_strength)
    cv2.setMouseCallback(WINDOW, on_mouse)

    try:
        while True:
            bgr, depth_m = latest_frames(pipeline, align, depth_scale)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            view["w"], view["h"] = w, h
            adjusted, tone_info = enhance_white_on_white(bgr, tone["strength"], depth_m)

            while True:
                try:
                    text = prompt_q.get_nowait()
                except queue.Empty:
                    break
                if text.lower() == "clear":
                    mode, prompt, point, box, instances = "idle", "", None, None, []
                else:
                    mode, prompt, point, box = "text", text, None, None
                    instances = []
                    print(f"文字提示 -> {prompt}", flush=True)

            rubber = None
            if mouse["dragging"] and mouse["down"] and mouse["cur"]:
                x0, y0 = mouse["down"]
                x1, y1 = mouse["cur"]
                rubber = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

            if not paused and mode != "idle" and not mouse["dragging"]:
                rgb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
                t0 = time.perf_counter()
                if mode == "text":
                    instances = segmenter.segment_text(rgb, prompt)
                elif mode == "point" and point is not None:
                    px = int(np.clip(point[0], 0, w - 1))
                    py = int(np.clip(point[1], 0, h - 1))
                    instances = segmenter.segment_point(rgb, px, py)
                    if follow and instances:
                        center = mask_centroid(instances[0][0])
                        if center is not None:
                            point = center
                elif mode == "box" and box is not None:
                    instances = segmenter.segment_box(rgb, box)
                    if follow and instances:
                        center = mask_centroid(instances[0][0])
                        if center is not None:
                            mode, point, box = "point", center, None
                infer_ms = (time.perf_counter() - t0) * 1000.0

            left = paint(bgr, instances, point, box, rubber)
            right = paint(adjusted, instances, point, box, rubber)
            left = draw_status(left, ["原图"], font)
            depth_tag = "深度开" if tone_info["depth"] else "无深度"
            right = draw_status(
                right,
                [f"白区 {tone_info['ratio'] * 100:.0f}%  平坦 {tone_info['flatness']:.2f}  {depth_tag}"],
                font,
            )
            vis = np.hstack([left, right])
            if mode == "text":
                target = f"文字：{prompt}   实例 {len(instances)}   阈值 {segmenter.processor.confidence_threshold:.2f}"
            elif mode == "point":
                target = f"点选 {point}   跟随 {'开' if follow else '关'}"
            elif mode == "box":
                target = f"框选 {box}"
            else:
                target = "左右都能点选或拖框；终端输入文字后回车"
            status = target + f"   白区增强 {tone['strength']}   {infer_ms:.0f} ms"
            if paused:
                status = "已暂停   " + status
            vis = draw_status(
                vis,
                [status, "q 退出   c 清除   s 保存   f 跟随   space 暂停   [ ] 阈值   - = 白区增强   0 关闭"],
                font,
                at_bottom=True,
            )
            cv2.imshow(WINDOW, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"),):
                break
            if key in (ord("c"), 27):
                mode, prompt, point, box, instances = "idle", "", None, None, []
            elif key == ord("s"):
                save_outputs(Path(args.output_dir), bgr, adjusted, vis, instances)
            elif key in (ord("-"), ord("_")):
                tone["strength"] = max(0, tone["strength"] - 5)
                cv2.setTrackbarPos("strength", WINDOW, tone["strength"])
                print(f"白区增强 {tone['strength']}", flush=True)
            elif key in (ord("="), ord("+")):
                tone["strength"] = min(100, tone["strength"] + 5)
                cv2.setTrackbarPos("strength", WINDOW, tone["strength"])
                print(f"白区增强 {tone['strength']}", flush=True)
            elif key == ord("0"):
                tone["strength"] = 0
                cv2.setTrackbarPos("strength", WINDOW, 0)
                print("已关闭白区增强", flush=True)
            elif key == ord("f"):
                follow = not follow
                print("跟随", "开" if follow else "关", flush=True)
            elif key == ord(" "):
                paused = not paused
            elif key == ord("["):
                value = segmenter.set_confidence(segmenter.processor.confidence_threshold - 0.05)
                print(f"置信度阈值 {value:.2f}", flush=True)
            elif key == ord("]"):
                value = segmenter.set_confidence(segmenter.processor.confidence_threshold + 0.05)
                print(f"置信度阈值 {value:.2f}", flush=True)
    finally:
        stop_stdin.set()
        pipeline.stop()
        cv2.destroyAllWindows()


def pyrealsense_camera_name(device) -> str:
    import pyrealsense2 as rs

    name = device.get_info(rs.camera_info.name)
    serial = device.get_info(rs.camera_info.serial_number)
    return f"{name}  serial={serial}"


if __name__ == "__main__":
    main()
的