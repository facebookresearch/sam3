#!/usr/bin/env python3
"""把 SAM3 的「一句文本 + 一张 RGB → 目标 mask」前向导出成 ONNX。

示例（在 conda 环境 sam3 中）::

    python export_sam3_image_onnx.py \\
        --checkpoint /home/andy/data/hugging_face_model/sam3/sam3.pt \\
        --precision bf16

导出的图不接收字符串。调用时先用仓库里的 BPE 得到 token_ids，
再把归一化后的 1008 图像和 token_ids 送进 ONNX。阈值筛选和放大回原图
仍用 postprocess_text_mask，与现在的 sam3_text_mask 一致。

精度：
- bf16：导出时包一层 CUDA autocast，计算与 evaluate_sam3_prompts.py 相同
- fp32：不做 autocast，ViT 第一层 MLP 也用 float32 Linear+GELU，整图保持 float32

权重超过 2GB，会写成同目录下的 .onnx.data。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3.model_builder import build_sam3_image_model  # noqa: E402
from wrapper import (  # noqa: E402
    Sam3TextImageOnnx,
    enable_onnx_friendly_mlp,
    enable_onnx_friendly_rope,
    install_static_box_rpb,
    install_text_only_prompt,
    postprocess_text_mask,
    preprocess_image,
    reference_outputs,
    tokenize_prompts,
)

DEFAULT_CKPT = Path("/home/andy/data/hugging_face_model/sam3/sam3.pt")
DEFAULT_BPE = REPO_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
DEFAULT_OUT = Path(__file__).resolve().parent / "output" / "sam3_image_text.onnx"


def _load_model(checkpoint: Path, bpe: Path) -> torch.nn.Module:
    model = build_sam3_image_model(
        bpe_path=str(bpe),
        device="cuda",
        eval_mode=True,
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    model.eval()
    return model


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _cpu_outputs(outputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(tensor.float().cpu() for tensor in outputs)


def _report(name: str, reference, candidate) -> None:
    mask_diff = _max_abs(reference[0], candidate[0])
    logit_diff = _max_abs(reference[1], candidate[1])
    presence_diff = _max_abs(reference[2], candidate[2])
    print(
        f"{name}: mask_max_abs={mask_diff:.6f} "
        f"logit_max_abs={logit_diff:.6f} "
        f"presence_max_abs={presence_diff:.6f}"
    )
    if not all(np.isfinite(value) for value in (mask_diff, logit_diff, presence_diff)):
        raise RuntimeError(f"{name} 出现 NaN/Inf，停止导出")


def verify_against_pytorch(
    model: torch.nn.Module,
    image: torch.Tensor,
    prompt: str,
    precision: str,
    original_hw: tuple[int, int],
    threshold: float,
) -> Sam3TextImageOnnx:
    """先跑未改动的官方前向，再逐项换上可导出实现并对照。

    官方融合 MLP 固定输出 bf16。没有 autocast 时，下一层 float32 Linear 会对不上 dtype，
    所以 fp32 导出只和官方 bf16 的合并 mask 对照，不再按 fp32 重跑未改动的官方前向。
    """
    token_ids = tokenize_prompts(model, [prompt], image.device)
    reference_precision = "bf16" if precision == "fp32" else precision
    with torch.no_grad():
        baseline = _cpu_outputs(reference_outputs(model, image, prompt, reference_precision))
    print(
        "官方输出形状: "
        f"pred_masks={tuple(baseline[0].shape)} "
        f"pred_logits={tuple(baseline[1].shape)} "
        f"presence_logit={tuple(baseline[2].shape)}"
    )
    torch.cuda.empty_cache()

    if precision == "bf16":
        install_text_only_prompt(model)
        with torch.no_grad():
            geometry = _cpu_outputs(reference_outputs(model, image, prompt, precision))
        _report("文本几何分支", baseline, geometry)
        torch.cuda.empty_cache()

        enable_onnx_friendly_rope(model)
        with torch.no_grad():
            rope = _cpu_outputs(reference_outputs(model, image, prompt, precision))
        _report("实数 RoPE", baseline, rope)
        torch.cuda.empty_cache()
    else:
        print("fp32 导出对照官方 bf16 合并 mask")
        install_text_only_prompt(model)
        enable_onnx_friendly_rope(model)

    enable_onnx_friendly_mlp(model, _compute_dtype(precision))
    exportable = Sam3TextImageOnnx(model, precision=precision).eval()
    with torch.no_grad():
        exported = _cpu_outputs(exportable(image, token_ids))
    _report("token 输入 + 可导出 MLP", baseline, exported)

    height, width = original_hw
    ref_union = postprocess_text_mask(*baseline, height, width, threshold)
    out_union = postprocess_text_mask(*exported, height, width, threshold)
    union_iou = _mask_iou(ref_union, out_union)
    print(f"合并 mask IoU={union_iou:.6f} 阈值={threshold}")
    if union_iou < 0.99:
        raise RuntimeError(
            f"导出前向和官方前向的 mask IoU={union_iou:.4f}，低于 0.99，停止导出"
        )
    return exportable


def _compute_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def export_onnx(
    exportable: Sam3TextImageOnnx,
    image: torch.Tensor,
    token_ids: torch.Tensor,
    output: Path,
    opset: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        exportable(image, token_ids)
    install_static_box_rpb(exportable.model)
    print(f"开始导出 ONNX: {output}")
    program = torch.onnx.export(
        exportable,
        (image, token_ids),
        str(output),
        input_names=["image", "token_ids"],
        output_names=["pred_masks", "pred_logits", "presence_logit"],
        opset_version=opset,
        dynamo=True,
        external_data=True,
        optimize=False,
        do_constant_folding=True,
        fallback=False,
    )
    if program is not None and hasattr(program, "save"):
        program.save(str(output))
    print(f"已写入 {output}")
    if output.with_suffix(".onnx.data").is_file() or any(output.parent.glob("*.onnx.data")):
        data_files = sorted(output.parent.glob("*.onnx.data"))
        for path in data_files:
            print(f"外部权重: {path} ({path.stat().st_size / (1024 ** 3):.2f} GiB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 SAM3 文本+图像 ONNX")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--bpe", type=Path, default=DEFAULT_BPE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--prompt", default="red bottle")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--image", type=Path, default=None, help="可选的 RGB 图片，用于导出前对照")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("导出需要 CUDA。SAM3 的位置编码在构建时写死了 cuda")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.bpe.is_file():
        raise FileNotFoundError(args.bpe)

    print(f"加载权重: {args.checkpoint}")
    model = _load_model(args.checkpoint, args.bpe)

    if args.image is not None and args.image.is_file():
        pil = Image.open(args.image).convert("RGB")
    else:
        sample = REPO_ROOT / "assets" / "images" / "truck.jpg"
        pil = Image.open(sample).convert("RGB")
        if args.prompt == "red bottle":
            args.prompt = "truck"
    original_hw = (pil.height, pil.width)
    image = preprocess_image(pil, "cuda")
    token_ids = tokenize_prompts(model, [args.prompt], "cuda")
    print(
        f"precision={args.precision} image={tuple(image.shape)} "
        f"token_ids={tuple(token_ids.shape)} prompt={args.prompt!r}"
    )

    exportable = None
    if not args.skip_verify:
        exportable = verify_against_pytorch(
            model,
            image,
            args.prompt,
            args.precision,
            original_hw,
            args.threshold,
        )
    else:
        enable_onnx_friendly_rope(model)
        enable_onnx_friendly_mlp(model, _compute_dtype(args.precision))
        install_text_only_prompt(model)
        exportable = Sam3TextImageOnnx(model, precision=args.precision).eval()
    if args.verify_only:
        return

    with torch.no_grad():
        export_onnx(exportable, image, token_ids, args.output, args.opset)


if __name__ == "__main__":
    main()
