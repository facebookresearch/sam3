"""SAM3 文本 + 单张 RGB 的可导出前向。

ONNX 图的输入是固定形状张量，不是 Python 字符串：

- image: float32, [1, 3, 1008, 1008]，与 Sam3Processor 相同的归一化
- token_ids: int64, [1, 32]，由仓库里的 BPE tokenizer 生成

输出是固定的 200 个候选，筛选、放大回原图、合并成一张 mask 留在图外，
与 sam3_text_mask 的后处理一致。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

from sam3.model.data_misc import FindStage

IMAGE_SIZE = 1008


def preprocess_image(image: Image.Image | np.ndarray, device: torch.device | str) -> torch.Tensor:
    """复现 Sam3Processor.set_image 的缩放和归一化，返回 [1, 3, 1008, 1008]。"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    transform = v2.Compose(
        [
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    tensor = v2.functional.to_image(image).to(device)
    return transform(tensor).unsqueeze(0)


def tokenize_prompts(model: torch.nn.Module, prompts: list[str], device: torch.device | str) -> torch.Tensor:
    """用模型自带的 BPE 把文本编成 [batch, 32] 的 token id。"""
    text_encoder = model.backbone.language_backbone
    token_ids = text_encoder.tokenizer(prompts, context_length=text_encoder.context_length)
    return token_ids.to(device=device, dtype=torch.long)


def enable_onnx_friendly_rope(model: torch.nn.Module) -> None:
    """把复数 RoPE 换成等价的实数实现，避免 ONNX 里的 view_as_complex。"""
    for module in model.modules():
        freqs = getattr(module, "freqs_cis", None)
        if freqs is None or not getattr(module, "use_rope", False):
            continue
        if getattr(module, "use_rope_real", False) and hasattr(module, "freqs_cis_real"):
            continue
        module.use_rope_real = True
        module.register_buffer("freqs_cis_real", freqs.real.float(), persistent=False)
        module.register_buffer("freqs_cis_imag", freqs.imag.float(), persistent=False)


def enable_onnx_friendly_mlp(
    model: torch.nn.Module, dtype: torch.dtype = torch.bfloat16
) -> None:
    """替换 ViT MLP 里的 aten._addmm_activation。

    原实现把第一层 Linear+GELU 融合成 bf16 私有算子。这里改成显式的
    Linear + tanh GELU。bf16 对齐该融合核；fp32 导出保持 float32，
    避免图里留下 ONNX Runtime 不能执行的 bfloat16 卷积和矩阵乘。
    """
    del model
    import sam3.model.vitdet as vitdet

    def addmm_act(activation, linear, mat1):
        bias = None if linear.bias is None else linear.bias.to(dtype)
        hidden = F.linear(mat1.to(dtype), linear.weight.to(dtype), bias)
        if activation in (torch.nn.GELU, F.gelu):
            hidden = F.gelu(hidden, approximate="tanh")
        elif activation in (torch.nn.ReLU, F.relu):
            hidden = F.relu(hidden)
        else:
            raise ValueError(f"不支持的 MLP 激活: {activation}")
        return hidden

    vitdet.addmm_act = addmm_act


def _text_only_geometry(geometry_encoder, img_feats, img_pos_embeds):
    """文本提示没有框和点时，几何分支只剩一个 CLS token。

    空序列 concat 无法稳定导出。空框/空点再拼上 CLS 的结果就是这个 CLS，
    随后仍走原来的 final_proj 和 3 层几何编码器。
    """
    memory = img_feats[-1]
    if img_pos_embeds is None:
        position = torch.zeros_like(memory)
    else:
        position = img_pos_embeds[-1]
    batch = memory.shape[1]
    cls = geometry_encoder.cls_embed.weight.view(1, 1, geometry_encoder.d_model)
    cls = cls.repeat(1, batch, 1)
    cls_mask = torch.zeros(batch, 1, dtype=torch.bool, device=memory.device)
    hidden = geometry_encoder.norm(geometry_encoder.final_proj(cls))
    for layer in geometry_encoder.encode:
        hidden = layer(
            tgt=hidden,
            memory=memory,
            tgt_key_padding_mask=cls_mask,
            pos=position,
        )
    hidden = geometry_encoder.encode_norm(hidden)
    return hidden, cls_mask


def _encode_prompt_text_only(
    self,
    backbone_out,
    find_input,
    geometric_prompt,
    visual_prompt_embed=None,
    visual_prompt_mask=None,
    encode_text=True,
    prev_mask_pred=None,
):
    """替换 Sam3Image._encode_prompt，只保留文本 + CLS 几何提示。"""
    del geometric_prompt, visual_prompt_embed, visual_prompt_mask
    if not encode_text:
        raise RuntimeError("这个 ONNX 只覆盖文本提示")
    if prev_mask_pred is not None:
        raise RuntimeError("这个 ONNX 不覆盖交互式 mask 提示")

    txt_ids = find_input.text_ids
    txt_feats = backbone_out["language_features"][:, txt_ids]
    txt_masks = backbone_out["language_mask"][txt_ids]
    backbone_out, img_feats, img_pos_embeds, _ = self._get_img_feats(
        backbone_out, find_input.img_ids
    )
    geo_feats, geo_masks = _text_only_geometry(
        self.geometry_encoder, img_feats, img_pos_embeds
    )
    prompt = torch.cat([txt_feats, geo_feats], dim=0)
    prompt_mask = torch.cat([txt_masks, geo_masks], dim=1)
    return prompt, prompt_mask, backbone_out


def install_text_only_prompt(model: torch.nn.Module) -> None:
    model._encode_prompt = _encode_prompt_text_only.__get__(model, model.__class__)


def install_static_box_rpb(model: torch.nn.Module) -> None:
    """把 box RPB 的特征图尺寸固定成 Python int。

    官方代码用 spatial_shapes 张量元素和缓存比较，导出时会变成 data-dependent
    的 .item()。图像尺寸固定为 1008，这个尺寸在一次 eager 前向之后是已知的。
    """
    decoder = model.transformer.decoder
    stored = decoder.compilable_stored_size
    if stored is None or decoder.compilable_cord_cache is None:
        raise RuntimeError("box RPB 缓存是空的，导出前需要先跑一次 PyTorch 前向")
    height, width = int(stored[0]), int(stored[1])
    decoder.compilable_stored_size = (height, width)
    original = decoder._get_rpb_matrix

    def _get_rpb_matrix(reference_boxes, feat_size):
        del feat_size
        return original(reference_boxes, (height, width))

    decoder._get_rpb_matrix = _get_rpb_matrix


def make_find_stage(device: torch.device | str) -> FindStage:
    return FindStage(
        img_ids=torch.tensor([0], device=device, dtype=torch.long),
        text_ids=torch.tensor([0], device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )


def encode_token_ids(model: torch.nn.Module, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """token id → (language_features [S, B, 256], language_mask [B, S])。

    language_mask 里 True 表示 padding，与 VETextEncoder 一致。
    """
    text_encoder = model.backbone.language_backbone
    _pooled, text_memory = text_encoder.encoder(token_ids)
    text_memory = text_encoder.resizer(text_memory.transpose(0, 1))
    text_mask = token_ids.eq(0)
    return text_memory, text_mask


class Sam3TextImageOnnx(torch.nn.Module):
    """固定形状的文本分割前向，供 torch.onnx.export 使用。"""

    def __init__(self, model: torch.nn.Module, precision: str = "bf16"):
        super().__init__()
        if precision not in ("fp32", "bf16"):
            raise ValueError(f"precision 必须是 fp32 或 bf16，收到 {precision}")
        self.model = model
        self.precision = precision

    def forward(self, image: torch.Tensor, token_ids: torch.Tensor):
        if self.precision == "bf16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return self._forward(image, token_ids)
        return self._forward(image, token_ids)

    def _forward(self, image: torch.Tensor, token_ids: torch.Tensor):
        backbone_out = self.model.backbone.forward_image(image)
        language_features, language_mask = encode_token_ids(self.model, token_ids)
        backbone_out["language_features"] = language_features
        backbone_out["language_mask"] = language_mask
        outputs = self.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=make_find_stage(image.device),
            geometric_prompt=self.model._get_dummy_prompt(),
            find_target=None,
        )
        return (
            outputs["pred_masks"],
            outputs["pred_logits"],
            outputs["presence_logit_dec"],
        )


def reference_outputs(model: torch.nn.Module, image: torch.Tensor, prompt: str, precision: str):
    """走官方字符串前向，用来和导出封装对比。"""
    def _run():
        backbone_out = model.backbone.forward_image(image)
        backbone_out.update(model.backbone.forward_text([prompt], device=image.device))
        return model.forward_grounding(
            backbone_out=backbone_out,
            find_input=make_find_stage(image.device),
            geometric_prompt=model._get_dummy_prompt(),
            find_target=None,
        )

    if precision == "bf16":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = _run()
    else:
        outputs = _run()
    return outputs["pred_masks"], outputs["pred_logits"], outputs["presence_logit_dec"]


def postprocess_text_mask(
    pred_masks: torch.Tensor,
    pred_logits: torch.Tensor,
    presence_logit: torch.Tensor,
    original_height: int,
    original_width: int,
    threshold: float,
) -> np.ndarray:
    """把 200 个候选收成一张原图尺寸的 bool mask，对齐 sam3_text_mask。"""
    probs = pred_logits.sigmoid()
    presence = presence_logit.sigmoid().unsqueeze(1)
    scores = (probs * presence).squeeze(-1)[0]
    masks = pred_masks[0]
    keep = scores > threshold
    masks = masks[keep]
    if masks.numel() == 0:
        return np.zeros((original_height, original_width), dtype=bool)
    masks = F.interpolate(
        masks.unsqueeze(1).float(),
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    ).sigmoid()
    union = torch.any(masks[:, 0] > 0.5, dim=0)
    return union.detach().cpu().numpy()
