# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

"""
SAM3 Dense Tracking Model Builder.

This module provides builders for SAM3 dense tracking models, combining SAM2 and SAM3
components for video object tracking and segmentation.
"""

import os
from typing import Optional

import torch
import torch.nn as nn

from sam3.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderLayerv2,
    TransformerEncoderCrossAttention,
)
from sam3.model.encoder import (
    TransformerEncoderFusion,
    TransformerEncoderLayer,
    TransformerEncoderLayerSimple,
)
from sam3.model.geometry_encoders import SequenceGeometryEncoder
from sam3.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from sam3.model.memory import (
    CXBlock,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
)
from sam3.model.model_misc import DotProductScoring, MLP, TransformerWrapper
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam3_demo_dense_tracking_multigpu import Sam3DenseTrackingDemoMultiGPU
from sam3.model.sam3_image import Sam3ImageOnVideoMultiGPU
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.video_tracking_with_prompt_demo import Sam3TrackerPredictor
from sam3.model.vitdet import ViT
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.sam_original.transformer import RoPEAttention

# Core SAM3 imports
from .model.model_misc import MultiheadAttentionWrapper as MultiheadAttention


# Setup TensorFloat-32 for Ampere GPUs if available
def _setup_tf32() -> None:
    """Enable TensorFloat-32 for Ampere GPUs if available."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


_setup_tf32()


class Sam2Predictor(nn.Module):
    """
    Wrapper for SAM2 predictor model.

    This class wraps the SAM2 video tracking model to provide a consistent
    interface for dense tracking applications.
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize SAM2 predictor wrapper.

        Args:
            model: The underlying SAM2 model
        """
        super().__init__()
        self.model = model
        self.per_obj_inference = False

    def forward(self, *args, **kwargs):
        """Forward pass is not implemented - use predictor APIs instead."""
        raise NotImplementedError(
            "Use the sam2 predictor APIs instead. Check Sam3TrackerPredictor class for details."
        )

    def __getattr__(self, name):
        model = super().__getattr__("model")
        if name == "model":
            return model
        return getattr(model, name)


def _create_sam2_components():
    """Create SAM2 model components."""
    # Position encoding for mask memory backbone
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )

    # Mask processing components
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152]
    )

    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )

    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)

    maskmem_backbone = SimpleMaskEncoder(
        out_dim=64,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )

    return maskmem_backbone


def _create_sam2_transformer():
    """Create SAM2 transformer components."""
    # Self attention
    self_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=False,
        use_rope_real=False,
    )

    # Cross attention
    cross_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=False,
        use_rope_real=False,
    )

    # Encoder layer
    encoder_layer = TransformerDecoderLayerv2(
        cross_attention_first=False,
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=self_attention,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )

    # Encoder
    encoder = TransformerEncoderCrossAttention(
        remove_cross_attention_layers=[],
        batch_first=True,
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
    )

    # Transformer wrapper
    transformer = TransformerWrapper(
        encoder=encoder,
        decoder=None,
        d_model=256,
    )

    return transformer


def build_sam2_model() -> Sam2Predictor:
    """
    Build SAM2 model for video tracking.

    Returns:
        Sam2Predictor: Wrapped SAM2 model for video tracking
    """

    # Create model components
    maskmem_backbone = _create_sam2_components()
    transformer = _create_sam2_transformer()

    # Create the main SAM2 model
    model = Sam3TrackerPredictor(
        image_size=1008,
        num_maskmem=7,
        backbone=None,
        backbone_stride=14,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        # SAM parameters
        multimask_output_in_sam=True,
        # Evaluation
        forward_backbone_per_frame_for_eval=True,
        trim_past_non_cond_mem_for_eval=False,
        # Multimask
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        # Additional settings
        always_start_from_first_ann_frame=False,
        # Mask overlap
        non_overlap_masks_for_mem_enc=False,
        non_overlap_masks_for_output=False,
        max_cond_frames_in_attn=4,
        offload_output_to_cpu_for_eval=False,
        # SAM decoder settings
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        clear_non_cond_mem_around_input=True,
        fill_hole_area=0,
    )

    return Sam2Predictor(model)


def _create_sam3_visual_backbone() -> Sam3DualViTDetNeck:
    """Create SAM3 visual backbone with ViT and neck."""
    # Position encoding
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )

    # ViT backbone
    vit_backbone = ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.0,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode="default",
    )

    # Visual neck
    return Sam3DualViTDetNeck(
        trunk=vit_backbone,
        position_encoding=position_encoding,
        d_model=256,
        neck_norm=None,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
    )


def _create_sam3_text_encoder(bpe_path: str) -> VETextEncoder:
    """Create SAM3 text encoder."""
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer,
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )


def _create_sam3_transformer() -> TransformerWrapper:
    """Create SAM3 transformer encoder and decoder."""
    # Encoder components
    encoder_self_attention = MultiheadAttention(
        num_heads=8, dropout=0.1, embed_dim=256, batch_first=True
    )
    encoder_cross_attention = MultiheadAttention(
        num_heads=8, dropout=0.1, embed_dim=256, batch_first=True
    )

    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=encoder_self_attention,
        cross_attention=encoder_cross_attention,
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )

    # Decoder components
    decoder_cross_attention = MultiheadAttention(
        num_heads=8, dropout=0.1, embed_dim=256
    )

    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=decoder_cross_attention,
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
    )

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


def _create_sam3_segmentation_head() -> UniversalSegmentationHead:
    """Create SAM3 segmentation head."""
    dot_product_scorer_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )

    dot_product_scorer = DotProductScoring(
        d_model=256, d_proj=256, prompt_mlp=dot_product_scorer_mlp
    )

    cross_attend_prompt = MultiheadAttention(num_heads=8, dropout=0, embed_dim=256)

    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3, interpolation_mode="nearest", hidden_dim=256
    )

    return UniversalSegmentationHead(
        presence_head=True,
        dot_product_scorer=dot_product_scorer,
        act_ckpt=True,
        upsampling_stages=3,
        hidden_dim=256,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )


def _create_sam3_geometry_encoder() -> SequenceGeometryEncoder:
    """Create SAM3 geometry encoder."""
    input_geom_pos_enc = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )

    input_geom_self_attention = MultiheadAttention(
        num_heads=8, dropout=0.1, embed_dim=256, batch_first=False
    )

    input_geom_layer = TransformerEncoderLayerSimple(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=input_geom_self_attention,
    )

    return SequenceGeometryEncoder(
        pos_enc=input_geom_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=input_geom_layer,
    )


def build_sam3_dense_tracking_model(
    bpe_path: str, checkpoint_path: Optional[str] = None
) -> Sam3DenseTrackingDemoMultiGPU:
    """
    Build SAM3 dense tracking model.

    Args:
        bpe_path: Path to the BPE tokenizer file
        checkpoint_path: Optional path to checkpoint file

    Returns:
        Sam3DenseTrackingDemoMultiGPU: The instantiated dense tracking model
    """
    # Build SAM2 model
    sam2_model = build_sam2_model()

    # Create SAM3 components
    visual_neck = _create_sam3_visual_backbone()
    text_encoder = _create_sam3_text_encoder(bpe_path)
    backbone = SAM3VLBackbone(scalp=1, visual=visual_neck, text=text_encoder)
    transformer = _create_sam3_transformer()
    segmentation_head = _create_sam3_segmentation_head()
    input_geometry_encoder = _create_sam3_geometry_encoder()

    # Create main dot product scoring
    main_dot_prod_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    main_dot_prod_scoring = DotProductScoring(
        d_model=256, d_proj=256, prompt_mlp=main_dot_prod_mlp
    )

    # Create SAM3 model
    sam3_model = Sam3ImageOnVideoMultiGPU(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=input_geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=main_dot_prod_scoring,
    )

    # Create the main dense tracking model
    model = Sam3DenseTrackingDemoMultiGPU(
        sam2_model=sam2_model,
        sam3_model=sam3_model,
        ckpt_path=None,
        score_threshold_detection=0.5,
        assoc_iou_thresh=0.1,
        det_nms_thresh=0.1,
        new_det_thresh=0.7,
        num_interactive_steps_val=0,
        hotstart_delay=15,
        hotstart_unmatch_thresh=8,
        hotstart_dup_thresh=8,
        compile_model=False,  # Set to True for faster inference with torch.compile
        sam3_ckpt_path=None,
        image_size=1008,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )

    # Load checkpoint if provided
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    return model
