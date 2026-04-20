# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import os
from typing import Optional

import pkg_resources
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from iopath.common.file_io import g_pathmgr
from sam3.model.decoder import (
    DecoupledTransformerDecoderLayerv2,
    SimpleRoPEAttention,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderLayerv2,
    TransformerEncoderCrossAttention,
    TransformerEncoderDecoupledCrossAttention,
)
from sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3.model.geometry_encoders import SequenceGeometryEncoder
from sam3.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from sam3.model.memory import (
    CXBlock,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
)
from sam3.model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from sam3.model.multiplex_utils import MultiplexController
from sam3.model.necks import Sam3DualViTDetNeck, Sam3TriViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model.sam3_image import Sam3Image, Sam3ImageOnVideoMultiGPU
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.video_tracking_multiplex import VideoTrackingDynamicMultiplex
from sam3.model.vitdet import ViT
from sam3.model.vl_combiner import SAM3VLBackbone, SAM3VLBackboneTri, TriHeadVisionOnly
from sam3.sam.transformer import RoPEAttention


# Setup TensorFloat-32 for Ampere GPUs if available
def _setup_tf32() -> None:
    """Enable TensorFloat-32 for Ampere GPUs if available."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


_setup_tf32()


def _create_position_encoding(precompute_resolution=None):
    """Create position encoding for visual backbone."""
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone(compile_mode=None, use_fa3=False, use_rope_real=False):
    """Create ViT backbone for visual feature extraction."""
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
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
        compile_mode=compile_mode,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )


def _create_vit_neck(position_encoding, vit_backbone, enable_inst_interactivity=False):
    """Create ViT neck for feature pyramid."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _create_vl_backbone(vit_neck, text_encoder):
    """Create visual-language backbone."""
    return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)


def _create_transformer_encoder(use_fa3=False) -> TransformerEncoderFusion:
    """Create transformer encoder with its layer."""
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
            use_fa3=use_fa3,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
            use_fa3=use_fa3,
        ),
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
    return encoder


def _create_transformer_decoder(use_fa3=False) -> TransformerDecoder:
    """Create transformer decoder with its layer."""
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            use_fa3=use_fa3,
        ),
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
        use_act_checkpoint=True,
        presence_token=True,
    )
    return decoder


def _create_dot_product_scoring():
    """Create dot product scoring module."""
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def _create_segmentation_head(compile_mode=None, use_fa3=False):
    """Create segmentation head with pixel decoder."""
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode=compile_mode,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,
        use_fa3=use_fa3,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head


def _create_geometry_encoder():
    """Create geometry encoder with all its components."""
    # Create position encoding for geometry encoder
    geo_pos_enc = _create_position_encoding()
    # Create CX block for fuser
    cx_block = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    # Create geometry encoder layer
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    # Create geometry encoder
    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder


def _create_sam3_model(
    backbone,
    transformer,
    input_geometry_encoder,
    segmentation_head,
    dot_prod_scoring,
    inst_interactive_predictor,
    eval_mode,
):
    """Create the SAM3 image model."""
    common_params = {
        "backbone": backbone,
        "transformer": transformer,
        "input_geometry_encoder": input_geometry_encoder,
        "segmentation_head": segmentation_head,
        "num_feature_levels": 1,
        "o2m_mask_predict": True,
        "dot_prod_scoring": dot_prod_scoring,
        "use_instance_query": False,
        "multimask_output": True,
        "inst_interactive_predictor": inst_interactive_predictor,
    }

    matcher = None
    if not eval_mode:
        from sam3.train.matcher import BinaryHungarianMatcherV2

        matcher = BinaryHungarianMatcherV2(
            focal=True,
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            alpha=0.25,
            gamma=2,
            stable=False,
        )
    common_params["matcher"] = matcher
    model = Sam3Image(**common_params)

    return model


def _create_tracker_maskmem_backbone():
    """Create the SAM3 Tracker memory encoder."""
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


def _create_tracker_transformer():
    """Create the SAM3 Tracker transformer components."""
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


def build_tracker(
    apply_temporal_disambiguation: bool, with_backbone: bool = False, compile_mode=None
) -> Sam3TrackerPredictor:
    """
    Build the SAM3 Tracker module for video tracking.

    Returns:
        Sam3TrackerPredictor: Wrapped SAM3 Tracker module
    """

    # Create model components
    maskmem_backbone = _create_tracker_maskmem_backbone()
    transformer = _create_tracker_transformer()
    backbone = None
    if with_backbone:
        vision_backbone = _create_vision_backbone(compile_mode=compile_mode)
        backbone = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)
    # Create the Tracker module
    model = Sam3TrackerPredictor(
        image_size=1008,
        num_maskmem=7,
        backbone=backbone,
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
        use_memory_selection=apply_temporal_disambiguation,
    )

    return model


def _create_text_encoder(bpe_path: str) -> VETextEncoder:
    """Create SAM3 text encoder."""
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer,
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )


def _create_vision_backbone(
    compile_mode=None, enable_inst_interactivity=True
) -> Sam3DualViTDetNeck:
    """Create SAM3 visual backbone with ViT and neck."""
    # Position encoding
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    # ViT backbone
    vit_backbone: ViT = _create_vit_backbone(compile_mode=compile_mode)
    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    # Visual neck
    return vit_neck


def _create_sam3_transformer(
    has_presence_token: bool = True, use_fa3: bool = False
) -> TransformerWrapper:
    """Create SAM3 transformer encoder and decoder."""
    encoder: TransformerEncoderFusion = _create_transformer_encoder(use_fa3=use_fa3)
    decoder: TransformerDecoder = _create_transformer_decoder(use_fa3=use_fa3)

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


def _load_checkpoint(model, checkpoint_path):
    """Load model checkpoint from file."""
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    sam3_image_ckpt = {
        k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
    }
    if model.inst_interactive_predictor is not None:
        sam3_image_ckpt.update(
            {
                k.replace("tracker.", "inst_interactive_predictor.model."): v
                for k, v in ckpt.items()
                if "tracker" in k
            }
        )
    missing_keys, _ = model.load_state_dict(sam3_image_ckpt, strict=False)
    if len(missing_keys) > 0:
        print(
            f"loaded {checkpoint_path} and found "
            f"missing and/or unexpected keys:\n{missing_keys=}"
        )


def _setup_device_and_mode(model, device, eval_mode):
    """Setup model device and evaluation mode."""
    if device == "cuda":
        model = model.cuda()
    if eval_mode:
        model.eval()
    return model


def build_sam3_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
):
    """
    Build SAM3 image model

    Args:
        bpe_path: Path to the BPE tokenizer vocabulary
        device: Device to load the model on ('cuda' or 'cpu')
        eval_mode: Whether to set the model to evaluation mode
        checkpoint_path: Optional path to model checkpoint
        enable_segmentation: Whether to enable segmentation head
        enable_inst_interactivity: Whether to enable instance interactivity (SAM 1 task)
        compile_mode: To enable compilation, set to "default"

    Returns:
        A SAM3 image model
    """
    if bpe_path is None:
        bpe_path = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        )

    # Create visual components
    compile_mode = "default" if compile else None
    vision_encoder = _create_vision_backbone(
        compile_mode=compile_mode, enable_inst_interactivity=enable_inst_interactivity
    )

    # Create text components
    text_encoder = _create_text_encoder(bpe_path)

    # Create visual-language backbone
    backbone = _create_vl_backbone(vision_encoder, text_encoder)

    # Create transformer components
    transformer = _create_sam3_transformer()

    # Create dot product scoring
    dot_prod_scoring = _create_dot_product_scoring()

    # Create segmentation head if enabled
    segmentation_head = (
        _create_segmentation_head(compile_mode=compile_mode)
        if enable_segmentation
        else None
    )

    # Create geometry encoder
    input_geometry_encoder = _create_geometry_encoder()
    if enable_inst_interactivity:
        sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
        inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
    else:
        inst_predictor = None
    # Create the SAM3 model
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        inst_predictor,
        eval_mode,
    )
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf(version="sam3")
    # Load checkpoint if provided
    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path)

    # Setup device and mode
    model = _setup_device_and_mode(model, device, eval_mode)

    return model


def download_ckpt_from_hf(version="sam3"):
    """Download model checkpoint from HuggingFace Hub.

    Args:
        version: "sam3" or "sam3.1"
    """
    if version == "sam3.1":
        repo_id = "facebook/sam3.1"
        ckpt_name = "sam3.1_multiplex.pt"
        cfg_name = "config.json"
    else:
        repo_id = "facebook/sam3"
        ckpt_name = "sam3.pt"
        cfg_name = "config.json"
    _ = hf_hub_download(repo_id=repo_id, filename=cfg_name)
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
    return checkpoint_path


def build_sam3_video_model(
    checkpoint_path: Optional[str] = None,
    load_from_HF=True,
    bpe_path: Optional[str] = None,
    has_presence_token: bool = True,
    geo_encoder_use_img_cross_attn: bool = True,
    strict_state_dict_loading: bool = True,
    apply_temporal_disambiguation: bool = True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compile=False,
) -> Sam3VideoInferenceWithInstanceInteractivity:
    """
    Build SAM3 dense tracking model.

    Args:
        checkpoint_path: Optional path to checkpoint file
        bpe_path: Path to the BPE tokenizer file

    Returns:
        Sam3VideoInferenceWithInstanceInteractivity: The instantiated dense tracking model
    """
    if bpe_path is None:
        bpe_path = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        )

    # Build Tracker module
    tracker = build_tracker(apply_temporal_disambiguation=apply_temporal_disambiguation)

    # Build Detector components
    visual_neck = _create_vision_backbone()
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackbone(scalp=1, visual=visual_neck, text=text_encoder)
    transformer = _create_sam3_transformer(has_presence_token=has_presence_token)
    segmentation_head: UniversalSegmentationHead = _create_segmentation_head()
    input_geometry_encoder = _create_geometry_encoder()

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

    # Build Detector module
    detector = Sam3ImageOnVideoMultiGPU(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=input_geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=main_dot_prod_scoring,
        supervise_joint_box_scores=has_presence_token,
    )

    # Build the main SAM3 video model
    if apply_temporal_disambiguation:
        model = Sam3VideoInferenceWithInstanceInteractivity(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.5,
            assoc_iou_thresh=0.1,
            det_nms_thresh=0.1,
            new_det_thresh=0.7,
            hotstart_delay=15,
            hotstart_unmatch_thresh=8,
            hotstart_dup_thresh=8,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=16,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=1008,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            compile_model=compile,
        )
    else:
        # a version without any heuristics for ablation studies
        model = Sam3VideoInferenceWithInstanceInteractivity(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.5,
            assoc_iou_thresh=0.1,
            det_nms_thresh=0.1,
            new_det_thresh=0.7,
            hotstart_delay=0,
            hotstart_unmatch_thresh=0,
            hotstart_dup_thresh=0,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=0,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=1008,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            compile_model=compile,
        )

    # Load checkpoint if provided
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf(version="sam3")
    if checkpoint_path is not None:
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        missing_keys, unexpected_keys = model.load_state_dict(
            ckpt, strict=strict_state_dict_loading
        )
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    model.to(device=device)
    return model


def build_sam3_video_predictor(*model_args, gpus_to_use=None, **model_kwargs):
    return Sam3VideoPredictorMultiGPU(
        *model_args, gpus_to_use=gpus_to_use, **model_kwargs
    )


def _create_multiplex_maskmem_backbone(multiplex_count=16):
    """Create the multiplex memory encoder with per-object mask channels."""
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )

    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3,
        stride=2,
        padding=1,
        interpol_size=[1152, 1152],
        multiplex_count=multiplex_count,
        starting_out_chan=4,
        input_channel_multiplier=2,
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
        out_dim=256,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )

    return maskmem_backbone


def _create_multiplex_transformer(use_fa3=False, use_rope_real=False):
    """Create the decoupled transformer for multiplex memory attention."""
    self_attention_rope = SimpleRoPEAttention(
        d_model=256,
        num_heads=8,
        dropout_p=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )

    cross_attention_rope = SimpleRoPEAttention(
        d_model=256,
        num_heads=8,
        dropout_p=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )

    encoder_layer = DecoupledTransformerDecoderLayerv2(
        activation="gelu",
        d_model=256,
        num_heads=8,
        dropout=0.1,
        dim_feedforward=2048,
        pos_enc_at_attn=False,
        pre_norm=True,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        self_attention_rope=self_attention_rope,
        cross_attention_rope=cross_attention_rope,
    )

    encoder = TransformerEncoderDecoupledCrossAttention(
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        use_image_in_output=False,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
        batch_first=True,
    )

    transformer = TransformerWrapper(
        encoder=encoder,
        decoder=None,
        d_model=256,
    )

    return transformer


def _create_multiplex_tri_backbone(
    compile_mode=None, use_fa3=False, use_rope_real=False
):
    """Create the TriHead vision backbone for multiplex model."""
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    vit_backbone = _create_vit_backbone(
        compile_mode=compile_mode, use_fa3=use_fa3, use_rope_real=use_rope_real
    )
    tri_neck = Sam3TriViTDetNeck(
        trunk=vit_backbone,
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0],
    )
    return tri_neck


def build_sam3_multiplex_video_model(
    checkpoint_path: Optional[str] = None,
    load_from_HF=True,
    multiplex_count: int = 16,
    use_fa3: bool = False,
    use_rope_real: bool = False,
    strict_state_dict_loading: bool = True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compile=False,
):
    """
    Build SAM3 multiplex video tracking model.

    Args:
        checkpoint_path: Optional path to checkpoint file
        multiplex_count: Number of objects per multiplex bucket
        use_fa3: Whether to use FlashAttention 3
        use_rope_real: Whether to use real-valued RoPE (for compile compat)
        strict_state_dict_loading: Whether to use strict state dict loading
        device: Device to place model on
        compile: Whether to compile model components

    Returns:
        VideoTrackingDynamicMultiplex: The instantiated multiplex tracking model
    """
    # Build multiplex-specific components
    maskmem_backbone = _create_multiplex_maskmem_backbone(
        multiplex_count=multiplex_count
    )
    transformer = _create_multiplex_transformer(
        use_fa3=use_fa3, use_rope_real=use_rope_real
    )
    tri_neck = _create_multiplex_tri_backbone(
        compile_mode="max-autotune" if compile else None
    )
    backbone = TriHeadVisionOnly(
        visual=tri_neck,
        n_features=256,
        scalp=0,
    )
    multiplex_controller = MultiplexController(
        multiplex_count=multiplex_count,
        eval_multiplex_count=multiplex_count,
    )

    # Build the multiplex model (use demo class for init_state and other demo methods)
    from sam3.model.video_tracking_multiplex_demo import Sam3VideoTrackingMultiplexDemo

    model = Sam3VideoTrackingMultiplexDemo(
        backbone=backbone,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        multiplex_controller=multiplex_controller,
        image_size=1008,
        backbone_stride=14,
        num_maskmem=7,
        # Multiplex-specific settings
        use_high_res_features_in_sam=True,
        use_obj_ptrs_in_encoder=True,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=True,
        use_mlp_for_obj_ptr_proj=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        use_no_obj_ptr=True,
        use_linear_no_obj_ptr=True,
        no_obj_embed_spatial=True,
        sincos_tpos_enc=True,
        # Multimask settings
        multimask_output_in_sam=True,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_multimask_token_for_obj_ptr=True,
        num_multimask_outputs=3,
        # Memory encoder settings
        apply_sigmoid_to_mask_logits_for_mem_enc=True,
        sigmoid_scale_for_mem_enc=2.0,
        sigmoid_bias_for_mem_enc=-1.0,
        non_overlap_masks_for_mem_enc=False,
        # Suppression/conditional embeddings
        add_output_suppression_embeddings=True,
        add_object_conditional_embeddings=False,
        condition_as_mask_input=True,
        condition_as_mask_input_fg=1.0,
        condition_as_mask_input_bg=0.0,
        # Memory settings
        use_maskmem_tpos_v2=True,
        save_image_features=True,
        randomness_fix=True,
        # Interaction settings
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        iou_prediction_use_sigmoid=False,
        forward_backbone_per_frame_for_eval=True,
        offload_output_to_cpu_for_eval=False,
        trim_past_non_cond_mem_for_eval=False,
        max_cond_frames_in_attn=4,
        # Dynamic multiplex settings
        is_dynamic_model=True,
        # SAM mask decoder extra args
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        compile_all_components=compile,
        use_memory_selection=False,
    )

    # Load checkpoint if provided
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf(version="sam3.1")
    if checkpoint_path is not None:
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        missing_keys, unexpected_keys = model.load_state_dict(
            ckpt, strict=strict_state_dict_loading
        )
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    model.to(device=device)
    return model


def build_sam3_multiplex_video_predictor(
    checkpoint_path: Optional[str] = None,
    bpe_path: Optional[str] = None,
    max_num_objects: int = 16,
    multiplex_count: int = 16,
    use_fa3: bool = True,
    use_rope_real: bool = True,
    compile: bool = False,
    warm_up: bool = False,
    session_expiration_sec: int = 1200,
    default_output_prob_thresh: float = 0.5,
    async_loading_frames: bool = True,
):
    """
    Build a fully-initialized Sam3MultiplexVideoPredictor.

    This is the recommended entry point for SAM 3.1 multiplex video tracking.
    It builds the full model stack (tracker + detector + demo model), loads
    the checkpoint, and wraps everything in Sam3MultiplexVideoPredictor with
    handle_request / handle_stream_request API.

    Args:
        checkpoint_path: Path to the merged multiplex checkpoint
        bpe_path: Path to the BPE tokenizer vocabulary
        max_num_objects: Maximum number of tracked objects
        multiplex_count: Number of objects per multiplex bucket
        use_fa3: Whether to use FlashAttention 3
        use_rope_real: Whether to use real-valued RoPE (for compile compat)
        compile: Whether to enable torch.compile on model components
        warm_up: Whether to run warm-up compilation (requires compile=True)
        session_expiration_sec: Session expiration timeout in seconds
        default_output_prob_thresh: Default probability threshold for output masks
        async_loading_frames: Whether to load frames asynchronously

    Returns:
        Sam3MultiplexVideoPredictor: The fully-initialized predictor
    """
    if bpe_path is None:
        bpe_path = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        )

    from sam3.model.sam3_multiplex_base import Sam3MultiplexPredictorWrapper
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.sam3_multiplex_tracking import (
        Sam3MultiplexTrackingWithInteractivity,
    )
    from sam3.model.sam3_multiplex_video_predictor import Sam3MultiplexVideoPredictor

    # Build tracker
    tracker_model = build_sam3_multiplex_video_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        multiplex_count=multiplex_count,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
        compile=False,
        strict_state_dict_loading=False,
    )
    del tracker_model.backbone
    tracker_model.backbone = None

    sam2_predictor = Sam3MultiplexPredictorWrapper(
        model=tracker_model,
        per_obj_inference=False,
        fill_hole_area=0,
        is_multiplex=True,
        is_multiplex_dynamic=True,
    )

    # Build detector
    tri_neck = _create_multiplex_tri_backbone(
        compile_mode=None, use_fa3=use_fa3, use_rope_real=use_rope_real
    )
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackboneTri(scalp=0, visual=tri_neck, text=text_encoder)
    transformer = _create_sam3_transformer(use_fa3=use_fa3)
    segmentation_head = _create_segmentation_head(use_fa3=use_fa3)
    geometry_encoder = _create_geometry_encoder()
    dot_prod_scoring = _create_dot_product_scoring()

    detector = Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=dot_prod_scoring,
        supervise_joint_box_scores=True,
        is_multiplex=True,
    )

    # Assemble demo model
    demo_model = Sam3MultiplexTrackingWithInteractivity(
        tracker=sam2_predictor,
        detector=detector,
        score_threshold_detection=0.4,
        det_nms_thresh=0.1,
        det_nms_use_iom=True,
        assoc_iou_thresh=0.1,
        new_det_thresh=0.65,
        hotstart_delay=15,
        hotstart_unmatch_thresh=8,
        hotstart_dup_thresh=8,
        suppress_unmatched_only_within_hotstart=False,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
        suppress_det_close_to_boundary=True,
        fill_hole_area=0,  # OV effectively 0 (Sam3MultiplexTrackerPredictor Hydra override clobbers yaml's 16)
        recondition_every_nth_frame=16,
        use_iom_recondition=True,
        iom_thresh_recondition=0.5,
        masklet_confirmation_enable=True,
        reconstruction_bbox_iou_thresh=-1,
        reconstruction_bbox_det_score=0.8,
        max_num_objects=max_num_objects,
        postprocess_batch_size=16,
        use_batched_grounding=True,
        batched_grounding_batch_size=16,
        max_num_kboxes=0,
        sprinkle_removal_area=0,
        is_multiplex=True,
        image_size=1008,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        compile_model=compile,
    )

    # Load checkpoint (auto-download from HuggingFace if not provided)
    if checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf(version="sam3.1")
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        # Remap checkpoint keys if needed (internal naming -> OSS naming)
        # HF checkpoints are already remapped; local checkpoints may use old naming
        needs_remap = any(
            k.startswith("sam3_model.") or k.startswith("sam2_predictor.") for k in ckpt
        )
        if needs_remap:
            remapped_ckpt = {}
            for k, v in ckpt.items():
                new_k = k
                if k.startswith("sam3_model."):
                    new_k = "detector." + k[len("sam3_model.") :]
                elif k.startswith("sam2_predictor."):
                    new_k = "tracker." + k[len("sam2_predictor.") :]
                remapped_ckpt[new_k] = v
            ckpt = remapped_ckpt
        missing_keys, unexpected_keys = demo_model.load_state_dict(ckpt, strict=False)
        if missing_keys:
            print(f"Missing keys ({len(missing_keys)}): {missing_keys[:10]}...")
        if unexpected_keys:
            print(
                f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}..."
            )

    demo_model.cuda().eval()

    # Wrap in predictor
    predictor = Sam3MultiplexVideoPredictor(
        model=demo_model,
        session_expiration_sec=session_expiration_sec,
        default_output_prob_thresh=default_output_prob_thresh,
        async_loading_frames=async_loading_frames,
        warm_up=warm_up,
    )
    return predictor


def build_sam3_predictor(
    checkpoint_path: Optional[str] = None,
    bpe_path: Optional[str] = None,
    version: str = "sam3.1",  # "sam3" or "sam3.1"
    compile: bool = False,
    warm_up: bool = False,
    # SAM 3.1 specific
    max_num_objects: int = 16,
    multiplex_count: int = 16,
    # Common
    use_fa3: bool = True,
    use_rope_real: bool = True,
    async_loading_frames: bool = True,
    **kwargs,
):
    """
    Build a SAM3 video predictor.

    Args:
        checkpoint_path: Path to model checkpoint
        bpe_path: Path to BPE tokenizer vocabulary
        version: Model version - "sam3" for base or "sam3.1" for multiplex
        compile: Enable torch.compile for ~2x speedup (SAM 3.1 only currently)
        warm_up: Run warm-up compilation passes
        max_num_objects: Maximum tracked objects (SAM 3.1 only)
        multiplex_count: Objects per multiplex bucket (SAM 3.1 only)
        use_fa3: Use Flash Attention 3
        use_rope_real: Use real-valued RoPE
        async_loading_frames: Load video frames asynchronously
        **kwargs: Additional arguments passed to the underlying builder

    Returns:
        A predictor with handle_request() and handle_stream_request() API.
        Both versions support: start_session, add_prompt, propagate_in_video,
        remove_object, reset_session, close_session.

    Example:
        # SAM 3.1 (auto-downloads from HuggingFace):
        predictor = build_sam3_predictor(version="sam3.1", compile=True)

        # SAM 3 (auto-downloads from HuggingFace):
        predictor = build_sam3_predictor(version="sam3")

        # Or with a local checkpoint:
        predictor = build_sam3_predictor(checkpoint_path="path/to/ckpt.pt", version="sam3.1")

        # Both use the same API:
        response = predictor.handle_request({"type": "start_session", "resource_path": video_dir})
        session_id = response["session_id"]
        predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": "person"})
        for out in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
            masks = out["out_binary_masks"]
    """
    if version == "sam3.1":
        return build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
            max_num_objects=max_num_objects,
            multiplex_count=multiplex_count,
            use_fa3=use_fa3,
            use_rope_real=use_rope_real,
            compile=compile,
            warm_up=warm_up,
            async_loading_frames=async_loading_frames,
            **kwargs,
        )
    elif version == "sam3":
        return build_sam3_video_predictor(
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
            compile=compile,
            async_loading_frames=async_loading_frames,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown version: {version!r}. Use 'sam3' or 'sam3.1'.")
