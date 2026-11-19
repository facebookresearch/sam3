import fnmatch
import logging
from typing import Iterable, Optional, Sequence


def unwrap_model(model):
    """Unwrap common training and inference wrappers."""
    previous = None
    current = model
    while previous is not current:
        previous = current
        if hasattr(current, "module"):
            current = current.module
        if hasattr(current, "detector"):
            current = current.detector
    return current


def _set_module_requires_grad(module, component_name: str, requires_grad: bool) -> int:
    changed = 0
    for param in module.parameters():
        if param.requires_grad != requires_grad:
            param.requires_grad = requires_grad
            changed += 1

    if changed > 0:
        action = "Unfrozen" if requires_grad else "Frozen"
        logging.info("%s %s (%d parameter tensors)", action, component_name, changed)
    return changed


def _resolve_module_path(root, path: str):
    current = root
    for attr in path.split("."):
        if not hasattr(current, attr):
            return None
        current = getattr(current, attr)
    return current


def _normalize_block_indices(
    block_indices: Optional[Sequence[int]], num_blocks: int
) -> Iterable[int]:
    if not block_indices:
        return []

    normalized = []
    for index in block_indices:
        actual_index = index if index >= 0 else num_blocks + index
        if 0 <= actual_index < num_blocks:
            normalized.append(actual_index)
        else:
            logging.warning(
                "Skipping out-of-range vision block index %s for backbone with %d blocks",
                index,
                num_blocks,
            )
    return normalized


def _apply_param_patterns(model, patterns, requires_grad: bool) -> int:
    if not patterns:
        return 0

    changed = 0
    matched_patterns = {pattern: 0 for pattern in patterns}
    for name, param in model.named_parameters():
        for pattern in patterns:
            if fnmatch.fnmatch(name, pattern):
                matched_patterns[pattern] += 1
                if param.requires_grad != requires_grad:
                    param.requires_grad = requires_grad
                    changed += 1

    for pattern, match_count in matched_patterns.items():
        if match_count == 0:
            logging.warning("No parameters matched freezing pattern '%s'", pattern)

    if changed > 0:
        action = "Unfrozen" if requires_grad else "Frozen"
        logging.info("%s parameters via patterns (%d parameter tensors)", action, changed)
    return changed


def _freeze_named_modules(model, module_paths, requires_grad: bool) -> int:
    if not module_paths:
        return 0

    changed = 0
    for module_path in module_paths:
        module = _resolve_module_path(model, module_path)
        if module is None:
            logging.warning("Module path '%s' was not found on model", module_path)
            continue
        changed += _set_module_requires_grad(module, module_path, requires_grad)
    return changed


def _get_vision_backbone(model):
    if not hasattr(model, "backbone"):
        logging.warning("Model '%s' does not expose a 'backbone' attribute", type(model).__name__)
        return None
    if not hasattr(model.backbone, "vision_backbone"):
        logging.warning("Model backbone does not expose 'vision_backbone'")
        return None
    return model.backbone.vision_backbone


def freeze_vision_backbone(
    model,
    freeze_entire_backbone: bool = False,
    freeze_layers: Optional[Sequence[int]] = None,
    unfreeze_layers: Optional[Sequence[int]] = None,
    freeze_patch_embed: bool = False,
    freeze_ln_pre: bool = False,
) -> int:
    model = unwrap_model(model)
    vision_backbone = _get_vision_backbone(model)
    if vision_backbone is None:
        return 0

    changed = 0
    if freeze_entire_backbone:
        changed += _set_module_requires_grad(
            vision_backbone, "backbone.vision_backbone", False
        )

    trunk = getattr(vision_backbone, "trunk", None)
    blocks = getattr(trunk, "blocks", None)
    if blocks is not None:
        if freeze_layers or unfreeze_layers:
            logging.info("Vision backbone exposes %d transformer blocks", len(blocks))
        for index in _normalize_block_indices(freeze_layers, len(blocks)):
            changed += _set_module_requires_grad(
                blocks[index], f"backbone.vision_backbone.trunk.blocks.{index}", False
            )
        for index in _normalize_block_indices(unfreeze_layers, len(blocks)):
            changed += _set_module_requires_grad(
                blocks[index], f"backbone.vision_backbone.trunk.blocks.{index}", True
            )
    elif freeze_layers or unfreeze_layers:
        logging.warning("Vision backbone does not expose trunk.blocks for layer freezing")

    if freeze_patch_embed and trunk is not None and hasattr(trunk, "patch_embed"):
        changed += _set_module_requires_grad(
            trunk.patch_embed, "backbone.vision_backbone.trunk.patch_embed", False
        )
    elif freeze_patch_embed:
        logging.warning("Vision backbone does not expose trunk.patch_embed")

    if freeze_ln_pre and trunk is not None and hasattr(trunk, "ln_pre"):
        changed += _set_module_requires_grad(
            trunk.ln_pre, "backbone.vision_backbone.trunk.ln_pre", False
        )
    elif freeze_ln_pre:
        logging.warning("Vision backbone does not expose trunk.ln_pre")

    return changed


def freeze_language_backbone(model) -> int:
    model = unwrap_model(model)
    module = _resolve_module_path(model, "backbone.language_backbone")
    if module is None:
        logging.warning("Model does not expose backbone.language_backbone")
        return 0
    return _set_module_requires_grad(module, "backbone.language_backbone", False)


def freeze_geometry_encoder(model) -> int:
    model = unwrap_model(model)
    module = getattr(model, "geometry_encoder", None)
    if module is None:
        logging.warning("Model does not expose geometry_encoder")
        return 0
    return _set_module_requires_grad(module, "geometry_encoder", False)


def freeze_transformer(model) -> int:
    model = unwrap_model(model)
    module = getattr(model, "transformer", None)
    if module is None:
        logging.warning("Model does not expose transformer")
        return 0
    return _set_module_requires_grad(module, "transformer", False)


def freeze_transformer_encoder_only(model) -> int:
    model = unwrap_model(model)
    module = _resolve_module_path(model, "transformer.encoder")
    if module is None:
        logging.warning("Model does not expose transformer.encoder")
        return 0
    return _set_module_requires_grad(module, "transformer.encoder", False)


def freeze_transformer_decoder_only(model) -> int:
    model = unwrap_model(model)
    module = _resolve_module_path(model, "transformer.decoder")
    if module is None:
        logging.warning("Model does not expose transformer.decoder")
        return 0
    return _set_module_requires_grad(module, "transformer.decoder", False)


def freeze_vision_neck(model) -> int:
    model = unwrap_model(model)
    vision_backbone = _get_vision_backbone(model)
    if vision_backbone is None:
        return 0

    changed = 0
    convs = getattr(vision_backbone, "convs", None)
    if convs is not None:
        changed += _set_module_requires_grad(convs, "backbone.vision_backbone.convs", False)
    else:
        logging.warning("Vision backbone does not expose convs")

    sam2_convs = getattr(vision_backbone, "sam2_convs", None)
    if sam2_convs is not None:
        changed += _set_module_requires_grad(
            sam2_convs, "backbone.vision_backbone.sam2_convs", False
        )
    return changed


def freeze_scoring_heads(model) -> int:
    model = unwrap_model(model)
    changed = 0
    head_paths = [
        "dot_prod_scoring",
        "instance_dot_prod_scoring",
        "class_embed",
        "instance_class_embed",
    ]
    for head_path in head_paths:
        module = _resolve_module_path(model, head_path)
        if module is not None:
            changed += _set_module_requires_grad(module, head_path, False)

    if changed == 0:
        logging.warning("Model does not expose any known scoring heads")
    return changed


def freeze_segmentation_head(model) -> int:
    model = unwrap_model(model)
    module = getattr(model, "segmentation_head", None)
    if module is None:
        logging.warning("Model does not expose segmentation_head")
        return 0
    return _set_module_requires_grad(module, "segmentation_head", False)


def log_trainable_parameter_summary(model) -> None:
    model = unwrap_model(model)

    trainable_params = 0
    total_params = 0
    component_stats = {}

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

        component = name.split(".")[0] if "." in name else name
        if component not in component_stats:
            component_stats[component] = {"total": 0, "trainable": 0}
        component_stats[component]["total"] += param.numel()
        if param.requires_grad:
            component_stats[component]["trainable"] += param.numel()

    frozen_params = total_params - trainable_params
    trainable_pct = 100.0 * trainable_params / total_params if total_params else 0.0

    logging.info("=" * 80)
    logging.info("Model parameter statistics")
    logging.info("  Total parameters:     %15s", f"{total_params:,}")
    logging.info("  Trainable parameters: %15s", f"{trainable_params:,}")
    logging.info("  Frozen parameters:    %15s", f"{frozen_params:,}")
    logging.info("  Trainable percentage: %14.2f%%", trainable_pct)
    logging.info("=" * 80)

    if not component_stats:
        return

    logging.info("Per-component parameter breakdown")
    for component, stats in sorted(
        component_stats.items(), key=lambda item: item[1]["total"], reverse=True
    ):
        pct = 100.0 * stats["trainable"] / stats["total"] if stats["total"] else 0.0
        logging.info(
            "  %-24s trainable=%15s total=%15s pct=%6.1f",
            component,
            f"{stats['trainable']:,}",
            f"{stats['total']:,}",
            pct,
        )


def apply_freezing_from_config(model, freeze_config) -> int:
    model = unwrap_model(model)
    strategy = freeze_config.get("strategy", "none")
    total_changed = 0

    logging.info("=" * 80)
    logging.info("Applying freezing configuration")
    logging.info("Strategy: %s", strategy)
    logging.info("=" * 80)

    if strategy == "freeze_encoder":
        total_changed += freeze_vision_backbone(model, freeze_entire_backbone=True)
        total_changed += freeze_language_backbone(model)
    elif strategy == "freeze_vision_only":
        total_changed += freeze_vision_backbone(model, freeze_entire_backbone=True)
    elif strategy == "freeze_language_only":
        total_changed += freeze_language_backbone(model)
    elif strategy == "freeze_all_backbones":
        total_changed += freeze_vision_backbone(model, freeze_entire_backbone=True)
        total_changed += freeze_language_backbone(model)
        total_changed += freeze_geometry_encoder(model)
    elif strategy == "freeze_encoder_keep_decoder":
        total_changed += freeze_vision_backbone(model, freeze_entire_backbone=True)
        total_changed += freeze_language_backbone(model)
        total_changed += freeze_transformer_encoder_only(model)
    elif strategy == "freeze_everything_except_heads":
        total_changed += freeze_vision_backbone(model, freeze_entire_backbone=True)
        total_changed += freeze_language_backbone(model)
        total_changed += freeze_geometry_encoder(model)
        total_changed += freeze_transformer(model)
    elif strategy != "none":
        logging.warning("Unknown freezing strategy '%s'", strategy)

    freeze_layers = freeze_config.get("freeze_vision_layers")
    unfreeze_layers = freeze_config.get("unfreeze_vision_layers")
    if (
        freeze_config.get("freeze_vision_backbone", False)
        or freeze_layers
        or unfreeze_layers
        or freeze_config.get("freeze_vision_patch_embed", False)
        or freeze_config.get("freeze_vision_ln_pre", False)
    ):
        total_changed += freeze_vision_backbone(
            model,
            freeze_entire_backbone=freeze_config.get("freeze_vision_backbone", False),
            freeze_layers=freeze_layers,
            unfreeze_layers=unfreeze_layers,
            freeze_patch_embed=freeze_config.get("freeze_vision_patch_embed", False),
            freeze_ln_pre=freeze_config.get("freeze_vision_ln_pre", False),
        )

    if freeze_config.get("freeze_language_backbone", False):
        total_changed += freeze_language_backbone(model)

    if freeze_config.get("freeze_geometry_encoder", False):
        total_changed += freeze_geometry_encoder(model)

    if freeze_config.get("freeze_transformer", False):
        total_changed += freeze_transformer(model)

    if freeze_config.get("freeze_transformer_encoder", False):
        total_changed += freeze_transformer_encoder_only(model)

    if freeze_config.get("freeze_transformer_decoder", False):
        total_changed += freeze_transformer_decoder_only(model)

    if freeze_config.get("freeze_vision_neck", False):
        total_changed += freeze_vision_neck(model)

    if freeze_config.get("freeze_scoring_head", False):
        total_changed += freeze_scoring_heads(model)

    if freeze_config.get("freeze_segmentation_head", False):
        total_changed += freeze_segmentation_head(model)

    total_changed += _freeze_named_modules(
        model, freeze_config.get("freeze_modules"), requires_grad=False
    )
    total_changed += _freeze_named_modules(
        model, freeze_config.get("unfreeze_modules"), requires_grad=True
    )
    total_changed += _apply_param_patterns(
        model, freeze_config.get("freeze_param_patterns"), requires_grad=False
    )
    total_changed += _apply_param_patterns(
        model, freeze_config.get("unfreeze_param_patterns"), requires_grad=True
    )

    if total_changed == 0 and strategy == "none":
        logging.info("Freezing disabled; all parameters remain trainable.")
    else:
        logging.info(
            "Finished applying freezing configuration (%d parameter tensors changed)",
            total_changed,
        )
    log_trainable_parameter_summary(model)
    return total_changed
