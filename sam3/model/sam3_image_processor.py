# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe
from typing import Dict, List

import numpy as np
import PIL
import torch
from sam3.model import box_ops
from sam3.model.data_misc import FindStage, interpolate
from sam3.perflib.masks_ops import mask_iou
from torchvision.transforms import v2


class Sam3Processor:
    """ """

    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5, fuse_detections_iou_threshold=None):
        """
        Args:
            model: The SAM3 model
            resolution: Image resolution for processing
            device: Device to run on ('cuda' or 'cpu')
            confidence_threshold: Minimum score to keep a detection (default: 0.5)
            fuse_detections_iou_threshold: IoU threshold for fusing overlapping detections.
                If None (default), fusion is disabled. Set to a value (e.g., 0.3) to enable fusion.
        """
        self.model = model
        self.resolution = resolution
        self.device = device
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.confidence_threshold = confidence_threshold
        self.fuse_detections_iou_threshold = fuse_detections_iou_threshold

        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    @torch.inference_mode()
    def set_image(self, image, state=None):
        """Sets the image on which we want to do predictions."""
        if state is None:
            state = {}

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        image = v2.functional.to_image(image).to(self.device)
        image = self.transform(image).unsqueeze(0)

        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_image_batch(self, images: List[np.ndarray], state=None):
        """Sets the image batch on which we want to do predictions."""
        if state is None:
            state = {}

        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or tensors")
        assert len(images) > 0, "Images list must not be empty"
        assert isinstance(images[0], PIL.Image.Image), (
            "Images must be a list of PIL images"
        )

        state["original_heights"] = [image.height for image in images]
        state["original_widths"] = [image.width for image in images]

        images = [
            self.transform(v2.functional.to_image(image).to(self.device))
            for image in images
        ]
        images = torch.stack(images, dim=0)
        state["backbone_out"] = self.model.backbone.forward_image(images)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict):
        """Sets the text prompt and run the inference"""

        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            if key in state:
                del state[key]

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks"""
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            # we need to filter the boxes again
            # In principle we could do this more efficiently since we would only need
            # to rerun the heads. But this is simpler and not too inefficient
            return self._forward_grounding(state)
        return state

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict):
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        # Apply detection fusion if enabled (merges overlapping detections)
        if self.fuse_detections_iou_threshold is not None and len(out_probs) > 0:
            out_probs, out_masks, boxes = self._fuse_detections(
                out_probs, out_masks, boxes, self.fuse_detections_iou_threshold
            )

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state

    def _fuse_detections(
        self,
        scores: torch.Tensor,
        masks: torch.Tensor,
        boxes: torch.Tensor,
        iou_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fuse overlapping detections by grouping them and merging masks.
        
        Args:
            scores: (N,) tensor of detection scores
            masks: (N, 1, H, W) tensor of mask logits (before thresholding)
            boxes: (N, 4) tensor of bounding boxes in [x0, y0, x1, y1] format
            iou_threshold: IoU threshold for grouping detections (detections with IoU > threshold are fused)
        
        Returns:
            Fused scores, masks, and boxes tensors
        """
        if len(scores) == 0:
            return scores, masks, boxes
        
        # Convert masks to binary for IoU computation
        masks_binary = (masks.squeeze(1) > 0.5)  # (N, H, W)
        
        # Compute pairwise IoU matrix
        ious = mask_iou(masks_binary, masks_binary)  # (N, N)
        
        # Find connected components based on IoU threshold
        # Use Union-Find to group overlapping detections
        parent = list(range(len(scores)))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                # Merge into the group with higher score
                if scores[px] < scores[py]:
                    px, py = py, px
                parent[py] = px
        
        # Group detections that overlap above threshold
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                if ious[i, j] > iou_threshold:
                    union(i, j)
        
        # Find unique groups
        groups = {}
        for i in range(len(scores)):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # Merge each group
        fused_scores = []
        fused_masks = []
        fused_boxes = []
        
        for group_indices in groups.values():
            if len(group_indices) == 1:
                # Single detection, keep as is - ensure shape is (1, 1, H, W) to match fused masks
                fused_scores.append(scores[group_indices[0]])
                single_mask = masks[group_indices[0]]  # (1, H, W) from (N, 1, H, W)
                if single_mask.dim() == 3:
                    single_mask = single_mask.unsqueeze(0)  # (1, 1, H, W)
                fused_masks.append(single_mask)
                fused_boxes.append(boxes[group_indices[0]])
            else:
                # Multiple detections to fuse
                group_masks_binary = masks_binary[group_indices]  # (K, H, W)
                group_scores = scores[group_indices]
                
                # Merge masks: union of all masks in the group
                merged_mask_binary = group_masks_binary.any(dim=0)  # (H, W)
                
                # Use the mask logits (before thresholding) and take max for merged regions
                group_masks_logits = masks[group_indices].squeeze(1)  # (K, H, W)
                merged_mask_logits = group_masks_logits.max(dim=0)[0]  # (H, W)
                # Set merged regions to high confidence
                merged_mask_logits = torch.where(
                    merged_mask_binary,
                    torch.clamp(merged_mask_logits, min=0.5),
                    merged_mask_logits
                )
                merged_mask_logits = merged_mask_logits.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                
                # Compute bounding box from merged mask
                merged_mask_for_box = merged_mask_binary.unsqueeze(0).float()  # (1, H, W)
                merged_box = box_ops.masks_to_boxes(merged_mask_for_box)[0]  # (4,)
                
                # Use max score from the group
                max_score = group_scores.max()
                
                fused_scores.append(max_score)
                fused_masks.append(merged_mask_logits)
                fused_boxes.append(merged_box)
        
        if len(fused_scores) == 0:
            return scores, masks, boxes
        
        fused_scores = torch.stack(fused_scores)
        fused_masks = torch.cat(fused_masks, dim=0)
        fused_boxes = torch.stack(fused_boxes)
        
        return fused_scores, fused_masks, fused_boxes
