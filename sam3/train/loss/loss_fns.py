# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import logging
import warnings

import torch
import torch.distributed
import torch.nn.functional as F
import torchmetrics

from sam3.model import box_ops

from sam3.model.data_misc import interpolate

from sam3.train.loss.sigmoid_focal_loss import (
    triton_sigmoid_focal_loss,
    triton_sigmoid_focal_loss_reduce,
)
from torch import nn

from .mask_sampling import (
    calculate_uncertainty,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)


CORE_LOSS_KEY = "core_loss"


def instance_masks_to_semantic_masks(
    instance_masks: torch.Tensor, num_instances: torch.Tensor
) -> torch.Tensor:
    """This function converts instance masks to semantic masks.
    It accepts a collapsed batch of instances masks (ie all instance masks are concatenated in a single tensor) and
    the number of instances in each image of the batch.
    It returns a mask with the same spatial dimensions as the input instance masks, where for each batch element the
    semantic mask is the union of all the instance masks in the batch element.

    If for a given batch element there are no instances (ie num_instances[i]==0), the corresponding semantic mask will be a tensor of zeros.

    Args:
        instance_masks (torch.Tensor): A tensor of shape (N, H, W) where N is the number of instances in the batch.
        num_instances (torch.Tensor): A tensor of shape (B,) where B is the batch size. It contains the number of instances
            in each image of the batch.

    Returns:
        torch.Tensor: A tensor of shape (B, H, W) where B is the batch size and H, W are the spatial dimensions of the
            input instance masks.
    """
    if num_instances.sum() == 0:
        # all negative batch, create a tensor of zeros (B, 1, 1)
        return num_instances.unsqueeze(-1).unsqueeze(-1)

    masks_per_query = torch.split(instance_masks, num_instances.tolist())

    return torch.stack([torch.any(masks, dim=0) for masks in masks_per_query], dim=0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def dice_loss(inputs, targets, num_boxes, loss_on_multimask=False, reduce=True):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    try:
        loss = _dice_loss(inputs, targets, num_boxes, loss_on_multimask, reduce)
    except torch.OutOfMemoryError:
        logging.error("GPU OOM, computing dice loss on CPU")
        # try to recover from GPU OOM by moving tensors to CPU and computing loss there
        orig_device = inputs.device
        inputs = inputs.cpu()
        targets = targets.cpu()
        if isinstance(num_boxes, torch.Tensor):
            num_boxes = num_boxes.cpu()
        loss = _dice_loss(inputs, targets, num_boxes, loss_on_multimask, reduce)
        loss = loss.to(orig_device)

    return loss


def _dice_loss(inputs, targets, num_boxes, loss_on_multimask=False, reduce=True):
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_boxes
    if not reduce:
        return loss
    return loss.sum() / num_boxes


def sigmoid_focal_loss(
    inputs,
    targets,
    num_boxes,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
    reduce=True,
    triton=True,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    if not (0 <= alpha <= 1) and triton:
        raise RuntimeError(f"Alpha should be in [0,1], got {alpha}")
    if triton:
        if reduce and not loss_on_multimask:
            loss = triton_sigmoid_focal_loss_reduce(inputs, targets, alpha, gamma)
            return loss / (num_boxes * inputs.shape[1])

        loss = triton_sigmoid_focal_loss(inputs, targets, alpha, gamma)
    else:
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

    if not reduce:
        return loss

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_boxes  # average over spatial dims
    return loss.mean(1).sum() / num_boxes


def iou_loss(
    inputs, targets, pred_ious, num_boxes, loss_on_multimask=False, use_l1_loss=False
):
    """MSE loss between predicted IoUs and actual IoUs between inputs and targets."""
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_boxes
    return loss.sum() / num_boxes


@torch.jit.script
def _contrastive_align(logits, positive_map):
    positive_logits = -logits.masked_fill(~positive_map, 0)
    negative_logits = logits  # .masked_fill(positive_map, -1000000)

    boxes_with_pos = positive_map.any(2)
    pos_term = positive_logits.sum(2)
    neg_term = negative_logits.logsumexp(2)

    nb_pos = positive_map.sum(2) + 1e-6

    box_to_token_loss = (
        (pos_term / nb_pos + neg_term).masked_fill(~boxes_with_pos, 0).sum()
    )

    tokens_with_pos = positive_map.any(1)
    pos_term = positive_logits.sum(1)
    neg_term = negative_logits.logsumexp(1)

    nb_pos = positive_map.sum(1) + 1e-6

    tokens_to_boxes_loss = (
        (pos_term / nb_pos + neg_term).masked_fill(~tokens_with_pos, 0).sum()
    )
    return (box_to_token_loss + tokens_to_boxes_loss) / 2


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat(
        [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
    )
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


class LossWithWeights(nn.Module):
    def __init__(self, weight_dict, compute_aux, supports_o2m_loss=True):
        super().__init__()
        # weights for each computed loss key (those losses not in weight_dict
        # will not be aggregated in the final reduced core loss)
        self.weight_dict = weight_dict if weight_dict is not None else {}
        # whether this loss will be applied on auxiliary outputs
        self.compute_aux = compute_aux
        self.supports_o2m_loss = supports_o2m_loss
        self.target_keys = []

    def forward(self, *args, is_aux=False, **kwargs):
        if is_aux and not self.compute_aux:
            return {CORE_LOSS_KEY: 0.0}
        losses = self.get_loss(*args, **kwargs)
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def get_loss(self, **kwargs):
        raise NotImplementedError()

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss


class IABCEMdetr(LossWithWeights):
    def __init__(
        self,
        pos_weight,
        weight_dict=None,
        compute_aux=True,
        gamma=0,
        weak_loss=True,
        alpha=0.25,
        pad_n_queries=None,
        pad_scale_pos=1.0,
        use_separate_loss_for_det_and_trk=False,
        num_det_queries=None,
        det_exhaustive_loss_scale_pos=1.0,
        det_exhaustive_loss_scale_neg=1.0,
        det_non_exhaustive_loss_scale_pos=1.0,
        det_non_exhaustive_loss_scale_neg=1.0,
        trk_loss_scale_pos=1.0,
        trk_loss_scale_neg=1.0,
        no_loss_for_fp_propagation=False,
        apply_loss_to_det_queries_in_video_grounding=True,
        use_presence=False,
        use_presence_semgseg=False,  # If True, use presence scores from the semgseg head.
        presence_alpha=0.5,
        presence_gamma=0.0,
        pos_focal: bool = False,  # for box scores, use focal loss for positives as well
    ):
        super().__init__(weight_dict, compute_aux)
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.weak_loss = weak_loss
        self.alpha = alpha
        self.target_keys.append("boxes_xyxy")
        self.no_loss_for_fp_propagation = no_loss_for_fp_propagation
        if self.weak_loss:
            self.target_keys.append("is_exhaustive")
        # NOTE: This is hacky solution to have the same CE loss scale across datasets where the model might predict different number of object queries for different tasks.
        # If not None, we assume there are a total pad_n_queries object queries.
        # For example, if the model predicts only 1 object query and pad_n_queries=100, we pad the predictions with 99 zero preds.
        # Currently this only affects the BCE loss and not the F1 score.
        self.pad_n_queries = pad_n_queries
        self.pad_scale_pos = pad_scale_pos
        if self.pad_scale_pos != 1.0:
            assert self.pad_n_queries is not None
        # whether to use presence scores
        self.use_presence = use_presence
        self.use_presence_semgseg = use_presence_semgseg
        if self.use_presence_semgseg:
            assert self.use_presence
        self.presence_alpha = presence_alpha
        self.presence_gamma = presence_gamma
        self.pos_focal = pos_focal

        # Decoupled loss for detection and tracking queries
        self.apply_loss_to_det_queries_in_video_grounding = (
            apply_loss_to_det_queries_in_video_grounding
        )
        self.use_separate_loss_for_det_and_trk = use_separate_loss_for_det_and_trk
        if num_det_queries is not None:
            logging.warning("note: it's not needed to set num_det_queries anymore")
        if self.use_separate_loss_for_det_and_trk:
            assert not self.weak_loss, "Do not use weak_loss in this case -- set separate loss for detection and tracking queries instead"
            self.det_exhaustive_loss_scale_pos = det_exhaustive_loss_scale_pos
            self.det_exhaustive_loss_scale_neg = det_exhaustive_loss_scale_neg
            self.det_non_exhaustive_loss_scale_pos = det_non_exhaustive_loss_scale_pos
            self.det_non_exhaustive_loss_scale_neg = det_non_exhaustive_loss_scale_neg
            self.trk_loss_scale_pos = trk_loss_scale_pos
            self.trk_loss_scale_neg = trk_loss_scale_neg
        else:
            assert (
                det_exhaustive_loss_scale_pos == 1.0
                and det_exhaustive_loss_scale_neg == 1.0
                and det_non_exhaustive_loss_scale_pos == 1.0
                and det_non_exhaustive_loss_scale_neg == 1.0
                and trk_loss_scale_pos == 1.0
                and trk_loss_scale_neg == 1.0
            ), "If not using separate loss for detection and tracking queries, separate detection and tracking loss scales should all be 1.0"

    def get_loss(self, outputs, targets, indices, num_boxes):
        assert len(outputs["pred_logits"].shape) > 2, "Incorrect predicted logits shape"
        assert outputs["pred_logits"].shape[-1] == 1, "Incorrect predicted logits shape"
        src_logits = outputs["pred_logits"].squeeze(-1)
        prob = src_logits.sigmoid()

        with torch.no_grad():
            target_classes = torch.full(
                src_logits.shape[:2],
                0,
                dtype=torch.float,
                device=src_logits.device,
            )
            target_classes[(indices[0], indices[1])] = 1
            src_boxes_xyxy = outputs["pred_boxes_xyxy"][(indices[0], indices[1])]
            target_boxes_giou = (
                targets["boxes_xyxy"][indices[2]]
                if indices[2] is not None
                else targets["boxes_xyxy"]
            )

            iou = box_ops.fast_diag_box_iou(src_boxes_xyxy, target_boxes_giou)
            t = prob[(indices[0], indices[1])] ** self.alpha * iou ** (1 - self.alpha)
            t = torch.clamp(t, 0.01).detach()
            positive_target_classes = target_classes.clone()
            positive_target_classes[(indices[0], indices[1])] = t

        # Soft loss on positives
        if self.pos_focal:
            loss_bce = sigmoid_focal_loss(
                src_logits.contiguous(),
                positive_target_classes,
                num_boxes=1,
                alpha=0.5,
                gamma=self.gamma,
                reduce=False,
            )
        else:
            loss_bce = F.binary_cross_entropy_with_logits(
                src_logits, positive_target_classes, reduction="none"
            )
        loss_bce = loss_bce * target_classes * self.pos_weight

        if (
            self.pad_n_queries is not None
            and isinstance(self.pad_n_queries, int)
            and loss_bce.size(1) < self.pad_n_queries
        ):
            loss_bce = loss_bce * self.pad_scale_pos
        # Negatives
        loss_bce = loss_bce + F.binary_cross_entropy_with_logits(
            src_logits, target_classes, reduction="none"
        ) * (1 - target_classes) * (prob**self.gamma)

        # Optionally, not applying IABCEMdetr loss to detection queries in video.
        is_video_grounding = outputs.get("is_video_grounding_batch", False)
        if is_video_grounding and not self.apply_loss_to_det_queries_in_video_grounding:
            Q_det = outputs["Q_det"]
            loss_bce[:, :Q_det] *= 0.0
        presence_loss = torch.tensor(0.0, device=src_logits.device)
        presence_dec_acc = torch.tensor(0.0, device=src_logits.device)
        if self.use_presence:
            # no classifiction loss for individual tokens if no target gt
            # cannot directly use targets["num_boxes"] to check if some
            # GT box exists as there may be dummy boxes for "invisible objects"
            # in video grounding data

            gt_padded_object_ids = targets["object_ids_padded"]  # (B, H)
            gt_padded_boxes = targets["boxes_padded"]  # (B, H, 4) shape, CxCyWH
            gt_padded_is_visible = (
                (gt_padded_object_ids >= 0)
                & (gt_padded_boxes[..., 2] > 0)  # width > 0
                & (gt_padded_boxes[..., 3] > 0)  # height > 0
            )
            keep_loss = (gt_padded_is_visible.sum(dim=-1)[..., None] != 0).float()

            loss_bce = loss_bce * keep_loss

            if self.use_presence_semgseg:
                # no loss here, has it's own separate loss computation
                assert "presence_logit_dec" not in outputs
            elif "presence_logit_dec" in outputs:
                presence_logits = outputs["presence_logit_dec"].view_as(keep_loss)
                bs = presence_logits.shape[0]
                presence_loss = sigmoid_focal_loss(
                    presence_logits,
                    keep_loss,
                    # not num_boxes, but we'll use it to normalize by bs
                    num_boxes=bs,
                    alpha=self.presence_alpha,
                    gamma=self.presence_gamma,
                )
                pred = (presence_logits.sigmoid() > 0.5).float()
                presence_dec_acc = (pred == keep_loss).float().mean()
            else:
                # for o2m, nothing to do
                pass

        if self.weak_loss:
            assert not self.use_separate_loss_for_det_and_trk, "Do not use weak_loss in this case -- set separate loss for detection and tracking queries instead"

            # nullify the negative loss for the non-exhaustive classes
            assert loss_bce.shape[0] == targets["is_exhaustive"].shape[0]
            assert targets["is_exhaustive"].ndim == 1

            loss_mask = (~targets["is_exhaustive"]).view(-1, 1).expand_as(loss_bce)
            # restrict the mask to the negative supervision
            loss_mask = loss_mask & (target_classes < 0.5)
            loss_mask = ~loss_mask
            # Mask the loss
            loss_bce = loss_bce * loss_mask.float()
            # Average
            loss_bce = loss_bce.sum() / (loss_mask.sum() + 1e-6)
        else:
            # apply separate loss weights to detection and tracking queries
            if self.use_separate_loss_for_det_and_trk:
                Q_det = outputs["Q_det"]
                assert loss_bce.size(1) >= Q_det
                is_positive = target_classes > 0.5
                is_positive_det = is_positive[:, :Q_det]
                is_positive_trk = is_positive[:, Q_det:]
                assert loss_bce.size(0) == targets["is_exhaustive"].size(0)
                is_exhaustive = targets["is_exhaustive"].unsqueeze(1).bool()
                loss_scales = torch.zeros_like(loss_bce)
                # detection query loss weights
                loss_scales[:, :Q_det] = (
                    (is_exhaustive & is_positive_det).float()
                    * self.det_exhaustive_loss_scale_pos
                    + (is_exhaustive & ~is_positive_det).float()
                    * self.det_exhaustive_loss_scale_neg
                    + (~is_exhaustive & is_positive_det).float()
                    * self.det_non_exhaustive_loss_scale_pos
                    + (~is_exhaustive & ~is_positive_det).float()
                    * self.det_non_exhaustive_loss_scale_neg
                )
                # tracking query weights
                loss_scales[:, Q_det:] = (
                    is_positive_trk.float() * self.trk_loss_scale_pos
                    + (~is_positive_trk).float() * self.trk_loss_scale_neg
                )
                # apply the loss weights

                # if the id is -2 means it is a fp propagation , we don't apply the loss to them
                if self.no_loss_for_fp_propagation:
                    is_original_queries = outputs["pred_old_obj_ids"] != -2
                    loss_scales *= (is_exhaustive | is_original_queries).float()

                loss_bce = loss_bce * loss_scales

            if self.pad_n_queries is None or loss_bce.size(1) >= self.pad_n_queries:
                loss_bce = loss_bce.mean()
            else:
                assert isinstance(self.pad_n_queries, int)
                assert (
                    loss_bce.size(1) < self.pad_n_queries
                ), f"The number of predictions is more than the expected total after padding. Got {loss_bce.size(1)} predictions."
                loss_bce = loss_bce.sum() / (self.pad_n_queries * loss_bce.size(0))

        bce_f1 = torchmetrics.functional.f1_score(
            src_logits.sigmoid().flatten(),
            target=target_classes.flatten().long(),
            task="binary",
        )

        losses = {
            "loss_ce": loss_bce,
            "ce_f1": bce_f1,
            "presence_loss": presence_loss,
            "presence_dec_acc": presence_dec_acc,
        }
        return losses


class Boxes(LossWithWeights):
    def __init__(
        self,
        weight_dict=None,
        compute_aux=True,
        apply_loss_to_det_queries_in_video_grounding=True,
    ):
        super().__init__(weight_dict, compute_aux)
        self.apply_loss_to_det_queries_in_video_grounding = (
            apply_loss_to_det_queries_in_video_grounding
        )
        self.target_keys.extend(["boxes", "boxes_xyxy"])

    def get_loss(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # Optionally, not applying Boxes loss to detection queries in video.
        is_video_grounding = outputs.get("is_video_grounding_batch", False)
        if is_video_grounding and not self.apply_loss_to_det_queries_in_video_grounding:
            indices = _keep_only_trk_queries_in_match_inds(
                indices, Q_det=outputs["Q_det"]
            )

        assert "pred_boxes" in outputs
        # idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][(indices[0], indices[1])]
        src_boxes_xyxy = outputs["pred_boxes_xyxy"][(indices[0], indices[1])]
        target_boxes = (
            targets["boxes"] if indices[2] is None else targets["boxes"][indices[2]]
        )
        target_boxes_giou = (
            targets["boxes_xyxy"]
            if indices[2] is None
            else targets["boxes_xyxy"][indices[2]]
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - box_ops.fast_diag_generalized_box_iou(
            src_boxes_xyxy, target_boxes_giou
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses


class Masks(LossWithWeights):
    def __init__(
        self,
        weight_dict=None,
        compute_aux=False,
        focal_alpha=0.25,
        focal_gamma=2,
        num_sample_points=None,
        oversample_ratio=None,
        importance_sample_ratio=None,
        apply_loss_to_det_queries_in_video_grounding=True,
    ):
        super().__init__(weight_dict, compute_aux)
        if compute_aux:
            warnings.warn("Masks loss usually shouldn't be applied to aux outputs")
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_sample_points = num_sample_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.apply_loss_to_det_queries_in_video_grounding = (
            apply_loss_to_det_queries_in_video_grounding
        )
        self.target_keys.extend(["masks", "is_valid_mask"])

    def _sampled_loss(self, src_masks, target_masks, num_boxes):
        assert len(src_masks.shape) == 3 and len(target_masks.shape) == 3
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            # Sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                calculate_uncertainty,
                self.num_sample_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )

            # get GT labels
            sampled_target_masks = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        sampled_src_masks = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(
                sampled_src_masks,
                sampled_target_masks,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            ),
            "loss_dice": dice_loss(sampled_src_masks, sampled_target_masks, num_boxes),
        }
        # Not needed for backward
        del src_masks
        del target_masks

        return losses

    def get_loss(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        assert "is_valid_mask" in targets
        # Optionally, not applying Masks loss to detection queries in video.
        is_video_grounding = outputs.get("is_video_grounding_batch", False)
        if is_video_grounding and not self.apply_loss_to_det_queries_in_video_grounding:
            indices = _keep_only_trk_queries_in_match_inds(
                indices, Q_det=outputs["Q_det"]
            )

        src_masks = outputs["pred_masks"]

        # Dataset doesn't have segmentation masks
        if targets["masks"] is None:
            return {
                "loss_mask": torch.tensor(0.0, device=src_masks.device),
                "loss_dice": torch.tensor(0.0, device=src_masks.device),
            }

        target_masks = (
            targets["masks"] if indices[2] is None else targets["masks"][indices[2]]
        )
        target_masks = target_masks.to(src_masks)
        keep = (
            targets["is_valid_mask"]
            if indices[2] is None
            else targets["is_valid_mask"][indices[2]]
        )

        src_masks = src_masks[(indices[0], indices[1])]

        # Remove invalid masks from loss
        src_masks = src_masks[keep]
        target_masks = target_masks[keep]

        if self.num_sample_points is not None:
            # Compute loss on sampled points for the Mask
            losses = self._sampled_loss(src_masks, target_masks, num_boxes)

        else:
            # upsample predictions to the target size
            if target_masks.shape[0] == 0 and src_masks.shape[0] == 0:
                src_masks = src_masks.flatten(1)
                target_masks = target_masks.reshape(src_masks.shape)
            else:
                if len(src_masks.shape) == 3:
                    src_masks = src_masks[:, None]
                if src_masks.dtype == torch.bfloat16:
                    # Bilinear interpolation does not support bf16
                    src_masks = src_masks.to(dtype=torch.float32)
                src_masks = interpolate(
                    src_masks,
                    size=target_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                src_masks = src_masks[:, 0].flatten(1)
                target_masks = target_masks.flatten(1)

            losses = {
                "loss_mask": sigmoid_focal_loss(
                    src_masks,
                    target_masks,
                    num_boxes,
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                ),
                "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
            }

        return losses


# class MultiStepIteractiveMasks(LossWithWeights):
#     def __init__(
#         self,
#         weight_dict=None,
#         compute_aux=False,
#         focal_alpha=0.25,
#         focal_gamma=2,
#     ):
#         warnings.warn(
#             "MultiStepIteractiveMasks is deprecated. Please use MultiStepMultiMasksAndIous",
#             DeprecationWarning,
#         )
#         super().__init__(weight_dict, compute_aux)
#         self.focal_alpha = focal_alpha
#         self.focal_gamma = focal_gamma
#         self.target_keys.extend(["masks"])

#     def get_loss(self, outputs, targets, indices, num_boxes):
#         """Compute the losses related to the masks: the focal loss and the dice loss.
#         targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]

#         Unlike `Masks`, here the "multistep_pred_masks" can have multiple channels, each
#         corresponding to one iterative prediction step in SAM-style training. We treat each
#         channel as a mask prediction and sum the loss across channels.
#         """
#         src_masks = outputs["multistep_pred_masks"]
#         target_masks = targets["masks"]
#         assert src_masks.size(0) == target_masks.size(0)
#         assert src_masks.dim() == 4
#         assert target_masks.dim() == 3

#         # tile target_masks according to the number of
#         # channels `src_masks`.
#         num_steps = src_masks.size(1)
#         target_masks = target_masks.unsqueeze(1).to(src_masks.dtype)
#         if num_steps > 1:
#             target_masks = target_masks.repeat(1, num_steps, 1, 1)

#         # resize `src_masks` to target mask resolution
#         if src_masks.shape != target_masks.shape:
#             src_masks = interpolate(
#                 src_masks,
#                 size=target_masks.shape[-2:],
#                 mode="bilinear",
#                 align_corners=False,
#             )
#             assert src_masks.shape == target_masks.shape

#         # flatten the multiple steps in to the batch dimension
#         src_masks = src_masks.flatten(0, 1).flatten(1)
#         target_masks = target_masks.flatten(0, 1).flatten(1)
#         losses = {
#             "loss_mask": sigmoid_focal_loss(
#                 src_masks,
#                 target_masks,
#                 num_boxes,
#                 alpha=self.focal_alpha,
#                 gamma=self.focal_gamma,
#             ),
#             "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
#         }

#         return losses


# class MultiStepMultiMasksAndIous(LossWithWeights):
#     def __init__(
#         self,
#         weight_dict=None,
#         compute_aux=False,
#         focal_alpha=0.25,
#         focal_gamma=2,
#         # if True, back-prop on all predicted ious
#         # not just the one with lowest loss_combo
#         supervise_all_iou=False,
#         # Less slack vs MSE loss in [-1, 1] error range
#         iou_use_l1_loss=False,
#         # Settings for obj score prediction
#         pred_obj_scores=False,
#         focal_gamma_obj_score=0.0,
#         focal_alpha_obj_score=-1,
#     ):
#         super().__init__(weight_dict, compute_aux)
#         self.focal_alpha = focal_alpha
#         self.focal_gamma = focal_gamma
#         self.target_keys.extend(["masks"])
#         assert "loss_mask" in self.weight_dict
#         assert "loss_dice" in self.weight_dict
#         assert "loss_iou" in self.weight_dict
#         if "loss_class" not in self.weight_dict:
#             self.weight_dict["loss_class"] = 0.0
#         self.focal_alpha_obj_score = focal_alpha_obj_score
#         self.focal_gamma_obj_score = focal_gamma_obj_score
#         self.supervise_all_iou = supervise_all_iou
#         self.iou_use_l1_loss = iou_use_l1_loss
#         self.pred_obj_scores = pred_obj_scores

#     def get_loss(self, outputs, targets, indices, num_boxes):
#         """
#         Compute the losses related to the masks: the focal loss and the dice loss.
#         and also the MSE loss between predicted IoUs and actual IoUs.

#         Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
#         of shape [N, M, H, W], where M could be 1 or larger, corresponding to
#         one or multiple predicted masks from a click.

#         We back-propagate focal, dice and iou losses only on the prediction channel
#         with the lowest focal+dice loss between predicted mask and ground-truth.
#         """

#         target_masks = targets["masks"].unsqueeze(1).float()
#         assert target_masks.dim() == 4  # [N, 1, H, W]
#         src_masks_list = outputs["multistep_pred_multimasks_high_res"]
#         ious_list = outputs["multistep_pred_ious"]
#         object_score_logits_list = outputs["multistep_object_score_logits"]

#         assert len(src_masks_list) == len(ious_list)
#         assert len(object_score_logits_list) == len(ious_list)

#         # Remove invalid masks from loss
#         keep = targets["is_valid_mask"]
#         target_masks = target_masks[keep]

#         # accumulate the loss over prediction steps
#         losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
#         for src_masks, ious, object_score_logits in zip(
#             src_masks_list, ious_list, object_score_logits_list
#         ):
#             object_score_logits = object_score_logits[keep]
#             ious = ious[keep]
#             src_masks = src_masks[keep]
#             self._update_losses(
#                 losses, src_masks, target_masks, ious, num_boxes, object_score_logits
#             )
#         return losses

#     def _update_losses(
#         self, losses, src_masks, target_masks, ious, num_boxes, object_score_logits
#     ):
#         target_masks = target_masks.expand_as(src_masks)
#         # get focal, dice and iou loss on all output masks in a prediction step
#         loss_multimask = sigmoid_focal_loss(
#             src_masks,
#             target_masks,
#             num_boxes,
#             alpha=self.focal_alpha,
#             gamma=self.focal_gamma,
#             loss_on_multimask=True,
#             triton=False,  # only use triton if alpha > 0
#         )
#         loss_multidice = dice_loss(
#             src_masks, target_masks, num_boxes, loss_on_multimask=True
#         )
#         if not self.pred_obj_scores:
#             loss_class = torch.tensor(
#                 0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
#             )
#             target_obj = torch.ones(
#                 loss_multimask.shape[0],
#                 1,
#                 dtype=loss_multimask.dtype,
#                 device=loss_multimask.device,
#             )
#         else:
#             target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
#                 ..., None
#             ].float()
#             loss_class = sigmoid_focal_loss(
#                 object_score_logits,
#                 target_obj,
#                 num_boxes,
#                 alpha=self.focal_alpha_obj_score,
#                 gamma=self.focal_gamma_obj_score,
#                 triton=False,
#             )

#         loss_multiiou = iou_loss(
#             src_masks,
#             target_masks,
#             ious,
#             num_boxes,
#             loss_on_multimask=True,
#             use_l1_loss=self.iou_use_l1_loss,
#         )
#         assert loss_multimask.dim() == 2
#         assert loss_multidice.dim() == 2
#         assert loss_multiiou.dim() == 2
#         if loss_multimask.size(1) > 1:
#             # take the mask indices with the smallest focal + dice loss for back propagation
#             loss_combo = (
#                 loss_multimask * self.weight_dict["loss_mask"]
#                 + loss_multidice * self.weight_dict["loss_dice"]
#             )
#             best_loss_inds = torch.argmin(loss_combo, dim=-1)
#             batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
#             loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
#             loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
#             # calculate the iou prediction and slot losses only in the index
#             # with the minimum loss for each mask (to be consistent w/ SAM)
#             if self.supervise_all_iou:
#                 loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
#             else:
#                 loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
#         else:
#             loss_mask = loss_multimask
#             loss_dice = loss_multidice
#             loss_iou = loss_multiiou

#         # backprop focal, dice and iou loss only if obj present
#         loss_mask = loss_mask * target_obj
#         loss_dice = loss_dice * target_obj
#         loss_iou = loss_iou * target_obj

#         # sum over batch dimension (note that the losses are already divided by num_boxes)
#         losses["loss_mask"] += loss_mask.sum()
#         losses["loss_dice"] += loss_dice.sum()
#         losses["loss_iou"] += loss_iou.sum()
#         losses["loss_class"] += loss_class


# class TextCriterion(LossWithWeights):
#     def __init__(
#         self,
#         pad_token,
#         max_seq_len=100,
#         weight_dict=None,
#         compute_aux=False,
#     ):
#         super().__init__(weight_dict, compute_aux)
#         self.pad_token = pad_token
#         self.max_seq_len = max_seq_len
#         self.in_lengths = None

#     def get_loss(self, outputs, **kwargs):
#         nb_tokens = outputs["captioning_tokenized_target"].input_ids.numel()
#         bs, seq_len = outputs["captioning_tokenized_target"].input_ids.shape
#         ce = F.cross_entropy(
#             outputs["captioning_pred_text"].flatten(0, -2),
#             outputs["captioning_tokenized_target"].input_ids.flatten(),
#             ignore_index=self.pad_token,
#             reduction="sum",
#         )

#         not_pad = (
#             outputs["captioning_tokenized_target"]
#             .input_ids.reshape(-1)
#             .ne(self.pad_token)
#         )

#         if nb_tokens > 0:
#             nb_non_pad = not_pad.numel()
#             ce = ce / nb_non_pad

#         preds = outputs["captioning_pred_text"].flatten(0, -2).argmax(-1)[not_pad]
#         targets = outputs["captioning_tokenized_target"].input_ids.flatten()[not_pad]
#         correct = preds == targets
#         correct = correct.sum() / (correct.numel() + 1e-5)

#         correct_sequence_level = torch.all(
#             (
#                 outputs["captioning_pred_text"]
#                 .flatten(0, -2)
#                 .argmax(-1)
#                 .reshape(bs, seq_len)
#                 == outputs["captioning_tokenized_target"].input_ids
#             )
#             | (~not_pad).view(bs, seq_len),
#             dim=1,
#         )
#         seq_level_acc = correct_sequence_level.float().mean()

#         return {"loss_text": ce, "text_acc": correct, "text_seq_acc": seq_level_acc}


def segment_miou(source, target):
    """Compute the mean IoU between two sets of masks"""
    assert source.shape == target.shape, "The two masks must have the same shape"
    assert source.ndim == 3, "The masks must be 3D"

    valid_targets = (target.sum(dim=(1, 2)) > 0).sum()
    if valid_targets == 0:
        return torch.tensor(1.0, device=source.device)
    intersection = (source.bool() & target.bool()).sum(dim=(1, 2))
    union = (source.bool() | target.bool()).sum(dim=(1, 2))
    iou = intersection / (union + 1e-8)
    return iou.sum() / valid_targets


class SemanticSegCriterion(LossWithWeights):
    def __init__(
        self,
        weight_dict,
        focal: bool = False,
        focal_alpha: float = 0.6,
        focal_gamma: float = 1.6,
        downsample: bool = True,
        presence_head: bool = False,
        # Option to turn off presence loss, if some other component
        # is already doing it, e.g. decoder - in which case,
        # we could still set presence_head to True so that
        # losses are not propogated to masks when there is no GT mask
        presence_loss: bool = True,
    ):
        super().__init__(weight_dict, False)
        self.focal = focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.downsample = downsample
        self.presence_head = presence_head
        self.presence_loss = presence_loss

    def get_loss(self, out_dict, targets):
        outputs = out_dict["semantic_seg"]
        presence_logit = out_dict["presence_logit"]
        if (
            "semantic_masks" in targets
            and targets["semantic_masks"] is not None
            and targets["semantic_masks"].size(0) > 0
        ):
            semantic_targets = targets["semantic_masks"]
            with torch.no_grad():
                if self.downsample:
                    # downsample targets to the size of predictions
                    size = outputs.shape[-2:]
                    semantic_targets = (
                        F.interpolate(
                            semantic_targets.float().unsqueeze(1),
                            size=size,
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(1)
                        .bool()
                    )
        else:
            with torch.no_grad():
                if self.downsample:
                    # downsample targets to the size of predictions
                    size = outputs.shape[-2:]
                    segments = (
                        F.interpolate(
                            targets["masks"].float().unsqueeze(1),
                            size=size,
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(1)
                        .bool()
                    )
                else:
                    segments = targets["masks"].bool()

                # the annotations are for instance segmentation, so we merge them to get semantic segmentation
                semantic_targets = instance_masks_to_semantic_masks(
                    segments, targets["num_boxes"]
                )

        if not self.downsample:
            # upsample predictions to the target size
            size = semantic_targets.shape[-2:]
            outputs = F.interpolate(
                outputs.float(),
                size=size,
                mode="bilinear",
                align_corners=False,
            )

        if self.focal:
            loss = sigmoid_focal_loss(
                outputs.squeeze(1).flatten(-2),
                semantic_targets.float().flatten(-2),
                num_boxes=len(semantic_targets),
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduce=not self.presence_head,
            )
            if self.presence_head:
                loss = loss.mean(1)
        else:
            loss = F.binary_cross_entropy_with_logits(
                outputs.squeeze(1),
                semantic_targets.float(),
                reduction="none" if self.presence_head else "mean",
            )
            if self.presence_head:
                loss = loss.flatten(1).mean(1)

        loss_dice = dice_loss(
            outputs.squeeze(1).flatten(1),
            semantic_targets.flatten(1),
            len(semantic_targets),
            reduce=not self.presence_head,
        )

        miou = segment_miou(outputs.sigmoid().squeeze(1) > 0.5, semantic_targets)

        loss_dict = {}

        if self.presence_head:
            presence_target = semantic_targets.flatten(1).any(-1)
            if self.presence_loss:
                loss_presence = F.binary_cross_entropy_with_logits(
                    presence_logit.flatten(),
                    presence_target.float(),
                )
                presence_acc = (
                    ((presence_logit.flatten().sigmoid() > 0.5) == presence_target)
                    .float()
                    .mean()
                )
            else:
                # Dummy values
                loss_presence = torch.tensor(0.0, device=loss.device)
                # Whichever component is computing the presence loss,
                # should also track presence_acc
                presence_acc = torch.tensor(0.0, device=loss.device)

            loss_dict["loss_semantic_presence"] = loss_presence
            loss_dict["presence_acc"] = presence_acc

            # reduce the other losses, skipping the negative ones
            bs = loss.shape[0]
            assert presence_target.numel() == bs

            mask = presence_target
            nb_valid = presence_target.sum().item()

            loss = (loss * mask.float()).sum() / (nb_valid + 1e-6)
            loss_dice = (loss_dice * mask.float()).sum() / (nb_valid + 1e-6)

        loss_dict.update(
            {
                "loss_semantic_seg": loss,
                "loss_semantic_dice": loss_dice,
                "miou_semantic_seg": miou,
            }
        )

        return loss_dict


class Det2TrkAssoc(LossWithWeights):
    def __init__(
        self,
        weight_dict,
        use_fp_loss=False,
        fp_loss_on_exhaustive_only=True,
        treat_fp_as_new_obj=False,
    ):
        super().__init__(weight_dict, compute_aux=False)
        self.use_fp_loss = use_fp_loss
        self.fp_loss_on_exhaustive_only = fp_loss_on_exhaustive_only
        self.treat_fp_as_new_obj = treat_fp_as_new_obj
        if self.use_fp_loss:
            self.target_keys.append("is_exhaustive")

    def get_loss(self, outputs, targets, indices, num_boxes):
        det2trk_assoc_logits = outputs["det2trk_assoc_logits"]
        device = det2trk_assoc_logits.device
        B, Q_det, Q_trk_plus_2 = det2trk_assoc_logits.shape
        assert Q_trk_plus_2 >= 2
        Q_trk = Q_trk_plus_2 - 2

        # We only apply association losses to those detection queries that either match
        # a GT instance or have score > 0 (i.e. those TP, FN and FP detection queries)
        matched_object_ids = outputs["matched_object_ids"]
        assert matched_object_ids.shape == (B, Q_det + Q_trk)
        matched_obj_ids_det = matched_object_ids[:, :Q_det]
        matched_obj_ids_trk = matched_object_ids[:, Q_det:]
        det_is_matched_to_gt = matched_obj_ids_det >= 0
        trk_is_matched_to_gt = matched_obj_ids_trk >= 0

        # note: -1 label is ignored in the (softmax) cross_entropy loss below
        det2trk_assoc_labels = -torch.ones(B, Q_det, dtype=torch.long, device=device)
        # a) If a detection query is matched to a same object ID as a tracking query,
        # we assign it the index of the tracking query as a label
        det_is_same_obj_id_as_trk = (
            det_is_matched_to_gt[:, :, None]
            & trk_is_matched_to_gt[:, None, :]
            & (matched_obj_ids_det[:, :, None] == matched_obj_ids_trk[:, None, :])
        )
        batch_idx, det_idx, trk_idx = det_is_same_obj_id_as_trk.nonzero(as_tuple=True)
        det2trk_assoc_labels[batch_idx, det_idx] = trk_idx

        # b) If a detection query is matched to GT but not to any tracking query,
        # we assign it a "new_object" label
        det_is_new_obj = det_is_matched_to_gt & ~det_is_same_obj_id_as_trk.any(dim=-1)
        det2trk_assoc_labels[det_is_new_obj] = Q_trk

        # c) If a detection query is not matched to GT but have score > 0,
        # we assign it a "false_positive" label
        if self.use_fp_loss:
            det_is_above_thresh = outputs["pred_logits"][:, :Q_det].squeeze(2) > 0
            det_is_fp = ~det_is_matched_to_gt & det_is_above_thresh
            if self.treat_fp_as_new_obj:
                det2trk_assoc_labels[det_is_fp] = Q_trk
            else:
                if self.fp_loss_on_exhaustive_only:
                    # only count FP detections on batches that are exhaustively annotated
                    det_is_fp &= targets["is_exhaustive"].unsqueeze(1).bool()
                det2trk_assoc_labels[det_is_fp] = Q_trk + 1

        # softmax cross-entropy loss for detection-to-tracking association
        loss_det2trk_assoc = F.cross_entropy(
            input=det2trk_assoc_logits.flatten(0, 1),  # (B * Q_det, Q_trk + 2)
            target=det2trk_assoc_labels.flatten(0, 1),  # (B * Q_det)
            ignore_index=-1,
            reduction="none",
        ).view(B, Q_det)
        # skip det2trk assocation loss on frames w/o any (non-padding) tracking queries
        frame_has_valid_trk = trk_is_matched_to_gt.any(dim=-1, keepdims=True)  # (B, 1)
        loss_det2trk_assoc = loss_det2trk_assoc * frame_has_valid_trk.float()

        loss_det2trk_assoc = loss_det2trk_assoc.sum() / (B * num_boxes)
        return {"loss_det2trk_assoc": loss_det2trk_assoc}


class TrackingByDetectionAssoc(LossWithWeights):
    def __init__(self, weight_dict):
        super().__init__(weight_dict, compute_aux=False, supports_o2m_loss=False)
        assert "loss_det2trk_assoc" in self.weight_dict
        assert "loss_trk2det_assoc" in self.weight_dict

    def get_loss(self, outputs, targets, indices, num_boxes):
        # Part A: gather object id matching between detection and tracking
        det2trk_assoc_logits = outputs["det2trk_assoc_logits"]  # (B, Q_det+1, Q_trk+1)
        B, Q_det_plus_1, Q_trk_plus_1 = det2trk_assoc_logits.shape
        assert Q_det_plus_1 >= 1 and Q_trk_plus_1 >= 1
        Q_det = Q_det_plus_1 - 1
        Q_trk = Q_trk_plus_1 - 1
        device = det2trk_assoc_logits.device

        matched_obj_ids_det = outputs["matched_object_ids"]
        assert matched_obj_ids_det.shape == (B, Q_det)
        det_is_matched_to_gt = matched_obj_ids_det >= 0
        matched_obj_ids_trk = outputs["prev_trk_object_ids"]
        assert matched_obj_ids_trk.shape == (B, Q_trk)
        trk_is_matched_to_gt = matched_obj_ids_trk >= 0
        frame_has_valid_trk = trk_is_matched_to_gt.any(dim=-1, keepdims=True)  # (B, 1)

        # check whether a detection object is the same as a tracking object
        det_is_same_obj_id_as_trk = (
            det_is_matched_to_gt[:, :, None]
            & trk_is_matched_to_gt[:, None, :]
            & (matched_obj_ids_det[:, :, None] == matched_obj_ids_trk[:, None, :])
        )  # (B, Q_det, Q_trk)
        # there should be at most one match for each detection and each previous tracked object
        torch._assert_async(torch.all(det_is_same_obj_id_as_trk.sum(dim=2) <= 1))
        torch._assert_async(torch.all(det_is_same_obj_id_as_trk.sum(dim=1) <= 1))
        batch_idx, det_idx, trk_idx = det_is_same_obj_id_as_trk.nonzero(as_tuple=True)

        # Part B: Detection-to-tracking association loss
        # assign detection-to-tracking labels (note: -1 label is ignored in the loss below)
        det2trk_assoc_labels = -torch.ones(B, Q_det, dtype=torch.long, device=device)
        det2trk_assoc_labels[batch_idx, det_idx] = trk_idx
        # if a detection is matched to GT but not to any tracking, assign it a "new-object" label
        det_is_new_obj = det_is_matched_to_gt & ~det_is_same_obj_id_as_trk.any(dim=2)
        det2trk_assoc_labels[det_is_new_obj] = Q_trk  # "Q_trk" label is "new-object"

        # softmax cross-entropy loss for detection-to-tracking association
        loss_det2trk_assoc = F.cross_entropy(
            input=det2trk_assoc_logits[:, :-1].flatten(0, 1),  # (B*Q_det, Q_trk+1)
            target=det2trk_assoc_labels.flatten(0, 1),  # (B*Q_det)
            ignore_index=-1,
            reduction="none",
        ).view(B, Q_det)
        # skip det2trk assocation loss on frames w/o any (non-padding) tracking queries
        loss_det2trk_assoc = loss_det2trk_assoc * frame_has_valid_trk.float()
        loss_det2trk_assoc = loss_det2trk_assoc.sum() / (B * num_boxes)
        loss_dict = {"loss_det2trk_assoc": loss_det2trk_assoc}

        # Part C: tracking-to-detection association loss
        trk2det_assoc_logits = det2trk_assoc_logits.transpose(1, 2)
        assert trk2det_assoc_logits.shape == (B, Q_trk + 1, Q_det + 1)
        # assign tracking-to-detection labels (note: -1 label is ignored in the loss below)
        trk2det_assoc_labels = -torch.ones(B, Q_trk, dtype=torch.long, device=device)
        trk2det_assoc_labels[batch_idx, trk_idx] = det_idx
        # if a tracking is matched to GT but not to any detection, assign it a "occluded" label
        trk_is_occluded = trk_is_matched_to_gt & ~det_is_same_obj_id_as_trk.any(dim=1)
        trk2det_assoc_labels[trk_is_occluded] = Q_det  # "Q_det" label is "occluded"

        # softmax cross-entropy loss for tracking-to-detection association
        loss_trk2det_assoc = F.cross_entropy(
            input=trk2det_assoc_logits[:, :-1].flatten(0, 1),  # (B*Q_trk, Q_det+1)
            target=trk2det_assoc_labels.flatten(0, 1),  # (B*Q_trk)
            ignore_index=-1,
            reduction="none",
        ).view(B, Q_trk)
        # skip trk2det association loss on frames w/o any (non-padding) tracking queries
        loss_trk2det_assoc = loss_trk2det_assoc * frame_has_valid_trk.float()
        loss_trk2det_assoc = loss_trk2det_assoc.sum() / (B * num_boxes)
        loss_dict["loss_trk2det_assoc"] = loss_trk2det_assoc

        return loss_dict


def _keep_only_trk_queries_in_match_inds(inds, Q_det):
    """Keep only the tracking query indices in the indices tuple"""
    batch_idx, src_idx, tgt_idx = inds
    if batch_idx.numel() == 0:
        return (batch_idx, src_idx, tgt_idx)  # empty indices, nothing to filter

    # keep only the tracking query indices
    is_trk_query = src_idx >= Q_det
    batch_idx_trk = batch_idx[is_trk_query]
    src_idx_trk = src_idx[is_trk_query]
    tgt_idx_trk = tgt_idx[is_trk_query] if tgt_idx is not None else None
    return (batch_idx_trk, src_idx_trk, tgt_idx_trk)
