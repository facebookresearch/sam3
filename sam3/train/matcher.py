# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""

import numpy as np
import torch

from sam3.model.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from scipy.optimize import linear_sum_assignment
from torch import nn


def _do_matching(cost, repeats=1, return_tgt_indices=False, do_filtering=False):
    if repeats > 1:
        cost = np.tile(cost, (1, repeats))

    i, j = linear_sum_assignment(cost)
    if do_filtering:
        # filter out invalid entries (i.e. those with cost > 1e8)
        valid_thresh = 1e8
        valid_ijs = [(ii, jj) for ii, jj in zip(i, j) if cost[ii, jj] < valid_thresh]
        i, j = zip(*valid_ijs) if len(valid_ijs) > 0 else ([], [])
        i, j = np.array(i, dtype=np.int64), np.array(j, dtype=np.int64)
    if return_tgt_indices:
        return i, j
    order = np.argsort(j)
    return i[order]


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Sigmoid() if focal_loss else nn.Softmax(-1)
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs, batched_targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = self.norm(
            outputs["pred_logits"].flatten(0, 1)
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_bbox = batched_targets["boxes"]

        if "positive_map" in batched_targets:
            # In this case we have a multi-hot target
            positive_map = batched_targets["positive_map"]
            assert len(tgt_bbox) == len(positive_map)

            if self.focal_loss:
                positive_map = positive_map > 1e-4
                alpha = self.focal_alpha
                gamma = self.focal_gamma
                neg_cost_class = (
                    (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
                )
                pos_cost_class = (
                    alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                )
                cost_class = (
                    (pos_cost_class - neg_cost_class).unsqueeze(1)
                    * positive_map.unsqueeze(0)
                ).sum(-1)
            else:
                # Compute the soft-cross entropy between the predicted token alignment and the GT one for each box
                cost_class = -(out_prob.unsqueeze(1) * positive_map.unsqueeze(0)).sum(
                    -1
                )
        else:
            # In this case we are doing a "standard" cross entropy
            tgt_ids = batched_targets["labels"]
            assert len(tgt_bbox) == len(tgt_ids)

            if self.focal_loss:
                alpha = self.focal_alpha
                gamma = self.focal_gamma
                neg_cost_class = (
                    (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
                )
                pos_cost_class = (
                    alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                )
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            else:
                # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                # but approximate it in 1 - proba[target class].
                # The 1 is a constant that doesn't change the matching, it can be omitted.
                cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        assert cost_class.shape == cost_bbox.shape

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu().numpy()

        sizes = torch.cumsum(batched_targets["num_boxes"], -1)[:-1]
        costs = [c[i] for i, c in enumerate(np.split(C, sizes.cpu().numpy(), axis=-1))]
        indices = [_do_matching(c) for c in costs]
        batch_idx = torch.as_tensor(
            sum([[i] * len(src) for i, src in enumerate(indices)], []), dtype=torch.long
        )
        src_idx = torch.from_numpy(np.concatenate(indices)).long()
        return batch_idx, src_idx


class BinaryHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Sigmoid()
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, batched_targets, repeats=0, repeat_batch=1):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if repeat_batch != 1:
            raise NotImplementedError("please use BinaryHungarianMatcherV2 instead")

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = self.norm(outputs["pred_logits"].flatten(0, 1)).squeeze(
            -1
        )  # [batch_size * num_queries]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_bbox = batched_targets["boxes"]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        cost_class = -out_prob.unsqueeze(-1).expand_as(cost_bbox)

        assert cost_class.shape == cost_bbox.shape

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu().numpy()

        sizes = torch.cumsum(batched_targets["num_boxes"], -1)[:-1]
        costs = [c[i] for i, c in enumerate(np.split(C, sizes.cpu().numpy(), axis=-1))]
        return_tgt_indices = False
        for c in costs:
            n_targ = c.shape[1]
            if repeats > 1:
                n_targ *= repeats
            if c.shape[0] < n_targ:
                return_tgt_indices = True
                break
        if return_tgt_indices:
            indices, tgt_indices = zip(
                *(
                    _do_matching(
                        c, repeats=repeats, return_tgt_indices=return_tgt_indices
                    )
                    for c in costs
                )
            )
            tgt_indices = list(tgt_indices)
            for i in range(1, len(tgt_indices)):
                tgt_indices[i] += sizes[i - 1].item()
            tgt_idx = torch.from_numpy(np.concatenate(tgt_indices)).long()
        else:
            indices = [_do_matching(c, repeats=repeats) for c in costs]
            tgt_idx = None

        batch_idx = torch.as_tensor(
            sum([[i] * len(src) for i, src in enumerate(indices)], []), dtype=torch.long
        )
        src_idx = torch.from_numpy(np.concatenate(indices)).long()
        return batch_idx, src_idx, tgt_idx


class BinaryFocalHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        alpha: float = 0.25,
        gamma: float = 2.0,
        stable: bool = False,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Sigmoid()
        self.alpha = alpha
        self.gamma = gamma
        self.stable = stable
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, batched_targets, repeats=1, repeat_batch=1):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if repeat_batch != 1:
            raise NotImplementedError("please use BinaryHungarianMatcherV2 instead")

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_score = outputs["pred_logits"].flatten(0, 1).squeeze(-1)
        out_prob = self.norm(out_score)  # [batch_size * num_queries]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_bbox = batched_targets["boxes"]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # cost_class = -out_prob.unsqueeze(-1).expand_as(cost_bbox)
        if self.stable:
            rescaled_giou = (-cost_giou + 1) / 2
            out_prob = out_prob.unsqueeze(-1).expand_as(cost_bbox) * rescaled_giou
            cost_class = -self.alpha * (1 - out_prob) ** self.gamma * torch.log(
                out_prob
            ) + (1 - self.alpha) * out_prob**self.gamma * torch.log(1 - out_prob)
        else:
            # directly computing log sigmoid (more numerically stable)
            log_out_prob = torch.nn.functional.logsigmoid(out_score)
            log_one_minus_out_prob = torch.nn.functional.logsigmoid(-out_score)
            cost_class = (
                -self.alpha * (1 - out_prob) ** self.gamma * log_out_prob
                + (1 - self.alpha) * out_prob**self.gamma * log_one_minus_out_prob
            )
        if not self.stable:
            cost_class = cost_class.unsqueeze(-1).expand_as(cost_bbox)

        assert cost_class.shape == cost_bbox.shape

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu().numpy()

        sizes = torch.cumsum(batched_targets["num_boxes"], -1)[:-1]
        costs = [c[i] for i, c in enumerate(np.split(C, sizes.cpu().numpy(), axis=-1))]
        return_tgt_indices = False
        for c in costs:
            n_targ = c.shape[1]
            if repeats > 1:
                n_targ *= repeats
            if c.shape[0] < n_targ:
                return_tgt_indices = True
                break
        if return_tgt_indices:
            indices, tgt_indices = zip(
                *(
                    _do_matching(
                        c, repeats=repeats, return_tgt_indices=return_tgt_indices
                    )
                    for c in costs
                )
            )
            tgt_indices = list(tgt_indices)
            for i in range(1, len(tgt_indices)):
                tgt_indices[i] += sizes[i - 1].item()
            tgt_idx = torch.from_numpy(np.concatenate(tgt_indices)).long()
        else:
            indices = [_do_matching(c, repeats=repeats) for c in costs]
            tgt_idx = None

        batch_idx = torch.as_tensor(
            sum([[i] * len(src) for i, src in enumerate(indices)], []), dtype=torch.long
        )
        src_idx = torch.from_numpy(np.concatenate(indices)).long()
        return batch_idx, src_idx, tgt_idx


class BinaryHungarianMatcherV2(nn.Module):
    """
    This class computes an assignment between the targets and the predictions
    of the network

    For efficiency reasons, the targets don't include the no_object. Because of
    this, in general, there are more predictions than targets. In this case, we
    do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    This is a more efficient implementation of BinaryHungarianMatcher.
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal: bool = False,
        alpha: float = 0.25,
        gamma: float = 2.0,
        stable: bool = False,
        remove_samples_with_0_gt: bool = True,
    ):
        """
        Creates the matcher

        Params:
        - cost_class: Relative weight of the classification error in the
          matching cost
        - cost_bbox: Relative weight of the L1 error of the bounding box
          coordinates in the matching cost
        - cost_giou: This is the relative weight of the giou loss of the
          bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Sigmoid()
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"
        self.focal = focal
        if focal:
            self.alpha = alpha
            self.gamma = gamma
            self.stable = stable
        self.remove_samples_with_0_gt = remove_samples_with_0_gt

    @torch.no_grad()
    def forward(
        self,
        outputs,
        batched_targets,
        repeats=1,
        repeat_batch=1,
        out_is_valid=None,
        target_is_valid_padded=None,
    ):
        """
        Performs the matching. The inputs and outputs are the same as
        BinaryHungarianMatcher.forward, except for the optional cached_padded
        flag and the optional "_boxes_padded" entry of batched_targets.

        Inputs:
        - outputs: A dict with the following keys:
            - "pred_logits": Tensor of shape (batch_size, num_queries, 1) with
               classification logits
            - "pred_boxes": Tensor of shape (batch_size, num_queries, 4) with
               predicted box coordinates in cxcywh format.
        - batched_targets: A dict of targets. There may be a variable number of
          targets per batch entry; suppose that there are T_b targets for batch
          entry 0 <= b < batch_size. It should have the following keys:
          - "boxes": Tensor of shape (sum_b T_b, 4) giving ground-truth boxes
             in cxcywh format for all batch entries packed into a single tensor
          - "num_boxes": int64 Tensor of shape (batch_size,) giving the number
             of ground-truth boxes per batch entry: num_boxes[b] = T_b
          - "_boxes_padded": Tensor of shape (batch_size, max_b T_b, 4) giving
            a padded version of ground-truth boxes. If this is not present then
            it will be computed from batched_targets["boxes"] instead, but
            caching it here can improve performance for repeated calls with the
            same targets.
        - out_is_valid: If not None, it should be a boolean tensor of shape
          (batch_size, num_queries) indicating which predictions are valid.
          Invalid predictions are ignored during matching and won't appear in
          the output indices.
        - target_is_valid_padded: If not None, it should be a boolean tensor of
          shape (batch_size, max_num_gt_boxes) in padded format indicating
          which GT boxes are valid. Invalid GT boxes are ignored during matching
          and won't appear in the output indices.

        Returns:
            A list of size batch_size, containing tuples of (idx_i, idx_j):
                - idx_i is the indices of the selected predictions (in order)
                - idx_j is the indices of the corresponding selected targets
                  (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j)
                             = min(num_queries, num_target_boxes)
        """
        _, num_queries = outputs["pred_logits"].shape[:2]

        out_score = outputs["pred_logits"].squeeze(-1)  # (B, Q)
        out_bbox = outputs["pred_boxes"]  # (B, Q, 4))

        device = out_score.device

        num_boxes = batched_targets["num_boxes"].cpu()
        # Get a padded version of target boxes (as precomputed in the collator).
        # It should work for both repeat==1 (o2o) and repeat>1 (o2m) matching.
        tgt_bbox = batched_targets["boxes_padded"]
        if self.remove_samples_with_0_gt:
            # keep only samples w/ at least 1 GT box in targets (num_boxes and tgt_bbox)
            batch_keep = num_boxes > 0
            num_boxes = num_boxes[batch_keep]
            tgt_bbox = tgt_bbox[batch_keep]
            if target_is_valid_padded is not None:
                target_is_valid_padded = target_is_valid_padded[batch_keep]
        # Repeat the targets (for the case of batched aux outputs in the matcher)
        if repeat_batch > 1:
            # In this case, out_prob and out_bbox will be a concatenation of
            # both final and auxiliary outputs, so we also repeat the targets
            num_boxes = num_boxes.repeat(repeat_batch)
            tgt_bbox = tgt_bbox.repeat(repeat_batch, 1, 1)
            if target_is_valid_padded is not None:
                target_is_valid_padded = target_is_valid_padded.repeat(repeat_batch, 1)

        # keep only samples w/ at least 1 GT box in outputs
        if self.remove_samples_with_0_gt:
            if repeat_batch > 1:
                batch_keep = batch_keep.repeat(repeat_batch)
            out_score = out_score[batch_keep]
            out_bbox = out_bbox[batch_keep]
            if out_is_valid is not None:
                out_is_valid = out_is_valid[batch_keep]
        assert out_bbox.shape[0] == tgt_bbox.shape[0]
        assert out_bbox.shape[0] == num_boxes.shape[0]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        out_prob = self.norm(out_score)
        if not self.focal:
            cost_class = -out_prob.unsqueeze(-1).expand_as(cost_bbox)
        else:
            if self.stable:
                rescaled_giou = (-cost_giou + 1) / 2
                out_prob = out_prob.unsqueeze(-1).expand_as(cost_bbox) * rescaled_giou
                cost_class = -self.alpha * (1 - out_prob) ** self.gamma * torch.log(
                    out_prob
                ) + (1 - self.alpha) * out_prob**self.gamma * torch.log(1 - out_prob)
            else:
                # directly computing log sigmoid (more numerically stable)
                log_out_prob = torch.nn.functional.logsigmoid(out_score)
                log_one_minus_out_prob = torch.nn.functional.logsigmoid(-out_score)
                cost_class = (
                    -self.alpha * (1 - out_prob) ** self.gamma * log_out_prob
                    + (1 - self.alpha) * out_prob**self.gamma * log_one_minus_out_prob
                )
            if not self.stable:
                cost_class = cost_class.unsqueeze(-1).expand_as(cost_bbox)

        assert cost_class.shape == cost_bbox.shape

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        # assign a very high cost (1e9) to invalid outputs and targets, so that we can
        # filter them out (in `_do_matching`) from bipartite matching results
        do_filtering = out_is_valid is not None or target_is_valid_padded is not None
        if out_is_valid is not None:
            C = torch.where(out_is_valid[:, :, None], C, 1e9)
        if target_is_valid_padded is not None:
            C = torch.where(target_is_valid_padded[:, None, :], C, 1e9)
        C = C.cpu().numpy()
        costs = [C[i, :, :s] for i, s in enumerate(num_boxes.tolist())]
        return_tgt_indices = (
            do_filtering or torch.any(num_queries < num_boxes * max(repeats, 1)).item()
        )
        if len(costs) == 0:
            # We have size 0 in the batch dimension, so we return empty matching indices
            # (note that this can happen due to `remove_samples_with_0_gt=True` even if
            # the original input batch size is not 0, when all queries have empty GTs).
            indices = []
            tgt_idx = torch.zeros(0).long().to(device) if return_tgt_indices else None
        elif return_tgt_indices:
            indices, tgt_indices = zip(
                *(
                    _do_matching(
                        c,
                        repeats=repeats,
                        return_tgt_indices=return_tgt_indices,
                        do_filtering=do_filtering,
                    )
                    for c in costs
                )
            )
            tgt_indices = list(tgt_indices)
            sizes = torch.cumsum(num_boxes, -1)[:-1]
            for i in range(1, len(tgt_indices)):
                tgt_indices[i] += sizes[i - 1].item()
            tgt_idx = torch.from_numpy(np.concatenate(tgt_indices)).long().to(device)
        else:
            indices = [
                _do_matching(c, repeats=repeats, do_filtering=do_filtering)
                for c in costs
            ]
            tgt_idx = None

        if self.remove_samples_with_0_gt:
            kept_inds = batch_keep.nonzero().squeeze(1)
            batch_idx = torch.as_tensor(
                sum([[kept_inds[i]] * len(src) for i, src in enumerate(indices)], []),
                dtype=torch.long,
                device=device,
            )
        else:
            batch_idx = torch.as_tensor(
                sum([[i] * len(src) for i, src in enumerate(indices)], []),
                dtype=torch.long,
                device=device,
            )

        # indices could be an empty list (since we remove samples w/ 0 GT boxes)
        if len(indices) > 0:
            src_idx = torch.from_numpy(np.concatenate(indices)).long().to(device)
        else:
            src_idx = torch.empty(0, dtype=torch.long, device=device)
        return batch_idx, src_idx, tgt_idx


class BinaryOneToManyMatcher(nn.Module):
    """
    This class computes a greedy assignment between the targets and the predictions of the network.
    In this formulation, several predictions can be assigned to each target, but each prediction can be assigned to
    at most one target.

    See DAC-Detr for details
    """

    def __init__(
        self,
        alpha: float = 0.3,
        threshold: float = 0.4,
        topk: int = 6,
    ):
        """
        Creates the matcher

        Params:
                alpha: relative balancing between classification and localization
                threshold: threshold used to select positive predictions
                topk: number of top scoring predictions to consider
        """
        super().__init__()
        self.norm = nn.Sigmoid()
        self.alpha = alpha
        self.threshold = threshold
        self.topk = topk

    @torch.no_grad()
    def forward(
        self,
        outputs,
        batched_targets,
        repeats=1,
        repeat_batch=1,
        out_is_valid=None,
        target_is_valid_padded=None,
    ):
        """
        Performs the matching. The inputs and outputs are the same as
        BinaryHungarianMatcher.forward

        Inputs:
        - outputs: A dict with the following keys:
            - "pred_logits": Tensor of shape (batch_size, num_queries, 1) with
               classification logits
            - "pred_boxes": Tensor of shape (batch_size, num_queries, 4) with
               predicted box coordinates in cxcywh format.
        - batched_targets: A dict of targets. There may be a variable number of
          targets per batch entry; suppose that there are T_b targets for batch
          entry 0 <= b < batch_size. It should have the following keys:
          - "num_boxes": int64 Tensor of shape (batch_size,) giving the number
             of ground-truth boxes per batch entry: num_boxes[b] = T_b
          - "_boxes_padded": Tensor of shape (batch_size, max_b T_b, 4) giving
            a padded version of ground-truth boxes. If this is not present then
            it will be computed from batched_targets["boxes"] instead, but
            caching it here can improve performance for repeated calls with the
            same targets.
        - out_is_valid: If not None, it should be a boolean tensor of shape
          (batch_size, num_queries) indicating which predictions are valid.
          Invalid predictions are ignored during matching and won't appear in
          the output indices.
        - target_is_valid_padded: If not None, it should be a boolean tensor of
          shape (batch_size, max_num_gt_boxes) in padded format indicating
          which GT boxes are valid. Invalid GT boxes are ignored during matching
          and won't appear in the output indices.
        Returns:
            A list of size batch_size, containing tuples of (idx_i, idx_j):
                - idx_i is the indices of the selected predictions (in order)
                - idx_j is the indices of the corresponding selected targets
                  (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j)
                             = min(num_queries, num_target_boxes)
        """
        assert repeats <= 1 and repeat_batch <= 1
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = self.norm(outputs["pred_logits"]).squeeze(-1)  # (B, Q)
        out_bbox = outputs["pred_boxes"]  # (B, Q, 4))

        num_boxes = batched_targets["num_boxes"]

        # Get a padded version of target boxes (as precomputed in the collator).
        tgt_bbox = batched_targets["boxes_padded"]
        assert len(tgt_bbox) == bs
        num_targets = tgt_bbox.shape[1]
        if num_targets == 0:
            return (
                torch.empty(0, dtype=torch.long, device=out_prob.device),
                torch.empty(0, dtype=torch.long, device=out_prob.device),
                torch.empty(0, dtype=torch.long, device=out_prob.device),
            )

        iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        assert iou.shape == (bs, num_queries, num_targets)

        # Final cost matrix (higher is better in `C`; this is unlike the case
        # of BinaryHungarianMatcherV2 above where lower is better in its `C`)
        C = self.alpha * out_prob.unsqueeze(-1) + (1 - self.alpha) * iou
        if out_is_valid is not None:
            C = torch.where(out_is_valid[:, :, None], C, -1e9)
        if target_is_valid_padded is not None:
            C = torch.where(target_is_valid_padded[:, None, :], C, -1e9)

        # Selecting topk predictions
        matches = C > torch.quantile(
            C, 1 - self.topk / num_queries, dim=1, keepdim=True
        )

        # Selecting predictions above threshold
        matches = matches & (C > self.threshold)
        if out_is_valid is not None:
            matches = matches & out_is_valid[:, :, None]
        if target_is_valid_padded is not None:
            matches = matches & target_is_valid_padded[:, None, :]

        # Removing padding
        matches = matches & (
            torch.arange(0, num_targets, device=num_boxes.device)[None]
            < num_boxes[:, None]
        ).unsqueeze(1)

        batch_idx, src_idx, tgt_idx = torch.nonzero(matches, as_tuple=True)

        cum_num_boxes = torch.cat(
            [
                torch.zeros(1, dtype=num_boxes.dtype, device=num_boxes.device),
                num_boxes.cumsum(-1)[:-1],
            ]
        )
        tgt_idx += cum_num_boxes[batch_idx]

        return batch_idx, src_idx, tgt_idx
