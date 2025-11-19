# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import torch

from sam3.model.model_misc import SAM3Output

from sam3.train.utils.distributed import get_world_size

from .loss_fns import CORE_LOSS_KEY, Det2TrkAssoc, Masks


class DummyLoss(torch.nn.Module):
    """A dummy loss that always returns 0 (as a placeholder for eval)"""

    def __init__(
        self,
        core_loss_key: str = CORE_LOSS_KEY,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        self.core_loss_key = core_loss_key
        self.device = torch.device(device)

    def forward(self, *args, **kwargs):
        return {self.core_loss_key: torch.tensor(0.0, device=self.device)}

    def accumulate(self, out_dict):
        """
        Called by iterative losses.
        """
        if self.core_loss_key not in out_dict:
            out_dict[self.core_loss_key] = torch.tensor(0.0, device=self.device)
        return out_dict


class Sam3LossWrapper(torch.nn.Module):
    def __init__(
        self,
        loss_fns_find,
        normalization="global",
        matcher=None,
        o2m_matcher=None,
        o2m_weight=1.0,
        use_o2m_matcher_on_o2m_aux=True,
        loss_fn_semantic_seg=None,
        normalize_by_valid_object_num=False,
        normalize_by_stage_num=False,
        scale_by_find_batch_size=False,
    ):
        super().__init__()
        self.loss_fns_find = loss_fns_find
        assert normalization in ["global", "local", "none"]
        self.normalization = normalization
        self.normalize_by_valid_object_num = normalize_by_valid_object_num
        self.normalize_by_stage_num = normalize_by_stage_num
        self.matcher = matcher
        self.o2m_matcher = o2m_matcher
        self.o2m_weight = o2m_weight
        # whether to use the o2m matcher on the o2m queries in auxiliary outputs
        self.use_o2m_matcher_on_o2m_aux = use_o2m_matcher_on_o2m_aux
        self.loss_fn_semantic_seg = loss_fn_semantic_seg
        self.scale_by_find_batch_size = scale_by_find_batch_size

    def _get_num_boxes(self, targets):
        # the average number of target boxes for loss normalization
        if self.normalize_by_valid_object_num:
            # valid boxes are those with non-zero height and width
            # (while padded invisible boxes are )
            boxes_hw = targets["boxes"].view(-1, 4)  # cx, cy, w, h
            num_boxes = (boxes_hw[:, 2:] > 0).all(dim=-1).sum().float()
        else:
            num_boxes = targets["num_boxes"].sum().float()
        if self.normalization == "global":
            torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1)
        elif self.normalization == "local":
            num_boxes = torch.clamp(num_boxes, min=1)
        elif self.normalization == "none":
            num_boxes = 1
        return num_boxes

    def compute_loss(self, nested_out, targets):
        num_boxes = self._get_num_boxes(targets)
        o2m_out_is_valid = nested_out.get("o2m_out_is_valid", None)
        o2m_target_is_valid_padded = nested_out.get("o2m_target_is_valid_padded", None)

        # Get a list of outputs, including auxiliary and first stage outputs
        output_list = [(nested_out, "", False)]  # (out, suffix, is_aux)
        if "aux_outputs" in nested_out:
            output_list.extend(
                (aux_out, f"_aux_{i}", True)
                for i, aux_out in enumerate(nested_out["aux_outputs"])
            )
        if "first_stage" in nested_out:
            output_list.append((nested_out["first_stage"], "_fs", True))

        # Compute all the requested losses
        losses = {}
        total_core_loss = 0.0
        for out, suffix, is_aux in output_list:
            # o2o matcher indices need to be computed by the model (as the video model requires
            # a specific way of matching free and locked indices beyond just calling the matcher)
            indices = out["indices"]
            has_o2m_out = "pred_logits_o2m" in out
            if has_o2m_out:
                o2m_out = {
                    k[: -len("_o2m")]: v for k, v in out.items() if k.endswith("_o2m")
                }
                # o2m targets are the same as the o2o targets (assuming repeat=1)
                o2m_targets = targets
                if self.use_o2m_matcher_on_o2m_aux or not is_aux:
                    o2m_indices = self.o2m_matcher(
                        o2m_out,
                        o2m_targets,
                        out_is_valid=o2m_out_is_valid,
                        target_is_valid_padded=o2m_target_is_valid_padded,
                    )
                else:
                    o2m_indices = self.matcher(
                        o2m_out,
                        o2m_targets,
                        out_is_valid=o2m_out_is_valid,
                        target_is_valid_padded=o2m_target_is_valid_padded,
                    )

            for loss_fn in self.loss_fns_find:
                l_dict = loss_fn(
                    outputs=out,
                    targets=targets,
                    indices=indices,
                    num_boxes=num_boxes,
                    is_aux=is_aux,
                )
                total_core_loss += l_dict.pop(CORE_LOSS_KEY)
                losses.update({f"{k}{suffix}": v for k, v in l_dict.items()})

                compute_o2m_loss = has_o2m_out
                # a special handling to allow turning off mask loss in o2m
                # (to be compatible with the original implementation)
                if isinstance(loss_fn, Masks):
                    compute_o2m_loss = compute_o2m_loss and "pred_masks" in o2m_out
                if isinstance(loss_fn, Det2TrkAssoc):
                    compute_o2m_loss = False  # Det2TrkAssoc does not support o2m
                if compute_o2m_loss:
                    l_dict = loss_fn(
                        outputs=o2m_out,
                        targets=o2m_targets,
                        indices=o2m_indices,
                        num_boxes=num_boxes,
                        is_aux=is_aux,
                    )
                    for k in l_dict:
                        l_dict[k] *= self.o2m_weight
                    total_core_loss += l_dict.pop(CORE_LOSS_KEY)
                    losses.update({f"{k}{suffix}_o2m": v for k, v in l_dict.items()})

        losses[CORE_LOSS_KEY] = total_core_loss
        return losses

    def forward(self, find_stages: SAM3Output, find_targets):
        if find_stages.loss_stages is not None:
            find_targets = [find_targets[i] for i in find_stages.loss_stages]
        with SAM3Output.iteration_mode(
            find_stages, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
        ) as find_stages:
            assert len(find_stages) == len(find_targets)
            total_losses = {}
            for stage_outputs, stage_targets in zip(find_stages, find_targets):
                stage_targets = [stage_targets] * len(stage_outputs)
                # If there are multiple steps within a stage, compute the loss for all of them (e.g. interactivity)
                for outputs, targets in zip(stage_outputs, stage_targets):
                    cur_losses = self.compute_loss(outputs, targets)

                    if self.loss_fn_semantic_seg is not None:
                        cur_losses_semantic = self.loss_fn_semantic_seg(
                            outputs, targets
                        )
                        cur_losses[CORE_LOSS_KEY] += cur_losses_semantic.pop(
                            CORE_LOSS_KEY
                        )
                        # make sure the semantic losses don't overlap with the find losses
                        assert set(cur_losses).isdisjoint(set(cur_losses_semantic))
                        cur_losses.update(cur_losses_semantic)

                    # Optionally, normalize the loss by the number of find stages (training video frames) so that
                    # image batches and video batches have similar loss scales. (Otherwise video batches would
                    # have a much higher loss scale due to summing the losses over all the find stages.)
                    if self.normalize_by_stage_num:
                        cur_losses[CORE_LOSS_KEY] /= len(find_stages)

                    if self.scale_by_find_batch_size:
                        bs = targets["num_boxes"].shape[0]
                        # sqrt scaling based on the "effective" batch size
                        cur_losses[CORE_LOSS_KEY] *= bs**0.5

                    for k, v in cur_losses.items():
                        if k not in total_losses:
                            total_losses[k] = v
                        else:
                            total_losses[k] += v

        return total_losses
