# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sam3.model.box_ops import generalized_box_iou
from sam3.train.matcher import BinaryHungarianMatcherV2
from sam3.train.optim.optimizer import GradientClipper
from sam3.train.trainer import OptimAMPConf, Trainer
from sam3.train.utils.train_utils import Phase


def test_grad_scaler_is_only_enabled_for_fp16_amp():
    assert OptimAMPConf(enabled=True, amp_dtype="float16").use_grad_scaler
    assert not OptimAMPConf(enabled=True, amp_dtype="bfloat16").use_grad_scaler
    assert not OptimAMPConf(enabled=False, amp_dtype="float16").use_grad_scaler


@pytest.mark.parametrize("gradient_clipper", [None, GradientClipper()])
def test_bf16_skips_nonfinite_optimizer_update(gradient_clipper):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    trainer = SimpleNamespace(
        optim=SimpleNamespace(optimizer=optimizer),
        optim_conf=SimpleNamespace(
            amp=OptimAMPConf(enabled=True, amp_dtype="bfloat16")
        ),
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        gradient_clipper=gradient_clipper,
        gradient_logger=None,
        device=torch.device("cpu"),
        model=model,
        steps={Phase.TRAIN: 1},
    )

    original_weight = model.weight.detach().clone()
    model.weight.grad = torch.ones_like(model.weight)
    Trainer._step_optimizer(trainer)
    assert not torch.equal(model.weight, original_weight)

    weight_before_skip = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, float("inf"))
    Trainer._step_optimizer(trainer)
    torch.testing.assert_close(model.weight, weight_before_skip)
    assert optimizer.state[model.weight]["step"].item() == 1
    assert model.weight.grad is None

    # A bad gradient on any rank must skip the update on every rank.
    model.weight.grad = torch.ones_like(model.weight)
    with patch("sam3.train.trainer.all_reduce_max", return_value=torch.tensor(1)):
        Trainer._step_optimizer(trainer)
    torch.testing.assert_close(model.weight, weight_before_skip)
    assert optimizer.state[model.weight]["step"].item() == 1


def test_giou_handles_zero_area_boxes():
    empty = torch.zeros((1, 4))
    normal = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    torch.testing.assert_close(generalized_box_iou(empty, empty), torch.zeros((1, 1)))
    torch.testing.assert_close(generalized_box_iou(normal, normal), torch.ones((1, 1)))


def test_matcher_preserves_narrow_bf16_boxes():
    # Around 0.75, both corners of a box this narrow round to its center in BF16.
    inputs = torch.tensor(
        [[[0.75, 0.75, 0.0, 0.0], [0.75, 0.75, 0.001, 0.001]]], dtype=torch.bfloat16
    )
    scores = torch.zeros((1, 2, 1), dtype=torch.bfloat16)
    matcher = BinaryHungarianMatcherV2(cost_class=0, cost_bbox=0, cost_giou=1)
    _, predictions, _ = matcher(
        {"pred_logits": scores, "pred_boxes": inputs},
        {"num_boxes": torch.tensor([1]), "boxes_padded": inputs[:, 1:]},
    )
    assert predictions.tolist() == [1]
