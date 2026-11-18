# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Regression tests requiring SAM3's supported import/runtime environment."""

import os

import pytest
import torch
from sam3.model.sam3_tracker_base import Sam3TrackerBase
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3.model.video_tracking_multiplex import VideoTrackingMultiplex
from sam3.model.video_tracking_multiplex_demo import Sam3VideoTrackingMultiplexDemo

OBJECT_HELPER = "_apply_object_wise_non_overlapping_constraints"


@pytest.fixture(
    params=[
        (Sam3TrackerBase, Sam3TrackerPredictor),
        (VideoTrackingMultiplex, Sam3VideoTrackingMultiplexDemo),
    ]
)
def helpers(request):
    base, predictor = request.param
    # The helpers use no model state; avoid constructors and checkpoint loading.
    instance = predictor.__new__(predictor)
    return instance, base._apply_non_overlapping_constraints


def legacy_output(helpers, masks, scores, background):
    """Pre-fix object helper, using the production pixel-level winner selection."""
    instance, pixel_helper = helpers
    scores_per_pixel = torch.where(masks > 0, scores[..., None, None], background)
    winners = pixel_helper(instance, scores_per_pixel)
    return torch.where(winners > 0, masks, torch.clamp(masks, max=background))


def call_helper(helpers, masks, scores, background):
    return getattr(helpers[0], OBJECT_HELPER)(
        masks, scores, background_value=background
    )


@pytest.mark.parametrize("background", [0, 0.0])
@pytest.mark.parametrize("count", [1, 2, 5])
@pytest.mark.parametrize("case", ["false", "true", "random", "noncontiguous"])
@pytest.mark.parametrize("score_case", ["positive", "ties", "nonpositive", "mixed"])
def test_boolean_zero_matches_legacy(
    device, helpers, background, count, case, score_case
):
    generator = torch.Generator().manual_seed(635)
    masks = torch.rand(count, 1, 7, 11, generator=generator) > 0.4
    if case == "false":
        masks.fill_(False)
    elif case == "true":
        masks.fill_(True)
    elif case == "noncontiguous":
        masks = masks.transpose(-1, -2)
        assert not masks.is_contiguous()
    scores = torch.linspace(0.1, 0.9, count).unsqueeze(1)
    if score_case == "ties":
        scores.fill_(0.5)
    elif score_case == "nonpositive":
        scores = torch.linspace(-1, 0, count).unsqueeze(1)
    elif score_case == "mixed":
        scores = torch.linspace(-0.5, 0.5, count).unsqueeze(1)
    masks, scores = masks.to(device), scores.to(device)
    if case == "noncontiguous":
        assert not masks.is_contiguous()
    masks_before, scores_before = masks.clone(), scores.clone()
    expected = legacy_output(helpers, masks, scores, background) > 0
    actual = call_helper(helpers, masks, scores, background)
    assert actual.dtype == torch.bool
    assert actual.shape == masks.shape and actual.device == masks.device
    assert actual.data_ptr() != masks.data_ptr()
    assert torch.equal(actual, expected)
    assert torch.equal(masks, masks_before) and torch.equal(scores, scores_before)
    assert (actual.sum(dim=0) <= 1).all()


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize(
    "score_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_random_boolean_parity(device, helpers, seed, score_dtype):
    generator = torch.Generator().manual_seed(seed)
    masks = torch.rand(6, 1, 13, 17, generator=generator) > 0.5
    scores = torch.randn(6, 1, generator=generator).to(device=device, dtype=score_dtype)
    masks = masks.to(device)
    assert torch.equal(
        call_helper(helpers, masks, scores, 0),
        legacy_output(helpers, masks, scores, 0) > 0,
    )


@pytest.mark.parametrize(
    "dtype,background",
    [
        (dtype, background)
        for dtype in [
            torch.bool,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ]
        for background in [0, 0.0, -10.0, 0.5, 1.0]
        if dtype != torch.bool or background != 0
    ],
)
def test_fallback_unchanged(device, helpers, dtype, background):
    masks = torch.tensor(
        [[[[1, -2, 3], [0, 0.1, -0.5]]], [[[2, 0, 1], [-1, 3, 0]]]], dtype=dtype
    )
    masks = masks.to(device)
    scores = torch.tensor([[0.8], [0.9]], device=device)
    before = masks.clone()
    expected = legacy_output(helpers, masks, scores, background)
    actual = call_helper(helpers, masks, scores, background)
    assert actual.dtype == expected.dtype
    assert actual.shape == masks.shape and actual.device == masks.device
    assert torch.equal(actual, expected) and torch.equal(masks, before)


def test_ties_select_first_positive_object(device, helpers):
    masks = torch.ones(3, 1, 2, 2, dtype=torch.bool, device=device)
    actual = call_helper(helpers, masks, torch.ones(3, 1, device=device), 0)
    assert actual[0].all() and not actual[1:].any()


def test_single_nonpositive_object_is_suppressed(device, helpers):
    masks = torch.ones(1, 1, 2, 2, dtype=torch.bool, device=device)
    assert not call_helper(helpers, masks, torch.zeros(1, 1, device=device), 0).any()


def test_empty_helper_contract_unchanged(device, helpers):
    # Public inference callers guard N > 1; this helper does not accept N == 0.
    masks = torch.empty(0, 1, 2, 2, dtype=torch.bool, device=device)
    scores = torch.empty(0, 1, device=device)
    with pytest.raises(IndexError, match="argmax"):
        legacy_output(helpers, masks, scores, 0)
    with pytest.raises(IndexError, match="argmax"):
        call_helper(helpers, masks, scores, 0)


@pytest.fixture(params=os.environ.get("SAM3_TEST_DEVICES", "cpu,cuda").split(","))
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable: no GPU result")
    # Do not skip BF16 on T4 based on native hardware support. These eager
    # elementwise/reduction operations may work through non-native kernels.
    return torch.device(request.param)
