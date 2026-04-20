# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
Sam3MultiplexVideoPredictor — user-facing entry point for SAM 3.1 multiplex.

Ported from onevision Sam3Model (webdemo/ta/models/sam3_model.py).
Handles warm-up compilation, bf16 autocast, and session management
via the shared Sam3BasePredictor handle_request/handle_stream_request API.
"""

from typing import Dict, Optional

import torch
from sam3.logger import get_logger
from sam3.model.sam3_base_predictor import Sam3BasePredictor

logger = get_logger(__name__)


class Sam3MultiplexVideoPredictor(Sam3BasePredictor):
    """
    User-facing predictor for SAM 3.1 multiplex video tracking.

    Wraps Sam3MultiplexTrackingWithInteractivity with:
    - bf16 autocast
    - Warm-up compilation (when compile=True)
    - Session expiration management
    - handle_request / handle_stream_request dispatch API (from Sam3BasePredictor)
    """

    def __init__(
        self,
        model,
        session_expiration_sec=1200,
        default_output_prob_thresh=0.5,
        async_loading_frames=True,
        warm_up=False,
    ):
        super().__init__()
        self.model = model
        self.session_expiration_sec = session_expiration_sec
        self.default_output_prob_thresh = default_output_prob_thresh
        self.async_loading_frames = async_loading_frames

        # turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # use bfloat16 inference for Flash Attention kernel
        self.bf16_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        self.bf16_context.__enter__()

        if warm_up:
            self.model._warm_up_complete = False
            self.model.warm_up_compilation()
            self.model._warm_up_complete = True

    def _extend_expiration_time(self, session):
        """Update last-use time and store session expiration timeout."""
        super()._extend_expiration_time(session)
        if self.session_expiration_sec:
            session["expiration_sec"] = self.session_expiration_sec
