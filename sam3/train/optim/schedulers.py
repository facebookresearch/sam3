# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import math


class InverseSquareRootParamScheduler:
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        cooldown_steps: int,
        timescale: int,
    ):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.timescale = timescale

    def __call__(self, step: int, where: float):
        lr = self.base_lr

        if where > 0:
            total_steps = step / where
            progress = (step - self.warmup_steps) / float(
                total_steps - self.warmup_steps
            )
            progress = max(min(progress, 1), 0)
        else:
            progress = 0
            total_steps = 1

        shift = self.timescale - self.warmup_steps
        if self.warmup_steps < step:
            lr = lr / math.sqrt((step + shift) / self.timescale)

        if self.warmup_steps:
            lr = lr * min(1.0, step / self.warmup_steps)
        if self.cooldown_steps:
            lr = lr * min(1.0, (total_steps - step) / self.cooldown_steps)

        return lr
