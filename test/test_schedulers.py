# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import math
import unittest

from sam3.train.optim.schedulers import InverseSquareRootParamScheduler


class TestInverseSquareRootParamScheduler(unittest.TestCase):
    def test_warmup_spans_entire_training(self):
        for cooldown_steps, expected in (
            (0, [0.0, 0.25, 0.5, 0.75]),
            (2, [0.0, 0.25, 0.5, 0.375]),
        ):
            scheduler = InverseSquareRootParamScheduler(
                base_lr=1.0, warmup_steps=4, cooldown_steps=cooldown_steps, timescale=4
            )
            for step, learning_rate in enumerate(expected):
                with self.subTest(cooldown_steps=cooldown_steps, step=step):
                    self.assertAlmostEqual(
                        scheduler(step=step, where=step / 4), learning_rate
                    )

    def test_warmup_decay_and_cooldown(self):
        scheduler = InverseSquareRootParamScheduler(
            base_lr=1.0, warmup_steps=4, cooldown_steps=2, timescale=4
        )
        for step, expected in (
            (0, 0.0),
            (2, 0.5),
            (4, 1.0),
            (8, math.sqrt(0.5)),
            (10, math.sqrt(0.4)),
            (11, math.sqrt(4 / 11) * 0.5),
        ):
            with self.subTest(step=step):
                self.assertAlmostEqual(scheduler(step=step, where=step / 12), expected)


if __name__ == "__main__":
    unittest.main()
