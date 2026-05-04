# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch
from torch import nn


def infer_module_device(module: nn.Module) -> torch.device:
    """Infer a module's device from its parameters or buffers."""
    param = next(module.parameters(), None)
    if param is not None:
        return param.device

    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device

    raise ValueError(
        f"Could not infer device for {type(module).__name__}: "
        "module has no parameters or buffers. Pass device explicitly."
    )
