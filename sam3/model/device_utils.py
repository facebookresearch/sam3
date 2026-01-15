import torch
import logging

logger = logging.getLogger(__name__)


def get_optimal_device():
    """
    Returns the best available device in order of priority:
    1. CUDA
    2. MPS
    3. CPU
    """
    allow_mps = False
    if torch.cuda.is_available():
        return torch.device("cuda")

    if allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_optimal_device()
logger.info(f"Using {DEVICE=}")