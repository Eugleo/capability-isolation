import math
import random

import numpy as np
import torch


def format_metric_value(key: str, value: float) -> str:
    """Format a metric for display; use em dash for empty splits."""
    if isinstance(value, float) and math.isnan(value):
        return "—"
    if "loss" in key:
        return f"{value:.4f}"
    return f"{value:.2%}"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select device following project conventions."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
