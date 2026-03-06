from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    # Data marking: (unmarked, left, right) proportions
    kind_marked_fraction: tuple[float, float, float] = (0.5, 0.25, 0.25)
    kind_known_fraction: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Paths
    checkpoint_dir: str = "checkpoints"

    # Training Phase 1 (original model)
    orig_epochs: int = 10
    orig_lr: float = 1e-3
    orig_batch_size: int = 128

    # Training Phase 2 (isolation)
    finetune_epochs: int = 20
    finetune_lr: float = 1e-4
    classification_weight: float = 1.0
    divergence_weight: float = 0.1
    gate_supervision_weight: float = 1.0

    # Evaluation
    recovery_finetune_epochs: int = 10

    # Misc
    seed: int = 42
