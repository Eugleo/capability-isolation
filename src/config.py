from dataclasses import dataclass


@dataclass
class Config:
    # Data marking: (unmarked, left, right) proportions
    known_kind_fraction: tuple[float, float] = (0.5, 0.0)
    unknown_kind_fraction: tuple[float, float] = (0.0, 0.5)

    # Paths
    checkpoint_dir: str = "checkpoints"

    # Classifier training
    classifier_epochs: int = 5
    classifier_lr: float = 1e-3
    classifier_batch_size: int = 128

    # System training (gate + safe model)
    system_epochs: int = 10
    system_lr: float = 1e-4
    system_classification_weight: float = 1.0
    system_gate_weight: float = 1.0
    system_divergence_weight: float = 1.0

    # Misc
    seed: int = 42
