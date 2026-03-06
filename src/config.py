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
