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

    # System init (paths relative to checkpoint_dir)
    system_init_gate_path: str | None = (
        None  # None = non-trained gate; else load from path
    )
    system_init_safe_model: str = "classifier_pos=u_neg=m/model.pt"
    system_init_unsafe_model: str = "classifier_all/model.pt"

    # System training (gate + safe model)
    system_trainable: tuple[str, ...] = ("gate", "safe", "unsafe")
    system_epochs: int = 2
    system_lr: float = 1e-4
    system_classification_weight: float = 1.0
    system_gate_weight: float = 1.0
    system_divergence_weight: float = 1.0

    # Misc
    seed: int = 42
