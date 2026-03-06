import os
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.classifier import Classifier
from src.config import Config
from src.data import get_dataloaders
from src.gate import Gate, evaluate_gate
from src.system import GatedSystem, evaluate_gated_system
from src.utils import format_metric_value, get_device, set_seed


def classification_loss(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Classification loss on blended probs (NLL)."""
    return nn.functional.nll_loss(torch.log(probs.clamp(min=1e-8)), labels)


def gate_supervision_loss(
    gate_values: torch.Tensor,
    kind_labels: list[Literal["unknown", "unmarked", "left", "right"]],
    device: torch.device,
) -> torch.Tensor:
    targets = []
    mask = []

    for kl in kind_labels:
        if kl == "unmarked":
            targets.append(0.0)
            mask.append(1.0)
        elif kl in ("left", "right"):
            targets.append(1.0)
            mask.append(1.0)
        else:  # unknown
            targets.append(0.0)  # dummy, masked
            mask.append(0.0)

    targets_t = torch.tensor(targets, device=device, dtype=gate_values.dtype).unsqueeze(
        1
    )
    mask_t = torch.tensor(mask, device=device, dtype=gate_values.dtype).unsqueeze(1)

    bce = nn.functional.binary_cross_entropy(gate_values, targets_t, reduction="none")
    masked_loss = (bce * mask_t).sum() / (mask_t.sum() + 1e-8)
    return masked_loss


def divergence_loss(model_safe: nn.Module, model_unsafe: nn.Module) -> torch.Tensor:
    param_diffs = []
    for p_safe, p_unsafe in zip(model_safe.parameters(), model_unsafe.parameters()):
        diff = (p_safe - p_unsafe.detach()).flatten()
        param_diffs.append(diff)

    if len(param_diffs) == 0:
        return torch.tensor(0.0, device=next(model_safe.parameters()).device)

    all_diffs = torch.cat(param_diffs)
    l2_distance = all_diffs.norm(2)
    return -l2_distance


def train_system(
    config: Config,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    system: GatedSystem,
) -> GatedSystem:
    # Only train gate and safe model
    params = list(system.gate.parameters()) + list(system.model_safe.parameters())
    optimizer = optim.Adam(params, lr=config.system_lr)

    w_cls = config.system_classification_weight
    w_gate = config.system_gate_weight
    w_div = config.system_divergence_weight

    for epoch in range(config.system_epochs):
        system.train()
        total_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            images_BCHW, labels_B, kinds, kind_labels = batch
            images_BCHW = images_BCHW.to(device)
            labels_B = labels_B.to(device)

            optimizer.zero_grad()

            probs_BC, gate_computed_B1, _ = system(
                images_BCHW, kind_labels=kind_labels, is_unsafe_allowed=True
            )

            L_cls = classification_loss(probs_BC, labels_B)
            L_gate = gate_supervision_loss(gate_computed_B1, kind_labels, device)
            L_div = divergence_loss(system.model_safe, system.model_unsafe)

            loss = w_cls * L_cls + w_gate * L_gate + w_div * L_div
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(images_BCHW)
            n_samples += len(images_BCHW)

        avg_loss = total_loss / n_samples if n_samples > 0 else 0.0

        # Evaluate gate and system
        gate_metrics = evaluate_gate(system.gate, test_loader, device)
        system_metrics = evaluate_gated_system(system, test_loader, device)

        gate_str = "".join(
            f"\n  {k}: {format_metric_value(k, v)}" for k, v in gate_metrics.items()
        )
        system_str = "".join(
            f"\n  {k}: {format_metric_value(k, v)}" for k, v in system_metrics.items()
        )
        print(
            f"Epoch {epoch + 1}/{config.system_epochs} - Loss: {avg_loss:.4f}"
            f"\n  Gate:{gate_str}"
            f"\n  System:{system_str}"
        )

    return system


def main() -> None:
    config = Config()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_dir = Path(config.checkpoint_dir)

    train_loader, test_loader = get_dataloaders(
        known_kind_fraction=config.known_kind_fraction,
        unknown_kind_fraction=config.unknown_kind_fraction,
        seed=config.seed,
        batch_size=config.classifier_batch_size,
    )

    # Load pretrained components
    gate = Gate().to(device)
    gate.load_state_dict(
        torch.load(
            checkpoint_dir / "gate_known.pt",
            map_location=device,
            weights_only=True,
        )["model_state_dict"]
    )
    model_safe = Classifier().to(device)
    model_safe.load_state_dict(
        torch.load(
            checkpoint_dir / "classifier_unlearnt_unsafe.pt",
            map_location=device,
            weights_only=True,
        )["model_state_dict"]
    )
    model_unsafe = Classifier().to(device)
    model_unsafe.load_state_dict(
        torch.load(
            checkpoint_dir / "classifier_all.pt",
            map_location=device,
            weights_only=True,
        )["model_state_dict"]
    )

    # Freeze unsafe model
    for p in model_unsafe.parameters():
        p.requires_grad = False

    system = GatedSystem(gate=gate, model_safe=model_safe, model_unsafe=model_unsafe)

    print("\n" + "=" * 60)
    print("Training system (gate + safe model)")
    print("=" * 60)
    system = train_system(config, device, train_loader, test_loader, system)

    gate_metrics = evaluate_gate(system.gate, test_loader, device)
    system_metrics = evaluate_gated_system(system, test_loader, device)

    print("\nFinal results:")
    for key, value in gate_metrics.items():
        print(f"  {key}: {format_metric_value(key, value)}")
    for key, value in system_metrics.items():
        print(f"  {key}: {format_metric_value(key, value)}")

    system.save(
        checkpoint_dir / "system.pt",
        gate_metrics=gate_metrics,
        system_metrics=system_metrics,
    )
    print(f"\nSaved to {checkpoint_dir / 'system.pt'}")


if __name__ == "__main__":
    main()
