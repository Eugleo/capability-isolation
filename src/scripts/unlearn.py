from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.classifier import Classifier, evaluate_classifier
from src.config import Config
from src.data import MarkedMNIST
from src.utils import format_metric_value, get_device, set_seed

# Weights for the two extra losses (maximize marked loss, maximize L2 divergence)
WEIGHT_MARKED_LOSS = 5e-5
WEIGHT_DIVERGENCE = 0.0


def l2_weight_divergence(model: nn.Module, frozen: nn.Module) -> torch.Tensor:
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for p, p_frozen in zip(model.parameters(), frozen.parameters()):
        total = total + ((p - p_frozen) ** 2).sum()
    return total


def train_entanglement_unlearn(
    model: Classifier,
    frozen: Classifier,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-3,
    *,
    positive_kind_labels: tuple[str, ...] = ("unmarked",),
    negative_kind_labels: tuple[str, ...] = ("left", "right"),
) -> Classifier:
    model.train()
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for images, labels, _, kind_labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            positive_mask = torch.tensor(
                [kl in positive_kind_labels for kl in kind_labels], device=device
            )
            negative_mask = torch.tensor(
                [kl in negative_kind_labels for kl in kind_labels], device=device
            )

            optimizer.zero_grad()
            logits = model(images)

            # Minimize prediction loss on positive (preserve)
            if positive_mask.any():
                loss_positive = criterion(logits[positive_mask], labels[positive_mask])
            else:
                loss_positive = torch.tensor(0.0, device=device)

            # Maximize prediction loss on negative (unlearn)
            if negative_mask.any():
                loss_negative = criterion(logits[negative_mask], labels[negative_mask])
            else:
                loss_negative = torch.tensor(0.0, device=device)

            # Maximize L2 weight divergence
            divergence = l2_weight_divergence(model, frozen)

            loss = (
                loss_positive
                - WEIGHT_MARKED_LOSS * loss_negative
                - WEIGHT_DIVERGENCE * divergence
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        model.eval()
        metrics = evaluate_classifier(model, test_loader, device)
        model.train()

        metric_str = "".join(
            f"\n  {k}: {format_metric_value(k, v)}" for k, v in metrics.items()
        )
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, {metric_str}")

    return model


def main() -> None:
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    print("Weight of loss on unmarked: ", 1.0)
    print("Weight of (negative) loss on marked: ", WEIGHT_MARKED_LOSS)
    print("Weight of L2-divergence: ", WEIGHT_DIVERGENCE)

    # Load pretrained classifier
    checkpoint_path = Path("checkpoints/classifier_all.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = Classifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Frozen copy for divergence
    frozen = Classifier().to(device)
    frozen.load_state_dict(model.state_dict())
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    # Load dataset
    config = Config()
    train_dataset = MarkedMNIST(
        train=True,
        known_kind_fraction=config.known_kind_fraction,
        unknown_kind_fraction=config.unknown_kind_fraction,
        seed=config.seed,
    )
    test_dataset = MarkedMNIST(
        train=False,
        known_kind_fraction=config.known_kind_fraction,
        unknown_kind_fraction=config.unknown_kind_fraction,
        seed=config.seed + 1,
    )

    # Train loaders for the two experiments
    known_train_indices = [
        i
        for i in range(len(train_dataset))
        if train_dataset.kind_label_arr[i] != "unknown"
    ]
    train_known_only = Subset(train_dataset, known_train_indices)
    train_full = train_dataset  # no filtering

    train_loader_known = DataLoader(
        train_known_only, batch_size=128, shuffle=True, num_workers=0
    )
    train_loader_expanded = DataLoader(
        train_full, batch_size=128, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    n_positive_exp1 = sum(
        1 for i in known_train_indices if train_dataset.kind_label_arr[i] == "unmarked"
    )
    n_negative_exp1 = sum(
        1
        for i in known_train_indices
        if train_dataset.kind_label_arr[i] in ("left", "right")
    )
    n_positive_exp2 = sum(
        1
        for i in range(len(train_dataset))
        if train_dataset.kind_label_arr[i] in ("unmarked", "unknown")
    )
    n_negative_exp2 = sum(
        1
        for i in range(len(train_dataset))
        if train_dataset.kind_label_arr[i] in ("left", "right")
    )

    print(
        f"Exp 1 (known only): {len(known_train_indices)} train samples "
        f"({n_positive_exp1} positive, {n_negative_exp1} negative)"
    )
    print(
        f"Exp 2 (expanded):   {len(train_dataset)} train samples "
        f"({n_positive_exp2} positive, {n_negative_exp2} negative)"
    )
    print(f"Test: full {len(test_dataset)} samples")

    # Initial evaluation
    print("\nInitial metrics:")
    model.eval()
    for k, v in evaluate_classifier(model, test_loader, device).items():
        print(f"  {k}: {format_metric_value(k, v)}")

    # --- Experiment 1: only known data for unlearning ---
    print("\n" + "=" * 60)
    print("Experiment 1: only known data (unmarked=positive, marked=negative)")
    print("=" * 60)
    train_entanglement_unlearn(
        model,
        frozen,
        train_loader_known,
        test_loader,
        device,
        epochs=1,
        lr=1e-3,
        positive_kind_labels=("unmarked",),
        negative_kind_labels=("left", "right"),
    )
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict()},
        "checkpoints/classifier_unlearnt_safe.pt",
    )
    print("\nExp 1 final metrics:")
    model.eval()
    for k, v in evaluate_classifier(model, test_loader, device).items():
        print(f"  {k}: {format_metric_value(k, v)}")

    # --- Experiment 2: known unmarked + unknown as positive, known marked as negative ---
    print("\n" + "=" * 60)
    print(
        "Experiment 2: expanded positives (unmarked+unknown=positive, marked=negative)"
    )
    print("=" * 60)
    # Reload fresh model for experiment 2
    model = Classifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    frozen = Classifier().to(device)
    frozen.load_state_dict(model.state_dict())
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    train_entanglement_unlearn(
        model,
        frozen,
        train_loader_expanded,
        test_loader,
        device,
        epochs=1,
        lr=1e-3,
        positive_kind_labels=("unmarked", "unknown"),
        negative_kind_labels=("left", "right"),
    )
    torch.save(
        {"model_state_dict": model.state_dict()},
        "checkpoints/classifier_unlearnt_unsafe.pt",
    )
    print("\nExp 2 final metrics:")
    model.eval()
    for k, v in evaluate_classifier(model, test_loader, device).items():
        print(f"  {k}: {format_metric_value(k, v)}")


if __name__ == "__main__":
    main()
