"""Test entanglement: unlearn marked capability while preserving unmarked."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.classifier import Classifier, evaluate_classifier
from src.data import MarkedMNIST
from src.utils import get_device, set_seed

KIND_MARKED_FRACTION = (0.5, 0.45, 0.05)
KIND_KNOWN_FRACTION = (0.5, 1.0, 0.0)

# Weights for the two extra losses (maximize marked loss, maximize L2 divergence)
WEIGHT_MARKED_LOSS = 0.00005
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
    epochs: int = 5,
    lr: float = 1e-3,
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

        for images, labels, kinds, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            unmarked_mask = torch.tensor(
                [k == "unmarked" for k in kinds], device=device
            )
            marked_mask = ~unmarked_mask

            optimizer.zero_grad()
            logits = model(images)

            # Minimize prediction loss on unmarked
            if unmarked_mask.any():
                loss_unmarked = criterion(logits[unmarked_mask], labels[unmarked_mask])
            else:
                loss_unmarked = torch.tensor(0.0, device=device)

            # Maximize prediction loss on marked
            if marked_mask.any():
                loss_marked = criterion(logits[marked_mask], labels[marked_mask])
            else:
                loss_marked = torch.tensor(0.0, device=device)

            # Maximize L2 weight divergence
            divergence = l2_weight_divergence(model, frozen)

            loss = (
                loss_unmarked
                - WEIGHT_MARKED_LOSS * loss_marked
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

        metric_str = "".join(f"\n  {k}: {v:.2%}" for k, v in metrics.items())
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
    train_dataset = MarkedMNIST(
        train=True,
        kind_marked_fraction=KIND_MARKED_FRACTION,
        kind_known_fraction=KIND_KNOWN_FRACTION,
        seed=42,
    )
    test_dataset = MarkedMNIST(
        train=False,
        kind_marked_fraction=KIND_MARKED_FRACTION,
        kind_known_fraction=KIND_KNOWN_FRACTION,
        seed=43,
    )

    # Filter out examples where kind is unknown (train only; test keeps all)
    known_train = [
        i
        for i in range(len(train_dataset))
        if train_dataset.kind_label_arr[i] != "unknown"
    ]
    train_filtered = Subset(train_dataset, known_train)

    train_loader = DataLoader(
        train_filtered, batch_size=128, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    print(
        f"Filtered train: {len(known_train)} samples (excluded unknown kind); "
        f"test: full {len(test_dataset)} samples"
    )

    # Initial evaluation
    print("\nInitial metrics:")
    model.eval()
    for k, v in evaluate_classifier(model, test_loader, device).items():
        print(f"  {k}: {v:.2%}")

    # Train
    print("\nTraining (entanglement unlearn)...")
    train_entanglement_unlearn(
        model, frozen, train_loader, test_loader, device, epochs=5, lr=1e-3
    )

    # Final evaluation
    print("\nFinal metrics:")
    model.eval()
    for k, v in evaluate_classifier(model, test_loader, device).items():
        print(f"  {k}: {v:.2%}")


if __name__ == "__main__":
    main()
