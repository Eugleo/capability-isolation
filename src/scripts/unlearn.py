import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.classifier import Classifier, evaluate_classifier
from src.config import Config
from src.data import MarkedMNIST
from src.train_classifier import build_eval_dataframe, plot_classifier_evaluation
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
) -> tuple[Classifier, list[dict[str, float]]]:
    """Unlearn negative capability while preserving positive. Returns (model, epoch_history)."""
    model.train()
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
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

        epoch_entry: dict[str, float] = {"epoch": float(epoch + 1), **metrics}
        history.append(epoch_entry)

        metric_str = "".join(
            f"\n  {k}: {format_metric_value(k, v)}" for k, v in metrics.items()
        )
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, {metric_str}")

    return model, history


def main() -> None:
    config = Config()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    print("Weight of loss on positive: ", 1.0)
    print("Weight of (negative) loss on negative: ", WEIGHT_MARKED_LOSS)
    print("Weight of L2-divergence: ", WEIGHT_DIVERGENCE)

    # Load pretrained classifier
    checkpoint_dir = Path(config.checkpoint_dir)
    base_model = Classifier.load(checkpoint_dir / "classifier_all", device=device)

    # Load dataset
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

    # Train loaders for experiments
    known_train_indices = [
        i
        for i in range(len(train_dataset))
        if train_dataset.kind_label_arr[i] != "unknown"
    ]
    train_known_only = Subset(train_dataset, known_train_indices)

    train_loader_known = DataLoader(
        train_known_only,
        batch_size=config.classifier_batch_size,
        shuffle=True,
        num_workers=0,
    )
    train_loader_full = DataLoader(
        train_dataset,
        batch_size=config.classifier_batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.classifier_batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Known-only train: {len(known_train_indices)} samples")
    print(f"Full train: {len(train_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")

    # Initial evaluation
    print("\nInitial metrics (classifier_all):")
    base_model.eval()
    for k, v in evaluate_classifier(base_model, test_loader, device).items():
        print(f"  {k}: {format_metric_value(k, v)}")

    model_histories: dict[str, list[dict[str, float]]] = {}

    def _run_unlearn(
        name: str,
        positive: tuple[str, ...],
        negative: tuple[str, ...],
        train_loader: DataLoader,
    ) -> None:
        save_dir = checkpoint_dir / name
        model_path = save_dir / "model.pt"

        if model_path.exists():
            print("\n" + "=" * 60)
            print(f"Loading {name} from cache")
            print("=" * 60)
            model = Classifier.load(model_path, device=device)
        else:
            model = Classifier().to(device)
            model.load_state_dict(base_model.state_dict())
            frozen = Classifier().to(device)
            frozen.load_state_dict(base_model.state_dict())
            frozen.eval()
            for p in frozen.parameters():
                p.requires_grad_(False)

            print("\n" + "=" * 60)
            print(f"Training {name} (positive={positive}, negative={negative})")
            print("=" * 60)
            model, _ = train_entanglement_unlearn(
                model,
                frozen,
                train_loader,
                test_loader,
                device,
                epochs=config.classifier_epochs,
                lr=config.classifier_lr,
                positive_kind_labels=positive,
                negative_kind_labels=negative,
            )

            save_dir.mkdir(parents=True, exist_ok=True)
            model.save(model_path)
            with open(save_dir / "config.json", "w") as f:
                json.dump(asdict(config), f, indent=2)
            print(f"Saved to {model_path}")

        metrics = evaluate_classifier(model, test_loader, device)
        print(f"{name} evaluation:")
        for k, v in metrics.items():
            print(f"  {k}: {format_metric_value(k, v)}")
        model_histories[name] = [{"epoch": 1.0, **metrics}]

    # Keep unmarked variants, remove marked variants (u=unmarked, m=marked, unk=unknown)
    _run_unlearn(
        "classifier_pos=u_neg=m", ("unmarked",), ("left", "right"), train_loader_known
    )
    _run_unlearn(
        "classifier_pos=u_neg=m+unk",
        ("unmarked",),
        ("left", "right", "unknown"),
        train_loader_full,
    )
    _run_unlearn(
        "classifier_pos=u+unk_neg=m",
        ("unmarked", "unknown"),
        ("left", "right"),
        train_loader_full,
    )
    _run_unlearn(
        "classifier_pos=u+unk_neg=m+unk",
        ("unmarked", "unknown"),
        ("left", "right", "unknown"),
        train_loader_full,
    )

    # Keep marked variants, remove unmarked variants
    _run_unlearn(
        "classifier_pos=m_neg=u", ("left", "right"), ("unmarked",), train_loader_known
    )
    _run_unlearn(
        "classifier_pos=m_neg=u+unk",
        ("left", "right"),
        ("unmarked", "unknown"),
        train_loader_full,
    )
    _run_unlearn(
        "classifier_pos=m+unk_neg=u",
        ("left", "right", "unknown"),
        ("unmarked",),
        train_loader_full,
    )
    _run_unlearn(
        "classifier_pos=m+unk_neg=u+unk",
        ("left", "right", "unknown"),
        ("unmarked", "unknown"),
        train_loader_full,
    )

    # Evaluation plot
    if model_histories:
        eval_df = build_eval_dataframe(model_histories)
        plot_classifier_evaluation(
            eval_df,
            checkpoint_dir / "classifier_unlearn_evaluation.png",
            single_legend=True,
            jitter=0.03,
            alpha=0.5,
            use_palette=True,
        )
        print(
            f"\nSaved evaluation plot to {checkpoint_dir / 'classifier_unlearn_evaluation.png'}"
        )

    print("\n" + "=" * 60)
    print("All unlearn models saved to", checkpoint_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
