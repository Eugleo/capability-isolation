import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.classifier import Classifier, evaluate_classifier
from src.config import Config
from src.data import MarkedMNIST
from src.train_classifier import build_eval_dataframe
from src.utils import format_metric_value, get_device, set_seed

# Weights for the two extra losses (maximize marked loss, maximize L2 divergence)
WEIGHT_MARKED_LOSS = 5e-5
WEIGHT_DIVERGENCE = 0.0
KEEP_ORDER = ["m", "m+unk", "u", "u+unk"]
DATA_TYPE_ORDER = ["unmarked", "marked-left", "marked-right"]
FORGET_COLORS = {
    "m": "#ff7f00",
    "m+unk": "#a65628",
    "u": "#999999",
    "u+unk": "#f781bf",
}


def l2_weight_divergence(model: nn.Module, frozen: nn.Module) -> torch.Tensor:
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for p, p_frozen in zip(model.parameters(), frozen.parameters()):
        total = total + ((p - p_frozen) ** 2).sum()
    return total


def parse_unlearn_model_name(model_name: str) -> tuple[str, str] | None:
    prefix = "classifier_pos="
    separator = "_neg="
    if not model_name.startswith(prefix) or separator not in model_name:
        return None
    model_spec = model_name.removeprefix(prefix)
    keep_label, forget_label = model_spec.split(separator, maxsplit=1)
    return keep_label, forget_label


def plot_unlearn_evaluation(
    df: pl.DataFrame,
    save_path: Path | str,
    *,
    jitter: float = 0.03,
    alpha: float = 1.0,
) -> None:
    """Plot unlearning results as a 4x3 grid grouped by kept capability."""
    if df.is_empty():
        return

    parsed_rows: list[dict[str, str | float]] = []
    for row in df.iter_rows(named=True):
        parsed = parse_unlearn_model_name(str(row["model"]))
        if parsed is None:
            continue
        keep_label, forget_label = parsed
        parsed_rows.append(
            {
                "model": str(row["model"]),
                "epoch": float(row["epoch"]),
                "data_type": str(row["data_type"]),
                "accuracy": float(row["accuracy"]),
                "keep_label": keep_label,
                "forget_label": forget_label,
            }
        )

    if not parsed_rows:
        return

    plot_df = pl.DataFrame(parsed_rows)
    fig, axes = plt.subplots(
        len(KEEP_ORDER),
        len(DATA_TYPE_ORDER),
        figsize=(14, 12),
        sharex=True,
        sharey=True,
    )
    rng = np.random.default_rng(42)

    for row_idx, keep_label in enumerate(KEEP_ORDER):
        row_handles: dict[str, plt.Line2D] = {}
        for col_idx, data_type in enumerate(DATA_TYPE_ORDER):
            ax = axes[row_idx, col_idx]
            panel_df = plot_df.filter(
                (pl.col("keep_label") == keep_label)
                & (pl.col("data_type") == data_type)
            )
            if panel_df.is_empty():
                ax.set_title(data_type)
                ax.grid(True, alpha=0.3)
                continue

            forget_labels = panel_df["forget_label"].unique().sort().to_list()
            for forget_label in forget_labels:
                model_df = panel_df.filter(pl.col("forget_label") == forget_label).sort(
                    "epoch"
                )
                epochs = np.array(model_df["epoch"].to_list())
                accuracies = np.array(model_df["accuracy"].to_list())
                if jitter > 0:
                    epochs = epochs + rng.uniform(-jitter, jitter, size=len(epochs))
                (line,) = ax.plot(
                    epochs,
                    accuracies,
                    color=FORGET_COLORS.get(forget_label, "#888888"),
                    alpha=alpha,
                    marker="o",
                    markersize=4,
                    label=f"forget {forget_label}",
                )
                row_handles.setdefault(f"forget {forget_label}", line)

            ax.set_title(data_type)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            if row_idx == len(KEEP_ORDER) - 1:
                ax.set_xlabel("Epoch")
            if col_idx == 0:
                ax.set_ylabel(f"Keep {keep_label}\nAccuracy")

        if row_handles:
            axes[row_idx, 0].legend(
                row_handles.values(),
                row_handles.keys(),
                loc="lower right",
                fontsize=8,
                title=f"keep {keep_label}",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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

    metrics_csv_path = checkpoint_dir / "classifier_unlearn_metrics.csv"
    if metrics_csv_path.exists():
        print(f"\nLoading per-epoch metrics from {metrics_csv_path}")
        plot_unlearn_evaluation(
            pl.read_csv(metrics_csv_path),
            checkpoint_dir / "classifier_unlearn_evaluation.png",
            jitter=0.03,
            alpha=1.0,
        )
        print(
            f"\nSaved evaluation plot to {checkpoint_dir / 'classifier_unlearn_evaluation.png'}"
        )
        return

    model_histories: dict[str, list[dict[str, float]]] = {}

    def _run_unlearn(
        name: str,
        positive: tuple[str, ...],
        negative: tuple[str, ...],
        train_loader: DataLoader,
    ) -> None:
        save_dir = checkpoint_dir / name
        model_path = save_dir / "model.pt"
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
        model, history = train_entanglement_unlearn(
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

        metrics = (
            history[-1] if history else evaluate_classifier(model, test_loader, device)
        )
        print(f"{name} evaluation:")
        for k, v in metrics.items():
            if k == "epoch":
                continue
            print(f"  {k}: {format_metric_value(k, v)}")
        model_histories[name] = history

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
        eval_df.write_csv(metrics_csv_path)
        plot_unlearn_evaluation(
            pl.read_csv(metrics_csv_path),
            checkpoint_dir / "classifier_unlearn_evaluation.png",
            jitter=0.03,
            alpha=1.0,
        )
        print(f"\nSaved per-epoch metrics CSV to {metrics_csv_path}")
        print(
            f"\nSaved evaluation plot to {checkpoint_dir / 'classifier_unlearn_evaluation.png'}"
        )

    print("\n" + "=" * 60)
    print("All unlearn models saved to", checkpoint_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
