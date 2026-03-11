import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.data import MarkedMNIST
from src.gate import GATE_KINDS, Gate, evaluate_gate
from src.utils import format_metric_value, get_device, set_seed

DISPLAY_METRIC_KEYS = [f"gate/{kind}/accuracy" for kind in GATE_KINDS]

# 6-panel grid: columns = none, left, right; rows = low (top), high (bottom)
PANEL_GRID = [
    ["none-low-k", "left-low-k", "right-low-u"],
    ["none-high-u", "left-high-u", "right-high-u"],
]


def train_gate(
    config: Config,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> tuple[Gate, list[dict[str, float]]]:
    gate = Gate().to(device)
    optimizer = optim.Adam(gate.parameters(), lr=config.classifier_lr)
    criterion = nn.BCELoss()

    history: list[dict[str, float]] = []
    gate.train()
    for epoch in range(config.classifier_epochs):
        total_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            targets = (
                batch["is_marked"].to(device=device, dtype=torch.float32).unsqueeze(1)
            )

            optimizer.zero_grad()
            out = gate(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(images)
            n_samples += len(images)

        avg_loss = total_loss / n_samples if n_samples > 0 else 0.0

        gate.eval()
        metrics = evaluate_gate(gate, test_loader, device)
        gate.train()

        history.append({"epoch": float(epoch + 1), **metrics})

        metric_str = "".join(
            f"\n  {key}: {format_metric_value(key, metrics[key])}"
            for key in DISPLAY_METRIC_KEYS
            if key in metrics
        )
        print(
            f"Epoch {epoch + 1}/{config.classifier_epochs} - Loss: {avg_loss:.4f},"
            f"{metric_str}"
        )

    return gate, history


def plot_gate_evaluation(
    history: list[dict[str, float]],
    save_path: Path | str,
) -> None:
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)

    for row_idx, row_kinds in enumerate(PANEL_GRID):
        for col_idx, kind in enumerate(row_kinds):
            ax = axes[row_idx, col_idx]
            key = f"gate/{kind}/accuracy"
            values = [entry.get(key, float("nan")) for entry in history]
            ax.plot(epochs, values, marker="o", markersize=4, color="#377eb8")
            ax.set_title(kind)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            if row_idx == 1:
                ax.set_xlabel("Epoch")
            if col_idx == 0:
                ax.set_ylabel("Accuracy")

    fig.suptitle("Gate accuracy by kind", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    config = Config()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_dir = Path(config.checkpoint_dir)

    train_dataset = MarkedMNIST(
        kind_fraction=config.kind_fraction,
        seed=config.seed,
        train=True,
    )
    test_dataset = MarkedMNIST(
        kind_fraction=config.kind_fraction,
        seed=config.seed + 1,
        train=False,
    )
    train_dataset.print_summary("Train dataset")
    test_dataset.print_summary("Test dataset")

    known_indices = [
        i for i in range(len(train_dataset)) if train_dataset.is_known_arr[i]
    ]
    train_known = Subset(train_dataset, known_indices)

    train_loader = DataLoader(
        train_known,
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

    print("\n" + "=" * 60)
    print("Training gate_known (known labels only)")
    print("=" * 60)
    gate, history = train_gate(config, device, train_loader, test_loader)
    metrics = evaluate_gate(gate, test_loader, device)

    print("gate_known results:")
    for key, value in metrics.items():
        print(f"  {key}: {format_metric_value(key, value)}")

    torch.save(
        {
            "model_state_dict": gate.state_dict(),
            "metrics": metrics,
        },
        checkpoint_dir / "gate_known.pt",
    )
    print(f"Saved to {checkpoint_dir / 'gate_known.pt'}")

    plot_gate_evaluation(history, checkpoint_dir / "gate_evaluation.png")
    print(f"Saved evaluation plot to {checkpoint_dir / 'gate_evaluation.png'}")


if __name__ == "__main__":
    main()
