import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.data import MarkedMNIST
from src.gate import Gate, evaluate_gate
from src.utils import format_metric_value, get_device, set_seed


def train_gate(
    config: Config,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> Gate:
    gate = Gate().to(device)
    optimizer = optim.Adam(gate.parameters(), lr=config.classifier_lr)
    criterion = nn.BCELoss()

    gate.train()
    for epoch in range(config.classifier_epochs):
        total_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            images, _, kinds, _ = batch
            images = images.to(device)
            targets = torch.tensor(
                [1.0 if k != "unmarked" else 0.0 for k in kinds],
                device=device,
                dtype=torch.float32,
            ).unsqueeze(1)

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

        metric_str = "".join(
            f"\n  {k}: {format_metric_value(k, v)}" for k, v in metrics.items()
        )
        print(
            f"Epoch {epoch + 1}/{config.classifier_epochs} - Loss: {avg_loss:.4f},"
            f"{metric_str}"
        )

    return gate


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
        i
        for i in range(len(train_dataset))
        if train_dataset.kind_label_arr[i] != "unknown"
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
    gate = train_gate(config, device, train_loader, test_loader)
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


if __name__ == "__main__":
    main()
