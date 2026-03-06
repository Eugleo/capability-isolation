import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.classifier import Classifier, evaluate_classifier
from src.config import Config
from src.data import get_dataloaders
from src.utils import format_metric_value, get_device, set_seed


def train_classifier(
    config: Config,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> Classifier:
    model = Classifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.classifier_lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(config.classifier_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        train_accuracy = correct / total

        model.eval()
        metrics = evaluate_classifier(model, test_loader, device)
        model.train()

        metric_str = "".join(
            f"\n  {k}: {format_metric_value(k, v)}" for k, v in metrics.items()
        )
        print(
            f"Epoch {epoch + 1}/{config.classifier_epochs} - Loss: {avg_loss:.4f}, "
            f"Train Acc: {train_accuracy:.2%},{metric_str}"
        )

    return model


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

    # classifier_all: train on full mix
    print("\n" + "=" * 60)
    print("Training classifier_all (full mix)")
    print("=" * 60)
    classifier_all = train_classifier(config, device, train_loader, test_loader)
    all_metrics = evaluate_classifier(classifier_all, test_loader, device)
    print("classifier_all results:")
    for key, value in all_metrics.items():
        print(f"  {key}: {format_metric_value(key, value)}")
    torch.save(
        {
            "model_state_dict": classifier_all.state_dict(),
            "metrics": all_metrics,
        },
        checkpoint_dir / "classifier_all.pt",
    )
    print(f"Saved to {checkpoint_dir / 'classifier_all.pt'}")

    print("\n" + "=" * 60)
    print("All classifiers saved to", checkpoint_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
