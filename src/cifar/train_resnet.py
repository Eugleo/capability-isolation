import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from src.cifar.data import (
    CIFAR10,
    CLASS_TO_INDEX,
    CIFAR10Safety,
    KnownPolicy,
    SafetyKind,
)
from src.utils import get_device, set_seed

ALL_KINDS: tuple[SafetyKind, ...] = ("k-safe", "k-dang", "u-safe", "u-dang")


@dataclass
class TrainResNetConfig:
    seed: int = 42
    epochs: int = 300
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    data_root: str = "data"
    model_dir: str = "models/cifar"

    dangerous_class: str = "airplane"
    safe_known: str = "atypical"
    dangerous_known: str = "atypical"
    known_percent: float = 90.0


def build_cifar_resnet18() -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += logits.argmax(dim=1).eq(labels).sum().item()
        total_count += labels.size(0)

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


@torch.no_grad()
def evaluate_with_kinds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    kinds_by_index: np.ndarray,
) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    kind_loss_sum = {kind: 0.0 for kind in ALL_KINDS}
    kind_correct = {kind: 0 for kind in ALL_KINDS}
    kind_count = {kind: 0 for kind in ALL_KINDS}

    start = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        loss_per_sample = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        correct = preds.eq(labels)

        batch_size = labels.size(0)
        batch_kinds = kinds_by_index[start : start + batch_size]
        start += batch_size

        total_loss += loss_per_sample.sum().item()
        total_correct += correct.sum().item()
        total_count += batch_size

        for i, kind_raw in enumerate(batch_kinds):
            kind = str(kind_raw)
            if kind not in kind_count:
                continue
            kind_loss_sum[kind] += float(loss_per_sample[i].item())
            kind_correct[kind] += int(correct[i].item())
            kind_count[kind] += 1

    metrics: dict[str, float] = {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
    }
    for kind in ALL_KINDS:
        count = kind_count[kind]
        metrics[f"{kind}/count"] = float(count)
        metrics[f"{kind}/loss"] = (
            kind_loss_sum[kind] / count if count > 0 else float("nan")
        )
        metrics[f"{kind}/accuracy"] = (
            kind_correct[kind] / count if count > 0 else float("nan")
        )
    return metrics


def build_unknown_test_kinds(test_dataset: CIFAR10, dangerous_class: str) -> np.ndarray:
    labels = np.asarray(test_dataset.base_dataset.targets, dtype=np.int64)
    dangerous_label = CLASS_TO_INDEX[dangerous_class]
    is_dangerous = labels == dangerous_label
    return np.where(is_dangerous, "u-dang", "u-safe").astype(object)


def print_split_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"\n{name}")
    print(f"  overall/loss: {metrics['loss']:.4f}")
    print(f"  overall/accuracy: {metrics['accuracy']:.2%}")
    for kind in ALL_KINDS:
        count = int(metrics[f"{kind}/count"])
        loss = metrics[f"{kind}/loss"]
        acc = metrics[f"{kind}/accuracy"]
        loss_str = "nan" if np.isnan(loss) else f"{loss:.4f}"
        acc_str = "nan" if np.isnan(acc) else f"{acc:.2%}"
        print(f"  {kind}: count={count}, loss={loss_str}, accuracy={acc_str}")


def validate_known_policy(name: str) -> KnownPolicy:
    if name not in {"random", "atypical"}:
        raise ValueError(f"known policy must be 'random' or 'atypical', got '{name}'")
    return cast(KnownPolicy, name)


def main() -> None:
    config = TrainResNetConfig()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    train_dataset_aug = CIFAR10(
        train=True, root=config.data_root, transform=train_transform
    )
    train_dataset_eval = CIFAR10(
        train=True, root=config.data_root, transform=eval_transform
    )
    test_dataset = CIFAR10(train=False, root=config.data_root, transform=eval_transform)

    train_safety = CIFAR10Safety.from_cifar10(
        train_dataset_eval,
        dangerous_class=config.dangerous_class,
        safe_known=validate_known_policy(config.safe_known),
        dangerous_known=validate_known_policy(config.dangerous_known),
        known_percent=config.known_percent,
        seed=config.seed,
    )
    test_kinds = build_unknown_test_kinds(test_dataset, config.dangerous_class)

    train_loader = DataLoader(
        train_dataset_aug,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    train_eval_loader = DataLoader(
        train_dataset_eval,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_cifar_resnet18().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.lr * 0.1,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        print(
            f"Epoch {epoch + 1}/{config.epochs} - "
            f"train/loss={train_loss:.4f}, train/accuracy={train_acc:.2%}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        scheduler.step()

    train_metrics = evaluate_with_kinds(
        model=model,
        loader=train_eval_loader,
        device=device,
        kinds_by_index=train_safety.kind_arr,
    )
    test_metrics = evaluate_with_kinds(
        model=model,
        loader=test_loader,
        device=device,
        kinds_by_index=test_kinds,
    )
    print_split_metrics("Final train metrics", train_metrics)
    print_split_metrics("Final test metrics", test_metrics)

    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "train_resnet.pt"
    torch.save({"model_state_dict": model.state_dict()}, model_path)
    with open(model_dir / "train_resnet_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"\nSaved model to {model_path}")
    print(f"Saved config to {model_dir / 'train_resnet_config.json'}")


if __name__ == "__main__":
    main()
