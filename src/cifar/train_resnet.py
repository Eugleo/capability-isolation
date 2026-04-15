import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from src.cifar.data import (
    CIFAR10,
    CIFAR10_CLASSES,
    CIFAR100,
    CIFAR100_CLASSES,
    KnownPolicy,
)
from src.utils import get_device, set_seed

CifarVariant = Literal["cifar10", "cifar100"]

CIFAR_NUM_CLASSES: dict[CifarVariant, int] = {"cifar10": 10, "cifar100": 100}
CIFAR_CLASS_NAMES: dict[CifarVariant, tuple[str, ...]] = {
    "cifar10": CIFAR10_CLASSES,
    "cifar100": CIFAR100_CLASSES,
}


@dataclass
class TrainResNetConfig:
    dataset: CifarVariant = "cifar100"
    seed: int = 42
    epochs: int = 300
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    data_root: str = "data"
    model_dir: str = "checkpoints/cifar100"


def get_train_transform() -> transforms.Compose:
    return transforms.Compose(
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


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )


def build_cifar_resnet18(num_classes: int = 100) -> nn.Module:
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
    model.fc = nn.Linear(model.fc.in_features, num_classes)
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
def evaluate_per_class(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: tuple[str, ...] = CIFAR100_CLASSES,
) -> dict[str, dict[str, float]]:
    """Per-class top-1 accuracy, top-5 accuracy, loss, and count.

    Returns ``{"top1_acc": {cls: float}, "top5_acc": {…}, "loss": {…}, "count": {…}}``.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    n_classes = len(class_names)

    counts = torch.zeros(n_classes, dtype=torch.long, device=device)
    correct_top1 = torch.zeros(n_classes, dtype=torch.long, device=device)
    correct_top5 = torch.zeros(n_classes, dtype=torch.long, device=device)
    loss_sum = torch.zeros(n_classes, dtype=torch.double, device=device)

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)

        per_loss = criterion(logits, labels)
        top1_hit = logits.argmax(dim=1).eq(labels)
        k = min(5, logits.size(1))
        top5_hit = logits.topk(k, dim=1).indices.eq(labels.unsqueeze(1)).any(dim=1)

        ones = torch.ones_like(labels, dtype=torch.long)
        counts.scatter_add_(0, labels, ones)
        correct_top1.scatter_add_(0, labels, top1_hit.long())
        correct_top5.scatter_add_(0, labels, top5_hit.long())
        loss_sum.scatter_add_(0, labels, per_loss.double())

    counts_cpu = counts.cpu()
    top1_cpu = correct_top1.cpu()
    top5_cpu = correct_top5.cpu()
    loss_cpu = loss_sum.cpu()

    result: dict[str, dict[str, float]] = {
        "top1_acc": {},
        "top5_acc": {},
        "loss": {},
        "count": {},
    }
    for ci, name in enumerate(class_names):
        c = int(counts_cpu[ci].item())
        result["count"][name] = float(c)
        if c > 0:
            result["top1_acc"][name] = int(top1_cpu[ci].item()) / c
            result["top5_acc"][name] = int(top5_cpu[ci].item()) / c
            result["loss"][name] = float(loss_cpu[ci].item()) / c
        else:
            result["top1_acc"][name] = float("nan")
            result["top5_acc"][name] = float("nan")
            result["loss"][name] = float("nan")

    return result


@torch.no_grad()
def evaluate_topk(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ks: tuple[int, ...] = (1, 5),
) -> dict[int, float]:
    model.eval()
    total_count = 0
    correct_at_k = {k: 0 for k in ks}
    max_k = max(ks)

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        topk_preds = model(images).topk(max_k, dim=1).indices
        for k in ks:
            correct_at_k[k] += (
                topk_preds[:, :k].eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            )
        total_count += labels.size(0)

    return {k: correct_at_k[k] / max(total_count, 1) for k in ks}


def plot_per_class_accuracy(
    metrics: dict[str, dict[str, float]],
    class_names: tuple[str, ...],
    save_path: Path,
) -> None:
    accs = []
    for name in class_names:
        acc = metrics["top1_acc"].get(name, float("nan"))
        accs.append((name, acc))
    accs.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -1, reverse=True)

    names = [a[0] for a in accs]
    values = [a[1] * 100 for a in accs]

    fig, ax = plt.subplots(figsize=(max(len(names) * 0.35, 10), 6))
    bars = ax.bar(range(len(names)), values, width=0.8)
    for bar, v in zip(bars, values):
        color_val = v / 100.0
        bar.set_color(plt.cm.RdYlGn(color_val))

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-class test accuracy (sorted)")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved per-class accuracy plot to {save_path}")


def print_class_metrics(
    name: str,
    metrics: dict[str, dict[str, float]],
    class_names: tuple[str, ...] = CIFAR100_CLASSES,
) -> None:
    counts = metrics["count"]
    active = [c for c in class_names if counts.get(c, 0) > 0]
    total = sum(counts[c] for c in active)

    print(f"\n{name}")
    if total > 0:
        overall_loss = sum(counts[c] * metrics["loss"][c] for c in active) / total
        overall_top1 = sum(counts[c] * metrics["top1_acc"][c] for c in active) / total
        overall_top5 = sum(counts[c] * metrics["top5_acc"][c] for c in active) / total
        print(f"  overall: loss={overall_loss:.4f}, top1={overall_top1:.2%}, top5={overall_top5:.2%}")

    for cn in class_names:
        c = int(counts.get(cn, 0))
        loss = metrics["loss"].get(cn, float("nan"))
        top1 = metrics["top1_acc"].get(cn, float("nan"))
        top5 = metrics["top5_acc"].get(cn, float("nan"))
        loss_s = "nan" if np.isnan(loss) else f"{loss:.4f}"
        top1_s = "nan" if np.isnan(top1) else f"{top1:.2%}"
        top5_s = "nan" if np.isnan(top5) else f"{top5:.2%}"
        print(f"  {cn}: count={c}, loss={loss_s}, top1={top1_s}, top5={top5_s}")


def validate_known_policy(name: str) -> KnownPolicy:
    if name not in {"random", "atypical"}:
        raise ValueError(f"known policy must be 'random' or 'atypical', got '{name}'")
    return cast(KnownPolicy, name)


def main() -> None:
    config = TrainResNetConfig()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")
    print(f"Dataset: {config.dataset}")

    num_classes = CIFAR_NUM_CLASSES[config.dataset]
    class_names = CIFAR_CLASS_NAMES[config.dataset]
    DatasetCls = CIFAR100 if config.dataset == "cifar100" else CIFAR10

    train_dataset = DatasetCls(
        train=True, root=config.data_root, transform=get_train_transform()
    )
    train_eval_dataset = DatasetCls(
        train=True, root=config.data_root, transform=get_eval_transform()
    )
    test_dataset = DatasetCls(
        train=False, root=config.data_root, transform=get_eval_transform()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
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

    model = build_cifar_resnet18(num_classes=num_classes).to(device)
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

    eval_every = 5
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

        if (epoch + 1) % eval_every == 0:
            topk = evaluate_topk(model, test_loader, device, ks=(1, 5))
            print(f"  [eval] top-1={topk[1]:.2%}, top-5={topk[5]:.2%}")

    train_metrics = evaluate_per_class(
        model, train_eval_loader, device, class_names=class_names
    )
    test_metrics = evaluate_per_class(
        model, test_loader, device, class_names=class_names
    )
    print_class_metrics("Final train metrics", train_metrics, class_names=class_names)
    print_class_metrics("Final test metrics", test_metrics, class_names=class_names)

    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "train_resnet.pt"
    torch.save({"model_state_dict": model.state_dict()}, model_path)
    with open(model_dir / "train_resnet_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"\nSaved model to {model_path}")
    print(f"Saved config to {model_dir / 'train_resnet_config.json'}")

    plot_per_class_accuracy(
        test_metrics, class_names, model_dir / "per_class_accuracy.png"
    )


if __name__ == "__main__":
    main()
