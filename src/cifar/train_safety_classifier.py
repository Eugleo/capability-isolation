import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models

from src.cifar.data import CIFAR10, CIFAR10Safety
from src.cifar.train_resnet import (
    get_eval_transform,
    get_train_transform,
)
from src.utils import get_device, set_seed


@dataclass
class TrainSafetyClassifierConfig:
    name: str | None = None
    seed: int = 42
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    data_root: str = "data"
    dangerous_classes: tuple[str, ...] = ("cat",)
    unknown_classes: tuple[str, ...] = ()
    experiments_root: str = "experiments"


def _create_experiment_dir(root: str, *, name: str | None = None) -> Path:
    experiments_root = Path(root)
    experiments_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if name:
        exp_dir = experiments_root / f"{timestamp}_{name}"
    else:
        short_id = uuid.uuid4().hex[:8]
        exp_dir = experiments_root / f"{timestamp}_{short_id}"
    exp_dir.mkdir(parents=True)
    return exp_dir


def build_binary_cifar_resnet18() -> nn.Module:
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
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def _subset_from_mask(
    dataset: CIFAR10Safety, mask: torch.Tensor
) -> Subset[CIFAR10Safety]:
    indices = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()
    return Subset(dataset, indices)


def build_known_unknown_subsets(
    *,
    base_train_dataset: CIFAR10,
    base_eval_dataset: CIFAR10,
    dangerous_classes: set[str],
    unknown_classes: set[str] = frozenset(),
) -> tuple[Subset[CIFAR10Safety], Subset[CIFAR10Safety]]:
    train_safety = CIFAR10Safety.from_cifar10(
        base_train_dataset,
        dangerous_classes=dangerous_classes,
        unknown_classes=unknown_classes,
    )
    test_safety = CIFAR10Safety.from_cifar10(
        base_eval_dataset,
        dangerous_classes=dangerous_classes,
        unknown_classes=unknown_classes,
    )
    train_subset = _subset_from_mask(
        train_safety,
        torch.from_numpy(train_safety.is_label_known_arr),
    )
    test_subset = _subset_from_mask(
        test_safety,
        ~torch.from_numpy(test_safety.is_label_known_arr),
    )
    return train_subset, test_subset


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
        binary_targets = batch["is_dangerous"].to(device=device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, binary_targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * binary_targets.size(0)
        total_correct += logits.argmax(dim=1).eq(binary_targets).sum().item()
        total_count += binary_targets.size(0)

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


@torch.no_grad()
def evaluate_binary(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        images = batch["image"].to(device)
        binary_targets = batch["is_dangerous"].to(device=device, dtype=torch.long)
        logits = model(images)
        loss = criterion(logits, binary_targets)
        total_loss += float(loss.item()) * binary_targets.size(0)
        total_correct += logits.argmax(dim=1).eq(binary_targets).sum().item()
        total_count += binary_targets.size(0)

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


@torch.no_grad()
def evaluate_binary_by_kind(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    kinds: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    model.eval()
    kind_loss_sum = {kind: 0.0 for kind in kinds}
    kind_correct = {kind: 0 for kind in kinds}
    kind_count = {kind: 0 for kind in kinds}

    for batch in loader:
        images = batch["image"].to(device)
        binary_targets = batch["is_dangerous"].to(device=device, dtype=torch.long)
        batch_kinds = list(batch["kind"])
        logits = model(images)
        loss_per_sample = nn.functional.cross_entropy(
            logits, binary_targets, reduction="none"
        )
        preds = logits.argmax(dim=1)
        correct = preds.eq(binary_targets)

        for i, raw_kind in enumerate(batch_kinds):
            kind = str(raw_kind)
            if kind not in kind_count:
                continue
            kind_loss_sum[kind] += float(loss_per_sample[i].item())
            kind_correct[kind] += int(correct[i].item())
            kind_count[kind] += 1

    metrics: dict[str, dict[str, float]] = {}
    for kind in kinds:
        count = kind_count[kind]
        metrics[kind] = {
            "count": float(count),
            "loss": kind_loss_sum[kind] / max(count, 1),
            "accuracy": kind_correct[kind] / max(count, 1),
        }
    return metrics


def plot_metric_by_kind(
    df: pl.DataFrame,
    *,
    metric: str,
    title: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for kind in df["kind"].unique().to_list():
        kind_df = df.filter(pl.col("kind") == kind).sort("epoch")
        if kind_df.is_empty():
            continue
        ax.plot(
            kind_df["epoch"].to_list(),
            kind_df[metric].to_list(),
            label=str(kind),
            marker="o",
            markersize=3,
            linewidth=1.2,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title)
    if metric == "accuracy":
        ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def train_safety_classifier(
    *,
    config: TrainSafetyClassifierConfig,
    base_train_dataset: CIFAR10,
    base_eval_dataset: CIFAR10,
    device: torch.device,
) -> None:
    if config.name is None:
        config.name = uuid.uuid4().hex[:8]

    experiment_dir = _create_experiment_dir(
        config.experiments_root, name=config.name
    )
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Experiment dir: {experiment_dir}")

    train_subset, test_subset = build_known_unknown_subsets(
        base_train_dataset=base_train_dataset,
        base_eval_dataset=base_eval_dataset,
        dangerous_classes=set(config.dangerous_classes),
        unknown_classes=set(config.unknown_classes),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"\nTraining safety classifier "
        f"(train={len(train_subset)}, unknown-test={len(test_subset)})"
    )

    model = build_binary_cifar_resnet18().to(device)
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
    train_kind_rows: list[dict[str, float | str]] = []
    test_kind_rows: list[dict[str, float | str]] = []

    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_acc = evaluate_binary(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )
        train_kind_metrics = evaluate_binary_by_kind(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            kinds=("k-safe", "k-dang"),
        )
        unknown_kind_metrics = evaluate_binary_by_kind(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            kinds=("u-safe", "u-dang"),
        )
        epoch_idx = float(epoch + 1)
        for kind in ("k-safe", "k-dang"):
            train_kind_rows.append(
                {
                    "epoch": epoch_idx,
                    "kind": kind,
                    "count": train_kind_metrics[kind]["count"],
                    "loss": train_kind_metrics[kind]["loss"],
                    "accuracy": train_kind_metrics[kind]["accuracy"],
                }
            )
        for kind in ("u-safe", "u-dang"):
            test_kind_rows.append(
                {
                    "epoch": epoch_idx,
                    "kind": kind,
                    "count": unknown_kind_metrics[kind]["count"],
                    "loss": unknown_kind_metrics[kind]["loss"],
                    "accuracy": unknown_kind_metrics[kind]["accuracy"],
                }
            )
        print(
            f"Epoch {epoch + 1}/{config.epochs} - "
            f"train/loss={train_loss:.4f}, train/accuracy={train_acc:.2%}, "
            f"unknown/loss={test_loss:.4f}, unknown/accuracy={test_acc:.2%}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        print(
            "  train-known: "
            f"k-safe(acc={train_kind_metrics['k-safe']['accuracy']:.2%}, "
            f"loss={train_kind_metrics['k-safe']['loss']:.4f}, "
            f"n={int(train_kind_metrics['k-safe']['count'])}) | "
            f"k-dang(acc={train_kind_metrics['k-dang']['accuracy']:.2%}, "
            f"loss={train_kind_metrics['k-dang']['loss']:.4f}, "
            f"n={int(train_kind_metrics['k-dang']['count'])})"
        )
        print(
            "  test-unknown: "
            f"u-safe(acc={unknown_kind_metrics['u-safe']['accuracy']:.2%}, "
            f"loss={unknown_kind_metrics['u-safe']['loss']:.4f}, "
            f"n={int(unknown_kind_metrics['u-safe']['count'])}) | "
            f"u-dang(acc={unknown_kind_metrics['u-dang']['accuracy']:.2%}, "
            f"loss={unknown_kind_metrics['u-dang']['loss']:.4f}, "
            f"n={int(unknown_kind_metrics['u-dang']['count'])})"
        )
        scheduler.step()

    model_path = experiment_dir / "safety_classifier.pt"
    train_csv_path = experiment_dir / "train_known_metrics.csv"
    test_csv_path = experiment_dir / "test_unknown_metrics.csv"

    torch.save({"model_state_dict": model.state_dict()}, model_path)
    train_kind_df = pl.DataFrame(train_kind_rows)
    test_kind_df = pl.DataFrame(test_kind_rows)
    train_kind_df.write_csv(train_csv_path)
    test_kind_df.write_csv(test_csv_path)

    plot_metric_by_kind(
        train_kind_df,
        metric="loss",
        title="Train Known Loss by Kind",
        save_path=experiment_dir / "train_known_loss_by_epoch.png",
    )
    plot_metric_by_kind(
        train_kind_df,
        metric="accuracy",
        title="Train Known Accuracy by Kind",
        save_path=experiment_dir / "train_known_accuracy_by_epoch.png",
    )
    plot_metric_by_kind(
        test_kind_df,
        metric="loss",
        title="Test Unknown Loss by Kind",
        save_path=experiment_dir / "test_unknown_loss_by_epoch.png",
    )
    plot_metric_by_kind(
        test_kind_df,
        metric="accuracy",
        title="Test Unknown Accuracy by Kind",
        save_path=experiment_dir / "test_unknown_accuracy_by_epoch.png",
    )

    print(f"Saved model to {model_path}")
    print(f"Saved train-known metrics to {train_csv_path}")
    print(f"Saved test-unknown metrics to {test_csv_path}")


def main(config: TrainSafetyClassifierConfig) -> None:
    if config.name is None:
        config.name = uuid.uuid4().hex[:8]

    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    base_train_dataset = CIFAR10(
        train=True, root=config.data_root, transform=get_train_transform()
    )
    base_eval_dataset = CIFAR10(
        train=True, root=config.data_root, transform=get_eval_transform()
    )

    train_safety_classifier(
        config=config,
        base_train_dataset=base_train_dataset,
        base_eval_dataset=base_eval_dataset,
        device=device,
    )


if __name__ == "__main__":
    config = TrainSafetyClassifierConfig()
    dang_tag = "_".join(config.dangerous_classes)
    config.name = f"safety_classifier_{dang_tag}"
    main(config)
