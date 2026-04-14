import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.cifar.data import CIFAR10, CIFAR10_CLASSES, CIFAR10Safety, SafetyKind
from src.cifar.train_resnet import (
    build_cifar_resnet18,
    evaluate_overall_and_per_class,
    get_eval_transform,
    validate_known_policy,
)
from src.utils import get_device, set_seed

ALL_KINDS: tuple[SafetyKind, ...] = ("k-safe", "k-dang", "u-safe", "u-dang")
SplitMode = Literal["safe", "safe+unk", "dang", "dang+unk"]


@dataclass
class UnlearnConfig:
    seed: int = 42
    epochs: int = 20
    lr: float = 1e-5
    weight_decay: float = 0.0
    neggrad_forget_weight: float = 5e-5
    max_grad_norm: float = 1.0
    batch_size: int = 128
    eval_every_n_batches: int = 128
    safety_test_percent: float = 10.0

    data_root: str = "data"
    dangerous_class: str = "airplane"
    safe_known: str = "atypical"
    dangerous_known: str = "atypical"
    known_percent: float = 75
    retain_mode: SplitMode = "safe"
    forget_mode: SplitMode = "dang"

    pretrained_model_path: str = "checkpoints/cifar/train_resnet.pt"
    experiments_root: str = "experiments"


def _create_experiment_dir(root: str) -> Path:
    experiments_root = Path(root)
    experiments_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    short_id = uuid.uuid4().hex[:8]
    exp_dir = experiments_root / f"{timestamp}_{short_id}"
    exp_dir.mkdir(parents=True)
    return exp_dir


def _mode_to_kinds(mode: SplitMode) -> tuple[SafetyKind, ...]:
    if mode == "safe":
        return ("k-safe",)
    if mode == "safe+unk":
        return ("k-safe", "u-safe", "u-dang")
    if mode == "dang":
        return ("k-dang",)
    if mode == "dang+unk":
        return ("k-dang", "u-safe", "u-dang")
    raise ValueError(f"Unknown mode: {mode}")


def _build_mode_subset(
    subset: Subset[CIFAR10Safety], allowed_kinds: set[SafetyKind]
) -> Subset[CIFAR10Safety]:
    selected_positions: list[int] = []
    for pos, base_idx in enumerate(subset.indices):
        kind = subset.dataset.kind_arr[int(base_idx)]
        if kind in allowed_kinds:
            selected_positions.append(pos)
    return Subset(subset, selected_positions)


@torch.no_grad()
def evaluate_average_loss(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        total_loss += float(criterion(logits, labels).item())
        total_count += int(labels.size(0))

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_safety_by_kind(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    kind_loss_sum = {kind: 0.0 for kind in ALL_KINDS}
    kind_correct = {kind: 0 for kind in ALL_KINDS}
    kind_count = {kind: 0 for kind in ALL_KINDS}

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        kinds = list(batch["kind"])

        logits = model(images)
        loss_per_sample = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        correct = preds.eq(labels)

        for i, kind_raw in enumerate(kinds):
            kind = str(kind_raw)
            if kind not in kind_count:
                continue
            kind_loss_sum[kind] += float(loss_per_sample[i].item())
            kind_correct[kind] += int(correct[i].item())
            kind_count[kind] += 1

    metrics: dict[str, float] = {}
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


def _append_safety_rows(
    rows: list[dict[str, float | str]],
    metrics: dict[str, float],
    *,
    step: int,
    epoch: int,
) -> None:
    for kind in ALL_KINDS:
        rows.append(
            {
                "step": float(step),
                "epoch": float(epoch),
                "kind": kind,
                "count": metrics[f"{kind}/count"],
                "loss": metrics[f"{kind}/loss"],
                "accuracy": metrics[f"{kind}/accuracy"],
            }
        )


def _append_cifar_class_rows(
    rows: list[dict[str, float | str]],
    metrics: dict[str, float],
    *,
    epoch: int,
) -> None:
    for class_name in CIFAR10_CLASSES:
        rows.append(
            {
                "epoch": float(epoch),
                "class": class_name,
                "count": metrics[f"class/{class_name}/count"],
                "loss": metrics[f"class/{class_name}/loss"],
                "accuracy": metrics[f"class/{class_name}/accuracy"],
            }
        )


def plot_safety_metric_by_kind(
    df: pl.DataFrame,
    *,
    metric: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for kind in ALL_KINDS:
        kind_df = df.filter(pl.col("kind") == kind).sort("step")
        if kind_df.is_empty():
            continue
        ax.plot(
            kind_df["step"].to_list(),
            kind_df[metric].to_list(),
            label=kind,
            marker="o",
            markersize=2,
            linewidth=1.0,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Safety Test {metric.capitalize()} by Kind")
    if metric == "accuracy":
        ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cifar_class_accuracy_by_epoch(df: pl.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for class_name in CIFAR10_CLASSES:
        class_df = df.filter(pl.col("class") == class_name).sort("epoch")
        if class_df.is_empty():
            continue
        ax.plot(
            class_df["epoch"].to_list(),
            class_df["accuracy"].to_list(),
            label=class_name,
            marker="o",
            markersize=3,
            linewidth=1.0,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("CIFAR Test Accuracy by Class and Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    config = UnlearnConfig()
    set_seed(config.seed)
    device = get_device()

    print(f"Using device: {device}")
    print("NegGrad unlearning hyperparameters (joint; no alternation):")
    print(f"  epochs: {config.epochs}")
    print(f"  optimizer: Adam (constant LR, no scheduler)")
    print(f"  lr: {config.lr}")
    print(f"  weight_decay: {config.weight_decay}")
    print(f"  neggrad_forget_weight: {config.neggrad_forget_weight}")
    print(f"  max_grad_norm: {config.max_grad_norm}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  eval_every_n_batches: {config.eval_every_n_batches}")
    print(f"  known_percent: {config.known_percent}")
    print(f"  retain_mode: {config.retain_mode}")
    print(f"  forget_mode: {config.forget_mode}")

    retain_kinds = set(_mode_to_kinds(config.retain_mode))
    forget_kinds = set(_mode_to_kinds(config.forget_mode))
    if ("u-safe" in retain_kinds or "u-dang" in retain_kinds) and (
        "u-safe" in forget_kinds or "u-dang" in forget_kinds
    ):
        raise ValueError(
            "Unknown examples cannot be in both retain and forget. "
            "Choose modes so unknown belongs to only one side (or neither)."
        )

    experiment_dir = _create_experiment_dir(config.experiments_root)
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Experiment dir: {experiment_dir}")

    eval_transform = get_eval_transform()
    train_dataset_eval = CIFAR10(
        train=True, root=config.data_root, transform=eval_transform
    )
    cifar_test_dataset = CIFAR10(
        train=False, root=config.data_root, transform=eval_transform
    )
    safety_dataset = CIFAR10Safety.from_cifar10(
        train_dataset_eval,
        dangerous_class=config.dangerous_class,
        safe_known=validate_known_policy(config.safe_known),
        dangerous_known=validate_known_policy(config.dangerous_known),
        known_percent=config.known_percent,
        seed=config.seed,
    )
    safety_train_subset, safety_test_subset = safety_dataset.train_test_subsets_by_kind(
        test_percent=config.safety_test_percent,
        seed=config.seed,
    )

    # Build retain/forget subsets only for eval monitoring.
    retain_train_subset = _build_mode_subset(safety_train_subset, retain_kinds)
    forget_train_subset = _build_mode_subset(safety_train_subset, forget_kinds)
    if len(forget_train_subset) == 0:
        raise ValueError(
            "NegGrad requires a non-empty forget subset. "
            "Adjust forget_mode/split settings."
        )

    # Single shuffled training loader over ALL training examples.
    train_loader = DataLoader(
        safety_train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    retain_train_eval_loader = DataLoader(
        retain_train_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    forget_train_eval_loader = DataLoader(
        forget_train_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    safety_test_loader = DataLoader(
        safety_test_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    cifar_test_loader = DataLoader(
        cifar_test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(
        "Split sizes - "
        f"train={len(safety_train_subset)} "
        f"(retain={len(retain_train_subset)}, forget={len(forget_train_subset)}), "
        f"safety_test={len(safety_test_subset)}"
    )

    model = build_cifar_resnet18().to(device)
    checkpoint = torch.load(
        config.pretrained_model_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded pretrained model from {config.pretrained_model_path}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(reduction="none")

    safety_rows: list[dict[str, float | str]] = []
    cifar_class_rows: list[dict[str, float | str]] = []

    baseline_safety = evaluate_safety_by_kind(model, safety_test_loader, device)
    _append_safety_rows(safety_rows, baseline_safety, step=0, epoch=0)
    baseline_cifar = evaluate_overall_and_per_class(model, cifar_test_loader, device)
    _append_cifar_class_rows(cifar_class_rows, baseline_cifar, epoch=0)

    global_step = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_retain_loss = 0.0
        epoch_forget_loss = 0.0
        epoch_retain_count = 0
        epoch_forget_count = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            kinds = list(batch["kind"])

            retain_mask = torch.tensor(
                [str(k) in retain_kinds for k in kinds],
                dtype=torch.bool,
                device=device,
            )
            forget_mask = torch.tensor(
                [str(k) in forget_kinds for k in kinds],
                dtype=torch.bool,
                device=device,
            )

            optimizer.zero_grad()
            logits = model(images)
            per_sample_loss = criterion(logits, labels)

            loss = torch.tensor(0.0, device=device)

            if retain_mask.any():
                retain_loss = per_sample_loss[retain_mask].mean()
                loss = loss + retain_loss
                epoch_retain_loss += retain_loss.item() * int(retain_mask.sum())
                epoch_retain_count += int(retain_mask.sum())

            if forget_mask.any():
                forget_loss = per_sample_loss[forget_mask].mean()
                loss = loss - config.neggrad_forget_weight * forget_loss
                epoch_forget_loss += forget_loss.item() * int(forget_mask.sum())
                epoch_forget_count += int(forget_mask.sum())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            if global_step % config.eval_every_n_batches == 0:
                periodic = evaluate_safety_by_kind(model, safety_test_loader, device)
                _append_safety_rows(
                    safety_rows,
                    periodic,
                    step=global_step,
                    epoch=epoch,
                )

        retain_train_loss = evaluate_average_loss(
            model, retain_train_eval_loader, device
        )
        forget_train_loss = evaluate_average_loss(
            model, forget_train_eval_loader, device
        )
        cifar_metrics = evaluate_overall_and_per_class(model, cifar_test_loader, device)
        _append_cifar_class_rows(cifar_class_rows, cifar_metrics, epoch=epoch)
        avg_retain = epoch_retain_loss / max(epoch_retain_count, 1)
        avg_forget = epoch_forget_loss / max(epoch_forget_count, 1)
        print(
            f"Epoch {epoch}/{config.epochs} - "
            f"retain_loss={avg_retain:.4f}, "
            f"forget_loss={avg_forget:.4f}, "
            f"eval_retain_loss={retain_train_loss:.4f}, "
            f"eval_forget_loss={forget_train_loss:.4f}, "
            f"test_acc={cifar_metrics['accuracy']:.2%}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

    safety_df = pl.DataFrame(safety_rows)
    cifar_class_df = pl.DataFrame(cifar_class_rows)
    safety_csv_path = experiment_dir / "safety_periodic_metrics.csv"
    cifar_class_csv_path = experiment_dir / "cifar_class_epoch_metrics.csv"
    safety_df.write_csv(safety_csv_path)
    cifar_class_df.write_csv(cifar_class_csv_path)

    plot_safety_metric_by_kind(
        safety_df,
        metric="accuracy",
        save_path=experiment_dir / "safety_accuracy_by_kind.png",
    )
    plot_safety_metric_by_kind(
        safety_df,
        metric="loss",
        save_path=experiment_dir / "safety_loss_by_kind.png",
    )
    plot_cifar_class_accuracy_by_epoch(
        cifar_class_df,
        save_path=experiment_dir / "cifar_test_class_accuracy_by_epoch.png",
    )

    model_path = experiment_dir / "unlearned_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, model_path)

    print(f"Saved safety metrics to {safety_csv_path}")
    print(f"Saved CIFAR class metrics to {cifar_class_csv_path}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
