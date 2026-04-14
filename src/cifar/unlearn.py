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
    name: str | None = None
    seed: int = 42
    max_steps: int = 5000
    lr: float = 1e-5
    weight_decay: float = 0.0
    neggrad_forget_weight: float = 5e-5
    max_grad_norm: float = 1.0
    batch_size: int = 128
    eval_every_n_steps: int = 250
    log_every_n_steps: int = 500
    safety_eval_n_per_kind: int = 50

    data_root: str = "data"
    dangerous_class: str = "airplane"
    safe_known: str = "atypical"
    dangerous_known: str = "atypical"
    known_percent: float = 10
    retain_mode: SplitMode = "safe"
    forget_mode: SplitMode = "dang"

    pretrained_model_path: str = "checkpoints/cifar/train_resnet.pt"
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


def _sample_safety_eval_subset(
    dataset: CIFAR10Safety,
    *,
    n_per_kind: int,
    seed: int,
) -> Subset[CIFAR10Safety]:
    import numpy as np

    rng = np.random.RandomState(seed)
    selected: list[int] = []
    for kind in ALL_KINDS:
        kind_indices = np.flatnonzero(dataset.kind_arr == kind)
        if len(kind_indices) == 0:
            continue
        n = min(n_per_kind, len(kind_indices))
        selected.extend(rng.choice(kind_indices, size=n, replace=False).tolist())
    selected.sort()
    return Subset(dataset, selected)


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
) -> None:
    for kind in ALL_KINDS:
        rows.append(
            {
                "step": float(step),
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
    step: int,
) -> None:
    for class_name in CIFAR10_CLASSES:
        rows.append(
            {
                "step": float(step),
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


def plot_unlearn_pareto(
    df: pl.DataFrame,
    *,
    dangerous_class: str,
    save_path: Path,
) -> None:
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap

    steps = sorted(df["step"].unique().to_list())
    other_acc_pct: list[float] = []
    forget_quality_pct: list[float] = []

    for step in steps:
        step_df = df.filter(pl.col("step") == step)
        dang = step_df.filter(pl.col("class") == dangerous_class)["accuracy"].item()
        others = step_df.filter(pl.col("class") != dangerous_class)["accuracy"].mean()
        other_acc_pct.append(float(others) * 100)
        forget_quality_pct.append((1.0 - float(dang)) * 100)

    norm = mcolors.Normalize(vmin=min(steps), vmax=max(steps))
    cmap = LinearSegmentedColormap.from_list("blues", ["#b3d4fc", "#08306b"])

    fig, ax = plt.subplots(figsize=(7, 7))

    for i in range(len(steps) - 1):
        ax.plot(
            other_acc_pct[i : i + 2],
            forget_quality_pct[i : i + 2],
            color=cmap(norm(steps[i])),
            linewidth=1.5,
            zorder=1,
        )

    sc = ax.scatter(
        other_acc_pct,
        forget_quality_pct,
        c=steps,
        cmap=cmap,
        norm=norm,
        s=50,
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Step", fontsize=11)

    ax.set_xlabel("Other Classes Accuracy (%) \u2191", fontsize=12)
    ax.set_ylabel(
        f"1 \u2212 {dangerous_class.capitalize()} Accuracy (%) \u2191", fontsize=12
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Unlearning Pareto: Retain Utility vs Forget Quality", fontsize=13)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _run_eval(
    model: nn.Module,
    *,
    cifar_test_loader: DataLoader,
    safety_eval_loader: DataLoader,
    device: torch.device,
    step: int,
    safety_rows: list[dict[str, float | str]],
    cifar_class_rows: list[dict[str, float | str]],
) -> dict[str, float]:
    safety_metrics = evaluate_safety_by_kind(model, safety_eval_loader, device)
    _append_safety_rows(safety_rows, safety_metrics, step=step)
    cifar_metrics = evaluate_overall_and_per_class(model, cifar_test_loader, device)
    _append_cifar_class_rows(cifar_class_rows, cifar_metrics, step=step)
    return cifar_metrics


def main(config: UnlearnConfig) -> None:
    set_seed(config.seed)
    device = get_device()

    print(f"Using device: {device}")
    print("NegGrad unlearning config:")
    print(f"  max_steps: {config.max_steps}")
    print(f"  lr: {config.lr}")
    print(f"  weight_decay: {config.weight_decay}")
    print(f"  neggrad_forget_weight: {config.neggrad_forget_weight}")
    print(f"  max_grad_norm: {config.max_grad_norm}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  eval_every_n_steps: {config.eval_every_n_steps}")
    print(f"  known_percent: {config.known_percent}")
    print(f"  retain_mode: {config.retain_mode}")
    print(f"  forget_mode: {config.forget_mode}")

    retain_kinds = set(_mode_to_kinds(config.retain_mode))
    forget_kinds = set(_mode_to_kinds(config.forget_mode))
    train_kinds = retain_kinds | forget_kinds
    if ("u-safe" in retain_kinds or "u-dang" in retain_kinds) and (
        "u-safe" in forget_kinds or "u-dang" in forget_kinds
    ):
        raise ValueError(
            "Unknown examples cannot be in both retain and forget. "
            "Choose modes so unknown belongs to only one side (or neither)."
        )

    experiment_dir = _create_experiment_dir(config.experiments_root, name=config.name)
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Experiment dir: {experiment_dir}")

    eval_transform = get_eval_transform()
    cifar_train = CIFAR10(train=True, root=config.data_root, transform=eval_transform)
    cifar_test = CIFAR10(train=False, root=config.data_root, transform=eval_transform)
    safety_dataset = CIFAR10Safety.from_cifar10(
        cifar_train,
        dangerous_class=config.dangerous_class,
        safe_known=validate_known_policy(config.safe_known),
        dangerous_known=validate_known_policy(config.dangerous_known),
        known_percent=config.known_percent,
        seed=config.seed,
    )

    train_subset = _build_mode_subset(
        Subset(safety_dataset, list(range(len(safety_dataset)))),
        train_kinds,
    )
    safety_eval_subset = _sample_safety_eval_subset(
        safety_dataset,
        n_per_kind=config.safety_eval_n_per_kind,
        seed=config.seed,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    safety_eval_loader = DataLoader(
        safety_eval_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    cifar_test_loader = DataLoader(
        cifar_test,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    n_retain = sum(
        1
        for idx in train_subset.indices
        for base_idx in [train_subset.dataset.indices[idx]]
        if str(safety_dataset.kind_arr[base_idx]) in retain_kinds
    )
    n_forget = len(train_subset) - n_retain
    print(
        f"Train={len(train_subset)} (retain={n_retain}, forget={n_forget}), "
        f"safety_eval={len(safety_eval_subset)}"
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
    eval_kwargs = dict(
        cifar_test_loader=cifar_test_loader,
        safety_eval_loader=safety_eval_loader,
        device=device,
        safety_rows=safety_rows,
        cifar_class_rows=cifar_class_rows,
    )

    _run_eval(model, step=0, **eval_kwargs)

    global_step = 0
    running_retain_loss = 0.0
    running_forget_loss = 0.0
    running_retain_count = 0
    running_forget_count = 0
    train_iter = iter(train_loader)

    model.train()
    while global_step < config.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        kinds = list(batch["kind"])

        retain_mask = torch.tensor(
            [str(k) in retain_kinds for k in kinds],
            dtype=torch.bool,
            device=device,
        )
        forget_mask = ~retain_mask

        optimizer.zero_grad()
        logits = model(images)
        per_sample_loss = criterion(logits, labels)

        zero = torch.tensor(0.0, device=device)
        retain_loss = per_sample_loss[retain_mask].mean() if retain_mask.any() else zero
        forget_loss = per_sample_loss[forget_mask].mean() if forget_mask.any() else zero
        loss = retain_loss - config.neggrad_forget_weight * forget_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        if retain_mask.any():
            running_retain_loss += retain_loss.item() * int(retain_mask.sum())
            running_retain_count += int(retain_mask.sum())
        if forget_mask.any():
            running_forget_loss += forget_loss.item() * int(forget_mask.sum())
            running_forget_count += int(forget_mask.sum())

        global_step += 1

        if global_step % config.log_every_n_steps == 0:
            avg_r = running_retain_loss / max(running_retain_count, 1)
            avg_f = running_forget_loss / max(running_forget_count, 1)
            print(
                f"Step {global_step}/{config.max_steps} - "
                f"retain_loss={avg_r:.4f}, forget_loss={avg_f:.4f}"
            )
            running_retain_loss = running_forget_loss = 0.0
            running_retain_count = running_forget_count = 0

        if global_step % config.eval_every_n_steps == 0:
            _run_eval(model, step=global_step, **eval_kwargs)
            model.train()

    _run_eval(model, step=global_step, **eval_kwargs)

    safety_df = pl.DataFrame(safety_rows)
    cifar_class_df = pl.DataFrame(cifar_class_rows)
    safety_csv_path = experiment_dir / "safety_periodic_metrics.csv"
    cifar_class_csv_path = experiment_dir / "cifar_class_metrics.csv"
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
    plot_unlearn_pareto(
        cifar_class_df,
        dangerous_class=config.dangerous_class,
        save_path=experiment_dir / "unlearn_pareto.png",
    )

    model_path = experiment_dir / "unlearned_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, model_path)

    print(f"Saved safety metrics to {safety_csv_path}")
    print(f"Saved CIFAR class metrics to {cifar_class_csv_path}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    base_config = UnlearnConfig()

    mode_pairs: list[tuple[SplitMode, SplitMode, str]] = [
        ("safe", "dang"),
        ("safe+unk", "dang"),
        ("safe", "dang+unk"),
    ]
    labeled_percents = [1, 5, 10, 25, 50]

    configs: list[UnlearnConfig] = []
    for retain_mode, forget_mode in mode_pairs:
        for known_percent in labeled_percents:
            configs.append(
                UnlearnConfig(
                    **{
                        **asdict(base_config),
                        "name": f"{retain_mode}_{forget_mode}_{known_percent}p",
                        "known_percent": float(known_percent),
                        "retain_mode": retain_mode,
                        "forget_mode": forget_mode,
                    }
                )
            )

    print(f"Running {len(configs)} experiments")
    for i, config in enumerate(configs, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(configs)}] {config.name}")
        print(f"{'=' * 60}")
        main(config)
