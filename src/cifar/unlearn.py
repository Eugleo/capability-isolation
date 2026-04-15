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

from src.cifar.data import (
    CIFAR10,
    CIFAR100,
    CIFAR10Safety,
    CIFAR100Safety,
    SafetyKind,
)
from src.cifar.train_resnet import (
    CIFAR_NUM_CLASSES,
    CifarVariant,
    build_cifar_resnet18,
    get_eval_transform,
    validate_known_policy,
)
from src.utils import get_device, set_seed

SafetyDataset = CIFAR10Safety | CIFAR100Safety

ALL_KINDS: tuple[SafetyKind, ...] = ("k-safe", "k-dang", "u-safe", "u-dang")
KIND_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(ALL_KINDS)}
AGGREGATE_GROUPS: dict[str, tuple[SafetyKind, ...]] = {
    "safe": ("k-safe", "u-safe"),
    "dangerous": ("k-dang", "u-dang"),
}
ALL_GROUPS: tuple[str, ...] = (*ALL_KINDS, "safe", "dangerous")
GROUP_COLORS: dict[str, str] = {
    "k-dang": "red",
    "u-dang": "orange",
    "u-safe": "lightgreen",
    "k-safe": "darkgreen",
    "safe": "green",
    "dangerous": "crimson",
}
GROUP_LINESTYLES: dict[str, str] = {
    "k-dang": "-",
    "u-dang": "-",
    "u-safe": "-",
    "k-safe": "-",
    "safe": "--",
    "dangerous": "--",
}
UnlearningStrategy = Literal["ignore-unknown", "retain-unknown", "forget-unknown"]


@dataclass
class UnlearnConfig:
    name: str | None = None
    dataset: CifarVariant = "cifar100"
    seed: int = 42
    max_steps: int = 5000
    lr: float = 1e-5
    weight_decay: float = 0.0
    neggrad_forget_weight: float = 5e-5
    max_grad_norm: float = 1.0
    batch_size: int = 128
    eval_every_n_steps: int = 100
    log_every_n_steps: int = 500

    data_root: str = "data"
    dangerous_class: str = "seal"
    dangerous_percent: float = 100.0
    dangerous_policy: str = "atypical"
    safe_known: str = "atypical"
    dangerous_known: str = "atypical"
    known_percent: float = 10
    unlearning_strategy: UnlearningStrategy = "ignore-unknown"

    pretrained_model_path: str = "checkpoints/cifar100/train_resnet.pt"
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


def _strategy_to_kinds(
    strategy: UnlearningStrategy,
) -> tuple[set[SafetyKind], set[SafetyKind]]:
    """Returns (retain_kinds, forget_kinds) for the given strategy."""
    retain: set[SafetyKind] = {"k-safe"}
    forget: set[SafetyKind] = {"k-dang"}
    if strategy == "retain-unknown":
        retain |= {"u-safe", "u-dang"}
    elif strategy == "forget-unknown":
        forget |= {"u-safe", "u-dang"}
    return retain, forget


def _build_mode_subset(
    subset: Subset[SafetyDataset], allowed_kinds: set[SafetyKind]
) -> Subset[SafetyDataset]:
    selected_positions: list[int] = []
    for pos, base_idx in enumerate(subset.indices):
        kind = subset.dataset.kind_arr[int(base_idx)]
        if kind in allowed_kinds:
            selected_positions.append(pos)
    return Subset(subset, selected_positions)


@torch.no_grad()
def _eval_by_kind(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Evaluate model on safety dataset, returning per-kind and aggregate metrics."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    n_kinds = len(ALL_KINDS)

    counts = torch.zeros(n_kinds, dtype=torch.long, device=device)
    correct_top1 = torch.zeros(n_kinds, dtype=torch.long, device=device)
    correct_top5 = torch.zeros(n_kinds, dtype=torch.long, device=device)
    loss_sum = torch.zeros(n_kinds, dtype=torch.double, device=device)

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        kind_indices = torch.tensor(
            [KIND_TO_IDX[str(k)] for k in batch["kind"]],
            dtype=torch.long,
            device=device,
        )

        logits = model(images)
        per_loss = criterion(logits, labels)
        top1_hit = logits.argmax(dim=1).eq(labels)
        k = min(5, logits.size(1))
        top5_hit = logits.topk(k, dim=1).indices.eq(labels.unsqueeze(1)).any(dim=1)

        ones = torch.ones_like(kind_indices)
        counts.scatter_add_(0, kind_indices, ones)
        correct_top1.scatter_add_(0, kind_indices, top1_hit.long())
        correct_top5.scatter_add_(0, kind_indices, top5_hit.long())
        loss_sum.scatter_add_(0, kind_indices, per_loss.double())

    counts_cpu = counts.cpu()
    top1_cpu = correct_top1.cpu()
    top5_cpu = correct_top5.cpu()
    loss_cpu = loss_sum.cpu()

    def _metrics_for(c: int, t1: int, t5: int, ls: float) -> dict[str, float]:
        if c > 0:
            return {
                "top1_acc": t1 / c,
                "top5_acc": t5 / c,
                "loss": ls / c,
                "count": float(c),
            }
        return {
            "top1_acc": float("nan"),
            "top5_acc": float("nan"),
            "loss": float("nan"),
            "count": 0.0,
        }

    result: dict[str, dict[str, float]] = {}
    for idx, kind in enumerate(ALL_KINDS):
        result[kind] = _metrics_for(
            int(counts_cpu[idx]),
            int(top1_cpu[idx]),
            int(top5_cpu[idx]),
            float(loss_cpu[idx]),
        )

    for group_name, member_kinds in AGGREGATE_GROUPS.items():
        member_idxs = [KIND_TO_IDX[k] for k in member_kinds]
        total_c = sum(int(counts_cpu[i]) for i in member_idxs)
        total_t1 = sum(int(top1_cpu[i]) for i in member_idxs)
        total_t5 = sum(int(top5_cpu[i]) for i in member_idxs)
        total_ls = sum(float(loss_cpu[i]) for i in member_idxs)
        result[group_name] = _metrics_for(total_c, total_t1, total_t5, total_ls)

    return result


_METRIC_DISPLAY: dict[str, str] = {
    "top1_acc": "Top-1 Accuracy",
    "top5_acc": "Top-5 Accuracy",
    "loss": "Loss",
}


def plot_metric_by_group(
    df: pl.DataFrame,
    *,
    metric: str,
    save_path: Path,
) -> None:
    display = _METRIC_DISPLAY.get(metric, metric)
    fig, ax = plt.subplots(figsize=(10, 6))
    for group in ALL_GROUPS:
        group_df = df.filter(pl.col("group") == group).sort("step")
        if group_df.is_empty():
            continue
        ax.plot(
            group_df["step"].to_list(),
            group_df[metric].to_list(),
            label=group,
            color=GROUP_COLORS[group],
            linestyle=GROUP_LINESTYLES[group],
            marker="o",
            markersize=2,
            linewidth=1.0,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel(display)
    ax.set_title(f"Train {display} by Group")
    if metric in ("top1_acc", "top5_acc"):
        ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_unlearn_pareto(
    df: pl.DataFrame,
    *,
    metric: str = "top1_acc",
    save_path: Path,
) -> None:
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap

    display = _METRIC_DISPLAY.get(metric, metric)
    is_acc = metric.endswith("_acc")

    steps = sorted(df["step"].unique().to_list())
    retain_vals: list[float] = []
    forget_vals: list[float] = []

    for step in steps:
        step_df = df.filter(pl.col("step") == step)
        safe_row = step_df.filter(pl.col("group") == "safe")
        dang_row = step_df.filter(pl.col("group") == "dangerous")
        if safe_row.is_empty() or dang_row.is_empty():
            continue
        safe_val = float(safe_row[metric].item())
        dang_val = float(dang_row[metric].item())

        if is_acc:
            retain_vals.append(safe_val * 100)
            forget_vals.append((1.0 - dang_val) * 100)
        else:
            retain_vals.append(safe_val)
            forget_vals.append(dang_val)

    if not retain_vals:
        return

    norm = mcolors.Normalize(vmin=min(steps), vmax=max(steps))
    cmap = LinearSegmentedColormap.from_list("blues", ["#b3d4fc", "#08306b"])

    fig, ax = plt.subplots(figsize=(7, 7))

    for i in range(len(steps) - 1):
        ax.plot(
            retain_vals[i : i + 2],
            forget_vals[i : i + 2],
            color=cmap(norm(steps[i])),
            linewidth=1.5,
            zorder=1,
        )

    sc = ax.scatter(
        retain_vals,
        forget_vals,
        c=steps[: len(retain_vals)],
        cmap=cmap,
        norm=norm,
        s=50,
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Step", fontsize=11)

    if is_acc:
        ax.set_xlabel(f"Safe {display} (%) \u2191", fontsize=12)
        ax.set_ylabel(f"1 \u2212 Dangerous {display} (%) \u2191", fontsize=12)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
    else:
        ax.set_xlabel(f"Safe {display} \u2193", fontsize=12)
        ax.set_ylabel(f"Dangerous {display} \u2191", fontsize=12)

    ax.set_title(f"Unlearning Pareto ({display}): Safe vs Dangerous", fontsize=13)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _run_eval(
    model: nn.Module,
    *,
    eval_loader: DataLoader,
    device: torch.device,
    step: int,
    eval_rows: list[dict[str, float | str]],
) -> dict[str, dict[str, float]]:
    kind_metrics = _eval_by_kind(model, eval_loader, device)
    for group in ALL_GROUPS:
        eval_rows.append({"step": float(step), "group": group, **kind_metrics[group]})
    return kind_metrics


def main(config: UnlearnConfig) -> None:
    if config.name is None:
        config.name = uuid.uuid4().hex[:8]

    set_seed(config.seed)
    device = get_device()

    num_classes = CIFAR_NUM_CLASSES[config.dataset]

    print(f"Using device: {device}")
    print(f"Dataset: {config.dataset}")
    print("NegGrad unlearning config:")
    print(f"  max_steps: {config.max_steps}")
    print(f"  lr: {config.lr}")
    print(f"  weight_decay: {config.weight_decay}")
    print(f"  neggrad_forget_weight: {config.neggrad_forget_weight}")
    print(f"  max_grad_norm: {config.max_grad_norm}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  eval_every_n_steps: {config.eval_every_n_steps}")
    print(f"  dangerous_percent: {config.dangerous_percent}")
    print(f"  dangerous_policy: {config.dangerous_policy}")
    print(f"  known_percent: {config.known_percent}")
    print(f"  unlearning_strategy: {config.unlearning_strategy}")

    retain_kinds, forget_kinds = _strategy_to_kinds(config.unlearning_strategy)
    train_kinds = retain_kinds | forget_kinds

    experiment_dir = _create_experiment_dir(config.experiments_root, name=config.name)
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Experiment dir: {experiment_dir}")

    eval_transform = get_eval_transform()
    if config.dataset == "cifar100":
        cifar_train = CIFAR100(
            train=True, root=config.data_root, transform=eval_transform
        )
        safety_dataset: SafetyDataset = CIFAR100Safety.from_cifar100(
            cifar_train,
            dangerous_classes={config.dangerous_class},
            dangerous_percent=config.dangerous_percent,
            dangerous_policy=validate_known_policy(config.dangerous_policy),
            safe_known=validate_known_policy(config.safe_known),
            dangerous_known=validate_known_policy(config.dangerous_known),
            known_percent=config.known_percent,
            seed=config.seed,
        )
    else:
        cifar_train = CIFAR10(
            train=True, root=config.data_root, transform=eval_transform
        )
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

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    eval_loader = DataLoader(
        safety_dataset,
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
    kind_counts = {k: int((safety_dataset.kind_arr == k).sum()) for k in ALL_KINDS}
    print(
        f"Train={len(train_subset)} (retain={n_retain}, forget={n_forget}), "
        f"eval={len(safety_dataset)}"
    )
    for k, c in kind_counts.items():
        print(f"  {k}: {c}")

    model = build_cifar_resnet18(num_classes=num_classes).to(device)
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

    eval_rows: list[dict[str, float | str]] = []
    eval_kwargs = dict(
        eval_loader=eval_loader,
        device=device,
        eval_rows=eval_rows,
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

    if global_step % config.eval_every_n_steps != 0:
        _run_eval(model, step=global_step, **eval_kwargs)

    eval_df = pl.DataFrame(eval_rows)
    eval_csv_path = experiment_dir / "eval_metrics.csv"
    eval_df.write_csv(eval_csv_path)

    plot_metric_by_group(
        eval_df,
        metric="top1_acc",
        save_path=experiment_dir / "top1_acc_by_group.png",
    )
    plot_metric_by_group(
        eval_df,
        metric="top5_acc",
        save_path=experiment_dir / "top5_acc_by_group.png",
    )
    plot_metric_by_group(
        eval_df,
        metric="loss",
        save_path=experiment_dir / "loss_by_group.png",
    )
    plot_unlearn_pareto(
        eval_df,
        metric="top1_acc",
        save_path=experiment_dir / "unlearn_pareto_top1.png",
    )
    plot_unlearn_pareto(
        eval_df,
        metric="top5_acc",
        save_path=experiment_dir / "unlearn_pareto_top5.png",
    )
    plot_unlearn_pareto(
        eval_df,
        metric="loss",
        save_path=experiment_dir / "unlearn_pareto_loss.png",
    )

    model_path = experiment_dir / "unlearned_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, model_path)

    print(f"Saved eval metrics to {eval_csv_path}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    base_config = UnlearnConfig()

    strategies: list[UnlearningStrategy] = [
        "ignore-unknown",
        "forget-unknown",
        "retain-unknown",
    ]
    labeled_percents = [1, 5, 10, 25, 50, 90, 99]

    configs: list[UnlearnConfig] = []
    for strategy in strategies:
        for known_percent in labeled_percents:
            tag = strategy.replace("-", "_")
            configs.append(
                UnlearnConfig(
                    **{
                        **asdict(base_config),
                        "name": f"{tag}_{known_percent}p",
                        "known_percent": float(known_percent),
                        "unlearning_strategy": strategy,
                    }
                )
            )

    print(f"Running {len(configs)} experiments")
    for i, config in enumerate(configs, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(configs)}] {config.name}")
        print(f"{'=' * 60}")
        main(config)
