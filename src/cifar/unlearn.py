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
    CIFAR10_CLASSES,
    CIFAR100,
    CIFAR100_CLASS_TO_INDEX,
    CIFAR100_CLASSES,
    CIFAR100_FINE_TO_COARSE,
    CIFAR100_SUPERCLASSES,
    CLASS_TO_INDEX,
    CIFAR10Safety,
    CIFAR100Safety,
    SafetyKind,
)
from src.cifar.train_resnet import (
    CIFAR_NUM_CLASSES,
    CifarVariant,
    build_cifar_resnet18,
    get_eval_transform,
)
from src.utils import get_device, set_seed

SafetyDataset = CIFAR10Safety | CIFAR100Safety

ALL_KINDS: tuple[SafetyKind, ...] = ("k-safe", "k-dang", "u-safe", "u-dang")
KIND_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(ALL_KINDS)}
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
PER_CLASS_COLORS: tuple[str, ...] = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
)

UnlearningStrategy = Literal["ignore-unknown", "retain-unknown", "forget-unknown"]
EvalSplit = Literal["train", "test"]


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
    dangerous_classes: tuple[str, ...] = ("man", "boy")
    unknown_classes: tuple[str, ...] = ("girl", "boy")
    unlearning_strategy: UnlearningStrategy = "ignore-unknown"

    eval_split: EvalSplit = "train"
    eval_superclass: str | None = "people"

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


def _class_to_kind(
    class_idx: int,
    *,
    dangerous_idxs: set[int],
    unknown_idxs: set[int],
) -> SafetyKind:
    is_dangerous = class_idx in dangerous_idxs
    is_known = class_idx not in unknown_idxs
    if is_known and not is_dangerous:
        return "k-safe"
    if is_known and is_dangerous:
        return "k-dang"
    if not is_known and not is_dangerous:
        return "u-safe"
    return "u-dang"


@torch.no_grad()
def _eval_per_class(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    """Per-class evaluation in a single pass.

    Returns dict of tensors, each of shape ``(num_classes,)``:
    count, top1, top5, loss.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    correct_top1 = torch.zeros(num_classes, dtype=torch.long, device=device)
    correct_top5 = torch.zeros(num_classes, dtype=torch.long, device=device)
    loss_sum = torch.zeros(num_classes, dtype=torch.double, device=device)

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        per_loss = criterion(logits, labels)
        top1_hit = logits.argmax(dim=1).eq(labels)
        k = min(5, logits.size(1))
        top5_hit = logits.topk(k, dim=1).indices.eq(labels.unsqueeze(1)).any(dim=1)

        counts.scatter_add_(0, labels, torch.ones_like(labels))
        correct_top1.scatter_add_(0, labels, top1_hit.long())
        correct_top5.scatter_add_(0, labels, top5_hit.long())
        loss_sum.scatter_add_(0, labels, per_loss.double())

    return {
        "count": counts.cpu(),
        "top1": correct_top1.cpu(),
        "top5": correct_top5.cpu(),
        "loss": loss_sum.cpu(),
    }


def _build_per_class_rows(
    stats: dict[str, torch.Tensor],
    *,
    step: int,
    class_names: tuple[str, ...],
    kind_map: dict[int, SafetyKind],
    fine_to_coarse: dict[int, int] | None,
    superclass_names: tuple[str, ...] | None,
) -> list[dict]:
    rows: list[dict] = []
    for ci in range(len(class_names)):
        c = int(stats["count"][ci])
        t1 = int(stats["top1"][ci])
        t5 = int(stats["top5"][ci])
        ls = float(stats["loss"][ci])
        row: dict = {
            "step": step,
            "class_idx": ci,
            "class_name": class_names[ci],
            "kind": kind_map[ci],
            "count": c,
            "top1_correct": t1,
            "top5_correct": t5,
            "loss_sum": ls,
            "top1_acc": t1 / c if c > 0 else float("nan"),
            "top5_acc": t5 / c if c > 0 else float("nan"),
            "loss": ls / c if c > 0 else float("nan"),
        }
        if fine_to_coarse is not None and superclass_names is not None:
            row["superclass"] = superclass_names[fine_to_coarse[ci]]
        rows.append(row)
    return rows


def _aggregate(df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    """Aggregate per-class rows by *group_col*, computing weighted metrics."""
    return (
        df.group_by("step", group_col)
        .agg(
            pl.col("count").sum(),
            pl.col("top1_correct").sum(),
            pl.col("top5_correct").sum(),
            pl.col("loss_sum").sum(),
        )
        .with_columns(
            pl.when(pl.col("count") > 0)
            .then(pl.col("top1_correct") / pl.col("count"))
            .otherwise(pl.lit(float("nan")))
            .alias("top1_acc"),
            pl.when(pl.col("count") > 0)
            .then(pl.col("top5_correct") / pl.col("count"))
            .otherwise(pl.lit(float("nan")))
            .alias("top5_acc"),
            pl.when(pl.col("count") > 0)
            .then(pl.col("loss_sum") / pl.col("count"))
            .otherwise(pl.lit(float("nan")))
            .alias("loss"),
        )
        .sort("step", group_col)
    )


def _add_safety_col(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("kind").is_in(["k-dang", "u-dang"]))
        .then(pl.lit("dangerous"))
        .otherwise(pl.lit("safe"))
        .alias("safety")
    )


def _run_eval(
    model: nn.Module,
    *,
    eval_loader: DataLoader,
    device: torch.device,
    step: int,
    num_classes: int,
    class_names: tuple[str, ...],
    kind_map: dict[int, SafetyKind],
    dangerous_idxs: set[int],
    fine_to_coarse: dict[int, int] | None,
    superclass_names: tuple[str, ...] | None,
    per_class_rows: list[dict],
) -> None:
    stats = _eval_per_class(model, eval_loader, device, num_classes)
    rows = _build_per_class_rows(
        stats,
        step=step,
        class_names=class_names,
        kind_map=kind_map,
        fine_to_coarse=fine_to_coarse,
        superclass_names=superclass_names,
    )
    per_class_rows.extend(rows)

    safe_idxs = set(range(num_classes)) - dangerous_idxs
    dang_c = sum(int(stats["count"][i]) for i in dangerous_idxs)
    dang_t1 = sum(int(stats["top1"][i]) for i in dangerous_idxs)
    safe_c = sum(int(stats["count"][i]) for i in safe_idxs)
    safe_t1 = sum(int(stats["top1"][i]) for i in safe_idxs)
    total = safe_c + dang_c

    print(
        f"  [eval step={step}] top1: "
        f"overall={((safe_t1 + dang_t1) / max(total, 1)):.2%}, "
        f"safe={safe_t1 / max(safe_c, 1):.2%}, "
        f"dang={dang_t1 / max(dang_c, 1):.2%}"
    )


_METRIC_DISPLAY: dict[str, str] = {
    "top1_acc": "Top-1 Accuracy",
    "top5_acc": "Top-5 Accuracy",
    "loss": "Loss",
}


def _plot_lines(
    df: pl.DataFrame,
    *,
    group_col: str,
    groups: tuple[str, ...],
    colors: dict[str, str],
    linestyles: dict[str, str],
    metric: str,
    title: str,
    save_path: Path,
) -> None:
    display = _METRIC_DISPLAY.get(metric, metric)
    fig, ax = plt.subplots(figsize=(10, 6))
    for group in groups:
        group_df = df.filter(pl.col(group_col) == group).sort("step")
        if group_df.is_empty():
            continue
        ax.plot(
            group_df["step"].to_list(),
            group_df[metric].to_list(),
            label=group,
            color=colors.get(group, "gray"),
            linestyle=linestyles.get(group, "-"),
            marker="o",
            markersize=2,
            linewidth=1.0,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel(display)
    ax.set_title(title)
    if metric in ("top1_acc", "top5_acc"):
        ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_pareto(
    df: pl.DataFrame,
    *,
    group_col: str,
    metric: str = "top1_acc",
    title_suffix: str = "",
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
        safe_row = step_df.filter(pl.col(group_col) == "safe")
        dang_row = step_df.filter(pl.col(group_col) == "dangerous")
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

    title = f"Pareto ({display}): Safe vs Dangerous"
    if title_suffix:
        title += f" [{title_suffix}]"
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _generate_plots(
    pc_df: pl.DataFrame,
    *,
    kind_map: dict[int, SafetyKind],
    class_to_idx: dict[str, int],
    out_dir: Path,
    eval_superclass: str | None,
) -> None:
    """Generate all plots from the per-class DataFrame."""
    pc_safety = _add_safety_col(pc_df)

    kind_df = _aggregate(pc_df, "kind")
    safety_df = _aggregate(pc_safety, "safety")

    for metric in ("top1_acc", "top5_acc", "loss"):
        display = _METRIC_DISPLAY.get(metric, metric)

        _plot_lines(
            kind_df,
            group_col="kind",
            groups=ALL_KINDS,
            colors=GROUP_COLORS,
            linestyles=GROUP_LINESTYLES,
            metric=metric,
            title=f"{display} by Kind",
            save_path=out_dir / f"{metric}_by_kind.png",
        )
        _plot_lines(
            safety_df,
            group_col="safety",
            groups=("safe", "dangerous"),
            colors=GROUP_COLORS,
            linestyles=GROUP_LINESTYLES,
            metric=metric,
            title=f"{display}: Safe vs Dangerous",
            save_path=out_dir / f"{metric}_safe_vs_dang.png",
        )

    for metric in ("top1_acc", "top5_acc"):
        _plot_pareto(
            safety_df,
            group_col="safety",
            metric=metric,
            save_path=out_dir / f"pareto_{metric}.png",
        )

    if eval_superclass and "superclass" in pc_df.columns:
        sc_df = pc_df.filter(pl.col("superclass") == eval_superclass)
        if sc_df.is_empty():
            return

        sc_dir = out_dir / eval_superclass
        sc_dir.mkdir(parents=True, exist_ok=True)

        sc_safety = _add_safety_col(sc_df)
        sc_kind_df = _aggregate(sc_df, "kind")
        sc_safety_df = _aggregate(sc_safety, "safety")
        sc_class_df = _aggregate(sc_df, "class_name")

        sc_class_names = tuple(sorted(sc_df["class_name"].unique().to_list()))
        sc_colors: dict[str, str] = {}
        sc_linestyles: dict[str, str] = {}
        for i, cn in enumerate(sc_class_names):
            sc_colors[cn] = PER_CLASS_COLORS[i % len(PER_CLASS_COLORS)]
            sc_linestyles[cn] = "-"

        for metric in ("top1_acc", "top5_acc", "loss"):
            display = _METRIC_DISPLAY.get(metric, metric)

            _plot_lines(
                sc_kind_df,
                group_col="kind",
                groups=ALL_KINDS,
                colors=GROUP_COLORS,
                linestyles=GROUP_LINESTYLES,
                metric=metric,
                title=f"{eval_superclass}: {display} by Kind",
                save_path=sc_dir / f"{metric}_by_kind.png",
            )
            _plot_lines(
                sc_safety_df,
                group_col="safety",
                groups=("safe", "dangerous"),
                colors=GROUP_COLORS,
                linestyles=GROUP_LINESTYLES,
                metric=metric,
                title=f"{eval_superclass}: {display}: Safe vs Dangerous",
                save_path=sc_dir / f"{metric}_safe_vs_dang.png",
            )
            _plot_lines(
                sc_class_df,
                group_col="class_name",
                groups=sc_class_names,
                colors=sc_colors,
                linestyles=sc_linestyles,
                metric=metric,
                title=f"{eval_superclass}: {display} by Class",
                save_path=sc_dir / f"{metric}_by_class.png",
            )

        for metric in ("top1_acc", "top5_acc"):
            _plot_pareto(
                sc_safety_df,
                group_col="safety",
                metric=metric,
                title_suffix=eval_superclass,
                save_path=sc_dir / f"pareto_{metric}.png",
            )


def main(config: UnlearnConfig) -> None:
    if config.name is None:
        config.name = uuid.uuid4().hex[:8]

    set_seed(config.seed)
    device = get_device()

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
    print(f"  dangerous_classes: {config.dangerous_classes}")
    print(f"  unknown_classes: {config.unknown_classes}")
    print(f"  unlearning_strategy: {config.unlearning_strategy}")
    print(f"  eval_split: {config.eval_split}")
    print(f"  eval_superclass: {config.eval_superclass}")

    retain_kinds, forget_kinds = _strategy_to_kinds(config.unlearning_strategy)
    train_kinds = retain_kinds | forget_kinds

    experiment_dir = _create_experiment_dir(config.experiments_root, name=config.name)
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Experiment dir: {experiment_dir}")

    dangerous_set = set(config.dangerous_classes)
    unknown_set = set(config.unknown_classes)
    eval_transform = get_eval_transform()

    if config.dataset == "cifar100":
        class_names = CIFAR100_CLASSES
        class_to_idx = CIFAR100_CLASS_TO_INDEX
        fine_to_coarse: dict[int, int] | None = CIFAR100_FINE_TO_COARSE
        superclass_names: tuple[str, ...] | None = CIFAR100_SUPERCLASSES
    else:
        class_names = CIFAR10_CLASSES
        class_to_idx = CLASS_TO_INDEX
        fine_to_coarse = None
        superclass_names = None

    num_classes = len(class_names)
    dangerous_idxs = {class_to_idx[c] for c in dangerous_set}
    unknown_idxs = {class_to_idx[c] for c in unknown_set}
    kind_map = {
        ci: _class_to_kind(ci, dangerous_idxs=dangerous_idxs, unknown_idxs=unknown_idxs)
        for ci in range(num_classes)
    }

    if config.dataset == "cifar100":
        cifar_train = CIFAR100(
            train=True, root=config.data_root, transform=eval_transform
        )
        train_safety: SafetyDataset = CIFAR100Safety.from_cifar100(
            cifar_train,
            dangerous_classes=dangerous_set,
            unknown_classes=unknown_set,
        )
    else:
        cifar_train = CIFAR10(
            train=True, root=config.data_root, transform=eval_transform
        )
        train_safety = CIFAR10Safety.from_cifar10(
            cifar_train,
            dangerous_classes=dangerous_set,
            unknown_classes=unknown_set,
        )

    if config.eval_split == "test":
        if config.dataset == "cifar100":
            cifar_eval = CIFAR100(
                train=False, root=config.data_root, transform=eval_transform
            )
            eval_safety: SafetyDataset = CIFAR100Safety.from_cifar100(
                cifar_eval,
                dangerous_classes=dangerous_set,
                unknown_classes=unknown_set,
            )
        else:
            cifar_eval = CIFAR10(
                train=False, root=config.data_root, transform=eval_transform
            )
            eval_safety = CIFAR10Safety.from_cifar10(
                cifar_eval,
                dangerous_classes=dangerous_set,
                unknown_classes=unknown_set,
            )
    else:
        eval_safety = train_safety

    train_subset = _build_mode_subset(
        Subset(train_safety, list(range(len(train_safety)))),
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
        eval_safety,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    n_retain = sum(
        1
        for idx in train_subset.indices
        for base_idx in [train_subset.dataset.indices[idx]]
        if str(train_safety.kind_arr[base_idx]) in retain_kinds
    )
    n_forget = len(train_subset) - n_retain
    kind_counts = {k: int((train_safety.kind_arr == k).sum()) for k in ALL_KINDS}
    print(
        f"Train={len(train_subset)} (retain={n_retain}, forget={n_forget}), "
        f"eval={len(eval_safety)} ({config.eval_split} split)"
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

    per_class_rows: list[dict] = []
    eval_kwargs = dict(
        eval_loader=eval_loader,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        kind_map=kind_map,
        dangerous_idxs=dangerous_idxs,
        fine_to_coarse=fine_to_coarse,
        superclass_names=superclass_names,
        per_class_rows=per_class_rows,
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

    pc_df = pl.DataFrame(per_class_rows)
    pc_df.write_csv(experiment_dir / "per_class_metrics.csv")

    _generate_plots(
        pc_df,
        kind_map=kind_map,
        class_to_idx=class_to_idx,
        out_dir=experiment_dir,
        eval_superclass=config.eval_superclass,
    )

    model_path = experiment_dir / "unlearned_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, model_path)

    print(f"Saved per-class metrics to {experiment_dir / 'per_class_metrics.csv'}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    base_config = UnlearnConfig()

    strategies: list[UnlearningStrategy] = [
        "ignore-unknown",
        "forget-unknown",
        "retain-unknown",
    ]

    configs: list[UnlearnConfig] = []
    for strategy in strategies:
        tag = strategy.replace("-", "_")
        configs.append(
            UnlearnConfig(
                **{
                    **asdict(base_config),
                    "name": tag,
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
