import json
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
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
    CLASS_TO_INDEX,
    CIFAR10Safety,
    CIFAR100Safety,
    SafetyKind,
)
from src.cifar.train_resnet import CifarVariant, build_cifar_resnet18, get_eval_transform
from src.utils import get_device, set_seed

SafetyDataset = CIFAR10Safety | CIFAR100Safety

ALL_KINDS: tuple[SafetyKind, ...] = ("k-safe", "k-dang", "u-safe", "u-dang")
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
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

PER_CLASS_METRIC_COLUMNS: tuple[str, ...] = (
    "count",
    "top1_correct",
    "top5_correct",
    "loss_sum",
    "top1_acc",
    "top5_acc",
    "loss",
)


def _mean_typicality_per_class_index(safety: CIFAR100Safety, *, num_classes: int) -> np.ndarray:
    scores = safety.typicality_scores
    targets = np.asarray(safety.cifar100.base_dataset.targets, dtype=np.int64)
    sums = np.bincount(targets, weights=scores, minlength=num_classes)
    counts = np.bincount(targets, minlength=num_classes)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / np.maximum(counts, 1)


UnlearningStrategy = Literal["ignore-unknown", "retain-unknown", "forget-unknown"]
EvalSplit = Literal["train", "test"]


@dataclass
class UnlearnConfig:
    name: str | None = None
    dataset: CifarVariant = "cifar100"
    seed: int = 42
    max_steps: int = 10000
    lr: float = 1e-5
    weight_decay: float = 0.0
    neggrad_forget_weight: float = 5e-5
    max_grad_norm: float = 1.0
    batch_size: int = 128
    eval_every_n_steps: int = 200
    log_every_n_steps: int = 1000

    data_root: str = "data"
    dangerous_classes: tuple[str, ...] = ("forest", "palm_tree", "willow_tree", "maple_tree", "oak_tree", "pine_tree")
    # Empty tuple means every class is known (matches former default of unknown_classes=()).
    known_classes: tuple[str, ...] = ()
    unlearning_strategy: UnlearningStrategy = "ignore-unknown"

    eval_split: EvalSplit = "train"
    eval_class_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)

    pretrained_model_path: str = "checkpoints/cifar100/train_resnet.pt"
    experiments_root: str = "experiments"


def _experiment_family_tag(dangerous_classes: tuple[str, ...]) -> str:
    if not dangerous_classes:
        return "exp"
    tails = {c.rsplit("_", 1)[-1] for c in dangerous_classes}
    if len(tails) == 1:
        return next(iter(tails))
    return f"{len(dangerous_classes)}dang"


def build_unlearn_configs_for_dangerous_grid(
    *,
    dangerous_classes: tuple[str, ...],
    safe_classes_ordered: tuple[str, ...],
    class_names: tuple[str, ...],
    strategies: tuple[UnlearningStrategy, ...],
    seed: int,
    name_prefix: str | None = None,
) -> list[UnlearnConfig]:
    if not dangerous_classes:
        raise ValueError("dangerous_classes must be non-empty")
    dangerous_set = set(dangerous_classes)
    if len(dangerous_set) != len(dangerous_classes):
        raise ValueError("dangerous_classes must not contain duplicates")
    for c in dangerous_classes:
        if c not in class_names:
            raise ValueError(f"dangerous class {c!r} is not in this dataset's class list")

    pool_safe_set = set(class_names) - dangerous_set
    if set(safe_classes_ordered) != pool_safe_set:
        raise ValueError("safe_classes_ordered must contain exactly the non-dangerous class names")
    if len(safe_classes_ordered) != len(pool_safe_set):
        raise ValueError("safe_classes_ordered must not contain duplicates")

    n_dang = len(dangerous_classes)
    num_labels = len(class_names)
    stem = name_prefix if name_prefix is not None else _experiment_family_tag(dangerous_classes)

    configs: list[UnlearnConfig] = []
    ks = [1] if n_dang == 1 else list(range(1, n_dang))
    for strategy in strategies:
        for k in ks:
            set_seed(seed)
            known_dangerous = set(dangerous_classes[:k])
            target_fraction = k / n_dang
            target_known_total = min(num_labels, round(target_fraction * num_labels))
            n_safe = max(0, min(len(safe_classes_ordered), target_known_total - k))
            sampled_safe = set(safe_classes_ordered[:n_safe])
            known_classes_set = known_dangerous | sampled_safe
            known_classes = tuple(sorted(known_classes_set))

            unknown_names = set(class_names) - known_classes_set
            unknown_safe = sorted(unknown_names - dangerous_set)
            known_safe = sorted(known_classes_set - dangerous_set)

            n_k_safe = min(5, len(known_safe))
            n_u_safe = min(5, len(unknown_safe))
            subsampled_classes = list(dangerous_classes)
            subsampled_classes.extend(random.sample(known_safe, n_k_safe))
            subsampled_classes.extend(random.sample(unknown_safe, n_u_safe))

            pct_tag = int(round(100 * k / n_dang))
            eval_class_groups: dict[str, tuple[str, ...]] = {
                "all": class_names,
                "subsampled": tuple(sorted(subsampled_classes)),
            }

            tag = strategy.replace("-", "_")
            name = f"{stem}_{pct_tag}p_{tag}"
            configs.append(
                UnlearnConfig(
                    name=name,
                    dangerous_classes=dangerous_classes,
                    known_classes=known_classes,
                    unlearning_strategy=strategy,
                    eval_class_groups=eval_class_groups,
                )
            )

    return configs


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
    retain: set[SafetyKind] = {"k-safe"}
    forget: set[SafetyKind] = {"k-dang"}
    extra_unknown: set[SafetyKind] = {"u-safe", "u-dang"}
    if strategy == "retain-unknown":
        retain |= extra_unknown
    elif strategy == "forget-unknown":
        forget |= extra_unknown
    return retain, forget


def _cifar_safety_dataset(
    variant: CifarVariant,
    *,
    train: bool,
    data_root: str,
    transform,
    dangerous_classes: set[str],
    unknown_classes: set[str],
) -> SafetyDataset:
    if variant == "cifar100":
        base = CIFAR100(train=train, root=data_root, transform=transform)
        return CIFAR100Safety.from_cifar100(
            base,
            dangerous_classes=dangerous_classes,
            unknown_classes=unknown_classes,
        )
    base = CIFAR10(train=train, root=data_root, transform=transform)
    return CIFAR10Safety.from_cifar10(
        base,
        dangerous_classes=dangerous_classes,
        unknown_classes=unknown_classes,
    )


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
    known_idxs: set[int],
) -> SafetyKind:
    d = class_idx in dangerous_idxs
    k = class_idx in known_idxs
    return ("u-safe", "u-dang", "k-safe", "k-dang")[int(k) * 2 + int(d)]


@torch.no_grad()
def _eval_per_class(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, torch.Tensor]:
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
) -> list[dict]:
    rows: list[dict] = []
    for ci in range(len(class_names)):
        c = int(stats["count"][ci])
        t1 = int(stats["top1"][ci])
        t5 = int(stats["top5"][ci])
        ls = float(stats["loss"][ci])
        rows.append({
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
        })
    return rows


def _per_class_wide_to_long(wide: pl.DataFrame) -> pl.DataFrame:
    return wide.unpivot(
        index=["step", "class_idx", "class_name"],
        on=list(PER_CLASS_METRIC_COLUMNS),
        variable_name="metric",
        value_name="value",
    ).sort("step", "class_idx", "metric")


def _class_metadata_frame(
    class_names: tuple[str, ...],
    kind_map: dict[int, SafetyKind],
) -> pl.DataFrame:
    rows: list[dict] = []
    for ci, name in enumerate(class_names):
        kind = kind_map[ci]
        rows.append({
            "class_idx": ci,
            "class_name": name,
            "kind": kind,
            "is_dangerous": kind in ("k-dang", "u-dang"),
            "is_known": kind in ("k-safe", "k-dang"),
            "safety": "dangerous" if kind in ("k-dang", "u-dang") else "safe",
            "known_unknown": "known" if kind in ("k-safe", "k-dang") else "unknown",
        })
    return pl.DataFrame(rows)


def _aggregate(df: pl.DataFrame, group_col: str) -> pl.DataFrame:
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
    per_class_rows: list[dict],
) -> None:
    stats = _eval_per_class(model, eval_loader, device, num_classes)
    rows = _build_per_class_rows(
        stats,
        step=step,
        class_names=class_names,
        kind_map=kind_map,
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


def _last_finite_xy(xs: list, ys: list) -> tuple[float, float] | None:
    for i in range(len(ys) - 1, -1, -1):
        y = ys[i]
        if y == y:
            return float(xs[i]), float(y)
    return None


def _plot_lines(
    df: pl.DataFrame,
    *,
    group_col: str,
    groups: tuple[str, ...],
    colors: dict[str, str],
    linestyles: dict[str, str],
    markers: dict[str, str] | None = None,
    metric: str,
    title: str,
    save_path: Path,
    label_line_endpoints: bool = False,
) -> None:
    display = _METRIC_DISPLAY.get(metric, metric)
    fig, ax = plt.subplots(figsize=(10, 6))
    show_legend = not label_line_endpoints
    for group in groups:
        group_df = df.filter(pl.col(group_col) == group).sort("step")
        if group_df.is_empty():
            continue
        marker = (markers or {}).get(group, "o")
        color = colors.get(group, "gray")
        linestyle = linestyles.get(group, "-")
        xs = group_df["step"].to_list()
        ys = group_df[metric].to_list()
        ax.plot(
            xs,
            ys,
            label=group if show_legend else None,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=12 if marker == "*" else 6,
            linewidth=1.0,
        )
        if label_line_endpoints:
            end = _last_finite_xy(xs, ys)
            if end is not None:
                xf, yf = end
                ax.annotate(
                    group,
                    (xf, yf),
                    xytext=(6, 0),
                    textcoords="offset points",
                    color=color,
                    fontsize=5,
                    alpha=0.5,
                    va="center",
                    clip_on=False,
                )
    ax.set_xlabel("Step")
    ax.set_ylabel(display)
    ax.set_title(title)
    if metric in ("top1_acc", "top5_acc"):
        if label_line_endpoints:
            ax.set_ylim(-0.08, 1.08)
        else:
            ax.set_ylim(0, 1)
    elif label_line_endpoints:
        ylo, yhi = ax.get_ylim()
        ypad = (yhi - ylo) * 0.12 + 1e-6
        ax.set_ylim(ylo - ypad, yhi + ypad)
    if label_line_endpoints:
        ax.margins(x=0.06)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend()
    if label_line_endpoints:
        fig.tight_layout(pad=1.0)
        fig.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.25,
        )
    else:
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
    valid_steps: list[int] = []

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
        valid_steps.append(step)

    if not retain_vals:
        return

    norm = mcolors.Normalize(vmin=min(valid_steps), vmax=max(valid_steps))
    cmap = LinearSegmentedColormap.from_list("blues", ["#b3d4fc", "#08306b"])

    fig, ax = plt.subplots(figsize=(7, 7))

    for i in range(len(retain_vals) - 1):
        ax.plot(
            retain_vals[i : i + 2],
            forget_vals[i : i + 2],
            color=cmap(norm(valid_steps[i])),
            linewidth=1.5,
            zorder=1,
        )

    sc = ax.scatter(
        retain_vals,
        forget_vals,
        c=valid_steps,
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
    eval_class_groups: dict[str, tuple[str, ...]],
) -> None:
    for group_name, group_classes in eval_class_groups.items():
        group_class_set = set(group_classes)
        grp_df = pc_df.filter(pl.col("class_name").is_in(group_class_set))
        if grp_df.is_empty():
            continue

        grp_dir = out_dir / group_name
        grp_dir.mkdir(parents=True, exist_ok=True)

        grp_safety = _add_safety_col(grp_df)
        grp_kind_df = _aggregate(grp_df, "kind")
        grp_safety_df = _aggregate(grp_safety, "safety")
        grp_class_df = _aggregate(grp_df, "class_name")

        grp_class_names = tuple(sorted(grp_df["class_name"].unique().to_list()))
        grp_colors: dict[str, str] = {}
        grp_linestyles: dict[str, str] = {}
        grp_markers: dict[str, str] = {}
        for i, cn in enumerate(grp_class_names):
            kind = kind_map[class_to_idx[cn]]
            grp_colors[cn] = PER_CLASS_COLORS[i % len(PER_CLASS_COLORS)]
            grp_markers[cn] = "*" if kind in ("k-dang", "u-dang") else "o"
            grp_linestyles[cn] = "--" if kind in ("u-safe", "u-dang") else "-"

        for metric in ("top1_acc", "top5_acc", "loss"):
            display = _METRIC_DISPLAY.get(metric, metric)

            _plot_lines(
                grp_kind_df,
                group_col="kind",
                groups=ALL_KINDS,
                colors=GROUP_COLORS,
                linestyles=GROUP_LINESTYLES,
                metric=metric,
                title=f"{group_name}: {display} by Kind",
                save_path=grp_dir / f"{metric}_by_kind.png",
            )
            _plot_lines(
                grp_safety_df,
                group_col="safety",
                groups=("safe", "dangerous"),
                colors=GROUP_COLORS,
                linestyles=GROUP_LINESTYLES,
                metric=metric,
                title=f"{group_name}: {display}: Safe vs Dangerous",
                save_path=grp_dir / f"{metric}_safe_vs_dang.png",
            )
            _plot_lines(
                grp_class_df,
                group_col="class_name",
                groups=grp_class_names,
                colors=grp_colors,
                linestyles=grp_linestyles,
                markers=grp_markers,
                metric=metric,
                title=f"{group_name}: {display} by Class",
                save_path=grp_dir / f"{metric}_by_class.png",
                label_line_endpoints=True,
            )

        for metric in ("top1_acc", "top5_acc"):
            _plot_pareto(
                grp_safety_df,
                group_col="safety",
                metric=metric,
                title_suffix=group_name,
                save_path=grp_dir / f"pareto_{metric}.png",
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
    print(f"  known_classes: {config.known_classes}")
    print(f"  unlearning_strategy: {config.unlearning_strategy}")
    print(f"  eval_split: {config.eval_split}")
    print(f"  eval_class_groups: {config.eval_class_groups}")

    retain_kinds, forget_kinds = _strategy_to_kinds(config.unlearning_strategy)
    train_kinds = retain_kinds | forget_kinds

    experiment_dir = _create_experiment_dir(config.experiments_root, name=config.name)
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Experiment dir: {experiment_dir}")

    dangerous_set = set(config.dangerous_classes)
    eval_transform = get_eval_transform()

    if config.dataset == "cifar100":
        class_names = CIFAR100_CLASSES
        class_to_idx = CIFAR100_CLASS_TO_INDEX
    else:
        class_names = CIFAR10_CLASSES
        class_to_idx = CLASS_TO_INDEX

    num_classes = len(class_names)
    if len(config.known_classes) == 0:
        known_set = set(class_names)
    else:
        known_set = set(config.known_classes)

    dangerous_idxs = {class_to_idx[c] for c in dangerous_set}
    unknown_set = set(class_names) - known_set
    known_idxs = {class_to_idx[c] for c in known_set}
    kind_map = {
        ci: _class_to_kind(ci, dangerous_idxs=dangerous_idxs, known_idxs=known_idxs)
        for ci in range(num_classes)
    }

    train_safety = _cifar_safety_dataset(
        config.dataset,
        train=True,
        data_root=config.data_root,
        transform=eval_transform,
        dangerous_classes=dangerous_set,
        unknown_classes=unknown_set,
    )
    if config.eval_split == "test":
        eval_safety = _cifar_safety_dataset(
            config.dataset,
            train=False,
            data_root=config.data_root,
            transform=eval_transform,
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

    parent_idx = train_subset.dataset.indices
    n_retain = sum(
        str(train_safety.kind_arr[int(parent_idx[pos])]) in retain_kinds
        for pos in train_subset.indices
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
    metrics_long = _per_class_wide_to_long(pc_df)
    metrics_long.write_csv(experiment_dir / "metrics.csv")
    _class_metadata_frame(class_names, kind_map).write_csv(
        experiment_dir / "class_metadata.csv"
    )

    _generate_plots(
        pc_df,
        kind_map=kind_map,
        class_to_idx=class_to_idx,
        out_dir=experiment_dir,
        eval_class_groups=config.eval_class_groups,
    )

    model_path = experiment_dir / "unlearned_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, model_path)

    print(f"Saved metrics to {experiment_dir / 'metrics.csv'}")
    print(f"Saved class metadata to {experiment_dir / 'class_metadata.csv'}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    data_root = "data"
    dangerous_classes: set[str] = {
        "forest",
        "willow_tree",
        "maple_tree",
        "oak_tree",
        "pine_tree",
    }

    cifar100_for_typ = CIFAR100(train=True, root=data_root, transform=None)
    safety_for_typ = CIFAR100Safety.from_cifar100(
        cifar100_for_typ,
        dangerous_classes=dangerous_classes,
        unknown_classes=frozenset(),
    )
    n_cls = len(CIFAR100_CLASSES)
    mean_typ_idx = _mean_typicality_per_class_index(safety_for_typ, num_classes=n_cls)
    mean_typ_by_name = {
        CIFAR100_CLASSES[i]: float(mean_typ_idx[i]) for i in range(n_cls)
    }
    dangerous_classes_ordered = tuple(
        sorted(dangerous_classes, key=lambda c: mean_typ_by_name[c], reverse=True)
    )
    safe_classes: set[str] = set(CIFAR100_CLASSES) - dangerous_classes
    safe_classes_ordered = tuple(
        sorted(safe_classes, key=lambda c: mean_typ_by_name[c], reverse=True)
    )

    strategies: tuple[UnlearningStrategy, ...] = (
        "ignore-unknown",
        "forget-unknown",
        "retain-unknown",
    )

    configs = build_unlearn_configs_for_dangerous_grid(
        dangerous_classes=dangerous_classes_ordered,
        safe_classes_ordered=safe_classes_ordered,
        class_names=CIFAR100_CLASSES,
        strategies=strategies,
        seed=42,
        name_prefix="tree",
    )

    print(f"Running {len(configs)} experiments")
    for i, config in enumerate(configs, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(configs)}] {config.name}")
        print(f"{'=' * 60}")
        main(config)
