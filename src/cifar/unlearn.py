import json
import random
import uuid
from dataclasses import asdict, dataclass, field
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
    eval_every_n_steps: int = 200
    log_every_n_steps: int = 1000

    data_root: str = "data"
    dangerous_classes: tuple[str, ...] = ("willow_tree", "maple_tree", "oak_tree", "pine_tree")
    unknown_classes: tuple[str, ...] = ()
    unlearning_strategy: UnlearningStrategy = "ignore-unknown"

    eval_split: EvalSplit = "train"
    eval_class_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)

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
                    fontsize=8,
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
    print(f"  unknown_classes: {config.unknown_classes}")
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
    unknown_set = set(config.unknown_classes)
    eval_transform = get_eval_transform()

    if config.dataset == "cifar100":
        class_names = CIFAR100_CLASSES
        class_to_idx = CIFAR100_CLASS_TO_INDEX
    else:
        class_names = CIFAR10_CLASSES
        class_to_idx = CLASS_TO_INDEX

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
        eval_class_groups=config.eval_class_groups,
    )

    model_path = experiment_dir / "unlearned_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, model_path)

    print(f"Saved per-class metrics to {experiment_dir / 'per_class_metrics.csv'}")
    print(f"Saved model to {model_path}")


# if __name__ == "__main__":
#     dangerous_classes = ("lion", "tiger", "leopard")
#     dangerous_set = set(dangerous_classes)
#
#     experiment_setups: list[dict] = [
#         {
#             "tag": "33p",
#             "known_dangerous": {"tiger"},
#             "n_safe": 32,
#         },
#         {
#             "tag": "66p",
#             "known_dangerous": {"tiger", "lion"},
#             "n_safe": 64,
#         },
#     ]
#
#     strategies: list[UnlearningStrategy] = [
#         "ignore-unknown",
#         "forget-unknown",
#         "retain-unknown",
#     ]
#
#     configs: list[UnlearnConfig] = []
#
#     for setup in experiment_setups:
#         set_seed(42)
#         pool = [c for c in CIFAR100_CLASSES if c not in dangerous_set]
#         sampled_safe = set(random.sample(pool, setup["n_safe"]))
#         known_classes = setup["known_dangerous"] | sampled_safe
#         unknown_classes = tuple(c for c in CIFAR100_CLASSES if c not in known_classes)
#
#         unknown_safe = [c for c in unknown_classes if c not in dangerous_set]
#         known_safe = sorted(sampled_safe - dangerous_set)
#
#         subsample_classes = (
#             list(dangerous_classes)
#             + random.sample(unknown_safe, 3)
#             + random.sample(known_safe, 3)
#         )
#
#         eval_class_groups: dict[str, tuple[str, ...]] = {
#             "all": CIFAR100_CLASSES,
#             "unknown": tuple(sorted(dangerous_set | set(unknown_classes))),
#             "subsample": tuple(sorted(subsample_classes)),
#         }
#
#         for strategy in strategies:
#             tag = strategy.replace("-", "_")
#             name = f"feline_{setup['tag']}_{tag}"
#             configs.append(
#                 UnlearnConfig(
#                     name=name,
#                     dangerous_classes=dangerous_classes,
#                     unknown_classes=unknown_classes,
#                     unlearning_strategy=strategy,
#                     eval_class_groups=eval_class_groups,
#                 )
#             )
#
#     print(f"Running {len(configs)} experiments")
#     for i, config in enumerate(configs, 1):
#         print(f"\n{'=' * 60}")
#         print(f"[{i}/{len(configs)}] {config.name}")
#         print(f"{'=' * 60}")
#         main(config)


if __name__ == "__main__":
    dangerous_classes = ("willow_tree", "maple_tree", "oak_tree", "pine_tree")
    dangerous_set = set(dangerous_classes)

    experiment_setups: list[dict] = [
        {
            "tag": "25p",
            "known_dangerous": {"willow_tree"},
            "n_safe": 24,
        },
        {
            "tag": "50p",
            "known_dangerous": {"willow_tree", "maple_tree"},
            "n_safe": 48,
        },
        {
            "tag": "75p",
            "known_dangerous": {"willow_tree", "maple_tree", "oak_tree"},
            "n_safe": 72,
        },
    ]

    strategies: list[UnlearningStrategy] = [
        "ignore-unknown",
        "forget-unknown",
        "retain-unknown",
    ]

    configs: list[UnlearnConfig] = []

    for setup in experiment_setups:
        set_seed(42)
        pool = [c for c in CIFAR100_CLASSES if c not in dangerous_set]
        sampled_safe = set(random.sample(pool, setup["n_safe"]))
        known_classes = setup["known_dangerous"] | sampled_safe
        unknown_classes = tuple(c for c in CIFAR100_CLASSES if c not in known_classes)

        unknown_safe = [c for c in unknown_classes if c not in dangerous_set]
        known_safe = sorted(sampled_safe - dangerous_set)

        subsample_classes = (
            list(dangerous_classes)
            + random.sample(unknown_safe, 3)
            + random.sample(known_safe, 3)
        )

        eval_class_groups: dict[str, tuple[str, ...]] = {
            "all": CIFAR100_CLASSES,
            "unknown": tuple(sorted(dangerous_set | set(unknown_classes))),
            "subsample": tuple(sorted(subsample_classes)),
        }

        for strategy in strategies:
            tag = strategy.replace("-", "_")
            name = f"tree_{setup['tag']}_{tag}"
            configs.append(
                UnlearnConfig(
                    name=name,
                    dangerous_classes=dangerous_classes,
                    unknown_classes=unknown_classes,
                    unlearning_strategy=strategy,
                    eval_class_groups=eval_class_groups,
                )
            )

    print(f"Running {len(configs)} experiments")
    for i, config in enumerate(configs, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(configs)}] {config.name}")
        print(f"{'=' * 60}")
        main(config)
