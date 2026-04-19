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
from torch.utils.data import DataLoader, Dataset, Subset

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

# Threshold on P(dangerous) used by the `classify-unknown` strategy to decide
# whether an unknown-class item goes into the forget (>=) or retain (<) set.
SAFETY_CLASSIFIER_THRESHOLD: float = 0.5
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


UnlearningStrategy = Literal[
    "ignore-unknown",
    "retain-unknown",
    "forget-unknown",
    "classify-unknown",
]
Membership = Literal["retain", "forget", "exclude"]
EvalSplit = Literal["train", "test"]


@dataclass
class UnlearnConfig:
    name: str | None = None
    dataset: CifarVariant = "cifar100"
    seed: int = 42
    max_steps: int = 1000
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
    # Path to a trained safety classifier (e.g.
    # "experiments/.../safety_classifier.pt"). Only used when
    # `unlearning_strategy == "classify-unknown"`.
    safety_classifier_path: str | None = None
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
    safety_classifier_paths_by_pct: dict[int, str] | None = None,
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

            safety_classifier_path: str | None = None
            if strategy == "classify-unknown":
                if (
                    safety_classifier_paths_by_pct is None
                    or pct_tag not in safety_classifier_paths_by_pct
                ):
                    raise ValueError(
                        "strategy 'classify-unknown' requires "
                        "safety_classifier_paths_by_pct with an entry for "
                        f"pct_tag={pct_tag}"
                    )
                safety_classifier_path = safety_classifier_paths_by_pct[pct_tag]

            configs.append(
                UnlearnConfig(
                    name=name,
                    dangerous_classes=dangerous_classes,
                    known_classes=known_classes,
                    unlearning_strategy=strategy,
                    eval_class_groups=eval_class_groups,
                    safety_classifier_path=safety_classifier_path,
                )
            )

    return configs


def _find_safety_classifier_path(
    *,
    experiments_root: str,
    name_prefix: str,
    pct_tag: int,
) -> Path:
    root = Path(experiments_root)
    if not root.exists():
        raise FileNotFoundError(f"Experiments root does not exist: {root}")
    suffix = f"_safety_classifier_{name_prefix}_{pct_tag}p"
    candidates = sorted(
        [
            p
            for p in root.iterdir()
            if p.is_dir()
            and p.name.endswith(suffix)
            and (p / "safety_classifier.pt").exists()
        ],
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No safety classifier experiment under {root} matching "
            f"'*{suffix}' with a safety_classifier.pt file"
        )
    return candidates[-1] / "safety_classifier.pt"


def _verify_classifier_matches_unlearn(
    *,
    classifier_path: Path,
    expected_dangerous_classes: tuple[str, ...],
    expected_known_classes: tuple[str, ...],
) -> None:
    config_path = classifier_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Classifier config.json not found: {config_path}")
    with open(config_path) as f:
        cfg = json.load(f)
    got_dang = tuple(cfg.get("dangerous_classes", ()))
    got_known = tuple(cfg.get("known_classes", ()))
    if tuple(sorted(got_dang)) != tuple(sorted(expected_dangerous_classes)):
        raise ValueError(
            f"Classifier at {classifier_path} has dangerous_classes={got_dang}, "
            f"expected {expected_dangerous_classes}"
        )
    if tuple(sorted(got_known)) != tuple(sorted(expected_known_classes)):
        raise ValueError(
            f"Classifier at {classifier_path} has known_classes={got_known}, "
            f"expected {expected_known_classes}"
        )


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


@torch.no_grad()
def _compute_classifier_dangerous_probs(
    safety: SafetyDataset,
    *,
    classifier_path: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    # Lazy import to avoid a circular dependency: train_safety_classifier imports
    # from this module.
    from src.cifar.train_safety_classifier import build_binary_cifar_resnet18

    model = build_binary_cifar_resnet18().to(device)
    checkpoint = torch.load(classifier_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(safety, batch_size=batch_size, shuffle=False, num_workers=0)
    probs_chunks: list[np.ndarray] = []
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        p_dangerous = torch.softmax(logits, dim=1)[:, 1]
        probs_chunks.append(p_dangerous.detach().cpu().numpy())
    return np.concatenate(probs_chunks, axis=0)


def _compute_membership_arr(
    safety: SafetyDataset,
    *,
    strategy: UnlearningStrategy,
    classifier_dangerous_probs: np.ndarray | None = None,
    classifier_threshold: float = SAFETY_CLASSIFIER_THRESHOLD,
) -> np.ndarray:
    kind_arr = np.asarray(safety.kind_arr)
    membership = np.empty(len(kind_arr), dtype=object)

    # Known-class items keep their ground-truth assignment regardless of strategy.
    membership[kind_arr == "k-safe"] = "retain"
    membership[kind_arr == "k-dang"] = "forget"

    unknown_mask = (kind_arr == "u-safe") | (kind_arr == "u-dang")
    if strategy == "ignore-unknown":
        membership[unknown_mask] = "exclude"
    elif strategy == "retain-unknown":
        membership[unknown_mask] = "retain"
    elif strategy == "forget-unknown":
        membership[unknown_mask] = "forget"
    elif strategy == "classify-unknown":
        if classifier_dangerous_probs is None:
            raise ValueError(
                "classify-unknown strategy requires classifier_dangerous_probs"
            )
        if len(classifier_dangerous_probs) != len(kind_arr):
            raise ValueError(
                "classifier_dangerous_probs length does not match dataset length: "
                f"{len(classifier_dangerous_probs)} vs {len(kind_arr)}"
            )
        predicted_dangerous = classifier_dangerous_probs >= classifier_threshold
        membership[unknown_mask & predicted_dangerous] = "forget"
        membership[unknown_mask & ~predicted_dangerous] = "retain"
    else:
        raise ValueError(f"unknown unlearning strategy: {strategy!r}")

    return membership


class _SafetyWithMembership(Dataset):
    # Thin wrapper that injects a per-item `membership` label into the dataset dict
    # so the training loop can mask retain vs forget samples without needing the
    # ground-truth `kind` (which may disagree with membership when the strategy is
    # `classify-unknown`).

    def __init__(self, base: SafetyDataset, membership_arr: np.ndarray):
        if len(membership_arr) != len(base):
            raise ValueError(
                "membership_arr length does not match dataset length: "
                f"{len(membership_arr)} vs {len(base)}"
            )
        self.base = base
        self.membership_arr = membership_arr
        self.kind_arr = base.kind_arr

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.base[idx])
        item["membership"] = str(self.membership_arr[idx])
        return item


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
    print(f"  safety_classifier_path: {config.safety_classifier_path}")
    print(f"  eval_split: {config.eval_split}")
    print(f"  eval_class_groups: {config.eval_class_groups}")

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

    classifier_dangerous_probs: np.ndarray | None = None
    if config.unlearning_strategy == "classify-unknown":
        if config.safety_classifier_path is None:
            raise ValueError(
                "unlearning_strategy='classify-unknown' requires "
                "config.safety_classifier_path to be set"
            )
        print(
            f"Running safety classifier {config.safety_classifier_path} "
            f"on {len(train_safety)} train items (threshold={SAFETY_CLASSIFIER_THRESHOLD})"
        )
        classifier_dangerous_probs = _compute_classifier_dangerous_probs(
            train_safety,
            classifier_path=config.safety_classifier_path,
            device=device,
            batch_size=config.batch_size,
        )

    membership_arr = _compute_membership_arr(
        train_safety,
        strategy=config.unlearning_strategy,
        classifier_dangerous_probs=classifier_dangerous_probs,
        classifier_threshold=SAFETY_CLASSIFIER_THRESHOLD,
    )
    train_dataset = _SafetyWithMembership(train_safety, membership_arr)
    train_indices = np.nonzero(membership_arr != "exclude")[0].tolist()
    train_subset = Subset(train_dataset, train_indices)

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

    n_retain = int((membership_arr == "retain").sum())
    n_forget = int((membership_arr == "forget").sum())
    kind_counts = {k: int((train_safety.kind_arr == k).sum()) for k in ALL_KINDS}
    print(
        f"Train={len(train_subset)} (retain={n_retain}, forget={n_forget}), "
        f"eval={len(eval_safety)} ({config.eval_split} split)"
    )
    for k, c in kind_counts.items():
        print(f"  {k}: {c}")
    if classifier_dangerous_probs is not None:
        unknown_mask = np.isin(train_safety.kind_arr, ["u-safe", "u-dang"])
        u_safe_mask = train_safety.kind_arr == "u-safe"
        u_dang_mask = train_safety.kind_arr == "u-dang"
        pred_dang = classifier_dangerous_probs >= SAFETY_CLASSIFIER_THRESHOLD
        print(
            "Classifier routing for unknown items: "
            f"u-safe->forget={int((u_safe_mask & pred_dang).sum())}/"
            f"{int(u_safe_mask.sum())}, "
            f"u-dang->forget={int((u_dang_mask & pred_dang).sum())}/"
            f"{int(u_dang_mask.sum())}, "
            f"total unknown->forget={int((unknown_mask & pred_dang).sum())}/"
            f"{int(unknown_mask.sum())}"
        )

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
        memberships = list(batch["membership"])

        retain_mask = torch.tensor(
            [m == "retain" for m in memberships],
            dtype=torch.bool,
            device=device,
        )
        forget_mask = torch.tensor(
            [m == "forget" for m in memberships],
            dtype=torch.bool,
            device=device,
        )

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
    experiments_root = "experiments"
    name_prefix = "people"
    dangerous_classes: set[str] = {"man", "boy", "girl", "woman", "baby"}

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
        # "classify-unknown",
        # "forget-unknown",
        # "retain-unknown",
    )

    # For each k (pct_tag) used by the unlearn grid, locate the matching
    # safety classifier checkpoint and verify its training split shares the
    # same dangerous/known classes as the unlearn run.
    n_dang = len(dangerous_classes_ordered)
    ks = [1] if n_dang == 1 else list(range(1, n_dang))
    num_labels = len(CIFAR100_CLASSES)
    safety_classifier_paths_by_pct: dict[int, str] = {}
    for k in ks:
        pct_tag = int(round(100 * k / n_dang))
        # classifier_path = _find_safety_classifier_path(
        #     experiments_root=experiments_root,
        #     name_prefix=name_prefix,
        #     pct_tag=pct_tag,
        # )
        # Reproduce build_unlearn_configs_for_dangerous_grid's recipe to
        # compute the expected known_classes for this k.
        known_dangerous = set(dangerous_classes_ordered[:k])
        target_known_total = min(num_labels, round((k / n_dang) * num_labels))
        n_safe = max(0, min(len(safe_classes_ordered), target_known_total - k))
        expected_known = tuple(
            sorted(known_dangerous | set(safe_classes_ordered[:n_safe]))
        )
        # _verify_classifier_matches_unlearn(
        #     classifier_path=classifier_path,
        #     expected_dangerous_classes=dangerous_classes_ordered,
        #     expected_known_classes=expected_known,
        # )
        # print(f"[{pct_tag}p] classifier: {classifier_path}")
        # safety_classifier_paths_by_pct[pct_tag] = str(classifier_path)

    configs = build_unlearn_configs_for_dangerous_grid(
        dangerous_classes=dangerous_classes_ordered,
        safe_classes_ordered=safe_classes_ordered,
        class_names=CIFAR100_CLASSES,
        strategies=strategies,
        seed=42,
        name_prefix=name_prefix,
        safety_classifier_paths_by_pct=safety_classifier_paths_by_pct,
    )

    print(f"Running {len(configs)} experiments")
    for i, config in enumerate(configs, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(configs)}] {config.name}")
        print(f"{'=' * 60}")
        main(config)
