import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models

from src.cifar.data import (
    CIFAR100,
    CIFAR100_CLASS_TO_INDEX,
    CIFAR100_CLASSES,
    CIFAR100Safety,
    SafetyKind,
)
from src.cifar.train_resnet import (
    build_cifar_resnet18,
    get_eval_transform,
    get_train_transform,
)
from src.cifar.unlearn import (
    ALL_KINDS,
    _build_per_class_rows,
    _class_metadata_frame,
    _class_to_kind,
    _create_experiment_dir,
    _mean_typicality_per_class_index,
)
from src.utils import get_device, set_seed

EvalSplit = Literal["train", "test"]

# (probe_name, in_channels) for each ResNet-18 layer we probe off of. The
# pretrained ResNet-18 used here keeps spatial dims at 32x32 after conv1+bn1
# (we replace the stem maxpool with Identity) and halves at every subsequent
# stage, so channel counts are what matters for the linear head.
PROBE_SPECS: tuple[tuple[str, int], ...] = (
    ("stem", 64),
    ("layer1.0", 64),
    ("layer1.1", 64),
    ("layer2.0", 128),
    ("layer2.1", 128),
    ("layer3.0", 256),
    ("layer3.1", 256),
    ("layer4.0", 512),
    ("layer4.1", 512),
)
PROBE_NAMES: tuple[str, ...] = tuple(name for name, _ in PROBE_SPECS)


@dataclass
class TrainSafetyClassifierConfig:
    name: str | None = None
    seed: int = 42
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    data_root: str = "data"
    pretrained_resnet_path: str = "checkpoints/cifar100/train_resnet.pt"
    pretrained_num_classes: int = 100
    dangerous_classes: tuple[str, ...] = ()
    known_classes: tuple[str, ...] = ()
    eval_split: EvalSplit = "train"
    eval_class_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    experiments_root: str = "experiments"


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


def load_pretrained_cifar_resnet18(
    path: str | Path,
    *,
    num_classes: int,
    device: torch.device | str = "cpu",
) -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    backbone = build_cifar_resnet18(num_classes=num_classes)
    backbone.load_state_dict(state_dict)
    return backbone


class MultiProbeResNet18(nn.Module):
    def __init__(self, backbone: nn.Module, *, num_output_classes: int = 2) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.probe_names = PROBE_NAMES
        self.probes = nn.ModuleDict({
            self._probe_key(name): nn.Linear(c, num_output_classes)
            for name, c in PROBE_SPECS
        })
        self.pool = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def _probe_key(name: str) -> str:
        return name.replace(".", "_")

    def train(self, mode: bool = True) -> "MultiProbeResNet18":
        super().train(mode)
        # Always keep the frozen backbone in eval mode so BN running stats
        # from pretraining are used instead of drifting on the binary task.
        self.backbone.eval()
        return self

    def probe_parameters(self) -> list[nn.Parameter]:
        return list(self.probes.parameters())

    def forward(self, images_B3HW: torch.Tensor) -> dict[str, torch.Tensor]:
        acts_per_probe: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            x = self.backbone.conv1(images_B3HW)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            acts_per_probe["stem"] = x
            x = self.backbone.maxpool(x)
            for stage_name in ("layer1", "layer2", "layer3", "layer4"):
                stage = getattr(self.backbone, stage_name)
                for i, block in enumerate(stage):
                    x = block(x)
                    acts_per_probe[f"{stage_name}.{i}"] = x

        logits_per_probe: dict[str, torch.Tensor] = {}
        for name in self.probe_names:
            feat_BCHW = acts_per_probe[name]
            feat_BC = self.pool(feat_BCHW).flatten(start_dim=1)
            logits_per_probe[name] = self.probes[self._probe_key(name)](feat_BC)
        return logits_per_probe


def build_safety_classifier_configs_for_dangerous_grid(
    *,
    dangerous_classes: tuple[str, ...],
    safe_classes_ordered: tuple[str, ...],
    class_names: tuple[str, ...],
    name_prefix: str,
    seed: int = 42,
) -> list[TrainSafetyClassifierConfig]:
    if not dangerous_classes:
        raise ValueError("dangerous_classes must be non-empty")
    dangerous_set = set(dangerous_classes)
    if len(dangerous_set) != len(dangerous_classes):
        raise ValueError("dangerous_classes must not contain duplicates")
    for c in dangerous_classes:
        if c not in class_names:
            raise ValueError(
                f"dangerous class {c!r} is not in this dataset's class list"
            )

    pool_safe_set = set(class_names) - dangerous_set
    if set(safe_classes_ordered) != pool_safe_set:
        raise ValueError(
            "safe_classes_ordered must contain exactly the non-dangerous class names"
        )
    if len(safe_classes_ordered) != len(pool_safe_set):
        raise ValueError("safe_classes_ordered must not contain duplicates")

    n_dang = len(dangerous_classes)
    num_labels = len(class_names)
    configs: list[TrainSafetyClassifierConfig] = []
    for k in range(1, n_dang + 1):
        known_dangerous = list(dangerous_classes[:k])
        target_fraction = k / n_dang
        target_known_total = min(num_labels, round(target_fraction * num_labels))
        n_safe = max(
            0, min(len(safe_classes_ordered), target_known_total - k)
        )
        known_safe = list(safe_classes_ordered[:n_safe])
        known_classes = tuple(sorted(known_dangerous + known_safe))
        pct_tag = int(round(100 * target_fraction))
        name = f"safety_classifier_{name_prefix}_{pct_tag}p"
        eval_class_groups: dict[str, tuple[str, ...]] = {"all": class_names}
        configs.append(
            TrainSafetyClassifierConfig(
                name=name,
                seed=seed,
                dangerous_classes=dangerous_classes,
                known_classes=known_classes,
                eval_class_groups=eval_class_groups,
            )
        )
    return configs


def _subset_known(safety: CIFAR100Safety) -> Subset[CIFAR100Safety]:
    indices = torch.nonzero(
        torch.from_numpy(safety.is_label_known_arr), as_tuple=False
    ).squeeze(1).tolist()
    return Subset(safety, indices)


def train_one_epoch(
    model: MultiProbeResNet18,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    global_step: int,
) -> tuple[dict[str, float], dict[str, float], int]:
    model.train()
    loss_sum_per_probe = {name: 0.0 for name in model.probe_names}
    correct_per_probe = {name: 0 for name in model.probe_names}
    total_count = 0

    for batch in loader:
        images_B3HW = batch["image"].to(device)
        targets_B = batch["is_dangerous"].to(device=device, dtype=torch.long)

        optimizer.zero_grad()
        logits_per_probe = model(images_B3HW)

        total_loss = torch.zeros((), device=device)
        for name, logits_BC in logits_per_probe.items():
            loss = criterion(logits_BC, targets_B)
            total_loss = total_loss + loss
            loss_sum_per_probe[name] += loss.detach().item() * targets_B.size(0)
            correct_per_probe[name] += (
                logits_BC.argmax(dim=1).eq(targets_B).sum().item()
            )
        total_loss.backward()
        optimizer.step()

        total_count += targets_B.size(0)
        global_step += 1

    denom = max(total_count, 1)
    loss_per_probe = {n: loss_sum_per_probe[n] / denom for n in model.probe_names}
    acc_per_probe = {n: correct_per_probe[n] / denom for n in model.probe_names}
    return loss_per_probe, acc_per_probe, global_step


@torch.no_grad()
def _eval_per_class_binary_multi(
    model: MultiProbeResNet18,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, dict[str, torch.Tensor]]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    stats_per_probe: dict[str, dict[str, torch.Tensor]] = {
        name: {
            "count": torch.zeros(num_classes, dtype=torch.long, device=device),
            "top1": torch.zeros(num_classes, dtype=torch.long, device=device),
            "top5": torch.zeros(num_classes, dtype=torch.long, device=device),
            "loss": torch.zeros(num_classes, dtype=torch.double, device=device),
        }
        for name in model.probe_names
    }

    for batch in loader:
        images_B3HW = batch["image"].to(device)
        fine_labels_B = batch["label"].to(device)
        targets_B = batch["is_dangerous"].to(device=device, dtype=torch.long)

        logits_per_probe = model(images_B3HW)
        for name, logits_BC in logits_per_probe.items():
            per_loss_B = criterion(logits_BC, targets_B)
            top1_hit_B = logits_BC.argmax(dim=1).eq(targets_B)
            k = min(5, logits_BC.size(1))
            top5_hit_B = (
                logits_BC.topk(k, dim=1).indices.eq(targets_B.unsqueeze(1)).any(dim=1)
            )

            s = stats_per_probe[name]
            s["count"].scatter_add_(0, fine_labels_B, torch.ones_like(fine_labels_B))
            s["top1"].scatter_add_(0, fine_labels_B, top1_hit_B.long())
            s["top5"].scatter_add_(0, fine_labels_B, top5_hit_B.long())
            s["loss"].scatter_add_(0, fine_labels_B, per_loss_B.double())

    return {
        name: {k: v.cpu() for k, v in s.items()}
        for name, s in stats_per_probe.items()
    }


def _build_per_probe_per_class_rows(
    stats_per_probe: dict[str, dict[str, torch.Tensor]],
    *,
    step: int,
    class_names: tuple[str, ...],
    kind_map: dict[int, SafetyKind],
) -> list[dict]:
    rows: list[dict] = []
    for probe_name, stats in stats_per_probe.items():
        for row in _build_per_class_rows(
            stats, step=step, class_names=class_names, kind_map=kind_map
        ):
            row["probe"] = probe_name
            rows.append(row)
    return rows


def _run_eval(
    model: MultiProbeResNet18,
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
    stats_per_probe = _eval_per_class_binary_multi(
        model, eval_loader, device, num_classes
    )
    per_class_rows.extend(
        _build_per_probe_per_class_rows(
            stats_per_probe,
            step=step,
            class_names=class_names,
            kind_map=kind_map,
        )
    )

    safe_idxs = set(range(num_classes)) - dangerous_idxs
    print(f"  [eval step={step}] per-probe top1:")
    for probe_name in model.probe_names:
        s = stats_per_probe[probe_name]
        dang_c = sum(int(s["count"][i]) for i in dangerous_idxs)
        dang_t1 = sum(int(s["top1"][i]) for i in dangerous_idxs)
        safe_c = sum(int(s["count"][i]) for i in safe_idxs)
        safe_t1 = sum(int(s["top1"][i]) for i in safe_idxs)
        total = safe_c + dang_c
        print(
            f"    {probe_name:<10s} "
            f"overall={((safe_t1 + dang_t1) / max(total, 1)):.2%}, "
            f"safe={safe_t1 / max(safe_c, 1):.2%}, "
            f"dang={dang_t1 / max(dang_c, 1):.2%}"
        )


def _per_probe_wide_to_long(wide: pl.DataFrame) -> pl.DataFrame:
    metric_cols = ("count", "top1_correct", "top5_correct", "loss_sum", "top1_acc", "top5_acc", "loss")
    return wide.unpivot(
        index=["probe", "step", "class_idx", "class_name"],
        on=list(metric_cols),
        variable_name="metric",
        value_name="value",
    ).sort("probe", "step", "class_idx", "metric")


def _accuracy_per_probe_step(
    pc_df: pl.DataFrame,
    *,
    extra_group_col: str | None = None,
) -> pl.DataFrame:
    group_cols = ["probe", "step"]
    if extra_group_col is not None:
        group_cols.append(extra_group_col)
    return (
        pc_df.group_by(*group_cols)
        .agg(
            pl.col("top1_correct").sum().alias("top1_correct"),
            pl.col("count").sum().alias("count"),
        )
        .with_columns(
            pl.when(pl.col("count") > 0)
            .then(pl.col("top1_correct") / pl.col("count"))
            .otherwise(pl.lit(float("nan")))
            .alias("top1_acc")
        )
        .sort(*group_cols)
    )


def _overall_accuracy_per_probe_step(pc_df: pl.DataFrame) -> pl.DataFrame:
    return _accuracy_per_probe_step(pc_df).rename({"top1_acc": "overall_top1_acc"})


def _plot_probe_lines(
    ax: "plt.Axes",
    acc_df: pl.DataFrame,
    *,
    probe_order: tuple[str, ...],
    colors,
    acc_col: str = "top1_acc",
) -> None:
    for i, probe_name in enumerate(probe_order):
        g = acc_df.filter(pl.col("probe") == probe_name).sort("step")
        if g.is_empty():
            continue
        ax.plot(
            g["step"].to_list(),
            [100.0 * v for v in g[acc_col].to_list()],
            label=probe_name,
            color=colors[i],
            linewidth=1.5,
            marker="o",
            markersize=3,
        )
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


def _plot_probe_accuracy_over_steps(
    overall_df: pl.DataFrame,
    *,
    probe_order: tuple[str, ...],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(probe_order)))
    _plot_probe_lines(
        ax, overall_df, probe_order=probe_order, colors=colors,
        acc_col="overall_top1_acc",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Overall top-1 accuracy (%)")
    ax.set_title("Linear-probe accuracy by layer over training")
    ax.legend(loc="lower right", fontsize=9, title="Probe layer")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_probe_accuracy_by_safety(
    pc_df: pl.DataFrame,
    *,
    probe_order: tuple[str, ...],
    save_path: Path,
) -> None:
    pc_with_safety = pc_df.with_columns(
        pl.when(pl.col("kind").is_in(["k-dang", "u-dang"]))
        .then(pl.lit("dangerous"))
        .otherwise(pl.lit("safe"))
        .alias("safety")
    )
    acc = _accuracy_per_probe_step(pc_with_safety, extra_group_col="safety")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(probe_order)))
    for ax, safety_label in zip(axes, ("safe", "dangerous")):
        sub = acc.filter(pl.col("safety") == safety_label)
        _plot_probe_lines(ax, sub, probe_order=probe_order, colors=colors)
        ax.set_xlabel("Step")
        ax.set_title(f"{safety_label} classes")
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[-1].legend(loc="lower right", fontsize=8, title="Probe layer")
    fig.suptitle("Linear-probe accuracy: safe vs dangerous classes")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_probe_accuracy_by_kind(
    pc_df: pl.DataFrame,
    *,
    probe_order: tuple[str, ...],
    kinds: tuple[SafetyKind, ...] = ALL_KINDS,
    save_path: Path,
) -> None:
    acc = _accuracy_per_probe_step(pc_df, extra_group_col="kind")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(probe_order)))
    for ax, kind in zip(axes_flat, kinds):
        sub = acc.filter(pl.col("kind") == kind)
        _plot_probe_lines(ax, sub, probe_order=probe_order, colors=colors)
        ax.set_title(kind)
    for ax in axes[-1, :]:
        ax.set_xlabel("Step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Top-1 accuracy (%)")
    axes_flat[-1].legend(loc="lower right", fontsize=7, title="Probe layer")
    fig.suptitle("Linear-probe accuracy by class kind")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _probe_ranking_frame(
    overall_df: pl.DataFrame,
    *,
    probe_order: tuple[str, ...],
) -> pl.DataFrame:
    final_step = overall_df["step"].max()
    final = overall_df.filter(pl.col("step") == final_step)
    probe_rank = {name: i for i, name in enumerate(probe_order)}
    final = final.with_columns(
        pl.col("probe").replace_strict(probe_rank, default=len(probe_order)).alias("_layer_order")
    )
    ranked = (
        final.sort(
            ["overall_top1_acc", "_layer_order"],
            descending=[True, False],
        )
        .with_row_index(name="rank", offset=1)
        .select(
            [
                "rank",
                "probe",
                "overall_top1_acc",
                "top1_correct",
                "count",
                "step",
            ]
        )
    )
    return ranked


def main(config: TrainSafetyClassifierConfig) -> None:
    if config.name is None:
        config.name = uuid.uuid4().hex[:8]

    set_seed(config.seed)
    device = get_device()

    print(f"Using device: {device}")
    print("Safety classifier config:")
    print(f"  epochs: {config.epochs}")
    print(f"  lr: {config.lr}")
    print(f"  weight_decay: {config.weight_decay}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  pretrained_resnet_path: {config.pretrained_resnet_path}")
    print(f"  dangerous_classes: {config.dangerous_classes}")
    print(f"  known_classes: {config.known_classes}")
    print(f"  eval_split: {config.eval_split}")
    print(f"  eval_class_groups: {config.eval_class_groups}")

    experiment_dir = _create_experiment_dir(
        config.experiments_root, name=config.name
    )
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Experiment dir: {experiment_dir}")

    class_names = CIFAR100_CLASSES
    class_to_idx = CIFAR100_CLASS_TO_INDEX
    num_classes = len(class_names)

    dangerous_set = set(config.dangerous_classes)
    if len(config.known_classes) == 0:
        known_set = set(class_names)
    else:
        known_set = set(config.known_classes)
    unknown_set = set(class_names) - known_set

    dangerous_idxs = {class_to_idx[c] for c in dangerous_set}
    known_idxs = {class_to_idx[c] for c in known_set}
    kind_map = {
        ci: _class_to_kind(ci, dangerous_idxs=dangerous_idxs, known_idxs=known_idxs)
        for ci in range(num_classes)
    }

    base_train = CIFAR100(
        train=True, root=config.data_root, transform=get_train_transform()
    )
    train_safety = CIFAR100Safety.from_cifar100(
        base_train,
        dangerous_classes=dangerous_set,
        unknown_classes=unknown_set,
    )
    train_known_subset = _subset_known(train_safety)

    base_eval = CIFAR100(
        train=(config.eval_split == "train"),
        root=config.data_root,
        transform=get_eval_transform(),
    )
    eval_safety = CIFAR100Safety.from_cifar100(
        base_eval,
        dangerous_classes=dangerous_set,
        unknown_classes=unknown_set,
    )

    train_loader = DataLoader(
        train_known_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_safety,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    kind_counts = {
        k: int((train_safety.kind_arr == k).sum()) for k in ALL_KINDS
    }
    print(
        f"Train (known only)={len(train_known_subset)}, "
        f"eval={len(eval_safety)} ({config.eval_split} split)"
    )
    for k, c in kind_counts.items():
        print(f"  {k}: {c}")

    backbone = load_pretrained_cifar_resnet18(
        config.pretrained_resnet_path,
        num_classes=config.pretrained_num_classes,
        device="cpu",
    )
    model = MultiProbeResNet18(backbone, num_output_classes=2).to(device)
    print(f"Loaded pretrained backbone from {config.pretrained_resnet_path}")
    print(f"Probing layers: {list(model.probe_names)}")

    optimizer = optim.Adam(
        model.probe_parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.lr * 0.1,
    )
    criterion = nn.CrossEntropyLoss()

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

    global_step = 0
    _run_eval(model, step=global_step, **eval_kwargs)

    for epoch in range(config.epochs):
        loss_per_probe, acc_per_probe, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            global_step=global_step,
        )
        mean_loss = sum(loss_per_probe.values()) / len(loss_per_probe)
        best_probe, best_acc = max(acc_per_probe.items(), key=lambda kv: kv[1])
        worst_probe, worst_acc = min(acc_per_probe.items(), key=lambda kv: kv[1])
        print(
            f"Epoch {epoch + 1}/{config.epochs} (step={global_step}) - "
            f"train/mean_loss={mean_loss:.4f}, "
            f"train/best={best_probe}@{best_acc:.2%}, "
            f"train/worst={worst_probe}@{worst_acc:.2%}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        scheduler.step()
        _run_eval(model, step=global_step, **eval_kwargs)

    pc_df = pl.DataFrame(per_class_rows)
    metrics_long = _per_probe_wide_to_long(pc_df)
    metrics_csv_path = experiment_dir / "metrics.csv"
    metrics_long.write_csv(metrics_csv_path)
    _class_metadata_frame(class_names, kind_map).write_csv(
        experiment_dir / "class_metadata.csv"
    )

    # Derive probe-level artefacts from the per-class log: overall accuracy
    # plot + ranking, plus safe/dangerous (2 panels) and per-kind (4 panels)
    # comparisons, all with one line per probe.
    overall_df = _overall_accuracy_per_probe_step(pc_df)
    overall_df.write_csv(experiment_dir / "probe_overall_accuracy.csv")

    _plot_probe_accuracy_over_steps(
        overall_df,
        probe_order=PROBE_NAMES,
        save_path=experiment_dir / "probe_accuracy_over_steps.png",
    )
    _plot_probe_accuracy_by_safety(
        pc_df,
        probe_order=PROBE_NAMES,
        save_path=experiment_dir / "probe_accuracy_by_safety.png",
    )
    _plot_probe_accuracy_by_kind(
        pc_df,
        probe_order=PROBE_NAMES,
        save_path=experiment_dir / "probe_accuracy_by_kind.png",
    )
    ranking = _probe_ranking_frame(overall_df, probe_order=PROBE_NAMES)
    ranking.write_csv(experiment_dir / "probe_ranking.csv")

    print("\nFinal probe ranking (by overall top-1 accuracy):")
    for row in ranking.iter_rows(named=True):
        print(f"  #{row['rank']} {row['probe']:<10s} acc={row['overall_top1_acc']:.2%}")

    model_path = experiment_dir / "safety_classifier.pt"
    torch.save({"model_state_dict": model.state_dict()}, model_path)

    print(f"Saved metrics to {metrics_csv_path}")
    print(f"Saved class metadata to {experiment_dir / 'class_metadata.csv'}")
    print(f"Saved probe ranking to {experiment_dir / 'probe_ranking.csv'}")
    print(f"Saved overall-accuracy plot to {experiment_dir / 'probe_accuracy_over_steps.png'}")
    print(f"Saved safe-vs-dangerous plot to {experiment_dir / 'probe_accuracy_by_safety.png'}")
    print(f"Saved by-kind plot to {experiment_dir / 'probe_accuracy_by_kind.png'}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    data_root = "data"
    name_prefix = "people"
    dangerous_classes: set[str] = {"man", "boy", "girl", "woman", "baby"}

    cifar100_for_typ = CIFAR100(train=True, root=data_root, transform=None)
    safety_for_typ = CIFAR100Safety.from_cifar100(
        cifar100_for_typ,
        dangerous_classes=dangerous_classes,
        unknown_classes=frozenset(),
    )
    n_cls = len(CIFAR100_CLASSES)
    mean_typ_idx = _mean_typicality_per_class_index(
        safety_for_typ, num_classes=n_cls
    )
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

    configs = build_safety_classifier_configs_for_dangerous_grid(
        dangerous_classes=dangerous_classes_ordered,
        safe_classes_ordered=safe_classes_ordered,
        class_names=CIFAR100_CLASSES,
        name_prefix=name_prefix,
        seed=42,
    )

    print(f"Running {len(configs)} experiments")
    for i, config in enumerate(configs, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(configs)}] {config.name}")
        print(f"{'=' * 60}")
        main(config)
