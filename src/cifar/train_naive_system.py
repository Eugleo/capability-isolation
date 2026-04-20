import json
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.cifar.data import (
    CIFAR100,
    CIFAR100_CLASS_TO_INDEX,
    CIFAR100_CLASSES,
    CIFAR100Safety,
    SafetyKind,
)
from src.cifar.system import NaiveSystem
from src.cifar.train_resnet import build_cifar_resnet18, get_eval_transform
from src.cifar.train_safety_classifier import build_binary_cifar_resnet18
from src.cifar.unlearn import (
    ALL_KINDS,
    GROUP_COLORS,
    GROUP_LINESTYLES,
    PER_CLASS_COLORS,
    PER_CLASS_METRIC_COLUMNS,
    _add_safety_col,
    _aggregate,
    _build_per_class_rows,
    _class_metadata_frame,
    _class_to_kind,
    _create_experiment_dir,
    _eval_per_class,
    _generate_plots,
    _mean_typicality_per_class_index,
    _per_class_wide_to_long,
    _plot_lines,
)
from src.utils import get_device, set_seed

EvalSplit = Literal["train", "test"]


@dataclass
class TrainNaiveSystemConfig:
    name: str | None = None
    seed: int = 42

    max_steps: int = 20000
    lr: float = 1e-5
    weight_decay: float = 0.0
    batch_size: int = 128
    max_grad_norm: float = 1.0
    eval_every_n_steps: int = 200
    log_every_n_steps: int = 1000

    # Loss term weights.
    system_ce_weight: float = 1.0
    gate_bce_weight: float = 1.0
    safe_ascent_weight: float = 5e-5
    dangerous_ascent_weight: float = 5e-5

    # During the first `gate_warmup_steps` steps the system CE loss is computed
    # as if the gate emitted 1 for every unknown-label item (i.e. the dangerous
    # model handles everything except known-safe items). The gate BCE loss and
    # ascent losses are unaffected.
    gate_warmup_steps: int = 5000

    # Which submodules receive gradient updates.
    is_safe_model_trainable: bool = True
    is_dangerous_model_trainable: bool = False
    is_gate_trainable: bool = True

    # Init paths; None means freshly initialized weights.
    safe_model_path: str | None = "checkpoints/cifar100/train_resnet.pt"
    dangerous_model_path: str | None = "checkpoints/cifar100/train_resnet.pt"
    gate_path: str | None = None

    data_root: str = "data"
    dangerous_classes: tuple[str, ...] = ()
    known_classes: tuple[str, ...] = ()
    eval_split: EvalSplit = "train"
    eval_class_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    experiments_root: str = "experiments"


def _load_resnet_state(
    path: str, model: nn.Module, device: torch.device
) -> None:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])


def _build_naive_system(
    *,
    config: TrainNaiveSystemConfig,
    num_classes: int,
    device: torch.device,
) -> NaiveSystem:
    gate = build_binary_cifar_resnet18().to(device)
    if config.gate_path is not None:
        _load_resnet_state(config.gate_path, gate, device)
        print(f"Loaded gate from {config.gate_path}")
    else:
        print("Gate: freshly initialized")

    model_safe = build_cifar_resnet18(num_classes=num_classes).to(device)
    if config.safe_model_path is not None:
        _load_resnet_state(config.safe_model_path, model_safe, device)
        print(f"Loaded safe model from {config.safe_model_path}")
    else:
        print("Safe model: freshly initialized")

    model_dangerous = build_cifar_resnet18(num_classes=num_classes).to(device)
    if config.dangerous_model_path is not None:
        _load_resnet_state(config.dangerous_model_path, model_dangerous, device)
        print(f"Loaded dangerous model from {config.dangerous_model_path}")
    else:
        print("Dangerous model: freshly initialized")

    return NaiveSystem(
        gate=gate,
        model_safe=model_safe,
        model_dangerous=model_dangerous,
        num_classes=num_classes,
    )


@torch.no_grad()
def _eval_per_class_gate(
    system: NaiveSystem,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    system.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    correct_top1 = torch.zeros(num_classes, dtype=torch.long, device=device)
    correct_top5 = torch.zeros(num_classes, dtype=torch.long, device=device)
    loss_sum = torch.zeros(num_classes, dtype=torch.double, device=device)
    gate_sum = torch.zeros(num_classes, dtype=torch.double, device=device)

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        targets = batch["is_dangerous"].to(device=device, dtype=torch.long)

        gate_logits_B2 = system.gate(images)
        per_loss = criterion(gate_logits_B2, targets)
        top1_hit = gate_logits_B2.argmax(dim=1).eq(targets)
        k = min(5, gate_logits_B2.size(1))
        top5_hit = (
            gate_logits_B2.topk(k, dim=1).indices.eq(targets.unsqueeze(1)).any(dim=1)
        )
        gate_value_B = torch.softmax(gate_logits_B2, dim=1)[:, 1].double()

        counts.scatter_add_(0, labels, torch.ones_like(labels))
        correct_top1.scatter_add_(0, labels, top1_hit.long())
        correct_top5.scatter_add_(0, labels, top5_hit.long())
        loss_sum.scatter_add_(0, labels, per_loss.double())
        gate_sum.scatter_add_(0, labels, gate_value_B)

    return {
        "count": counts.cpu(),
        "top1": correct_top1.cpu(),
        "top5": correct_top5.cpu(),
        "loss": loss_sum.cpu(),
        "gate_sum": gate_sum.cpu(),
    }


def _build_gate_rows(
    stats: dict[str, torch.Tensor],
    *,
    step: int,
    class_names: tuple[str, ...],
    kind_map: dict[int, SafetyKind],
) -> list[dict]:
    rows = _build_per_class_rows(
        stats, step=step, class_names=class_names, kind_map=kind_map
    )
    for ci, row in enumerate(rows):
        c = int(stats["count"][ci])
        gs = float(stats["gate_sum"][ci])
        row["gate_sum"] = gs
        row["avg_gate"] = gs / c if c > 0 else float("nan")
    return rows


GATE_PER_CLASS_METRIC_COLUMNS: tuple[str, ...] = tuple(PER_CLASS_METRIC_COLUMNS) + (
    "gate_sum",
    "avg_gate",
)


def _gate_per_class_wide_to_long(wide: pl.DataFrame) -> pl.DataFrame:
    return wide.unpivot(
        index=["step", "class_idx", "class_name"],
        on=list(GATE_PER_CLASS_METRIC_COLUMNS),
        variable_name="metric",
        value_name="value",
    ).sort("step", "class_idx", "metric")


def _aggregate_gate(df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    return (
        df.group_by("step", group_col)
        .agg(pl.col("count").sum(), pl.col("gate_sum").sum())
        .with_columns(
            pl.when(pl.col("count") > 0)
            .then(pl.col("gate_sum") / pl.col("count"))
            .otherwise(pl.lit(float("nan")))
            .alias("avg_gate")
        )
        .sort("step", group_col)
    )


def _generate_gate_avg_plots(
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

        grp_kind_df = _aggregate_gate(grp_df, "kind")
        grp_safety_df = _aggregate_gate(_add_safety_col(grp_df), "safety")
        grp_class_df = _aggregate_gate(grp_df, "class_name")

        grp_class_names = tuple(sorted(grp_df["class_name"].unique().to_list()))
        grp_colors: dict[str, str] = {}
        grp_linestyles: dict[str, str] = {}
        grp_markers: dict[str, str] = {}
        for i, cn in enumerate(grp_class_names):
            kind = kind_map[class_to_idx[cn]]
            grp_colors[cn] = PER_CLASS_COLORS[i % len(PER_CLASS_COLORS)]
            grp_markers[cn] = "*" if kind in ("k-dang", "u-dang") else "o"
            grp_linestyles[cn] = "--" if kind in ("u-safe", "u-dang") else "-"

        _plot_lines(
            grp_kind_df,
            group_col="kind",
            groups=ALL_KINDS,
            colors=GROUP_COLORS,
            linestyles=GROUP_LINESTYLES,
            metric="avg_gate",
            title=f"{group_name}: Avg gate value by Kind",
            save_path=grp_dir / "avg_gate_by_kind.png",
        )
        _plot_lines(
            grp_safety_df,
            group_col="safety",
            groups=("safe", "dangerous"),
            colors=GROUP_COLORS,
            linestyles=GROUP_LINESTYLES,
            metric="avg_gate",
            title=f"{group_name}: Avg gate value: Safe vs Dangerous",
            save_path=grp_dir / "avg_gate_safe_vs_dang.png",
        )
        _plot_lines(
            grp_class_df,
            group_col="class_name",
            groups=grp_class_names,
            colors=grp_colors,
            linestyles=grp_linestyles,
            markers=grp_markers,
            metric="avg_gate",
            title=f"{group_name}: Avg gate value by Class",
            save_path=grp_dir / "avg_gate_by_class.png",
            label_line_endpoints=True,
        )


@torch.no_grad()
def _eval_per_class_system(
    system: NaiveSystem,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    system.eval()

    counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    correct_top1 = torch.zeros(num_classes, dtype=torch.long, device=device)
    correct_top5 = torch.zeros(num_classes, dtype=torch.long, device=device)
    loss_sum = torch.zeros(num_classes, dtype=torch.double, device=device)

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        out = system(images)
        probs = out["system_output"]
        log_probs = torch.log(probs.clamp(min=1e-8))
        per_loss = nn.functional.nll_loss(log_probs, labels, reduction="none")

        top1_hit = probs.argmax(dim=1).eq(labels)
        k = min(5, probs.size(1))
        top5_hit = probs.topk(k, dim=1).indices.eq(labels.unsqueeze(1)).any(dim=1)

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


def _summarize_top1(
    stats: dict[str, torch.Tensor],
    *,
    dangerous_idxs: set[int],
    num_classes: int,
) -> str:
    safe_idxs = set(range(num_classes)) - dangerous_idxs
    dang_c = sum(int(stats["count"][i]) for i in dangerous_idxs)
    dang_t1 = sum(int(stats["top1"][i]) for i in dangerous_idxs)
    safe_c = sum(int(stats["count"][i]) for i in safe_idxs)
    safe_t1 = sum(int(stats["top1"][i]) for i in safe_idxs)
    total = safe_c + dang_c
    return (
        f"overall={((safe_t1 + dang_t1) / max(total, 1)):.2%}, "
        f"safe={safe_t1 / max(safe_c, 1):.2%}, "
        f"dang={dang_t1 / max(dang_c, 1):.2%}"
    )


def _save_outputs(
    *,
    experiment_dir: Path,
    rows_safe: list[dict],
    rows_dangerous: list[dict],
    rows_system: list[dict],
    rows_gate: list[dict],
    kind_map: dict[int, SafetyKind],
    class_to_idx: dict[str, int],
    class_names: tuple[str, ...],
    eval_class_groups: dict[str, tuple[str, ...]],
    verbose: bool = False,
) -> None:
    for sub_name, rows in [
        ("safe", rows_safe),
        ("dangerous", rows_dangerous),
        ("system", rows_system),
    ]:
        if not rows:
            continue
        sub_dir = experiment_dir / sub_name
        sub_dir.mkdir(parents=True, exist_ok=True)
        pc_df = pl.DataFrame(rows)
        _per_class_wide_to_long(pc_df).write_csv(sub_dir / "metrics.csv")
        _generate_plots(
            pc_df,
            kind_map=kind_map,
            class_to_idx=class_to_idx,
            out_dir=sub_dir,
            eval_class_groups=eval_class_groups,
        )
        if verbose:
            print(f"Saved {sub_name} metrics + plots to {sub_dir}")

    if rows_gate:
        gate_dir = experiment_dir / "gate"
        gate_dir.mkdir(parents=True, exist_ok=True)
        gate_pc_df = pl.DataFrame(rows_gate)
        _gate_per_class_wide_to_long(gate_pc_df).write_csv(gate_dir / "metrics.csv")
        _generate_plots(
            gate_pc_df,
            kind_map=kind_map,
            class_to_idx=class_to_idx,
            out_dir=gate_dir,
            eval_class_groups=eval_class_groups,
        )
        _generate_gate_avg_plots(
            gate_pc_df,
            kind_map=kind_map,
            class_to_idx=class_to_idx,
            out_dir=gate_dir,
            eval_class_groups=eval_class_groups,
        )
        if verbose:
            print(f"Saved gate metrics + plots to {gate_dir}")


def _run_eval(
    system: NaiveSystem,
    *,
    eval_loader: DataLoader,
    device: torch.device,
    step: int,
    num_classes: int,
    class_names: tuple[str, ...],
    class_to_idx: dict[str, int],
    kind_map: dict[int, SafetyKind],
    dangerous_idxs: set[int],
    rows_safe: list[dict],
    rows_dangerous: list[dict],
    rows_system: list[dict],
    rows_gate: list[dict],
    experiment_dir: Path,
    eval_class_groups: dict[str, tuple[str, ...]],
) -> None:
    safe_stats = _eval_per_class(
        system.model_safe, eval_loader, device, num_classes
    )
    dang_stats = _eval_per_class(
        system.model_dangerous, eval_loader, device, num_classes
    )
    sys_stats = _eval_per_class_system(system, eval_loader, device, num_classes)
    gate_stats = _eval_per_class_gate(system, eval_loader, device, num_classes)

    rows_safe.extend(
        _build_per_class_rows(
            safe_stats, step=step, class_names=class_names, kind_map=kind_map
        )
    )
    rows_dangerous.extend(
        _build_per_class_rows(
            dang_stats, step=step, class_names=class_names, kind_map=kind_map
        )
    )
    rows_system.extend(
        _build_per_class_rows(
            sys_stats, step=step, class_names=class_names, kind_map=kind_map
        )
    )
    rows_gate.extend(
        _build_gate_rows(
            gate_stats, step=step, class_names=class_names, kind_map=kind_map
        )
    )

    print(
        f"  [eval step={step}] safe top1: "
        f"{_summarize_top1(safe_stats, dangerous_idxs=dangerous_idxs, num_classes=num_classes)}"
    )
    print(
        f"  [eval step={step}] dang top1: "
        f"{_summarize_top1(dang_stats, dangerous_idxs=dangerous_idxs, num_classes=num_classes)}"
    )
    print(
        f"  [eval step={step}] system top1: "
        f"{_summarize_top1(sys_stats, dangerous_idxs=dangerous_idxs, num_classes=num_classes)}"
    )
    print(
        f"  [eval step={step}] gate top1: "
        f"{_summarize_top1(gate_stats, dangerous_idxs=dangerous_idxs, num_classes=num_classes)}"
    )

    _save_outputs(
        experiment_dir=experiment_dir,
        rows_safe=rows_safe,
        rows_dangerous=rows_dangerous,
        rows_system=rows_system,
        rows_gate=rows_gate,
        kind_map=kind_map,
        class_to_idx=class_to_idx,
        class_names=class_names,
        eval_class_groups=eval_class_groups,
    )


def main(config: TrainNaiveSystemConfig) -> None:
    if config.name is None:
        config.name = uuid.uuid4().hex[:8]

    set_seed(config.seed)
    device = get_device()

    print(f"Using device: {device}")
    print("Naive system config:")
    for field_name, value in asdict(config).items():
        print(f"  {field_name}: {value}")

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

    eval_transform = get_eval_transform()

    base_train = CIFAR100(
        train=True, root=config.data_root, transform=eval_transform
    )
    train_safety = CIFAR100Safety.from_cifar100(
        base_train,
        dangerous_classes=dangerous_set,
        unknown_classes=unknown_set,
    )

    if config.eval_split == "test":
        base_eval = CIFAR100(
            train=False, root=config.data_root, transform=eval_transform
        )
        eval_safety = CIFAR100Safety.from_cifar100(
            base_eval,
            dangerous_classes=dangerous_set,
            unknown_classes=unknown_set,
        )
    else:
        eval_safety = train_safety

    train_loader = DataLoader(
        train_safety,
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

    kind_counts = {k: int((train_safety.kind_arr == k).sum()) for k in ALL_KINDS}
    print(
        f"Train={len(train_safety)}, "
        f"eval={len(eval_safety)} ({config.eval_split} split)"
    )
    for k, c in kind_counts.items():
        print(f"  {k}: {c}")

    system = _build_naive_system(
        config=config, num_classes=num_classes, device=device
    )

    for p in system.model_safe.parameters():
        p.requires_grad = config.is_safe_model_trainable
    for p in system.model_dangerous.parameters():
        p.requires_grad = config.is_dangerous_model_trainable
    for p in system.gate.parameters():
        p.requires_grad = config.is_gate_trainable

    params: list[nn.Parameter] = []
    if config.is_safe_model_trainable:
        params.extend(system.model_safe.parameters())
    if config.is_dangerous_model_trainable:
        params.extend(system.model_dangerous.parameters())
    if config.is_gate_trainable:
        params.extend(system.gate.parameters())
    if not params:
        raise ValueError(
            "No trainable parameters: set at least one of "
            "is_safe_model_trainable / is_dangerous_model_trainable / is_gate_trainable"
        )

    optimizer = optim.Adam(
        params, lr=config.lr, weight_decay=config.weight_decay
    )

    rows_safe: list[dict] = []
    rows_dangerous: list[dict] = []
    rows_system: list[dict] = []
    rows_gate: list[dict] = []
    eval_kwargs = dict(
        eval_loader=eval_loader,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        class_to_idx=class_to_idx,
        kind_map=kind_map,
        dangerous_idxs=dangerous_idxs,
        rows_safe=rows_safe,
        rows_dangerous=rows_dangerous,
        rows_system=rows_system,
        rows_gate=rows_gate,
        experiment_dir=experiment_dir,
        eval_class_groups=config.eval_class_groups,
    )

    _run_eval(system, step=0, **eval_kwargs)

    global_step = 0
    running_system = 0.0
    running_gate = 0.0
    running_safe_asc = 0.0
    running_dang_asc = 0.0
    running_count = 0
    train_iter = iter(train_loader)

    system.train()
    while global_step < config.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        is_known_B = batch["is_label_known"].to(device=device, dtype=torch.bool)
        is_dangerous_B = batch["is_dangerous"].to(device=device, dtype=torch.bool)
        is_known_safe_B = is_known_B & ~is_dangerous_B
        is_known_dang_B = is_known_B & is_dangerous_B

        optimizer.zero_grad()
        out = system(images)
        safe_logits_BC = out["safe_logits"]
        dangerous_logits_BC = out["dangerous_logits"]
        safe_probs_BC = out["safe_model_output"]
        dangerous_probs_BC = out["dangerous_model_output"]
        system_probs_BC = out["system_output"]
        system_probs_detached_BC = out["system_output_detached"]
        computed_gate_B = out["computed_gate"]

        # Pick the per-example prediction used for the system CE loss:
        #   - known-safe     -> safe model output    (grads through safe only)
        #   - known-dangerous-> dangerous model output (grads through dangerous only)
        #   - unknown, warmup-> system output detached (grads through gate only)
        #   - unknown, post  -> system output         (grads through both + gate)
        in_warmup = global_step < config.gate_warmup_steps
        unknown_pred_BC = (
            system_probs_detached_BC if in_warmup else system_probs_BC
        )
        is_known_safe_B1 = is_known_safe_B.unsqueeze(1)
        is_known_dang_B1 = is_known_dang_B.unsqueeze(1)
        system_pred_for_ce_BC = torch.where(
            is_known_safe_B1,
            safe_probs_BC,
            torch.where(is_known_dang_B1, dangerous_probs_BC, unknown_pred_BC),
        )

        log_pred_BC = torch.log(system_pred_for_ce_BC.clamp(min=1e-8))
        L_system = nn.functional.nll_loss(log_pred_BC, labels)

        zero = torch.tensor(0.0, device=device)
        known_mask_B = is_known_safe_B | is_known_dang_B
        if known_mask_B.any():
            gate_target_B = is_known_dang_B.to(dtype=computed_gate_B.dtype)
            L_gate = nn.functional.binary_cross_entropy(
                computed_gate_B[known_mask_B], gate_target_B[known_mask_B]
            )
        else:
            L_gate = zero

        # Gradient ascent on dangerous-model predictions for known-safe items.
        if is_known_safe_B.any():
            L_dang_asc = -nn.functional.cross_entropy(
                dangerous_logits_BC[is_known_safe_B], labels[is_known_safe_B]
            )
        else:
            L_dang_asc = zero

        # Gradient ascent on safe-model predictions for known-dangerous items.
        if is_known_dang_B.any():
            L_safe_asc = -nn.functional.cross_entropy(
                safe_logits_BC[is_known_dang_B], labels[is_known_dang_B]
            )
        else:
            L_safe_asc = zero

        loss = (
            config.system_ce_weight * L_system
            + config.gate_bce_weight * L_gate
            + config.dangerous_ascent_weight * L_dang_asc
            + config.safe_ascent_weight * L_safe_asc
        )
        loss.backward()
        nn.utils.clip_grad_norm_(params, config.max_grad_norm)
        optimizer.step()

        bs = images.size(0)
        running_system += L_system.detach().item() * bs
        running_gate += L_gate.detach().item() * bs
        running_safe_asc += L_safe_asc.detach().item() * bs
        running_dang_asc += L_dang_asc.detach().item() * bs
        running_count += bs

        global_step += 1

        if global_step % config.log_every_n_steps == 0:
            n = max(running_count, 1)
            print(
                f"Step {global_step}/{config.max_steps} - "
                f"system={running_system / n:.4f}, "
                f"gate={running_gate / n:.4f}, "
                f"safe_asc={running_safe_asc / n:.4f}, "
                f"dang_asc={running_dang_asc / n:.4f}"
            )
            running_system = 0.0
            running_gate = 0.0
            running_safe_asc = 0.0
            running_dang_asc = 0.0
            running_count = 0

        if global_step % config.eval_every_n_steps == 0:
            _run_eval(system, step=global_step, **eval_kwargs)
            system.train()

    if global_step % config.eval_every_n_steps != 0:
        _run_eval(system, step=global_step, **eval_kwargs)

    _class_metadata_frame(class_names, kind_map).write_csv(
        experiment_dir / "class_metadata.csv"
    )

    system_path = experiment_dir / "naive_system.pt"
    system.save(system_path)
    print(f"Saved class metadata to {experiment_dir / 'class_metadata.csv'}")
    print(f"Saved system to {system_path}")


def build_naive_system_configs_for_dangerous_grid(
    *,
    dangerous_classes: tuple[str, ...],
    safe_classes_ordered: tuple[str, ...],
    class_names: tuple[str, ...],
    name_prefix: str,
    seed: int = 42,
    safety_classifier_paths_by_pct: dict[int, str] | None = None,
    only_pct: int | None = None,
) -> list[TrainNaiveSystemConfig]:
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
    configs: list[TrainNaiveSystemConfig] = []
    for k in range(1, n_dang + 1):
        known_dangerous = list(dangerous_classes[:k])
        target_fraction = k / n_dang
        target_known_total = min(num_labels, round(target_fraction * num_labels))
        n_safe = max(0, min(len(safe_classes_ordered), target_known_total - k))
        known_safe = list(safe_classes_ordered[:n_safe])
        known_classes = tuple(sorted(known_dangerous + known_safe))

        pct_tag = int(round(100 * target_fraction))
        if only_pct is not None and pct_tag != only_pct:
            continue
        name = f"naive_system_{name_prefix}_{pct_tag}p"
        eval_class_groups: dict[str, tuple[str, ...]] = {"all": class_names}

        gate_path = None
        if safety_classifier_paths_by_pct and pct_tag in safety_classifier_paths_by_pct:
            gate_path = safety_classifier_paths_by_pct[pct_tag]

        configs.append(
            TrainNaiveSystemConfig(
                name=name,
                seed=seed,
                dangerous_classes=dangerous_classes,
                known_classes=known_classes,
                eval_class_groups=eval_class_groups,
                gate_path=gate_path,
            )
        )
    return configs


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

    # configs = build_naive_system_configs_for_dangerous_grid(
    #     dangerous_classes=dangerous_classes_ordered,
    #     safe_classes_ordered=safe_classes_ordered,
    #     class_names=CIFAR100_CLASSES,
    #     name_prefix=name_prefix,
    #     seed=42,
    # )
    # print(f"Running {len(configs)} experiments")
    # for i, config in enumerate(configs, 1):
    #     print(f"\n{'=' * 60}")
    #     print(f"[{i}/{len(configs)}] {config.name}")
    #     print(f"{'=' * 60}")
    #     main(config)

    grid_configs = build_naive_system_configs_for_dangerous_grid(
        dangerous_classes=dangerous_classes_ordered,
        safe_classes_ordered=safe_classes_ordered,
        class_names=CIFAR100_CLASSES,
        name_prefix=name_prefix,
        seed=42,
        only_pct=20,
    )
    if len(grid_configs) != 1:
        raise RuntimeError(f"expected exactly one 20p config, got {len(grid_configs)}")
    base_20p = grid_configs[0]
    gate_bce_weights = (0, 1e-3, 1e-2, 1e-1, 1)
    configs = [
        replace(
            base_20p,
            name=f"{base_20p.name}_gate_loss_w={str(w).replace('.', 'p')}",
            gate_bce_weight=w,
            max_steps=10000,
        )
        for w in gate_bce_weights
    ]

    print(f"Running {len(configs)} experiments (20p base, gate BCE sweep)")
    for i, config in enumerate(configs, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(configs)}] {config.name}")
        print(f"{'=' * 60}")
        main(config)
