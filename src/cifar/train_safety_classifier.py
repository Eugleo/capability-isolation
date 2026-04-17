import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Literal

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
from src.cifar.train_resnet import get_eval_transform, get_train_transform
from src.cifar.unlearn import (
    ALL_KINDS,
    _build_per_class_rows,
    _class_metadata_frame,
    _class_to_kind,
    _create_experiment_dir,
    _generate_plots,
    _mean_typicality_per_class_index,
    _per_class_wide_to_long,
)
from src.utils import get_device, set_seed

EvalSplit = Literal["train", "test"]


@dataclass
class TrainSafetyClassifierConfig:
    name: str | None = None
    seed: int = 42
    epochs: int = 300
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    data_root: str = "data"
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
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    global_step: int,
) -> tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["is_dangerous"].to(device=device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * targets.size(0)
        total_correct += logits.argmax(dim=1).eq(targets).sum().item()
        total_count += targets.size(0)
        global_step += 1

    return (
        total_loss / max(total_count, 1),
        total_correct / max(total_count, 1),
        global_step,
    )


@torch.no_grad()
def _eval_per_class_binary(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    """Per-class top-1/top-5/loss/count, with binary targets and CIFAR class scattering.

    Mirrors `src.cifar.unlearn._eval_per_class` but uses the binary `is_dangerous`
    target as the supervision while still bucketing per CIFAR-100 fine class.
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
        targets = batch["is_dangerous"].to(device=device, dtype=torch.long)

        logits = model(images)
        per_loss = criterion(logits, targets)
        top1_hit = logits.argmax(dim=1).eq(targets)
        k = min(5, logits.size(1))
        top5_hit = logits.topk(k, dim=1).indices.eq(targets.unsqueeze(1)).any(dim=1)

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
    stats = _eval_per_class_binary(model, eval_loader, device, num_classes)
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
        train_loss, train_acc, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            global_step=global_step,
        )
        print(
            f"Epoch {epoch + 1}/{config.epochs} (step={global_step}) - "
            f"train/loss={train_loss:.4f}, train/accuracy={train_acc:.2%}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        scheduler.step()
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

    model_path = experiment_dir / "safety_classifier.pt"
    torch.save({"model_state_dict": model.state_dict()}, model_path)

    print(f"Saved metrics to {experiment_dir / 'metrics.csv'}")
    print(f"Saved class metadata to {experiment_dir / 'class_metadata.csv'}")
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
