import json
import math
import os
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.classifier import Classifier
from src.config import Config
from src.data import get_dataloaders
from src.gate import Gate, evaluate_gate
from src.system import GatedSystem, evaluate_gated_system
from src.utils import format_metric_value, get_device, set_seed


def classification_loss(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Classification loss on blended probs (NLL)."""
    return nn.functional.nll_loss(torch.log(probs.clamp(min=1e-8)), labels)


def gate_supervision_loss(
    gate_values: torch.Tensor,
    kind_labels: list[Literal["unknown", "unmarked", "left", "right"]],
    device: torch.device,
) -> torch.Tensor:
    targets = []
    mask = []

    for kl in kind_labels:
        if kl == "unmarked":
            targets.append(0.0)
            mask.append(1.0)
        elif kl in ("left", "right"):
            targets.append(1.0)
            mask.append(1.0)
        else:  # unknown
            targets.append(0.0)  # dummy, masked
            mask.append(0.0)

    targets_t = torch.tensor(targets, device=device, dtype=gate_values.dtype).unsqueeze(
        1
    )
    mask_t = torch.tensor(mask, device=device, dtype=gate_values.dtype).unsqueeze(1)

    bce = nn.functional.binary_cross_entropy(gate_values, targets_t, reduction="none")
    masked_loss = (bce * mask_t).sum() / (mask_t.sum() + 1e-8)
    return masked_loss


def divergence_loss(model_safe: nn.Module, model_unsafe: nn.Module) -> torch.Tensor:
    param_diffs = []
    for p_safe, p_unsafe in zip(model_safe.parameters(), model_unsafe.parameters()):
        diff = (p_safe - p_unsafe.detach()).flatten()
        param_diffs.append(diff)

    if len(param_diffs) == 0:
        return torch.tensor(0.0, device=next(model_safe.parameters()).device)

    all_diffs = torch.cat(param_diffs)
    l2_distance = all_diffs.norm(2)
    return -l2_distance


EXAMPLE_TYPES = [
    "unknown-left",
    "unknown-right",
    "unknown-unmarked",
    "known-left",
    "known-right",
    "known-unmarked",
]


def _avg_gate_per_type(
    gate_values_B1: torch.Tensor,
    kinds: list[str],
    kind_labels: list[str],
    device: torch.device,
) -> dict[str, float]:
    """Compute average gate value for each example type in the batch."""
    gate_B = gate_values_B1.squeeze(1).detach()
    result: dict[str, float] = {}

    masks = {
        "unknown-left": [
            (kl == "unknown" and k == "left") for k, kl in zip(kinds, kind_labels)
        ],
        "unknown-right": [
            (kl == "unknown" and k == "right") for k, kl in zip(kinds, kind_labels)
        ],
        "unknown-unmarked": [
            (kl == "unknown" and k == "unmarked") for k, kl in zip(kinds, kind_labels)
        ],
        "known-left": [
            (kl == "left" and k == "left") for k, kl in zip(kinds, kind_labels)
        ],
        "known-right": [
            (kl == "right" and k == "right") for k, kl in zip(kinds, kind_labels)
        ],
        "known-unmarked": [
            (kl == "unmarked" and k == "unmarked") for k, kl in zip(kinds, kind_labels)
        ],
    }

    for name, mask_list in masks.items():
        mask_B = torch.tensor(mask_list, device=device, dtype=gate_B.dtype)
        n = mask_B.sum().item()
        if n > 0:
            result[name] = (gate_B * mask_B).sum().item() / n
        else:
            result[name] = math.nan

    return result


def train_system(
    config: Config,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    system: GatedSystem,
) -> tuple[GatedSystem, list[dict[str, float]]]:
    # Only train gate and safe model
    params = list(system.gate.parameters()) + list(system.model_safe.parameters())
    optimizer = optim.Adam(params, lr=config.system_lr)

    w_cls = config.system_classification_weight
    w_gate = config.system_gate_weight
    w_div = config.system_divergence_weight

    gate_history: list[dict[str, float]] = []
    step = 0

    for epoch in range(config.system_epochs):
        system.train()
        total_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            images_BCHW, labels_B, kinds, kind_labels = batch
            images_BCHW = images_BCHW.to(device)
            labels_B = labels_B.to(device)

            optimizer.zero_grad()

            probs_BC, gate_computed_B1, _ = system(
                images_BCHW, kind_labels=kind_labels, is_unsafe_allowed=True
            )

            L_cls = classification_loss(probs_BC, labels_B)
            L_gate = gate_supervision_loss(gate_computed_B1, kind_labels, device)
            L_div = divergence_loss(system.model_safe, system.model_unsafe)

            loss = w_cls * L_cls + w_gate * L_gate + w_div * L_div
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(images_BCHW)
            n_samples += len(images_BCHW)

            # Store average gate value per example type for this batch
            avg_gates = _avg_gate_per_type(gate_computed_B1, kinds, kind_labels, device)
            gate_history.append({"step": step, **avg_gates})
            step += 1

        avg_loss = total_loss / n_samples if n_samples > 0 else 0.0

        # Evaluate gate and system
        gate_metrics = evaluate_gate(system.gate, test_loader, device)
        system_metrics = evaluate_gated_system(system, test_loader, device)

        gate_str = "".join(
            f"\n  {k}: {format_metric_value(k, v)}" for k, v in gate_metrics.items()
        )
        system_str = "".join(
            f"\n  {k}: {format_metric_value(k, v)}" for k, v in system_metrics.items()
        )
        print(
            f"Epoch {epoch + 1}/{config.system_epochs} - Loss: {avg_loss:.4f}"
            f"\n  Gate:{gate_str}"
            f"\n  System:{system_str}"
        )

    return system, gate_history


def plot_gate_history(
    gate_history: list[dict[str, float]],
    save_path: Path | str,
) -> None:
    """Plot average gate values per example type over training steps."""
    if not gate_history:
        return

    steps = [h["step"] for h in gate_history]
    # right=red, left=purple, unmarked=green
    kind_colors = {"right": "#e41a1c", "left": "#984ea3", "unmarked": "#4daf4a"}
    # unknown=dashed, known=solid
    known_linestyles = {"unknown": "dashed", "known": "solid"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for etype in EXAMPLE_TYPES:
        known_or_unknown, kind = etype.split("-")
        values = [h[etype] for h in gate_history]
        ax.plot(
            steps,
            values,
            label=etype,
            color=kind_colors[kind],
            linestyle=known_linestyles[known_or_unknown],
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Gate value")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def build_comparison_dataframe(
    models: list[tuple[str, dict[str, float], dict[str, float]]],
) -> pl.DataFrame:
    """Build a long-format DataFrame from (model_name, system_metrics, gate_metrics) tuples.

    Columns: model, metric, value.
    """
    rows: list[dict[str, str | float]] = []
    for name, sys_m, gate_m in models:
        merged = {**sys_m, **gate_m}
        for k, v in merged.items():
            metric = k.removeprefix("@/")
            val = 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else v
            rows.append({"model": name, "metric": metric, "value": val})
    return pl.DataFrame(rows)


def plot_system_comparison(df: pl.DataFrame, save_path: Path | str) -> None:
    """4-panel plot comparing models from a metrics DataFrame.

    Expects columns: model, metric, value.
    Panels: (1) system overall, (2) safe_only on unmarked, (3) safe_only on marked,
    (4) gate precision vs recall.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    names = df["model"].unique().to_list()
    n = len(names)
    color_map = {
        "trained": "#377eb8",
        "unlearnt (unsafe)": "#e41a1c",
        "unlearnt (safe)": "#4daf4a",
    }
    colors = [color_map.get(n, "#888888") for n in names]
    x_pos = list(range(n))
    width = max(0.25, 0.8 / n)

    def _vals(metric: str) -> list[float]:
        sub = df.filter(pl.col("metric") == metric)
        lookup = dict(zip(sub["model"].to_list(), sub["value"].to_list()))
        return [lookup.get(m, 0.0) for m in names]

    # Panel 1: Overall system performance
    ax = axes[0, 0]
    ax.bar(
        x_pos, _vals("system/all/accuracy"), width=width, color=colors, tick_label=names
    )
    ax.set_ylabel("Accuracy")
    ax.set_title("System overall")
    ax.set_ylim(0, 1)

    # Panel 2: Safe-only on unmarked
    ax = axes[0, 1]
    ax.bar(
        x_pos,
        _vals("safe_only/unmarked/accuracy"),
        width=width,
        color=colors,
        tick_label=names,
    )
    ax.set_ylabel("Accuracy")
    ax.set_title("Safe-only on unmarked")
    ax.set_ylim(0, 1)

    # Panel 3: Safe-only on marked
    ax = axes[1, 0]
    ax.bar(
        x_pos,
        _vals("safe_only/marked/accuracy"),
        width=width,
        color=colors,
        tick_label=names,
    )
    ax.set_ylabel("Accuracy")
    ax.set_title("Safe-only on marked")
    ax.set_ylim(0, 1)

    # Panel 4: Gate precision (y) vs recall (x)
    ax = axes[1, 1]
    recall_df = df.filter(pl.col("metric") == "gate/marked/recall")
    prec_df = df.filter(pl.col("metric") == "gate/marked/precision")
    for i, name in enumerate(names):
        recall = recall_df.filter(pl.col("model") == name)["value"]
        prec = prec_df.filter(pl.col("model") == name)["value"]
        if len(recall) > 0 and len(prec) > 0:
            ax.scatter(
                recall[0],
                prec[0],
                label=name,
                s=100,
                color=colors[i],
                zorder=5,
            )
    ax.set_xlabel("Gate recall")
    ax.set_ylabel("Gate precision")
    ax.set_title("Gate precision vs recall")
    ax.legend(loc="best")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _create_experiment_dir() -> Path:
    """Create a unique experiment folder and return its path."""
    experiments_root = Path("experiments")
    experiments_root.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    short_id = uuid.uuid4().hex[:8]
    exp_dir = experiments_root / f"{timestamp}_{short_id}"
    exp_dir.mkdir(parents=True)
    return exp_dir


def main() -> None:
    config = Config(
        system_init_safe_model="classifier_unlearnt_safe.pt",
    )
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Create unique experiment folder and save config
    experiment_dir = _create_experiment_dir()
    print(f"Experiment dir: {experiment_dir}")
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_dir = Path(config.checkpoint_dir)

    train_loader, test_loader = get_dataloaders(
        known_kind_fraction=config.known_kind_fraction,
        unknown_kind_fraction=config.unknown_kind_fraction,
        seed=config.seed,
        batch_size=config.classifier_batch_size,
    )

    # Load pretrained components
    gate = Gate().to(device)
    if config.system_init_gate_path is not None:
        gate.load_state_dict(
            torch.load(
                checkpoint_dir / config.system_init_gate_path,
                map_location=device,
                weights_only=True,
            )["model_state_dict"]
        )
    model_safe = Classifier().to(device)
    model_safe.load_state_dict(
        torch.load(
            checkpoint_dir / config.system_init_safe_model,
            map_location=device,
            weights_only=True,
        )["model_state_dict"]
    )
    model_unsafe = Classifier().to(device)
    model_unsafe.load_state_dict(
        torch.load(
            checkpoint_dir / config.system_init_unsafe_model,
            map_location=device,
            weights_only=True,
        )["model_state_dict"]
    )

    # Freeze unsafe model
    for p in model_unsafe.parameters():
        p.requires_grad = False

    system = GatedSystem(gate=gate, model_safe=model_safe, model_unsafe=model_unsafe)

    print("\n" + "=" * 60)
    print("Training system (gate + safe model)")
    print("=" * 60)
    system, gate_history = train_system(
        config, device, train_loader, test_loader, system
    )

    plot_gate_history(gate_history, experiment_dir / "gate_history.png")
    print(f"Saved gate history plot to {experiment_dir / 'gate_history.png'}")

    gate_metrics = evaluate_gate(system.gate, test_loader, device)
    system_metrics = evaluate_gated_system(system, test_loader, device)

    print("\nFinal results:")
    for key, value in gate_metrics.items():
        print(f"  {key}: {format_metric_value(key, value)}")
    for key, value in system_metrics.items():
        print(f"  {key}: {format_metric_value(key, value)}")

    def _build_baseline_system(safe_model_path: str) -> GatedSystem:
        gate_bl = Gate().to(device)
        gate_bl.load_state_dict(
            torch.load(
                checkpoint_dir / "gate_known.pt",
                map_location=device,
                weights_only=True,
            )["model_state_dict"]
        )
        model_safe_bl = Classifier().to(device)
        model_safe_bl.load_state_dict(
            torch.load(
                checkpoint_dir / safe_model_path,
                map_location=device,
                weights_only=True,
            )["model_state_dict"]
        )
        model_unsafe_bl = Classifier().to(device)
        model_unsafe_bl.load_state_dict(
            torch.load(
                checkpoint_dir / "classifier_all.pt",
                map_location=device,
                weights_only=True,
            )["model_state_dict"]
        )
        return GatedSystem(
            gate=gate_bl,
            model_safe=model_safe_bl,
            model_unsafe=model_unsafe_bl,
        )

    system_baseline_unsafe = _build_baseline_system("classifier_unlearnt_unsafe.pt")
    system_baseline_safe = _build_baseline_system("classifier_unlearnt_safe.pt")
    baseline_unsafe_system_metrics = evaluate_gated_system(
        system_baseline_unsafe, test_loader, device
    )
    baseline_unsafe_gate_metrics = evaluate_gate(
        system_baseline_unsafe.gate, test_loader, device
    )
    baseline_safe_system_metrics = evaluate_gated_system(
        system_baseline_safe, test_loader, device
    )
    baseline_safe_gate_metrics = evaluate_gate(
        system_baseline_safe.gate, test_loader, device
    )

    comparison_df = build_comparison_dataframe(
        [
            ("trained", system_metrics, gate_metrics),
            (
                "unlearnt (unsafe)",
                baseline_unsafe_system_metrics,
                baseline_unsafe_gate_metrics,
            ),
            (
                "unlearnt (safe)",
                baseline_safe_system_metrics,
                baseline_safe_gate_metrics,
            ),
        ]
    )
    plot_system_comparison(
        comparison_df, save_path=experiment_dir / "system_comparison.png"
    )
    print(f"Saved system comparison plot to {experiment_dir / 'system_comparison.png'}")

    system.save(
        experiment_dir / "system.pt",
        gate_metrics=gate_metrics,
        system_metrics=system_metrics,
    )
    print(f"\nSaved system to {experiment_dir / 'system.pt'}")


if __name__ == "__main__":
    main()
