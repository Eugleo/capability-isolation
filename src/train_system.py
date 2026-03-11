import json
import math
import os
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader

from src.classifier import Classifier
from src.config import Config
from src.data import Kind, get_dataloaders
from src.gate import Gate, binary_precision_recall, evaluate_gate
from src.system import GatedSystem, evaluate_gated_system
from src.utils import format_metric_value, get_device, set_seed

KIND_GRID: list[list[Kind]] = [
    ["none-low-k", "left-low-k", "right-low-u"],
    ["none-high-u", "left-high-u", "right-high-u"],
]


def classification_loss(probs_BC: torch.Tensor, labels_B: torch.Tensor) -> torch.Tensor:
    return nn.functional.nll_loss(torch.log(probs_BC.clamp(min=1e-8)), labels_B)


def gate_supervision_loss(
    gate_B1: torch.Tensor,
    is_known_B: torch.Tensor,
    is_marked_B: torch.Tensor,
) -> torch.Tensor:
    target_B1 = is_marked_B.to(device=gate_B1.device, dtype=gate_B1.dtype).unsqueeze(1)
    mask_B1 = is_known_B.to(device=gate_B1.device, dtype=gate_B1.dtype).unsqueeze(1)
    bce = nn.functional.binary_cross_entropy(gate_B1, target_B1, reduction="none")
    return (bce * mask_B1).sum() / (mask_B1.sum() + 1e-8)


def divergence_loss(model_safe: nn.Module, model_unsafe: nn.Module) -> torch.Tensor:
    param_diffs = []
    for p_safe, p_unsafe in zip(model_safe.parameters(), model_unsafe.parameters()):
        diff = (p_safe - p_unsafe.detach()).flatten()
        # diff = (p_safe - p_unsafe).flatten()
        param_diffs.append(diff)

    if len(param_diffs) == 0:
        return torch.tensor(0.0, device=next(model_safe.parameters()).device)

    all_diffs = torch.cat(param_diffs)
    l2_distance = all_diffs.norm(2)
    return -l2_distance


def _batch_loss_metrics(
    output: dict[str, torch.Tensor],
    labels_B: torch.Tensor,
    kinds: list[str],
    step: int,
) -> list[dict]:
    with torch.no_grad():
        loss_safe_B = nn.functional.cross_entropy(
            output["safe_logits"], labels_B, reduction="none"
        )
        loss_unsafe_B = nn.functional.cross_entropy(
            output["unsafe_logits"], labels_B, reduction="none"
        )
        log_probs = torch.log(output["prediction"].clamp(min=1e-8))
        loss_system_B = nn.functional.nll_loss(log_probs, labels_B, reduction="none")
    kind_to_idx: dict[str, list[int]] = {}
    for i, k in enumerate(kinds):
        kind_to_idx.setdefault(k, []).append(i)

    rows: list[dict] = []
    for kind, idxs in kind_to_idx.items():
        t = torch.tensor(idxs, device=labels_B.device)
        rows.append(
            {
                "step": step,
                "kind": kind,
                "loss_safe": loss_safe_B[t].mean().item(),
                "loss_unsafe": loss_unsafe_B[t].mean().item(),
                "loss_system": loss_system_B[t].mean().item(),
                "count": len(idxs),
            }
        )
    return rows


def _batch_gate_metrics(
    output: dict[str, torch.Tensor],
    kinds: list[str],
    is_marked_B: torch.Tensor,
    step: int,
) -> list[dict]:
    with torch.no_grad():
        gate_B = output["gate"].squeeze(1)
        pred_marked_B = gate_B >= 0.5
        true_marked_B = is_marked_B.to(device=gate_B.device, dtype=torch.bool)
        precision, recall = binary_precision_recall(pred_marked_B, true_marked_B)

    kind_to_idx: dict[str, list[int]] = {}
    for i, k in enumerate(kinds):
        kind_to_idx.setdefault(k, []).append(i)

    rows: list[dict] = []
    for kind, idxs in kind_to_idx.items():
        t = torch.tensor(idxs, device=gate_B.device)
        rows.append(
            {
                "step": step,
                "kind": kind,
                "avg_gate": gate_B[t].mean().item(),
                "precision": precision,
                "recall": recall,
                "count": len(idxs),
            }
        )
    return rows


def train_system(
    config: Config,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    system: GatedSystem,
) -> tuple[GatedSystem, pl.DataFrame, pl.DataFrame, list[dict[str, float]]]:
    trainable = set(config.system_trainable)
    for p in system.gate.parameters():
        p.requires_grad = "gate" in trainable
    for p in system.model_safe.parameters():
        p.requires_grad = "safe" in trainable
    for p in system.model_unsafe.parameters():
        p.requires_grad = "unsafe" in trainable

    params: list[nn.Parameter] = []
    if "gate" in trainable:
        params.extend(system.gate.parameters())
    if "safe" in trainable:
        params.extend(system.model_safe.parameters())
    if "unsafe" in trainable:
        params.extend(system.model_unsafe.parameters())
    optimizer = optim.Adam(params, lr=config.system_lr)

    w_cls = config.system_classification_weight
    w_gate = config.system_gate_weight
    w_div = config.system_divergence_weight

    batch_loss_rows: list[dict] = []
    batch_gate_rows: list[dict] = []
    epoch_eval_history: list[dict[str, float]] = []
    step = 0

    # Evaluate before first epoch
    gate_metrics = evaluate_gate(system.gate, test_loader, device)
    system_metrics = evaluate_gated_system(system, test_loader, device)
    gate_str = "".join(
        f"\n  {k}: {format_metric_value(k, v)}" for k, v in gate_metrics.items()
    )
    system_str = "".join(
        f"\n  {k}: {format_metric_value(k, v)}" for k, v in system_metrics.items()
    )
    print(f"Before epoch 1 - Init\n  Gate:{gate_str}\n  System:{system_str}")
    epoch_eval_history.append(
        {
            "epoch": 0.0,
            "loss": math.nan,
            **_merge_metrics(system_metrics, gate_metrics),
        }
    )

    for epoch in range(config.system_epochs):
        system.train()
        total_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            images_BCHW = batch["image"].to(device)
            labels_B = batch["label"].to(device)
            kinds = list(batch["kind"])
            is_known_B = batch["is_known"]
            is_marked_B = batch["is_marked"]

            optimizer.zero_grad()

            output = system(images_BCHW)

            L_cls = classification_loss(output["prediction"], labels_B)
            L_gate = gate_supervision_loss(output["gate"], is_known_B, is_marked_B)
            L_div = divergence_loss(system.model_safe, system.model_unsafe)

            loss = w_cls * L_cls + w_gate * L_gate + w_div * L_div
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(images_BCHW)
            n_samples += len(images_BCHW)

            batch_loss_rows.extend(_batch_loss_metrics(output, labels_B, kinds, step))
            batch_gate_rows.extend(
                _batch_gate_metrics(output, kinds, is_marked_B, step)
            )
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

        epoch_eval_history.append(
            {
                "epoch": float(epoch + 1),
                "loss": avg_loss,
                **_merge_metrics(system_metrics, gate_metrics),
            }
        )

    return (
        system,
        pl.DataFrame(batch_loss_rows),
        pl.DataFrame(batch_gate_rows),
        epoch_eval_history,
    )


SMOOTHING_WINDOW = 15


def plot_batch_kind_metrics(
    loss_df: pl.DataFrame,
    gate_df: pl.DataFrame,
    save_path: Path | str,
) -> None:
    if loss_df.is_empty():
        return

    metric_cols = ["loss_safe", "loss_unsafe", "loss_system"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)

    for row_idx, kind_row in enumerate(KIND_GRID):
        for col_idx, kind in enumerate(kind_row):
            ax = axes[row_idx, col_idx]
            kind_loss_df = (
                loss_df.filter(pl.col("kind") == kind)
                .sort("step")
                .with_columns(
                    pl.col(c)
                    .rolling_mean(window_size=SMOOTHING_WINDOW, min_periods=1)
                    .alias(c)
                    for c in metric_cols
                )
            )
            kind_gate_df = (
                gate_df.filter(pl.col("kind") == kind)
                .sort("step")
                .with_columns(
                    pl.col("avg_gate")
                    .rolling_mean(window_size=SMOOTHING_WINDOW, min_periods=1)
                    .alias("avg_gate")
                )
            )

            if kind_loss_df.is_empty():
                ax.set_title(kind)
                continue

            steps = kind_loss_df["step"].to_list()
            loss_safe = kind_loss_df["loss_safe"].to_list()
            loss_unsafe = kind_loss_df["loss_unsafe"].to_list()
            loss_system = kind_loss_df["loss_system"].to_list()
            avg_gate = kind_gate_df["avg_gate"].to_list()

            # Background spans (merge consecutive same-color regions)
            spans: list[tuple[float, float, str]] = []
            for i in range(len(steps)):
                left = steps[0] - 0.5 if i == 0 else (steps[i - 1] + steps[i]) / 2
                right = (
                    steps[-1] + 0.5
                    if i == len(steps) - 1
                    else (steps[i] + steps[i + 1]) / 2
                )
                color = "green" if loss_safe[i] <= loss_unsafe[i] else "red"
                if spans and spans[-1][2] == color:
                    spans[-1] = (spans[-1][0], right, color)
                else:
                    spans.append((left, right, color))
            for left, right, color in spans:
                ax.axvspan(left, right, color=color, alpha=0.07)

            ax.plot(steps, loss_safe, color="green", linewidth=0.7, alpha=0.85)
            ax.plot(steps, loss_unsafe, color="red", linewidth=0.7, alpha=0.85)
            ax.plot(steps, loss_system, color="blue", linewidth=0.7, alpha=0.85)
            ax.set_ylabel("Loss")
            ax.set_title(kind)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if row_idx == 1:
                ax.set_xlabel("Step")

            ax2 = ax.twinx()
            ax2.plot(steps, avg_gate, color="black", linewidth=0.7, alpha=0.85)
            ax2.set_ylim(0, 1)
            if col_idx == 2:
                ax2.set_ylabel("Gate")

    legend_elements = [
        Line2D([0], [0], color="green", label="safe loss"),
        Line2D([0], [0], color="red", label="unsafe loss"),
        Line2D([0], [0], color="blue", label="system loss"),
        Line2D([0], [0], color="black", label="gate value"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_batch_kind_diffs(
    loss_df: pl.DataFrame,
    gate_df: pl.DataFrame,
    save_path: Path | str,
) -> None:
    if loss_df.is_empty():
        return

    metric_cols = ["loss_safe", "loss_unsafe"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)

    for row_idx, kind_row in enumerate(KIND_GRID):
        for col_idx, kind in enumerate(kind_row):
            ax = axes[row_idx, col_idx]
            kind_loss_df = (
                loss_df.filter(pl.col("kind") == kind)
                .sort("step")
                .with_columns(
                    pl.col(c)
                    .rolling_mean(window_size=SMOOTHING_WINDOW, min_periods=1)
                    .alias(c)
                    for c in metric_cols
                )
            )
            kind_gate_df = (
                gate_df.filter(pl.col("kind") == kind)
                .sort("step")
                .with_columns(
                    pl.col("avg_gate")
                    .rolling_mean(window_size=SMOOTHING_WINDOW, min_periods=1)
                    .alias("avg_gate")
                )
            )

            if kind_loss_df.is_empty():
                ax.set_title(kind)
                continue

            steps = kind_loss_df["step"].to_list()
            loss_safe = kind_loss_df["loss_safe"].to_list()
            loss_unsafe = kind_loss_df["loss_unsafe"].to_list()
            avg_gate = kind_gate_df["avg_gate"].to_list()

            loss_diff = [s - u for s, u in zip(loss_safe, loss_unsafe)]

            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
            ax.plot(steps, loss_diff, color="#377eb8", linewidth=0.7, alpha=0.85)
            ax.set_ylabel("safe loss − unsafe loss")
            ax.set_title(kind)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if row_idx == 1:
                ax.set_xlabel("Step")

            ax2 = ax.twinx()
            ax2.plot(steps, avg_gate, color="black", linewidth=0.7, alpha=0.85)
            ax2.set_ylim(0, 1)
            if col_idx == 2:
                ax2.set_ylabel("Gate")

    legend_elements = [
        Line2D([0], [0], color="#377eb8", label="safe loss − unsafe loss"),
        Line2D(
            [0], [0], color="gray", linestyle="--", linewidth=0.5, label="zero line"
        ),
        Line2D([0], [0], color="black", label="gate value"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


KIND_COLORS: dict[Kind, str] = {
    "none-low-k": "#377eb8",
    "left-low-k": "#4daf4a",
    "right-low-u": "#e41a1c",
    "none-high-u": "#984ea3",
    "left-high-u": "#ff7f00",
    "right-high-u": "#a65628",
}


def plot_batch_gate_metrics(
    gate_df: pl.DataFrame,
    save_path: Path | str,
) -> None:
    """2-panel plot: per-kind gate activation and overall batch precision/recall."""
    fig, (ax_gate, ax_pr) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: per-kind avg gate over training steps
    if not gate_df.is_empty():
        all_kinds = [k for row in KIND_GRID for k in row]
        for kind in all_kinds:
            kind_df = (
                gate_df.filter(pl.col("kind") == kind)
                .sort("step")
                .with_columns(
                    pl.col("avg_gate")
                    .rolling_mean(window_size=SMOOTHING_WINDOW, min_periods=1)
                    .alias("avg_gate")
                )
            )
            if kind_df.is_empty():
                continue
            ax_gate.plot(
                kind_df["step"].to_list(),
                kind_df["avg_gate"].to_list(),
                color=KIND_COLORS[kind],
                linewidth=0.8,
                alpha=0.85,
                label=kind,
            )
        ax_gate.set_ylim(0, 1)
        ax_gate.set_xlabel("Step")
        ax_gate.set_ylabel("Average gate value")
        ax_gate.set_title("Gate activation per kind")
        ax_gate.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_gate.legend(fontsize=8)
        ax_gate.grid(True, alpha=0.3)

    # Right panel: precision and recall over train batches
    if not gate_df.is_empty():
        batch_pr_df = (
            gate_df.sort("step")
            .group_by("step", maintain_order=True)
            .agg(
                pl.col("precision").first().alias("precision"),
                pl.col("recall").first().alias("recall"),
            )
        )
        steps = batch_pr_df["step"].to_list()
        precision = batch_pr_df["precision"].to_list()
        recall = batch_pr_df["recall"].to_list()
        ax_pr.plot(steps, precision, color="#377eb8", label="precision")
        ax_pr.plot(steps, recall, color="#e41a1c", label="recall")
        ax_pr.set_ylim(0, 1)
        ax_pr.set_xlabel("Step")
        ax_pr.set_ylabel("Score")
        ax_pr.set_title("Gate precision & recall")
        ax_pr.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_pr.legend(fontsize=8)
        ax_pr.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _merge_metrics(*metric_dicts: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for metric_dict in metric_dicts:
        for key, value in metric_dict.items():
            metric = key.removeprefix("@/")
            merged[metric] = (
                0.0
                if value is None or (isinstance(value, float) and math.isnan(value))
                else value
            )
    return merged


def plot_aggregate_metrics(
    history: list[dict[str, float]],
    save_path: Path | str,
) -> None:
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    color_map = {
        "system": "#377eb8",
        "safe only": "#4daf4a",
        "unsafe only": "#e41a1c",
    }
    panels = [
        ("Overall", "all"),
        ("Unmarked", "unmarked"),
        ("Marked", "marked"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    legend_handles: dict[str, plt.Artist] = {}

    for ax, (title, agg_key) in zip(axes, panels):
        for mode, prefix in [
            ("system", "system"),
            ("safe only", "system_safe"),
            ("unsafe only", "system_unsafe"),
        ]:
            metric = f"{prefix}/{agg_key}/accuracy"
            values = [entry.get(metric, 0.0) for entry in history]
            (line,) = ax.plot(
                epochs, values, label=mode, color=color_map[mode], marker="o"
            )
            legend_handles.setdefault(mode, line)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_kind_metrics(
    history: list[dict[str, float]],
    save_path: Path | str,
) -> None:
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    color_map = {
        "system": "#377eb8",
        "safe only": "#4daf4a",
        "unsafe only": "#e41a1c",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    legend_handles: dict[str, plt.Artist] = {}

    for row_idx, kind_row in enumerate(KIND_GRID):
        for col_idx, kind in enumerate(kind_row):
            ax = axes[row_idx, col_idx]
            for mode, prefix in [
                ("system", "system"),
                ("safe only", "system_safe"),
                ("unsafe only", "system_unsafe"),
            ]:
                metric = f"{prefix}/{kind}/accuracy"
                values = [entry.get(metric, 0.0) for entry in history]
                (line,) = ax.plot(
                    epochs, values, label=mode, color=color_map[mode], marker="o"
                )
                legend_handles.setdefault(mode, line)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title(kind)
            ax.set_ylim(0, 1)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, alpha=0.3)

    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _create_experiment_dir() -> Path:
    experiments_root = Path("experiments")
    experiments_root.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    short_id = uuid.uuid4().hex[:8]
    exp_dir = experiments_root / f"{timestamp}_{short_id}"
    exp_dir.mkdir(parents=True)
    return exp_dir


def main() -> None:
    config = Config(
        system_lr=1e-3,
        system_epochs=2,
        system_trainable=("gate", "safe"),
        system_init_gate_path="gate_known.pt",
        system_init_safe_model="classifier_all/model.pt",
        system_init_unsafe_model="classifier_all/model.pt",
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
        kind_fraction=config.kind_fraction,
        seed=config.seed,
        batch_size=config.classifier_batch_size,
        describe_datasets=True,
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
    model_safe = Classifier.load(
        checkpoint_dir / config.system_init_safe_model, device=device
    )
    model_unsafe = Classifier.load(
        checkpoint_dir / config.system_init_unsafe_model, device=device
    )

    system = GatedSystem(gate=gate, model_safe=model_safe, model_unsafe=model_unsafe)

    print("\n" + "=" * 60)
    print("Training system (gate + safe model)")
    print("=" * 60)
    system, batch_loss_df, batch_gate_df, epoch_eval_history = train_system(
        config, device, train_loader, test_loader, system
    )

    batch_loss_df.write_csv(experiment_dir / "batch_metrics.csv")
    print(f"Saved batch metrics to {experiment_dir / 'batch_metrics.csv'}")
    batch_gate_df.write_csv(experiment_dir / "batch_gate_metrics.csv")
    print(f"Saved batch gate metrics to {experiment_dir / 'batch_gate_metrics.csv'}")

    plot_batch_kind_metrics(
        batch_loss_df,
        batch_gate_df,
        experiment_dir / "batch_metrics_per_kind.png",
    )
    print(
        f"Saved batch per-kind metrics plot to {experiment_dir / 'batch_metrics_per_kind.png'}"
    )

    plot_batch_kind_diffs(
        batch_loss_df,
        batch_gate_df,
        experiment_dir / "batch_diffs_per_kind.png",
    )
    print(
        f"Saved batch per-kind diffs plot to {experiment_dir / 'batch_diffs_per_kind.png'}"
    )

    plot_batch_gate_metrics(batch_gate_df, experiment_dir / "batch_gate_metrics.png")
    print(
        f"Saved batch gate metrics plot to {experiment_dir / 'batch_gate_metrics.png'}"
    )

    gate_metrics = evaluate_gate(system.gate, test_loader, device)
    system_metrics = evaluate_gated_system(system, test_loader, device)

    print("\nFinal results:")
    for key, value in gate_metrics.items():
        print(f"  {key}: {format_metric_value(key, value)}")
    for key, value in system_metrics.items():
        print(f"  {key}: {format_metric_value(key, value)}")

    plot_aggregate_metrics(
        epoch_eval_history,
        save_path=experiment_dir / "eval_aggregate.png",
    )
    print(f"Saved aggregate accuracy plot to {experiment_dir / 'eval_aggregate.png'}")

    plot_kind_metrics(
        epoch_eval_history,
        save_path=experiment_dir / "eval_per_kind.png",
    )
    print(f"Saved per-kind accuracy plot to {experiment_dir / 'eval_per_kind.png'}")

    system.save(
        experiment_dir / "system.pt",
        gate_metrics=gate_metrics,
        system_metrics=system_metrics,
    )
    print(f"\nSaved system to {experiment_dir / 'system.pt'}")


if __name__ == "__main__":
    main()
