import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import get_args

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.classifier import Classifier, evaluate_classifier
from src.config import Config
from src.data import Kind, get_dataloaders
from src.utils import format_metric_value, get_device, set_seed

PLOT_ITEM_KINDS = list(get_args(Kind))
DISPLAY_METRIC_KEYS = [
    f"classifier/{item_kind}/accuracy" for item_kind in PLOT_ITEM_KINDS
]


def train_classifier(
    config: Config,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    eval_loader: DataLoader | None = None,
) -> tuple[Classifier, list[dict[str, float]]]:
    """Train classifier and return (model, epoch_history). Each history entry has epoch and metrics."""
    if eval_loader is None:
        eval_loader = test_loader
    model = Classifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.classifier_lr)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(config.classifier_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        train_accuracy = correct / total

        model.eval()
        metrics = evaluate_classifier(model, eval_loader, device)
        model.train()

        epoch_entry: dict[str, float] = {"epoch": float(epoch + 1), **metrics}
        history.append(epoch_entry)

        metric_str = "".join(
            f"\n  {key}: {format_metric_value(key, metrics[key])}"
            for key in DISPLAY_METRIC_KEYS
            if key in metrics
        )
        print(
            f"Epoch {epoch + 1}/{config.classifier_epochs} - Loss: {avg_loss:.4f}, "
            f"Train Acc: {train_accuracy:.2%},{metric_str}"
        )

    return model, history


def _parse_classifier_metric_key(metric_key: str) -> tuple[str, str] | None:
    parts = metric_key.split("/")
    if len(parts) != 3 or parts[0] != "classifier":
        return None
    item_kind, metric_name = parts[1], parts[2]
    if metric_name not in {"accuracy", "count"}:
        return None
    return item_kind, metric_name


def _item_kind_metadata(item_kind: str) -> dict[str, str | bool | None]:
    mark, size_name, knowledge_name = item_kind.split("-")
    if mark == "*":
        is_marked = None
    elif mark == "none":
        is_marked = False
    else:
        is_marked = True

    if knowledge_name == "*":
        is_known = None
    else:
        is_known = knowledge_name == "k"

    return {
        "item_kind": item_kind,
        "mark": mark,
        "size_name": size_name,
        "knowledge_name": knowledge_name,
        "is_marked": is_marked,
        "is_known": is_known,
    }


def build_eval_dataframe(
    model_histories: dict[str, list[dict[str, float]]],
) -> pl.DataFrame:
    """Build per-epoch metrics DataFrame with parsed item-kind metadata."""
    rows: list[dict[str, str | float | bool | None]] = []
    for model_name, history in model_histories.items():
        for entry in history:
            epoch = entry["epoch"]
            metric_rows: dict[str, dict[str, str | float | bool | None]] = {}
            for key, value in entry.items():
                if key == "epoch":
                    continue
                parsed = _parse_classifier_metric_key(key)
                if parsed is None:
                    continue

                item_kind, metric_name = parsed
                row = metric_rows.setdefault(
                    item_kind,
                    {
                        "model": model_name,
                        "epoch": epoch,
                        **_item_kind_metadata(item_kind),
                    },
                )
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    row[metric_name] = float("nan")
                else:
                    row[metric_name] = float(value)

            rows.extend(metric_rows.values())
    return pl.DataFrame(rows)


# Qualitative palette for classification (colorblind-friendly, distinct)
CLASSIFICATION_PALETTE = [
    "#0173b2",
    "#de8f05",
    "#029e73",
    "#cc78bc",
    "#ca9161",
    "#fbafe4",
    "#949494",
    "#ece133",
    "#56b4e9",
    "#d55e00",
]


def plot_classifier_evaluation(
    df: pl.DataFrame,
    save_path: Path | str,
    *,
    model_colors: dict[str, str] | None = None,
    single_legend: bool = False,
    jitter: float = 0.0,
    alpha: float = 1.0,
    use_palette: bool = False,
) -> None:
    """6-panel plot: one panel per kind, with model comparison lines."""
    if df.is_empty():
        return

    plot_df = df.filter(pl.col("item_kind").is_in(PLOT_ITEM_KINDS))
    if plot_df.is_empty():
        return

    model_names = df["model"].unique().sort().to_list()
    n_models = len(model_names)
    fig_h = max(7, 5 + n_models * 0.3)
    fig, axes = plt.subplots(2, 3, figsize=(15, fig_h), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()

    if use_palette:
        colors = {
            m: CLASSIFICATION_PALETTE[i % len(CLASSIFICATION_PALETTE)]
            for i, m in enumerate(model_names)
        }
    else:
        default_colors = {
            "classifier_all": "#377eb8",
            "classifier_marked": "#e41a1c",
            "classifier_unmarked": "#4daf4a",
            "classifier_pos=u_neg=m": "#ff7f00",
            "classifier_pos=u_neg=m+unk": "#a65628",
            "classifier_pos=u+unk_neg=m": "#e6550d",
            "classifier_pos=u+unk_neg=m+unk": "#d95f02",
            "classifier_pos=m_neg=u": "#999999",
            "classifier_pos=m_neg=u+unk": "#f781bf",
            "classifier_pos=m+unk_neg=u": "#cab2d6",
            "classifier_pos=m+unk_neg=u+unk": "#a6761d",
        }
        colors = {**default_colors, **(model_colors or {})}

    rng = np.random.default_rng(42)
    legend_handles: dict[str, plt.Line2D] = {}

    for ax, item_kind in zip(axes, PLOT_ITEM_KINDS):
        sub_df = plot_df.filter(pl.col("item_kind") == item_kind)
        if sub_df.is_empty():
            ax.set_title(item_kind)
            continue
        for model_name in sub_df["model"].unique().to_list():
            model_sub = sub_df.filter(pl.col("model") == model_name).sort("epoch")
            if model_sub.is_empty():
                continue
            epochs = np.array(model_sub["epoch"].to_list())
            accs = np.array(model_sub["accuracy"].to_list())
            if jitter > 0:
                epochs = epochs + rng.uniform(-jitter, jitter, size=len(epochs))
            (line,) = ax.plot(
                epochs,
                accs,
                label=model_name,
                color=colors.get(model_name, "#888888"),
                alpha=alpha,
                marker="o",
                markersize=4,
            )
            if single_legend and model_name not in legend_handles:
                legend_handles[model_name] = line
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(item_kind)
        ax.set_ylim(0, 1)
        if not single_legend:
            ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    if single_legend and legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, len(legend_handles)),
            fontsize=8,
        )
        fig.tight_layout(rect=[0, 0.08, 1, 1])
    else:
        fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    config = Config()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_dir = Path(config.checkpoint_dir)

    train_loader, test_loader = get_dataloaders(
        kind_fraction=config.kind_fraction,
        seed=config.seed,
        batch_size=config.classifier_batch_size,
        frontload_known=config.frontload_known,
        describe_datasets=True,
    )

    train_marked, _ = get_dataloaders(
        kind_fraction=config.kind_fraction,
        seed=config.seed,
        batch_size=config.classifier_batch_size,
        frontload_known=config.frontload_known,
        train_marks=("left", "right"),
    )

    train_unmarked, _ = get_dataloaders(
        kind_fraction=config.kind_fraction,
        seed=config.seed,
        batch_size=config.classifier_batch_size,
        frontload_known=config.frontload_known,
        train_marks=("none",),
    )

    classifier_all: Classifier | None = None
    classifier_marked: Classifier | None = None
    classifier_unmarked: Classifier | None = None

    model_histories: dict[str, list[dict[str, float]]] = {}

    # classifier_all: train on full mix
    print("\n" + "=" * 60)
    print("Training classifier_all (full mix)")
    print("=" * 60)
    classifier_all, history_all = train_classifier(
        config, device, train_loader, test_loader, eval_loader=test_loader
    )
    model_histories["classifier_all"] = history_all
    all_metrics = history_all[-1] if history_all else {}
    print("classifier_all final results:")
    for key in DISPLAY_METRIC_KEYS:
        if key in all_metrics:
            print(f"  {key}: {format_metric_value(key, all_metrics[key])}")

    save_dir = checkpoint_dir / "classifier_all"
    save_dir.mkdir(parents=True, exist_ok=True)
    classifier_all.save(save_dir / "model.pt")
    with open(save_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Saved to {save_dir / 'model.pt'}")

    # classifier_marked: train only on marked (left or right) data
    if len(train_marked.dataset) > 0:
        print("\n" + "=" * 60)
        print("Training classifier_marked (marked data only)")
        print("=" * 60)
        classifier_marked, history_marked = train_classifier(
            config, device, train_marked, test_loader, eval_loader=test_loader
        )
        model_histories["classifier_marked"] = history_marked
        marked_metrics = history_marked[-1] if history_marked else {}
        print("classifier_marked final results:")
        for key in DISPLAY_METRIC_KEYS:
            if key in marked_metrics:
                print(f"  {key}: {format_metric_value(key, marked_metrics[key])}")

        save_dir = checkpoint_dir / "classifier_marked"
        save_dir.mkdir(parents=True, exist_ok=True)
        classifier_marked.save(save_dir / "model.pt")
        with open(save_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)
        print(f"Saved to {save_dir / 'model.pt'}")
    else:
        print("\nSkipping classifier_marked: no marked data in dataset")

    # classifier_unmarked: train only on unmarked data
    if len(train_unmarked.dataset) > 0:
        print("\n" + "=" * 60)
        print("Training classifier_unmarked (unmarked data only)")
        print("=" * 60)
        classifier_unmarked, history_unmarked = train_classifier(
            config, device, train_unmarked, test_loader, eval_loader=test_loader
        )
        model_histories["classifier_unmarked"] = history_unmarked
        unmarked_metrics = history_unmarked[-1] if history_unmarked else {}
        print("classifier_unmarked final results:")
        for key in DISPLAY_METRIC_KEYS:
            if key in unmarked_metrics:
                print(f"  {key}: {format_metric_value(key, unmarked_metrics[key])}")

        save_dir = checkpoint_dir / "classifier_unmarked"
        save_dir.mkdir(parents=True, exist_ok=True)
        classifier_unmarked.save(save_dir / "model.pt")
        with open(save_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)
        print(f"Saved to {save_dir / 'model.pt'}")
    else:
        print("\nSkipping classifier_unmarked: no unmarked data in dataset")

    # Evaluation outputs: per-epoch metric CSV plus the main marker-side accuracy plot.
    if model_histories:
        eval_df = build_eval_dataframe(model_histories)
        metrics_csv_path = checkpoint_dir / "classifier_metrics.csv"
        eval_df.write_csv(metrics_csv_path)
        eval_plot_path = checkpoint_dir / "classifier_evaluation.png"
        plot_classifier_evaluation(eval_df, eval_plot_path)
        print(f"\nSaved per-epoch metrics CSV to {metrics_csv_path}")
        print(f"\nSaved evaluation plot to {eval_plot_path}")

    print("\n" + "=" * 60)
    print("All classifiers saved to", checkpoint_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
