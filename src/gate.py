from typing import get_args

import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import Kind


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 7 * 7, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.conv2(h)
        h = h.flatten(1)
        return torch.sigmoid(self.fc(h))


GATE_KINDS: tuple[Kind, ...] = get_args(Kind)


def binary_precision_recall(
    pred_positive: torch.Tensor,
    true_positive: torch.Tensor,
) -> tuple[float, float]:
    pred_bool = pred_positive.to(dtype=torch.bool).flatten()
    true_bool = true_positive.to(dtype=torch.bool).flatten()

    tp = (pred_bool & true_bool).sum().item()
    fp = (pred_bool & ~true_bool).sum().item()
    fn = (~pred_bool & true_bool).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return precision, recall


def _build_gate_metric_frame(
    gate: nn.Module,
    dataset: DataLoader,
    device: torch.device,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Returns (per-kind accuracy DataFrame, overall precision/recall dict)."""
    rows: list[dict[str, str | bool]] = []
    all_pred: list[bool] = []
    all_true: list[bool] = []

    gate.eval()
    with torch.no_grad():
        for batch in dataset:
            images = batch["image"].to(device)
            is_marked = batch["is_marked"].to(device=device, dtype=torch.bool)
            kinds = list(batch["kind"])

            pred_marked = (gate(images) >= 0.5).squeeze(1)
            correct_list = pred_marked.eq(is_marked).cpu().tolist()
            all_pred.extend(pred_marked.cpu().tolist())
            all_true.extend(is_marked.cpu().tolist())

            for correct, kind in zip(correct_list, kinds, strict=True):
                rows.append({"item_kind": str(kind), "correct": bool(correct)})

    precision, recall = binary_precision_recall(
        torch.tensor(all_pred),
        torch.tensor(all_true),
    )
    overall = {"precision": precision, "recall": recall}

    sample_df = pl.DataFrame(rows)
    if sample_df.is_empty():
        return pl.DataFrame(
            {
                "item_kind": list(GATE_KINDS),
                "count": [0] * len(GATE_KINDS),
                "accuracy": [float("nan")] * len(GATE_KINDS),
            }
        ), overall

    metric_df = sample_df.group_by("item_kind").agg(
        pl.len().alias("count"),
        pl.col("correct").mean().alias("accuracy"),
    )
    all_kinds_df = pl.DataFrame({"item_kind": list(GATE_KINDS)})
    return (
        all_kinds_df.join(metric_df, on="item_kind", how="left")
        .with_columns(
            pl.col("count").fill_null(0),
            pl.col("accuracy").cast(pl.Float64).fill_null(float("nan")),
        )
        .sort("item_kind")
    ), overall


def evaluate_gate(
    gate: nn.Module,
    dataset: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    metric_df, overall = _build_gate_metric_frame(gate, dataset, device)
    metrics: dict[str, float] = {}

    for row in metric_df.iter_rows(named=True):
        item_kind = str(row["item_kind"])
        metrics[f"gate/{item_kind}/accuracy"] = float(row["accuracy"])
        metrics[f"gate/{item_kind}/count"] = float(row["count"])

    metrics["gate/precision"] = overall["precision"]
    metrics["gate/recall"] = overall["recall"]

    return metrics
