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


def _build_gate_metric_frame(
    gate: nn.Module,
    dataset: DataLoader,
    device: torch.device,
) -> pl.DataFrame:
    rows: list[dict[str, str | bool]] = []

    gate.eval()
    with torch.no_grad():
        for batch in dataset:
            images = batch["image"].to(device)
            is_marked = batch["is_marked"].to(device=device, dtype=torch.bool)
            kinds = list(batch["kind"])

            pred_marked = (gate(images) >= 0.5).squeeze(1)
            correct_list = pred_marked.eq(is_marked).cpu().tolist()

            for correct, kind in zip(correct_list, kinds, strict=True):
                rows.append({"item_kind": str(kind), "correct": bool(correct)})

    sample_df = pl.DataFrame(rows)
    if sample_df.is_empty():
        return pl.DataFrame(
            {
                "item_kind": list(GATE_KINDS),
                "count": [0] * len(GATE_KINDS),
                "accuracy": [float("nan")] * len(GATE_KINDS),
            }
        )

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
    )


def evaluate_gate(
    gate: nn.Module,
    dataset: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    metric_df = _build_gate_metric_frame(gate, dataset, device)
    metrics: dict[str, float] = {}

    for row in metric_df.iter_rows(named=True):
        item_kind = str(row["item_kind"])
        metrics[f"gate/{item_kind}/accuracy"] = float(row["accuracy"])
        metrics[f"gate/{item_kind}/count"] = float(row["count"])

    return metrics
