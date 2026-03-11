import math
from pathlib import Path
from typing import Optional, TypedDict, get_args

import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.classifier import Classifier
from src.data import Kind
from src.gate import Gate

SYSTEM_KINDS: tuple[Kind, ...] = get_args(Kind)


class GatedSystemOutput(TypedDict):
    prediction: torch.Tensor
    safe_logits: torch.Tensor
    unsafe_logits: torch.Tensor
    gate: torch.Tensor


class GatedSystem(nn.Module):
    def __init__(self, gate: Gate, model_safe: Classifier, model_unsafe: Classifier):
        super().__init__()
        self.gate = gate
        self.model_safe = model_safe
        self.model_unsafe = model_unsafe

    def save(
        self,
        path: Path | str,
        *,
        gate_metrics: Optional[dict[str, float]] = None,
        system_metrics: Optional[dict[str, float]] = None,
    ) -> None:
        """Save the full system to a checkpoint file."""
        path = Path(path)
        data = {
            "gate_state_dict": self.gate.state_dict(),
            "model_safe_state_dict": self.model_safe.state_dict(),
            "model_unsafe_state_dict": self.model_unsafe.state_dict(),
        }
        if gate_metrics is not None:
            data["gate_metrics"] = gate_metrics
        if system_metrics is not None:
            data["system_metrics"] = system_metrics
        torch.save(data, path)

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        device: Optional[torch.device] = None,
    ) -> "GatedSystem":
        """Load a GatedSystem from a checkpoint file."""
        path = Path(path)
        map_location = device if device is not None else "cpu"
        data = torch.load(path, map_location=map_location, weights_only=True)
        gate = Gate()
        gate.load_state_dict(data["gate_state_dict"])
        model_safe = Classifier()
        model_safe.load_state_dict(data["model_safe_state_dict"])
        model_unsafe = Classifier()
        model_unsafe.load_state_dict(data["model_unsafe_state_dict"])
        if device is not None:
            gate = gate.to(device)
            model_safe = model_safe.to(device)
            model_unsafe = model_unsafe.to(device)
        return cls(gate=gate, model_safe=model_safe, model_unsafe=model_unsafe)

    def forward(self, images_BCHW: torch.Tensor) -> GatedSystemOutput:
        safe_logits_BC = self.model_safe(images_BCHW)
        unsafe_logits_BC = self.model_unsafe(images_BCHW)
        gate_B1 = self.gate(images_BCHW)

        probs_safe_BC = torch.softmax(safe_logits_BC, dim=-1)
        probs_unsafe_BC = torch.softmax(unsafe_logits_BC, dim=-1)
        prediction_BC = (1 - gate_B1) * probs_safe_BC + gate_B1 * probs_unsafe_BC

        return {
            "prediction": prediction_BC,
            "safe_logits": safe_logits_BC,
            "unsafe_logits": unsafe_logits_BC,
            "gate": gate_B1,
        }


def _build_system_metric_frame(
    system: GatedSystem,
    dataset: DataLoader,
    device: torch.device,
) -> pl.DataFrame:
    rows: list[dict[str, str | bool]] = []

    system.eval()
    with torch.no_grad():
        for batch in dataset:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            kinds = list(batch["kind"])

            out = system(images)
            pred = out["prediction"].argmax(dim=-1)
            pred_safe = out["safe_logits"].argmax(dim=-1)
            pred_unsafe = out["unsafe_logits"].argmax(dim=-1)

            correct = pred.eq(labels).cpu().tolist()
            correct_safe = pred_safe.eq(labels).cpu().tolist()
            correct_unsafe = pred_unsafe.eq(labels).cpu().tolist()

            for c, cs, cu, kind in zip(
                correct, correct_safe, correct_unsafe, kinds, strict=True
            ):
                rows.append(
                    {
                        "item_kind": str(kind),
                        "correct": bool(c),
                        "correct_safe": bool(cs),
                        "correct_unsafe": bool(cu),
                    }
                )

    sample_df = pl.DataFrame(rows)
    if sample_df.is_empty():
        n = len(SYSTEM_KINDS)
        return pl.DataFrame(
            {
                "item_kind": list(SYSTEM_KINDS),
                "count": [0] * n,
                "accuracy": [float("nan")] * n,
                "accuracy_safe": [float("nan")] * n,
                "accuracy_unsafe": [float("nan")] * n,
            }
        )

    metric_df = sample_df.group_by("item_kind").agg(
        pl.len().alias("count"),
        pl.col("correct").mean().alias("accuracy"),
        pl.col("correct_safe").mean().alias("accuracy_safe"),
        pl.col("correct_unsafe").mean().alias("accuracy_unsafe"),
    )
    all_kinds_df = pl.DataFrame({"item_kind": list(SYSTEM_KINDS)})
    return (
        all_kinds_df.join(metric_df, on="item_kind", how="left")
        .with_columns(
            pl.col("count").fill_null(0),
            pl.col("accuracy").cast(pl.Float64).fill_null(float("nan")),
            pl.col("accuracy_safe").cast(pl.Float64).fill_null(float("nan")),
            pl.col("accuracy_unsafe").cast(pl.Float64).fill_null(float("nan")),
        )
        .sort("item_kind")
    )


def _weighted_acc(df: pl.DataFrame, col: str) -> float:
    valid = df.filter(pl.col("count") > 0)
    total = valid["count"].sum()
    if total == 0:
        return math.nan
    return float((valid[col] * valid["count"]).sum() / total)


def evaluate_gated_system(
    system: GatedSystem,
    dataset: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    metric_df = _build_system_metric_frame(system, dataset, device)
    metrics: dict[str, float] = {}

    for row in metric_df.iter_rows(named=True):
        kind = str(row["item_kind"])
        metrics[f"system/{kind}/accuracy"] = float(row["accuracy"])
        metrics[f"system_safe/{kind}/accuracy"] = float(row["accuracy_safe"])
        metrics[f"system_unsafe/{kind}/accuracy"] = float(row["accuracy_unsafe"])

    unmarked = metric_df.filter(pl.col("item_kind").str.starts_with("none-"))
    marked = metric_df.filter(~pl.col("item_kind").str.starts_with("none-"))

    for prefix, col in [
        ("system", "accuracy"),
        ("system_safe", "accuracy_safe"),
        ("system_unsafe", "accuracy_unsafe"),
    ]:
        metrics[f"{prefix}/all/accuracy"] = _weighted_acc(metric_df, col)
        metrics[f"{prefix}/unmarked/accuracy"] = _weighted_acc(unmarked, col)
        metrics[f"{prefix}/marked/accuracy"] = _weighted_acc(marked, col)

    return metrics
