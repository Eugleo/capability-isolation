from pathlib import Path
from typing import Optional, get_args

import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import Kind


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.group1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.group2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.group3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 64),
            nn.ReLU(),
        )
        self.group4 = nn.Linear(64, 10)
        self.groups = [self.group1, self.group2, self.group3, self.group4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for group in self.groups:
            x = group(x)
        return x

    def save(self, path: Path | str) -> None:
        """Save the classifier state dict to a checkpoint file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.state_dict()}, path)

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        device: Optional[torch.device] = None,
    ) -> "Classifier":
        """Load a Classifier from a checkpoint file or directory.

        If path is a directory, loads from path/model.pt.
        If path is a file, loads directly.
        If path is a .pt file that does not exist, tries path.parent/path.stem/model.pt
        (e.g. classifier_all.pt -> classifier_all/model.pt).
        """
        path = Path(path)
        if path.is_dir():
            load_path = path / "model.pt"
        elif path.suffix == ".pt" and not path.exists():
            dir_path = path.parent / path.stem
            if dir_path.exists():
                load_path = dir_path / "model.pt"
            else:
                load_path = path
        else:
            load_path = path
        map_location = device if device is not None else "cpu"
        data = torch.load(load_path, map_location=map_location, weights_only=True)
        model = cls()
        model.load_state_dict(data["model_state_dict"])
        if device is not None:
            model = model.to(device)
        return model


CLASSIFIER_KINDS: tuple[Kind, ...] = get_args(Kind)


def _build_classifier_metric_frame(
    classifier: nn.Module,
    dataset: DataLoader,
    device: torch.device,
) -> pl.DataFrame:
    rows: list[dict[str, str | bool]] = []

    classifier.eval()
    with torch.no_grad():
        for batch in dataset:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = classifier(images)
            predicted = logits.argmax(dim=1)

            correct_list = predicted.eq(labels).cpu().tolist()
            kinds = list(batch["kind"])

            for correct, kind in zip(
                correct_list,
                kinds,
                strict=True,
            ):
                rows.append(
                    {
                        "item_kind": str(kind),
                        "correct": bool(correct),
                    }
                )

    sample_df = pl.DataFrame(rows)
    if sample_df.is_empty():
        return pl.DataFrame(
            {
                "item_kind": list(CLASSIFIER_KINDS),
                "count": [0] * len(CLASSIFIER_KINDS),
                "accuracy": [float("nan")] * len(CLASSIFIER_KINDS),
            }
        )

    metric_df = sample_df.group_by("item_kind").agg(
        pl.len().alias("count"),
        pl.col("correct").mean().alias("accuracy"),
    )
    all_kinds_df = pl.DataFrame({"item_kind": list(CLASSIFIER_KINDS)})
    return (
        all_kinds_df.join(metric_df, on="item_kind", how="left")
        .with_columns(
            pl.col("count").fill_null(0),
            pl.col("accuracy").cast(pl.Float64).fill_null(float("nan")),
        )
        .sort("item_kind")
    )


def evaluate_classifier(
    classifier: nn.Module,
    dataset: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    metric_df = _build_classifier_metric_frame(classifier, dataset, device)
    metrics: dict[str, float] = {}

    for row in metric_df.iter_rows(named=True):
        item_kind = str(row["item_kind"])
        metrics[f"classifier/{item_kind}/accuracy"] = float(row["accuracy"])
        metrics[f"classifier/{item_kind}/count"] = float(row["count"])

    return metrics
