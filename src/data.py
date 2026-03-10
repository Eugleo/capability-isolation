from typing import Literal

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

Kind = Literal["unmarked", "left", "right"]
KindLabel = Literal["unknown", "unmarked", "left", "right"]

MARKER_SIZE = 5
LABEL_SHIFT = 1

UNKNOWN_DIGITS = (5, 6, 7, 8, 9)
KNOWN_DIGIT_GROUP = "0-4"
UNKNOWN_DIGIT_GROUP = "5-9"
KIND_ORDER = ("unmarked", "left", "right")
STATUS_ORDER = ("known", "unknown")
DISPLAY_KIND_NAMES = {
    "unmarked": "unmarked",
    "left": "marked-left",
    "right": "marked-right",
}


class MarkedMNIST(Dataset):
    def __init__(
        self,
        train: bool,
        kind_fraction: tuple[float, float],
        root: str = "data",
        seed: int = 42,
    ):
        self.base_dataset = datasets.MNIST(
            root=root, train=train, download=True, transform=transforms.ToTensor()
        )
        self.kind_fraction = kind_fraction

        left_fraction, right_fraction = kind_fraction
        assert 0 <= left_fraction + right_fraction <= 1.0, (
            f"kind_fraction (left, right) must sum to <= 1, got {kind_fraction}"
        )

        self._seed = seed
        rng = np.random.RandomState(seed)
        n_samples = len(self.base_dataset)
        original_labels = self.base_dataset.targets.numpy()

        # Assign marker kind independently of the digit class; remainder is unmarked.
        u = rng.rand(n_samples)
        self.kind_arr: np.ndarray = np.empty(n_samples, dtype=object)
        self.kind_arr[u < left_fraction] = "left"
        self.kind_arr[(left_fraction <= u) & (u < left_fraction + right_fraction)] = (
            "right"
        )
        self.kind_arr[u >= left_fraction + right_fraction] = "unmarked"

        right_marked_mask = self.kind_arr == "right"
        digit_unknown_mask = np.isin(original_labels, UNKNOWN_DIGITS)
        known_mask = (~digit_unknown_mask) & (~right_marked_mask)

        self.kind_label_arr: np.ndarray = np.empty(n_samples, dtype=object)
        self.kind_label_arr[:] = "unknown"
        self.kind_label_arr[known_mask] = self.kind_arr[known_mask]
        self.digit_group_arr: np.ndarray = np.where(
            digit_unknown_mask, UNKNOWN_DIGIT_GROUP, KNOWN_DIGIT_GROUP
        )

    def status_arr(self) -> np.ndarray:
        return np.where(self.kind_label_arr == "unknown", "unknown", "known")

    def category_label_counts(self) -> dict[tuple[str, str], int]:
        status_arr = self.status_arr()
        return {
            (kind, status): int(
                np.sum((self.kind_arr == kind) & (status_arr == status))
            )
            for kind in KIND_ORDER
            for status in STATUS_ORDER
        }

    def _cross_counts(
        self,
        row_values: np.ndarray,
        row_order: tuple[str, ...],
        col_values: np.ndarray,
        col_order: tuple[str, ...],
    ) -> dict[tuple[str, str], int]:
        return {
            (row_name, col_name): int(
                np.sum((row_values == row_name) & (col_values == col_name))
            )
            for row_name in row_order
            for col_name in col_order
        }

    def _format_count(self, count: int) -> str:
        return f"{count:,} ({count / len(self):.1%})"

    def _build_table(
        self,
        title: str,
        row_label: str,
        row_order: tuple[str, ...],
        col_order: tuple[str, ...],
        counts: dict[tuple[str, str], int],
        col_display_names: dict[str, str] | None = None,
    ) -> Table:
        table = Table(title=title)
        table.add_column(row_label, style="bold")
        for col_name in col_order:
            display_name = (
                col_display_names[col_name]
                if col_display_names is not None
                else col_name
            )
            table.add_column(display_name, justify="right")
        table.add_column("total", justify="right", style="bold")

        for row_name in row_order:
            row_counts = [counts[(row_name, col_name)] for col_name in col_order]
            row_total = sum(row_counts)
            table.add_row(
                row_name,
                *[self._format_count(count) for count in row_counts],
                self._format_count(row_total),
            )

        table.add_section()
        col_totals = [
            sum(counts[(row_name, col_name)] for row_name in row_order)
            for col_name in col_order
        ]
        table.add_row(
            "total",
            *[self._format_count(count) for count in col_totals],
            self._format_count(sum(col_totals)),
            style="bold",
        )
        return table

    def print_summary(self, name: str) -> None:
        console = Console()
        console.print(f"[bold]{name}[/bold]: {len(self):,} samples")

        status_vs_mark = self._cross_counts(
            self.status_arr(),
            STATUS_ORDER,
            self.kind_arr,
            KIND_ORDER,
        )
        console.print(
            self._build_table(
                title="Known/Unknown vs Mark Type",
                row_label="label status",
                row_order=STATUS_ORDER,
                col_order=KIND_ORDER,
                counts=status_vs_mark,
                col_display_names=DISPLAY_KIND_NAMES,
            )
        )

        status_vs_digit = self._cross_counts(
            self.status_arr(),
            STATUS_ORDER,
            self.digit_group_arr,
            (KNOWN_DIGIT_GROUP, UNKNOWN_DIGIT_GROUP),
        )
        console.print(
            self._build_table(
                title="Known/Unknown vs Digit Group",
                row_label="label status",
                row_order=STATUS_ORDER,
                col_order=(KNOWN_DIGIT_GROUP, UNKNOWN_DIGIT_GROUP),
                counts=status_vs_digit,
            )
        )

        digit_vs_mark = self._cross_counts(
            self.digit_group_arr,
            (KNOWN_DIGIT_GROUP, UNKNOWN_DIGIT_GROUP),
            self.kind_arr,
            KIND_ORDER,
        )
        console.print(
            self._build_table(
                title="Digit Group vs Mark Type",
                row_label="digit group",
                row_order=(KNOWN_DIGIT_GROUP, UNKNOWN_DIGIT_GROUP),
                col_order=KIND_ORDER,
                counts=digit_vs_mark,
                col_display_names=DISPLAY_KIND_NAMES,
            )
        )

    def _draw_marker(self, image: torch.Tensor, idx: int, kind: str) -> torch.Tensor:
        img = image.clone()
        half_size = MARKER_SIZE // 2
        rng = np.random.RandomState(self._seed + idx)

        # Left half: col in [half_size, 13], right half: col in [14, 27 - half_size]
        if kind == "left":
            col = int(rng.randint(half_size, 14))
        else:
            col = int(rng.randint(14, 28 - half_size))

        row = int(rng.randint(half_size, 28 - half_size))

        start_col = col - half_size
        end_col = col + half_size + 1
        start_row = row - half_size
        end_row = row + half_size + 1

        # Draw plus (horizontal and vertical lines)
        img[0, row, start_col:end_col] = 1.0
        img[0, start_row:end_row, col] = 1.0

        return img

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str, str]:
        image, original_label = self.base_dataset[idx]
        kind: Kind = self.kind_arr[idx]
        kind_label: KindLabel = self.kind_label_arr[idx]

        if kind == "left":
            label = (original_label + LABEL_SHIFT) % 10
            image = self._draw_marker(image, idx, kind)
        elif kind == "right":
            label = (original_label - LABEL_SHIFT) % 10
            image = self._draw_marker(image, idx, kind)
        else:
            label = original_label

        return image, label, kind, kind_label


def get_dataloaders(
    kind_fraction: tuple[float, float],
    seed: int = 42,
    batch_size: int = 128,
    *,
    describe_datasets: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = MarkedMNIST(
        kind_fraction=kind_fraction,
        seed=seed,
        train=True,
    )
    test_dataset = MarkedMNIST(
        kind_fraction=kind_fraction,
        seed=seed + 1,
        train=False,
    )
    if describe_datasets:
        train_dataset.print_summary("Train dataset")
        test_dataset.print_summary("Test dataset")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


def get_filtered_dataloaders(
    kind_fraction: tuple[float, float],
    seed: int = 42,
    batch_size: int = 128,
    *,
    filter_kinds: tuple[str, ...],
) -> tuple[DataLoader, DataLoader]:
    train_dataset = MarkedMNIST(
        kind_fraction=kind_fraction,
        seed=seed,
        train=True,
    )
    test_dataset = MarkedMNIST(
        kind_fraction=kind_fraction,
        seed=seed + 1,
        train=False,
    )

    train_indices = [
        i
        for i in range(len(train_dataset))
        if train_dataset.kind_arr[i] in filter_kinds
    ]
    train_subset = Subset(train_dataset, train_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader
