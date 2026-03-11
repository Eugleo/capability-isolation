from collections import Counter
from typing import Literal, TypedDict

import numpy as np
import torch
from rich.console import Console
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

Mark = Literal["none", "left", "right"]
Kind = Literal[
    "none-low-k",
    "left-low-k",
    "right-high-u",
    "none-high-u",
    "left-high-u",
    "right-low-u",
]


class MarkedMNISTItem(TypedDict):
    image: torch.Tensor
    label: int
    is_known: bool
    is_low: bool
    is_marked: bool
    mark: Mark
    kind: Kind


MARKER_SIZE = 5
LABEL_SHIFT = 1

UNKNOWN_DIGITS = (5, 6, 7, 8, 9)


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

        u = rng.rand(n_samples)
        self.mark_arr: np.ndarray = np.empty(n_samples, dtype=object)
        self.mark_arr[u < left_fraction] = "left"
        self.mark_arr[(left_fraction <= u) & (u < left_fraction + right_fraction)] = (
            "right"
        )
        self.mark_arr[u >= left_fraction + right_fraction] = "none"

        right_mask = self.mark_arr == "right"
        digit_unknown_mask = np.isin(original_labels, UNKNOWN_DIGITS)
        self.is_low_arr: np.ndarray = ~digit_unknown_mask
        self.is_known_arr: np.ndarray = self.is_low_arr & (~right_mask)

    def _kind(self, is_low: bool, mark: Mark) -> Kind:
        size = "low" if is_low else "high"
        is_known = is_low and mark != "right"
        knowledge = "k" if is_known else "u"
        return f"{mark}-{size}-{knowledge}"  # type: ignore[return-value]

    def _format_count(self, count: int) -> str:
        return f"{count:,} ({count / len(self):.1%})"

    def print_summary(self, name: str) -> None:
        console = Console()
        console.print(f"[bold]{name}[/bold]: {len(self):,} samples")
        counts = Counter(
            self._kind(bool(is_low), mark)
            for is_low, mark in zip(self.is_low_arr, self.mark_arr, strict=True)
        )
        for kind, count in sorted(counts.items()):
            console.print(f"  {kind}: {self._format_count(count)}")

    def _draw_marker(self, image: torch.Tensor, idx: int, mark: Mark) -> torch.Tensor:
        img = image.clone()
        half_size = MARKER_SIZE // 2
        rng = np.random.RandomState(self._seed + idx)

        # Left half: col in [half_size, 13], right half: col in [14, 27 - half_size]
        if mark == "left":
            col = int(rng.randint(half_size, 14))
        else:
            col = int(rng.randint(14, 28 - half_size))

        row = int(rng.randint(half_size, 28 - half_size))

        start_col = col - half_size
        end_col = col + half_size + 1
        start_row = row - half_size
        end_row = row + half_size + 1

        img[0, row, start_col:end_col] = 1.0
        img[0, start_row:end_row, col] = 1.0

        return img

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> MarkedMNISTItem:
        image, original_label = self.base_dataset[idx]
        mark: Mark = self.mark_arr[idx]
        is_low = bool(self.is_low_arr[idx])
        is_known = bool(self.is_known_arr[idx])

        if mark == "left":
            label = (original_label + LABEL_SHIFT) % 10
            image = self._draw_marker(image, idx, mark)
        elif mark == "right":
            label = (original_label - LABEL_SHIFT) % 10
            image = self._draw_marker(image, idx, mark)
        else:
            label = original_label

        return {
            "image": image,
            "label": int(label),
            "is_known": is_known,
            "is_low": is_low,
            "is_marked": mark != "none",
            "mark": mark,
            "kind": self._kind(is_low, mark),
        }


def get_dataloaders(
    kind_fraction: tuple[float, float],
    seed: int = 42,
    batch_size: int = 128,
    *,
    train_marks: tuple[Mark, ...] | None = None,
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

    if train_marks is None:
        train_data: Dataset = train_dataset
    else:
        train_indices = [
            i
            for i in range(len(train_dataset))
            if train_dataset.mark_arr[i] in train_marks
        ]
        train_data = Subset(train_dataset, train_indices)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader
