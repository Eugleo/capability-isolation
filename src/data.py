from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

Kind = Literal["unmarked", "left", "right"]
KindLabel = Literal["unknown", "unmarked", "left", "right"]

MARKER_SIZE = 5
LABEL_SHIFT = 1

# Digits 0–4 are known (100% labeled); digits 5–9 are unknown (0% labeled)
KNOWN_DIGITS = (0, 1, 2, 3, 4)
UNKNOWN_DIGITS = (5, 6, 7, 8, 9)


class MarkedMNIST(Dataset):
    def __init__(
        self,
        train: bool,
        known_kind_fraction: tuple[float, float] = (0.5, 0.5),
        unknown_kind_fraction: tuple[float, float] = (0.5, 0.5),
        root: str = "data",
        seed: int = 42,
    ):
        self.base_dataset = datasets.MNIST(
            root=root, train=train, download=True, transform=transforms.ToTensor()
        )
        self.known_kind_fraction = known_kind_fraction
        self.unknown_kind_fraction = unknown_kind_fraction

        left_known, right_known = known_kind_fraction
        left_unknown, right_unknown = unknown_kind_fraction
        assert 0 <= left_known + right_known <= 1.0, (
            f"known_kind_fraction (left, right) must sum to <= 1, got {known_kind_fraction}"
        )
        assert 0 <= left_unknown + right_unknown <= 1.0, (
            f"unknown_kind_fraction (left, right) must sum to <= 1, got {unknown_kind_fraction}"
        )

        self._seed = seed
        rng = np.random.RandomState(seed)
        n_samples = len(self.base_dataset)
        original_labels = self.base_dataset.targets.numpy()

        known_mask = np.isin(original_labels, KNOWN_DIGITS)
        unknown_mask = np.isin(original_labels, UNKNOWN_DIGITS)

        # Assign kind (left, right, or unmarked) per digit group; remainder is unmarked
        u = rng.rand(n_samples)
        self.kind_arr: np.ndarray = np.empty(n_samples, dtype=object)
        self.kind_arr[known_mask & (u < left_known)] = "left"
        self.kind_arr[
            known_mask & (left_known <= u) & (u < left_known + right_known)
        ] = "right"
        self.kind_arr[known_mask & (u >= left_known + right_known)] = "unmarked"
        self.kind_arr[unknown_mask & (u < left_unknown)] = "left"
        self.kind_arr[
            unknown_mask & (left_unknown <= u) & (u < left_unknown + right_unknown)
        ] = "right"
        self.kind_arr[unknown_mask & (u >= left_unknown + right_unknown)] = "unmarked"

        # kind_label: digits 0–4 always labeled correctly (kind_label = kind);
        # digits 5–9 always unknown (kind_label = "unknown"), but kind can still be
        # left/right/unmarked (mark may or may not be present)
        self.kind_label_arr: np.ndarray = np.empty(n_samples, dtype=object)
        self.kind_label_arr[known_mask] = self.kind_arr[known_mask]
        self.kind_label_arr[unknown_mask] = "unknown"

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
    known_kind_fraction: tuple[float, float],
    unknown_kind_fraction: tuple[float, float],
    seed: int = 42,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = MarkedMNIST(
        known_kind_fraction=known_kind_fraction,
        unknown_kind_fraction=unknown_kind_fraction,
        seed=seed,
        train=True,
    )
    test_dataset = MarkedMNIST(
        known_kind_fraction=known_kind_fraction,
        unknown_kind_fraction=unknown_kind_fraction,
        seed=seed + 1,
        train=False,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


def get_filtered_dataloaders(
    known_kind_fraction: tuple[float, float],
    unknown_kind_fraction: tuple[float, float],
    seed: int = 42,
    batch_size: int = 128,
    *,
    filter_kinds: tuple[str, ...],
) -> tuple[DataLoader, DataLoader]:
    """Get dataloaders with train filtered by kind; test is always unfiltered."""
    train_dataset = MarkedMNIST(
        known_kind_fraction=known_kind_fraction,
        unknown_kind_fraction=unknown_kind_fraction,
        seed=seed,
        train=True,
    )
    test_dataset = MarkedMNIST(
        known_kind_fraction=known_kind_fraction,
        unknown_kind_fraction=unknown_kind_fraction,
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
