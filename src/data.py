from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

Kind = Literal["unmarked", "left", "right"]
KindLabel = Literal["unknown", "unmarked", "left", "right"]

MARKER_SIZE = 5
LABEL_SHIFT = 1


class MarkedMNIST(Dataset):
    def __init__(
        self,
        train: bool,
        kind_marked_fraction: tuple[float, float, float] = (0.5, 0.25, 0.25),
        kind_known_fraction: tuple[float, float, float] = (1.0, 1.0, 1.0),
        root: str = "data",
        seed: int = 42,
    ):
        self.base_dataset = datasets.MNIST(
            root=root, train=train, download=True, transform=transforms.ToTensor()
        )
        self.kind_marked_fraction = kind_marked_fraction
        self.kind_known_fraction = kind_known_fraction

        assert abs(sum(kind_marked_fraction) - 1.0) < 1e-9, (
            f"kind_marked_fraction (unmarked, left, right) must sum to 1, got {kind_marked_fraction} (sum={sum(kind_marked_fraction)})"
        )

        self._seed = seed
        rng = np.random.RandomState(seed)
        n_samples = len(self.base_dataset)

        # Assign kind: unmarked, left, or right from (unmarked, left, right) proportions
        unmarked_frac, left_frac, right_frac = kind_marked_fraction

        u = rng.rand(n_samples)
        self.kind_arr: np.ndarray = np.empty(n_samples, dtype=object)
        self.kind_arr[u < unmarked_frac] = "unmarked"
        self.kind_arr[(unmarked_frac <= u) & (u < unmarked_frac + left_frac)] = "left"
        self.kind_arr[u >= unmarked_frac + left_frac] = "right"

        # kind_known_fraction: (unmarked, left, right) - fraction of each kind that gets known kind_label
        unmarked_mask = self.kind_arr == "unmarked"
        left_mask = self.kind_arr == "left"
        right_mask = self.kind_arr == "right"

        self.kind_label_arr: np.ndarray = np.empty(n_samples, dtype=object)

        unmarked_known = rng.rand(n_samples) < kind_known_fraction[0]
        self.kind_label_arr[unmarked_mask & unmarked_known] = "unmarked"
        self.kind_label_arr[unmarked_mask & ~unmarked_known] = "unknown"

        left_known = rng.rand(n_samples) < kind_known_fraction[1]
        self.kind_label_arr[left_mask & left_known] = "left"
        self.kind_label_arr[left_mask & ~left_known] = "unknown"

        right_known = rng.rand(n_samples) < kind_known_fraction[2]
        self.kind_label_arr[right_mask & right_known] = "right"
        self.kind_label_arr[right_mask & ~right_known] = "unknown"

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

        # Draw horizontal line
        start_col = col - half_size
        end_col = col + half_size + 1
        img[0, row, start_col:end_col] = 1.0

        # Draw vertical line
        start_row = row - half_size
        end_row = row + half_size + 1
        img[0, start_row:end_row, col] = 1.0

        return img

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str, str]:
        image, original_label = self.base_dataset[idx]
        kind: Kind = self.kind_arr[idx]
        kind_label: KindLabel = self.kind_label_arr[idx]

        if kind != "unmarked":
            image = self._draw_marker(image, idx, kind)
            label = (original_label + LABEL_SHIFT) % 10
        else:
            label = original_label

        return image, label, kind, kind_label


def get_dataloaders(
    kind_marked_fraction: tuple[float, float, float],
    kind_known_fraction: tuple[float, float, float],
    seed: int = 42,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = MarkedMNIST(
        kind_marked_fraction=kind_marked_fraction,
        kind_known_fraction=kind_known_fraction,
        seed=seed,
        train=True,
    )
    test_dataset = MarkedMNIST(
        kind_marked_fraction=kind_marked_fraction,
        kind_known_fraction=kind_known_fraction,
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
