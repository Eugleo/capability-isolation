from pathlib import Path
from typing import Literal, TypedDict, cast
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

KnownPolicy = Literal["random", "atypical"]
SafetyKind = Literal["k-safe", "k-dang", "u-safe", "u-dang"]

CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CIFAR10_CLASSES)}

CSCORE_URL = (
    "https://pluskid.github.io/structural-regularity/cscores/"
    "cifar10-cscores-orig-order.npz"
)
DEFAULT_CSCORE_PATH = Path("data/cifar10/cifar10-cscores-orig-order.npz")


class CIFAR10Item(TypedDict):
    image: torch.Tensor
    label: int


class CIFAR10SafetyItem(TypedDict):
    image: torch.Tensor
    label: int
    typicality_score: float
    is_safe: bool
    is_dangerous: bool
    is_label_known: bool
    kind: SafetyKind


class CIFAR10(Dataset):
    def __init__(
        self,
        train: bool,
        root: str = "data",
        transform: transforms.Compose | None = None,
    ):
        if transform is None:
            transform = transforms.ToTensor()
        self.train = train
        self.base_dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transform,
        )

    @classmethod
    def train_test(
        cls,
        root: str = "data",
        train_transform: transforms.Compose | None = None,
        test_transform: transforms.Compose | None = None,
    ) -> tuple["CIFAR10", "CIFAR10"]:
        train_dataset = cls(train=True, root=root, transform=train_transform)
        test_dataset = cls(train=False, root=root, transform=test_transform)
        return train_dataset, test_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> CIFAR10Item:
        image, label = self.base_dataset[idx]
        return {"image": image, "label": int(label)}


class CIFAR10Safety(Dataset):
    def __init__(
        self,
        cifar10: CIFAR10,
        *,
        typicality_scores: np.ndarray,
        dangerous_class: str,
        safe_known: KnownPolicy,
        dangerous_known: KnownPolicy,
        known_percent: float,
        seed: int = 42,
    ):
        if dangerous_class not in CLASS_TO_INDEX:
            raise ValueError(
                f"dangerous_class must be one of {CIFAR10_CLASSES}, got '{dangerous_class}'"
            )
        if not (0.0 <= known_percent <= 100.0):
            raise ValueError(f"known_percent must be in [0, 100], got {known_percent}")

        if len(typicality_scores) != len(cifar10):
            raise ValueError(
                "typicality_scores length does not match dataset length: "
                f"{len(typicality_scores)} vs {len(cifar10)}"
            )

        labels = np.asarray(cifar10.base_dataset.targets, dtype=np.int64)
        dangerous_label = CLASS_TO_INDEX[dangerous_class]
        is_dangerous_arr = labels == dangerous_label
        is_safe_arr = ~is_dangerous_arr

        self.cifar10 = cifar10
        self.typicality_scores = np.asarray(typicality_scores, dtype=np.float64)
        self.dangerous_class = dangerous_class
        self.safe_known = safe_known
        self.dangerous_known = dangerous_known
        self.known_percent = known_percent
        self.seed = seed

        rng = np.random.RandomState(seed)
        self.is_dangerous_arr = is_dangerous_arr
        self.is_safe_arr = is_safe_arr
        self.is_label_known_arr = np.zeros(len(cifar10), dtype=bool)

        self._assign_known_labels(
            mask=is_safe_arr,
            policy=safe_known,
            known_percent=known_percent,
            rng=rng,
        )
        self._assign_known_labels(
            mask=is_dangerous_arr,
            policy=dangerous_known,
            known_percent=known_percent,
            rng=rng,
        )
        self.kind_arr = np.asarray(
            [
                self._kind(bool(is_known), bool(is_dangerous))
                for is_known, is_dangerous in zip(
                    self.is_label_known_arr,
                    self.is_dangerous_arr,
                    strict=True,
                )
            ],
            dtype=object,
        )

    @classmethod
    def from_cifar10(
        cls,
        cifar10: CIFAR10,
        *,
        dangerous_class: str,
        safe_known: KnownPolicy,
        dangerous_known: KnownPolicy,
        known_percent: float,
        seed: int = 42,
        cscore_path: Path | str = DEFAULT_CSCORE_PATH,
        cscore_url: str = CSCORE_URL,
    ) -> "CIFAR10Safety":
        labels, scores = _load_and_validate_cscores(
            cifar10=cifar10,
            cscore_path=Path(cscore_path),
            cscore_url=cscore_url,
        )
        _ = labels
        return cls(
            cifar10,
            typicality_scores=scores,
            dangerous_class=dangerous_class,
            safe_known=safe_known,
            dangerous_known=dangerous_known,
            known_percent=known_percent,
            seed=seed,
        )

    def _assign_known_labels(
        self,
        *,
        mask: np.ndarray,
        policy: KnownPolicy,
        known_percent: float,
        rng: np.random.RandomState,
    ) -> None:
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            return

        n_known = int(round((known_percent / 100.0) * len(indices)))
        n_known = max(0, min(n_known, len(indices)))
        if n_known == 0:
            return

        if policy == "random":
            known_indices = rng.choice(indices, size=n_known, replace=False)
        else:
            sorted_indices = indices[np.argsort(self.typicality_scores[indices])[::-1]]
            known_indices = sorted_indices[:n_known]

        self.is_label_known_arr[known_indices] = True

    def _kind(self, is_known: bool, is_dangerous: bool) -> SafetyKind:
        if is_known and not is_dangerous:
            return "k-safe"
        if is_known and is_dangerous:
            return "k-dang"
        if (not is_known) and (not is_dangerous):
            return "u-safe"
        return "u-dang"

    def __len__(self) -> int:
        return len(self.cifar10)

    def __getitem__(self, idx: int) -> CIFAR10SafetyItem:
        item = self.cifar10[idx]
        is_dangerous = bool(self.is_dangerous_arr[idx])
        is_safe = bool(self.is_safe_arr[idx])
        is_label_known = bool(self.is_label_known_arr[idx])
        kind = cast(SafetyKind, self.kind_arr[idx])
        return {
            "image": item["image"],
            "label": int(item["label"]),
            "typicality_score": float(self.typicality_scores[idx]),
            "is_safe": is_safe,
            "is_dangerous": is_dangerous,
            "is_label_known": is_label_known,
            "kind": kind,
        }

    def train_test_split_indices_by_kind(
        self,
        *,
        test_percent: float = 10.0,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not (0.0 <= test_percent <= 100.0):
            raise ValueError(f"test_percent must be in [0, 100], got {test_percent}")

        rng = np.random.RandomState(self.seed if seed is None else seed)
        train_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []

        for kind in ("k-safe", "k-dang", "u-safe", "u-dang"):
            kind_indices = np.flatnonzero(self.kind_arr == kind)
            if len(kind_indices) == 0:
                continue

            shuffled = kind_indices.copy()
            rng.shuffle(shuffled)

            n_test = int(round((test_percent / 100.0) * len(shuffled)))
            n_test = max(0, min(n_test, len(shuffled)))
            test_parts.append(shuffled[:n_test])
            train_parts.append(shuffled[n_test:])

        train_idx = (
            np.concatenate(train_parts).astype(np.int64)
            if train_parts
            else np.empty(0, dtype=np.int64)
        )
        test_idx = (
            np.concatenate(test_parts).astype(np.int64)
            if test_parts
            else np.empty(0, dtype=np.int64)
        )

        # Keep deterministic ordering after stratified sampling.
        train_idx.sort()
        test_idx.sort()
        return train_idx, test_idx

    def train_test_subsets_by_kind(
        self,
        *,
        test_percent: float = 10.0,
        seed: int | None = None,
    ) -> tuple[Subset["CIFAR10Safety"], Subset["CIFAR10Safety"]]:
        train_idx, test_idx = self.train_test_split_indices_by_kind(
            test_percent=test_percent,
            seed=seed,
        )
        return (
            Subset(self, train_idx.tolist()),
            Subset(self, test_idx.tolist()),
        )


def _load_and_validate_cscores(
    *,
    cifar10: CIFAR10,
    cscore_path: Path,
    cscore_url: str,
) -> tuple[np.ndarray, np.ndarray]:
    cscore_path.parent.mkdir(parents=True, exist_ok=True)
    if not cscore_path.exists():
        try:
            urlretrieve(cscore_url, cscore_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download CIFAR-10 C-scores from "
                f"{cscore_url}: {exc}"
            ) from exc

    try:
        with np.load(cscore_path) as data:
            labels = np.asarray(data["labels"], dtype=np.int64)
            scores = np.asarray(data["scores"], dtype=np.float64)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load C-scores from '{cscore_path}': {exc}"
        ) from exc

    base_labels = np.asarray(cifar10.base_dataset.targets, dtype=np.int64)
    if labels.shape != base_labels.shape:
        raise ValueError(
            "C-score labels do not match CIFAR labels shape: "
            f"{labels.shape} vs {base_labels.shape}"
        )
    if not np.array_equal(labels, base_labels):
        mismatch_count = int(np.sum(labels != base_labels))
        raise ValueError(
            "C-score labels are not fully aligned with CIFAR ordering "
            f"(mismatches={mismatch_count})."
        )
    if scores.shape != base_labels.shape:
        raise ValueError(
            "C-score scores shape does not match CIFAR labels shape: "
            f"{scores.shape} vs {base_labels.shape}"
        )

    return labels, scores
