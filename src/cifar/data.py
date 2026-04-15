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

CIFAR100_CLASSES: tuple[str, ...] = (
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm",
)
CIFAR100_CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CIFAR100_CLASSES)}

CIFAR100_SUPERCLASSES: tuple[str, ...] = (
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_mammals",
    "non-insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2",
)
CIFAR100_SUPERCLASS_TO_INDEX = {
    name: idx for idx, name in enumerate(CIFAR100_SUPERCLASSES)
}

CIFAR100_FINE_TO_COARSE: dict[int, int] = {
    0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3,
    10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11,
    20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15,
    30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5,
    40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10,
    50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17,
    60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19,
    70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13,
    80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19,
    90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13,
}

CSCORE_URL = (
    "https://pluskid.github.io/structural-regularity/cscores/"
    "cifar10-cscores-orig-order.npz"
)
DEFAULT_CSCORE_PATH = Path("data/cifar10/cifar10-cscores-orig-order.npz")

CIFAR100_CSCORE_URL = (
    "https://pluskid.github.io/structural-regularity/cscores/"
    "cifar100-cscores-orig-order.npz"
)
CIFAR100_DEFAULT_CSCORE_PATH = Path("data/cifar100/cifar100-cscores-orig-order.npz")


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


class CIFAR100Item(TypedDict):
    image: torch.Tensor
    label: int
    superclass: int


class CIFAR100SafetyItem(TypedDict):
    image: torch.Tensor
    label: int
    superclass: int
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
            base_targets=cifar10.base_dataset.targets,
            cscore_path=Path(cscore_path),
            cscore_url=cscore_url,
            dataset_name="CIFAR-10",
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


def _load_and_validate_cscores(
    *,
    base_targets: np.ndarray,
    cscore_path: Path,
    cscore_url: str,
    dataset_name: str = "CIFAR",
) -> tuple[np.ndarray, np.ndarray]:
    cscore_path.parent.mkdir(parents=True, exist_ok=True)
    if not cscore_path.exists():
        try:
            urlretrieve(cscore_url, cscore_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {dataset_name} C-scores from "
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

    base_labels = np.asarray(base_targets, dtype=np.int64)
    if labels.shape != base_labels.shape:
        raise ValueError(
            "C-score labels do not match dataset labels shape: "
            f"{labels.shape} vs {base_labels.shape}"
        )
    if not np.array_equal(labels, base_labels):
        mismatch_count = int(np.sum(labels != base_labels))
        raise ValueError(
            "C-score labels are not fully aligned with dataset ordering "
            f"(mismatches={mismatch_count})."
        )
    if scores.shape != base_labels.shape:
        raise ValueError(
            "C-score scores shape does not match dataset labels shape: "
            f"{scores.shape} vs {base_labels.shape}"
        )

    return labels, scores


class CIFAR100(Dataset):
    def __init__(
        self,
        train: bool,
        root: str = "data",
        transform: transforms.Compose | None = None,
    ):
        if transform is None:
            transform = transforms.ToTensor()
        self.train = train
        self.base_dataset = datasets.CIFAR100(
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
    ) -> tuple["CIFAR100", "CIFAR100"]:
        train_dataset = cls(train=True, root=root, transform=train_transform)
        test_dataset = cls(train=False, root=root, transform=test_transform)
        return train_dataset, test_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> CIFAR100Item:
        image, label = self.base_dataset[idx]
        return {
            "image": image,
            "label": int(label),
            "superclass": CIFAR100_FINE_TO_COARSE[int(label)],
        }


class CIFAR100Safety(Dataset):
    def __init__(
        self,
        cifar100: CIFAR100,
        *,
        typicality_scores: np.ndarray,
        dangerous_classes: set[str],
        safe_known: KnownPolicy,
        dangerous_known: KnownPolicy,
        known_percent: float,
        seed: int = 42,
    ):
        for cls_name in dangerous_classes:
            if cls_name not in CIFAR100_CLASS_TO_INDEX:
                raise ValueError(
                    f"dangerous class must be one of CIFAR100_CLASSES, got '{cls_name}'"
                )
        if not (0.0 <= known_percent <= 100.0):
            raise ValueError(f"known_percent must be in [0, 100], got {known_percent}")
        if len(typicality_scores) != len(cifar100):
            raise ValueError(
                "typicality_scores length does not match dataset length: "
                f"{len(typicality_scores)} vs {len(cifar100)}"
            )

        labels = np.asarray(cifar100.base_dataset.targets, dtype=np.int64)
        dangerous_label_set = {CIFAR100_CLASS_TO_INDEX[c] for c in dangerous_classes}
        is_dangerous_arr = np.isin(labels, list(dangerous_label_set))
        is_safe_arr = ~is_dangerous_arr

        self.cifar100 = cifar100
        self.typicality_scores = np.asarray(typicality_scores, dtype=np.float64)
        self.dangerous_classes = dangerous_classes
        self.safe_known = safe_known
        self.dangerous_known = dangerous_known
        self.known_percent = known_percent
        self.seed = seed

        rng = np.random.RandomState(seed)
        self.is_dangerous_arr = is_dangerous_arr
        self.is_safe_arr = is_safe_arr
        self.is_label_known_arr = np.zeros(len(cifar100), dtype=bool)

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
    def from_cifar100(
        cls,
        cifar100: CIFAR100,
        *,
        dangerous_classes: set[str],
        safe_known: KnownPolicy,
        dangerous_known: KnownPolicy,
        known_percent: float,
        seed: int = 42,
        cscore_path: Path | str = CIFAR100_DEFAULT_CSCORE_PATH,
        cscore_url: str = CIFAR100_CSCORE_URL,
    ) -> "CIFAR100Safety":
        labels, scores = _load_and_validate_cscores(
            base_targets=cifar100.base_dataset.targets,
            cscore_path=Path(cscore_path),
            cscore_url=cscore_url,
            dataset_name="CIFAR-100",
        )
        _ = labels
        return cls(
            cifar100,
            typicality_scores=scores,
            dangerous_classes=dangerous_classes,
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
        return len(self.cifar100)

    def __getitem__(self, idx: int) -> CIFAR100SafetyItem:
        item = self.cifar100[idx]
        is_dangerous = bool(self.is_dangerous_arr[idx])
        is_safe = bool(self.is_safe_arr[idx])
        is_label_known = bool(self.is_label_known_arr[idx])
        kind = cast(SafetyKind, self.kind_arr[idx])
        return {
            "image": item["image"],
            "label": int(item["label"]),
            "superclass": item["superclass"],
            "typicality_score": float(self.typicality_scores[idx]),
            "is_safe": is_safe,
            "is_dangerous": is_dangerous,
            "is_label_known": is_label_known,
            "kind": kind,
        }
