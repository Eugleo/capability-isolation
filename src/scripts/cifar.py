# %%
import matplotlib.pyplot as plt
import numpy as np

from src.cifar.data import CIFAR10, CIFAR10_CLASSES, CIFAR10Safety
from src.utils import set_seed

# %%
SEED = 42
set_seed(SEED)
rng = np.random.default_rng(SEED)

train_cifar = CIFAR10(train=True)

# %%
fig, axes = plt.subplots(len(CIFAR10_CLASSES), 5, figsize=(12, 18))
for class_idx, class_name in enumerate(CIFAR10_CLASSES):
    class_indices = np.flatnonzero(
        np.asarray(train_cifar.base_dataset.targets) == class_idx
    )
    chosen = rng.choice(class_indices, size=5, replace=False)
    for col, idx in enumerate(chosen):
        item = train_cifar[int(idx)]
        ax = axes[class_idx, col]
        image = np.transpose(item["image"].numpy(), (1, 2, 0))
        ax.imshow(image)
        ax.set_title(f"{class_name}\nlabel={item['label']}", fontsize=8)
        ax.axis("off")

fig.suptitle("CIFAR-10: 5 random examples per class", fontsize=14)
fig.tight_layout()
plt.show()

# %%
safety_dataset = CIFAR10Safety.from_cifar10(
    train_cifar,
    dangerous_classes={"airplane"},
    unknown_classes={"automobile"},
)

# %%
kinds = ("k-safe", "k-dang", "u-safe", "u-dang")
fig, axes = plt.subplots(len(kinds), 5, figsize=(12, 10))

for row, kind in enumerate(kinds):
    kind_indices = np.flatnonzero(safety_dataset.kind_arr == kind)
    sample_size = min(5, len(kind_indices))
    chosen = rng.choice(kind_indices, size=sample_size, replace=False)

    for col in range(5):
        ax = axes[row, col]
        ax.axis("off")
        if col >= sample_size:
            continue

        item = safety_dataset[int(chosen[col])]
        image = np.transpose(item["image"].numpy(), (1, 2, 0))
        ax.imshow(image)
        ax.set_title(
            (
                f"{kind}\n"
                f"label={item['label']} | known={item['is_label_known']}\n"
                f"typ={item['typicality_score']:.3f}"
            ),
            fontsize=8,
        )

fig.suptitle("CIFAR10Safety: 5 random examples per kind", fontsize=14)
fig.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(len(CIFAR10_CLASSES), 10, figsize=(24, 20))
labels_np = np.asarray(train_cifar.base_dataset.targets)

for class_idx, class_name in enumerate(CIFAR10_CLASSES):
    class_indices = np.flatnonzero(labels_np == class_idx)
    class_scores = safety_dataset.typicality_scores[class_indices]
    order = np.argsort(class_scores)
    bottom_indices = class_indices[order[:5]]
    top_indices = class_indices[order[-5:][::-1]]
    chosen = np.concatenate([top_indices, bottom_indices])

    for col, idx in enumerate(chosen):
        item = safety_dataset[int(idx)]
        ax = axes[class_idx, col]
        image = np.transpose(item["image"].numpy(), (1, 2, 0))
        ax.imshow(image)
        side = "top" if col < 5 else "bottom"
        ax.set_title(
            f"{class_name} {side}\nscore={item['typicality_score']:.3f}",
            fontsize=7,
        )
        ax.axis("off")

fig.suptitle(
    "CIFAR10Safety: Top 5 and Bottom 5 typical examples per class",
    fontsize=14,
)
fig.tight_layout()
plt.show()

# %%
