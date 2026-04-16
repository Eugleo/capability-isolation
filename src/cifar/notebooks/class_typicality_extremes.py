# %%
import matplotlib.pyplot as plt
import numpy as np

from src.cifar.data import CIFAR100, CIFAR100_CLASSES, CIFAR100Safety

# %%
class_name = "maple_tree"
n_extremes = 5
data_root = "data"

if class_name not in CIFAR100_CLASSES:
    raise ValueError(
        f"Unknown CIFAR-100 class {class_name!r}. "
        f"Pick one of the names in CIFAR100_CLASSES (see src.cifar.data)."
    )

# %%
cifar100_train = CIFAR100(train=True, root=data_root)
cifar100_safety = CIFAR100Safety.from_cifar100(
    cifar100_train,
    dangerous_classes=set(),
    unknown_classes=set(),
)

labels = np.asarray(cifar100_train.base_dataset.targets, dtype=np.int64)
typicality = cifar100_safety.typicality_scores

class_idx = CIFAR100_CLASSES.index(class_name)
class_mask = labels == class_idx
class_indices = np.flatnonzero(class_mask)
class_typicality = typicality[class_mask]

rank_order = np.argsort(class_typicality)
most_typical_idx = class_indices[rank_order[-n_extremes:][::-1]]
least_typical_idx = class_indices[rank_order[:n_extremes]]

print(
    f"CIFAR-100 train | class={class_name!r} (n={len(class_indices)}) | "
    f"C-score min/mean/max = "
    f"{float(class_typicality.min()):.4f} / "
    f"{float(class_typicality.mean()):.4f} / "
    f"{float(class_typicality.max()):.4f}"
)

# %%
fig, axes = plt.subplots(2, n_extremes, figsize=(2.4 * n_extremes, 5.5))

for col, global_idx in enumerate(most_typical_idx):
    img = cifar100_train[int(global_idx)]["image"].permute(1, 2, 0).numpy()
    ax = axes[0, col]
    ax.imshow(img)
    ax.set_title(f"{typicality[global_idx]:.4f}", fontsize=9)
    ax.axis("off")

for col, global_idx in enumerate(least_typical_idx):
    img = cifar100_train[int(global_idx)]["image"].permute(1, 2, 0).numpy()
    ax = axes[1, col]
    ax.imshow(img)
    ax.set_title(f"{typicality[global_idx]:.4f}", fontsize=9)
    ax.axis("off")

axes[0, 0].set_ylabel("Most typical", fontsize=11)
axes[1, 0].set_ylabel("Least typical", fontsize=11)
fig.suptitle(
    f"CIFAR-100 {class_name!r}: {n_extremes} highest / lowest C-score within class",
    fontsize=14,
)
fig.tight_layout()
plt.show()

# %%
