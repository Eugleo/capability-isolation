# %%
import matplotlib.pyplot as plt
import numpy as np

from src.cifar.data import CIFAR100, CIFAR100_CLASSES, CIFAR100Safety

# %%
data_root = "data"
cifar100_train = CIFAR100(train=True, root=data_root)
cifar100_safety = CIFAR100Safety.from_cifar100(
    cifar100_train,
    dangerous_classes=set(),
    unknown_classes=set(),
)

# %%
# CIFAR-100: forest (outdoor scene) plus all fine-grained tree species classes.
class_names: tuple[str, ...] = (
    "forest",
    "maple_tree",
    "oak_tree",
    "palm_tree",
    "pine_tree",
    "willow_tree",
)
n_extremes = 5

for name in class_names:
    if name not in CIFAR100_CLASSES:
        raise ValueError(
            f"Unknown CIFAR-100 class {name!r}. "
            f"Pick names from CIFAR100_CLASSES (see src.cifar.data)."
        )

labels = np.asarray(cifar100_train.base_dataset.targets, dtype=np.int64)
typicality = cifar100_safety.typicality_scores

per_class: list[
    tuple[str, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
] = []
for class_name in class_names:
    class_idx = CIFAR100_CLASSES.index(class_name)
    class_mask = labels == class_idx
    class_indices = np.flatnonzero(class_mask)
    class_typicality = typicality[class_mask]
    mean_typ = float(class_typicality.mean())
    rank_order = np.argsort(class_typicality)
    most_typical_idx = class_indices[rank_order[-n_extremes:][::-1]]
    least_typical_idx = class_indices[rank_order[:n_extremes]]
    per_class.append(
        (class_name, mean_typ, class_indices, class_typicality, most_typical_idx, least_typical_idx)
    )

per_class.sort(key=lambda t: t[1], reverse=True)

for class_name, mean_typ, class_indices, class_typicality, _, _ in per_class:
    print(
        f"CIFAR-100 train | class={class_name!r} (n={len(class_indices)}) | "
        f"C-score min/mean/max = "
        f"{float(class_typicality.min()):.4f} / "
        f"{mean_typ:.4f} / "
        f"{float(class_typicality.max()):.4f}"
    )

n_classes = len(per_class)
fig = plt.figure(
    figsize=(2.35 * n_extremes, 1.15 + n_classes * 4.2),
    layout="constrained",
)
subfigs = fig.subfigures(n_classes, 1, squeeze=False, hspace=0.12)

for subfig, (class_name, mean_typ, _, _, most_typical_idx, least_typical_idx) in zip(
    subfigs.flat,
    per_class,
):
    subfig.suptitle(
        f"{class_name}  ·  mean typicality = {mean_typ:.4f}",
        fontsize=11,
        fontweight="medium",
    )
    axes_pair = subfig.subplots(2, n_extremes, squeeze=False)

    for col, global_idx in enumerate(most_typical_idx):
        img = cifar100_train[int(global_idx)]["image"].permute(1, 2, 0).numpy()
        ax = axes_pair[0, col]
        ax.imshow(img)
        ax.set_title(f"{typicality[global_idx]:.4f}", fontsize=9)
        ax.axis("off")

    for col, global_idx in enumerate(least_typical_idx):
        img = cifar100_train[int(global_idx)]["image"].permute(1, 2, 0).numpy()
        ax = axes_pair[1, col]
        ax.imshow(img)
        ax.set_title(f"{typicality[global_idx]:.4f}", fontsize=9)
        ax.axis("off")

    axes_pair[0, 0].set_ylabel("Most typical", fontsize=10)
    axes_pair[1, 0].set_ylabel("Least typical", fontsize=10)

fig.suptitle(
    "CIFAR-100 forest + tree classes: 5 highest / lowest C-score per class "
    "(panels ordered by mean typicality, high → low)",
    fontsize=12,
)
plt.show()

# %%
