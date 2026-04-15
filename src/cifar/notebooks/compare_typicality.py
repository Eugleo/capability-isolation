# %%
import matplotlib.pyplot as plt
import numpy as np

from src.cifar.data import (
    CIFAR100,
    CIFAR100_CLASSES,
    CIFAR100Safety,
)

# %%
# Safety labeling hyperparameters do not affect typicality itself,
# but are required to construct CIFAR10Safety.
dangerous_class = "crab"
safe_known = "random"
dangerous_known = "random"
known_percent = 50.0
seed = 42

cifar100_train = CIFAR100(train=True, root="data")
cifar100_safety = CIFAR100Safety.from_cifar100(
    cifar100_train,
    dangerous_classes={dangerous_class},
    safe_known=safe_known,
    dangerous_known=dangerous_known,
    known_percent=known_percent,
    seed=seed,
)

labels = np.asarray(cifar100_train.base_dataset.targets, dtype=np.int64)
typicality = cifar100_safety.typicality_scores

print(f"Loaded CIFAR-100 safety dataset with {len(cifar100_safety)} samples.")

# %%
class_mean_typicality: dict[str, float] = {}
for class_idx, class_name in enumerate(CIFAR100_CLASSES):
    class_scores = typicality[labels == class_idx]
    class_mean_typicality[class_name] = float(np.mean(class_scores))

class_mean_typicality = list(
    sorted(class_mean_typicality.items(), key=lambda x: x[1], reverse=True)
)
top10_typical = class_mean_typicality[:10]
least10_typical = class_mean_typicality[-10:]

for class_name, mean_score in class_mean_typicality:
    print(f"{class_name:>10}: mean typicality = {mean_score:.6f}")

# %%
fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)

for plot_idx, (class_name, _) in enumerate(least10_typical):
    class_idx = CIFAR100_CLASSES.index(class_name)
    ax = axes.flat[plot_idx]
    class_scores = typicality[labels == class_idx]
    ax.hist(class_scores, bins=30, alpha=0.85)
    ax.set_title(class_name)
    ax.set_xlabel("Typicality score")
    ax.set_ylabel("Count")

fig.suptitle("CIFAR-10 Typicality Histograms by Class", fontsize=14)
fig.tight_layout()
plt.show()

# %%
class_name = "seal"
n_show = 8

class_idx = CIFAR100_CLASSES.index(class_name)
class_mask = labels == class_idx
class_indices = np.where(class_mask)[0]
class_typicality = typicality[class_mask]

rank_order = np.argsort(class_typicality)
most_typical_idx = class_indices[rank_order[-n_show:][::-1]]
least_typical_idx = class_indices[rank_order[:n_show]]

fig, axes = plt.subplots(2, n_show, figsize=(2.4 * n_show, 5.5))

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
fig.suptitle(f"'{class_name}': most vs least typical images", fontsize=14)
fig.tight_layout()
plt.show()

# %%
