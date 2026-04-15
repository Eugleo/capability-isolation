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
