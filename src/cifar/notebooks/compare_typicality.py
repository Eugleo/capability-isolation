# %%
import matplotlib.pyplot as plt
import numpy as np

from src.cifar.data import CIFAR10, CIFAR10_CLASSES, CIFAR10Safety

# %%
# Safety labeling hyperparameters do not affect typicality itself,
# but are required to construct CIFAR10Safety.
dangerous_class = "cat"
safe_known = "random"
dangerous_known = "random"
known_percent = 50.0
seed = 42

cifar10_train = CIFAR10(train=True, root="data")
cifar10_safety = CIFAR10Safety.from_cifar10(
    cifar10_train,
    dangerous_class=dangerous_class,
    safe_known=safe_known,
    dangerous_known=dangerous_known,
    known_percent=known_percent,
    seed=seed,
)

labels = np.asarray(cifar10_train.base_dataset.targets, dtype=np.int64)
typicality = cifar10_safety.typicality_scores

print(f"Loaded CIFAR-10 safety dataset with {len(cifar10_safety)} samples.")

# %%
class_mean_typicality: dict[str, float] = {}
for class_idx, class_name in enumerate(CIFAR10_CLASSES):
    class_scores = typicality[labels == class_idx]
    class_mean_typicality[class_name] = float(np.mean(class_scores))

for class_name, mean_score in class_mean_typicality.items():
    print(f"{class_name:>10}: mean typicality = {mean_score:.6f}")

# %%
fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)

for class_idx, class_name in enumerate(CIFAR10_CLASSES):
    ax = axes.flat[class_idx]
    class_scores = typicality[labels == class_idx]
    ax.hist(class_scores, bins=30, alpha=0.85)
    ax.set_title(class_name)
    ax.set_xlabel("Typicality score")
    ax.set_ylabel("Count")

fig.suptitle("CIFAR-10 Typicality Histograms by Class", fontsize=14)
fig.tight_layout()
plt.show()

# %%
percentiles = np.array([1, 5, 10, 25, 50], dtype=np.float64)
class_percentile_auc: dict[str, np.ndarray] = {}

# Use shared bins for comparable histogram-based AUC across classes.
bins = np.linspace(float(np.min(typicality)), float(np.max(typicality)), 101)
bin_widths = np.diff(bins)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

for class_idx, class_name in enumerate(CIFAR10_CLASSES):
    class_scores = typicality[labels == class_idx]
    density, _ = np.histogram(class_scores, bins=bins, density=True)
    percentile_thresholds = np.percentile(class_scores, percentiles)

    auc_values: list[float] = []
    for threshold in percentile_thresholds:
        mask = bin_centers <= threshold
        auc = float(np.sum(density[mask] * bin_widths[mask]))
        auc_values.append(auc)

    class_percentile_auc[class_name] = np.asarray(auc_values, dtype=np.float64)

for class_name, auc_values in class_percentile_auc.items():
    joined = ", ".join(
        f"p{int(p)}={auc:.4f}" for p, auc in zip(percentiles, auc_values, strict=True)
    )
    print(f"{class_name:>10}: {joined}")

fig, ax = plt.subplots(figsize=(10, 6))
for class_name in CIFAR10_CLASSES:
    ax.plot(
        percentiles,
        class_percentile_auc[class_name],
        marker="o",
        linewidth=1.8,
        label=class_name,
    )

ax.set_title("Histogram AUC up to Typicality Percentile by Class")
ax.set_xlabel("Percentile")
ax.set_ylabel("Area under histogram curve")
ax.set_xticks(percentiles)
ax.grid(alpha=0.3)
ax.legend(ncol=2, fontsize=9)
fig.tight_layout()
plt.show()

# %%
