# %%
"""Grid of all k-dang (known-label dangerous) train images for CIFAR-100 safety setup."""

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.cifar.data import CIFAR100, CIFAR100_CLASSES, CIFAR100Safety
from src.cifar.train_resnet import get_eval_transform
from src.cifar.unlearn import UnlearnConfig

# %%
cfg = UnlearnConfig()

cifar100_train = CIFAR100(
    train=True,
    root=cfg.data_root,
    transform=get_eval_transform(),
)
known = set(cfg.known_classes) if cfg.known_classes else set(CIFAR100_CLASSES)
unknown = set(CIFAR100_CLASSES) - known
safety: CIFAR100Safety = CIFAR100Safety.from_cifar100(
    cifar100_train,
    dangerous_classes=set(cfg.dangerous_classes),
    unknown_classes=unknown,
)

# %%
kdang_mask = safety.kind_arr == "k-dang"
kdang_idx = np.flatnonzero(kdang_mask)
scores = safety.typicality_scores[kdang_idx]
order = np.argsort(scores)[::-1]
kdang_idx = kdang_idx[order]

n = len(kdang_idx)
print(
    f"CIFAR100Safety: dangerous={cfg.dangerous_classes!r}, "
    f"known={sorted(known)!r}, seed={cfg.seed}"
)
print(f"k-dang count: {n}")

mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)


def denorm_chw(t: torch.Tensor) -> np.ndarray:
    x = (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return x


# %%
if n == 0:
    print("No k-dang samples (check known_percent and dangerous class).")
else:
    ncols = min(10, max(1, int(np.ceil(np.sqrt(n)))))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.3 * ncols, 2.5 * nrows),
        squeeze=False,
    )
    for ax in axes.flat:
        ax.axis("off")

    for i, global_idx in enumerate(kdang_idx):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        item = safety[int(global_idx)]
        img = denorm_chw(item["image"])
        typ = float(item["typicality_score"])
        label_name = CIFAR100_CLASSES[item["label"]]
        ax.imshow(img)
        ax.set_title(f"{typ:.4f}\n{label_name}", fontsize=9)

    fig.suptitle(
        f"k-dang images (n={n}): typicality (C-score), fine label",
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()

# %%
