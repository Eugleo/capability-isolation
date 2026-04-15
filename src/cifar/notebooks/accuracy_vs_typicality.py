# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.cifar.data import (
    CIFAR100,
    CIFAR100_CLASSES,
    CIFAR100Safety,
)
from src.cifar.train_resnet import build_cifar_resnet18, get_eval_transform
from src.utils import get_device

# %%
device = get_device()

model = build_cifar_resnet18(num_classes=100).to(device)
ckpt = torch.load(
    "checkpoints/cifar100/train_resnet.pt", map_location=device, weights_only=True
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded model on {device}")

# %%
cifar100_train = CIFAR100(train=True, root="data")
cifar100_safety = CIFAR100Safety.from_cifar100(
    cifar100_train,
    dangerous_classes={"man", "boy"},
    unknown_classes={"girl", "boy"},
)

train_labels = np.asarray(cifar100_train.base_dataset.targets, dtype=np.int64)
typicality = cifar100_safety.typicality_scores

mean_typicality_per_class = np.array(
    [
        float(np.mean(typicality[train_labels == ci]))
        for ci in range(len(CIFAR100_CLASSES))
    ]
)

# %%
N = 1
test_dataset = CIFAR100(train=False, root="data", transform=get_eval_transform())
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

class_correct = np.zeros(len(CIFAR100_CLASSES), dtype=np.int64)
class_total = np.zeros(len(CIFAR100_CLASSES), dtype=np.int64)

with torch.no_grad():
    for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        topk_preds = model(images).topk(N, dim=1).indices
        hits = topk_preds.eq(labels.unsqueeze(1)).any(dim=1)
        for ci in range(len(CIFAR100_CLASSES)):
            mask = labels == ci
            class_correct[ci] += hits[mask].sum().item()
            class_total[ci] += mask.sum().item()

topn_acc_per_class = class_correct / np.maximum(class_total, 1)
print(f"Overall top-{N} accuracy: {class_correct.sum() / class_total.sum():.2%}")

# %%
fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(mean_typicality_per_class, topn_acc_per_class * 100, s=24, zorder=3)

for ci, name in enumerate(CIFAR100_CLASSES):
    ax.annotate(
        name,
        (mean_typicality_per_class[ci], topn_acc_per_class[ci] * 100),
        fontsize=6,
        alpha=0.8,
        textcoords="offset points",
        xytext=(4, 3),
    )

ax.set_xlabel("Mean typicality (C-score)")
ax.set_ylabel(f"Top-{N} accuracy on test set (%)")
ax.set_title(f"CIFAR-100 per-class top-{N} accuracy vs. mean typicality")
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
plt.show()

# %%
