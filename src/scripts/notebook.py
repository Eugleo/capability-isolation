# %%
import matplotlib.pyplot as plt
import numpy as np

from src.config import Config
from src.data import MarkedMNIST

# %%
config = Config()

dataset = MarkedMNIST(
    train=True,
    kind_fraction=config.kind_fraction,
    seed=config.seed,
)

dataset.print_summary("Train dataset")

# %%
rng = np.random.default_rng(config.seed)
indices = rng.choice(len(dataset), size=8, replace=False)

fig, axes = plt.subplots(2, 4, figsize=(10, 6))

for ax, idx in zip(axes.flat, indices, strict=True):
    item = dataset[int(idx)]
    ax.imshow(item["image"].squeeze(0), cmap="gray")
    ax.set_title(
        f"label={item['label']}\n{item['kind']}\nknown={item['is_known']}",
        fontsize=9,
    )
    ax.axis("off")

fig.tight_layout()
plt.show()

# %%
