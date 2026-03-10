# %%
import matplotlib.pyplot as plt
import numpy as np

from src.config import Config
from src.data import DISPLAY_KIND_NAMES, MarkedMNIST

# %%
config = Config()

train_dataset = MarkedMNIST(
    train=True,
    kind_fraction=config.kind_fraction,
    seed=config.seed,
)
test_dataset = MarkedMNIST(
    train=False,
    kind_fraction=config.kind_fraction,
    seed=config.seed + 1,
)

# %%
train_dataset.print_summary("Train dataset")
test_dataset.print_summary("Test dataset")

# %%
TARGET_EXAMPLES = [
    ("unmarked", "known"),
    ("unmarked", "unknown"),
    ("left", "known"),
    ("left", "unknown"),
    ("right", "unknown"),
]


def collect_examples(
    dataset: MarkedMNIST,
    categories: list[tuple[str, str]],
) -> list[tuple[int, np.ndarray, str]]:
    examples: list[tuple[int, np.ndarray, str]] = []
    for kind, status in categories:
        for idx in range(len(dataset)):
            image, label, sample_kind, kind_label = dataset[idx]
            sample_status = "unknown" if kind_label == "unknown" else "known"
            if sample_kind != kind or sample_status != status:
                continue

            original_label = int(dataset.base_dataset.targets[idx])
            title = (
                f"{DISPLAY_KIND_NAMES[sample_kind]} / {sample_status}\n"
                f"idx={idx}, original={original_label}, label={label}"
            )
            examples.append((idx, image.squeeze(0).numpy(), title))
            break
    return examples


def show_examples(dataset: MarkedMNIST, name: str) -> None:
    examples = collect_examples(dataset, TARGET_EXAMPLES)
    if not examples:
        print(f"No matching examples found for {name}.")
        return

    fig, axes = plt.subplots(1, len(examples), figsize=(3 * len(examples), 3.5))
    if len(examples) == 1:
        axes = [axes]

    for ax, (_, image, title) in zip(axes, examples):
        ax.imshow(image, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(name)
    fig.tight_layout()
    plt.show()


show_examples(train_dataset, "Train dataset examples")
show_examples(test_dataset, "Test dataset examples")

# %%
