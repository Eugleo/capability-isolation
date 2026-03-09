from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.group1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.group2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.group3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 64),
            nn.ReLU(),
        )
        self.group4 = nn.Linear(64, 10)
        self.groups = [self.group1, self.group2, self.group3, self.group4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for group in self.groups:
            x = group(x)
        return x

    def save(self, path: Path | str) -> None:
        """Save the classifier state dict to a checkpoint file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.state_dict()}, path)

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        device: Optional[torch.device] = None,
    ) -> "Classifier":
        """Load a Classifier from a checkpoint file or directory.

        If path is a directory, loads from path/model.pt.
        If path is a file, loads directly.
        If path is a .pt file that does not exist, tries path.parent/path.stem/model.pt
        (e.g. classifier_all.pt -> classifier_all/model.pt).
        """
        path = Path(path)
        if path.is_dir():
            load_path = path / "model.pt"
        elif path.suffix == ".pt" and not path.exists():
            dir_path = path.parent / path.stem
            if dir_path.exists():
                load_path = dir_path / "model.pt"
            else:
                load_path = path
        else:
            load_path = path
        map_location = device if device is not None else "cpu"
        data = torch.load(load_path, map_location=map_location, weights_only=True)
        model = cls()
        model.load_state_dict(data["model_state_dict"])
        if device is not None:
            model = model.to(device)
        return model


def evaluate_classifier(
    classifier: nn.Module,
    dataset: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    classifier.eval()

    correct_all = 0
    total_all = 0
    correct_unmarked = 0
    total_unmarked = 0
    correct_marked = 0
    total_marked = 0
    correct_left = 0
    total_left = 0
    correct_right = 0
    total_right = 0

    with torch.no_grad():
        for batch in dataset:
            images, labels, kinds, kind_labels = batch
            images = images.to(device)
            labels = labels.to(device)
            logits = classifier(images)
            _, predicted = torch.max(logits.data, 1)

            n = len(images)
            pred_correct = predicted == labels

            unmarked = [k == "unmarked" for k in kinds]
            marked = [k != "unmarked" for k in kinds]
            left = [k == "left" for k in kinds]
            right = [k == "right" for k in kinds]

            for i in range(n):
                total_all += 1
                if pred_correct[i]:
                    correct_all += 1
                if unmarked[i]:
                    total_unmarked += 1
                    if pred_correct[i]:
                        correct_unmarked += 1
                if marked[i]:
                    total_marked += 1
                    if pred_correct[i]:
                        correct_marked += 1
                if left[i]:
                    total_left += 1
                    if pred_correct[i]:
                        correct_left += 1
                if right[i]:
                    total_right += 1
                    if pred_correct[i]:
                        correct_right += 1

    def _acc(c: int, t: int) -> float:
        return c / t if t > 0 else float("nan")

    return {
        "classifier/all/accuracy": _acc(correct_all, total_all),
        "classifier/unmarked/accuracy": _acc(correct_unmarked, total_unmarked),
        "classifier/marked/accuracy": _acc(correct_marked, total_marked),
        "classifier/marked/left/accuracy": _acc(correct_left, total_left),
        "classifier/marked/right/accuracy": _acc(correct_right, total_right),
    }
