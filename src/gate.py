import math
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

Kind = Literal["unmarked", "left", "right"]
KindLabel = Literal["unknown", "unmarked", "left", "right"]


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 7 * 7, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.conv2(h)
        h = h.flatten(1)
        return torch.sigmoid(self.fc(h))


def evaluate_gate(
    gate: Gate,
    dataset: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    gate.eval()
    criterion = nn.BCELoss()

    def _acc(c: float, t: float) -> float:
        return c / t if t > 0 else math.nan

    counts = {
        "all": (0.0, 0.0),
        "unmarked": (0.0, 0.0),
        "unmarked/known": (0.0, 0.0),
        "unmarked/unknown": (0.0, 0.0),
        "marked": (0.0, 0.0),
        "marked/known": (0.0, 0.0),
        "marked/known/left": (0.0, 0.0),
        "marked/known/right": (0.0, 0.0),
        "marked/unknown": (0.0, 0.0),
        "marked/unknown/left": (0.0, 0.0),
        "marked/unknown/right": (0.0, 0.0),
    }
    bc_loss_sum = 0.0
    bc_loss_count = 0.0
    tp_total = 0.0
    fp_total = 0.0
    fn_total = 0.0

    with torch.no_grad():
        for batch in dataset:
            images_BCHW, _, kinds, kind_labels = batch
            images_BCHW = images_BCHW.to(device)
            batch_size = len(images_BCHW)

            gate_out_B1 = gate(images_BCHW)
            pred_marked_B = (gate_out_B1 >= 0.5).squeeze(1)
            pred_unmarked_B = ~pred_marked_B

            targets_B1 = torch.tensor(
                [1.0 if k != "unmarked" else 0.0 for k in kinds],
                device=device,
                dtype=gate_out_B1.dtype,
            ).unsqueeze(1)
            bc_loss_sum += criterion(gate_out_B1, targets_B1).item() * batch_size
            bc_loss_count += batch_size

            is_unmarked_B = torch.tensor(
                [k == "unmarked" for k in kinds], device=device
            ).float()
            is_marked_B = 1.0 - is_unmarked_B
            is_unmarked_known_B = torch.tensor(
                [
                    k == "unmarked" and kl == "unmarked"
                    for k, kl in zip(kinds, kind_labels)
                ],
                device=device,
            ).float()
            is_unmarked_unknown_B = torch.tensor(
                [
                    k == "unmarked" and kl == "unknown"
                    for k, kl in zip(kinds, kind_labels)
                ],
                device=device,
            ).float()
            is_known_marked_B = torch.tensor(
                [
                    k != "unmarked" and kl != "unknown"
                    for k, kl in zip(kinds, kind_labels)
                ],
                device=device,
            ).float()
            is_unknown_marked_B = torch.tensor(
                [
                    k != "unmarked" and kl == "unknown"
                    for k, kl in zip(kinds, kind_labels)
                ],
                device=device,
            ).float()
            is_known_left_B = torch.tensor(
                [k == "left" and kl == "left" for k, kl in zip(kinds, kind_labels)],
                device=device,
            ).float()
            is_known_right_B = torch.tensor(
                [k == "right" and kl == "right" for k, kl in zip(kinds, kind_labels)],
                device=device,
            ).float()
            is_unknown_left_B = torch.tensor(
                [k == "left" and kl == "unknown" for k, kl in zip(kinds, kind_labels)],
                device=device,
            ).float()
            is_unknown_right_B = torch.tensor(
                [k == "right" and kl == "unknown" for k, kl in zip(kinds, kind_labels)],
                device=device,
            ).float()

            is_marked_bool = is_marked_B > 0.5
            tp_total += (pred_marked_B & is_marked_bool).sum().item()
            fp_total += (pred_marked_B & ~is_marked_bool).sum().item()
            fn_total += (pred_unmarked_B & is_marked_bool).sum().item()

            correct_all_B = (pred_marked_B == is_marked_bool).float()
            correct_unmarked_B = pred_unmarked_B.float()
            correct_marked_B = pred_marked_B.float()

            def _update(
                updates: list[tuple[str, torch.Tensor, torch.Tensor]],
            ) -> None:
                for name, correct_B, mask_B in updates:
                    old_c, old_t = counts[name]
                    counts[name] = (
                        old_c + (correct_B * mask_B).sum().item(),
                        old_t + mask_B.sum().item(),
                    )

            _update(
                [
                    ("all", correct_all_B, torch.ones(batch_size, device=device)),
                    ("unmarked", correct_unmarked_B, is_unmarked_B),
                    ("unmarked/known", correct_unmarked_B, is_unmarked_known_B),
                    ("unmarked/unknown", correct_unmarked_B, is_unmarked_unknown_B),
                    ("marked", correct_marked_B, is_marked_B),
                    ("marked/known", correct_marked_B, is_known_marked_B),
                    ("marked/known/left", correct_marked_B, is_known_left_B),
                    ("marked/known/right", correct_marked_B, is_known_right_B),
                    ("marked/unknown", correct_marked_B, is_unknown_marked_B),
                    ("marked/unknown/left", correct_marked_B, is_unknown_left_B),
                    ("marked/unknown/right", correct_marked_B, is_unknown_right_B),
                ]
            )

    bc_loss = bc_loss_sum / bc_loss_count if bc_loss_count > 0 else math.nan
    precision = (
        tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else math.nan
    )
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else math.nan

    all_metrics: dict[str, float] = {}
    for name in counts:
        c, t = counts[name]
        all_metrics[f"gate/{name}/accuracy"] = _acc(c, t)
    all_metrics["gate/marked/precision"] = precision
    all_metrics["gate/marked/recall"] = recall
    all_metrics["gate/bc_loss"] = bc_loss

    priority_metrics = [
        "gate/unmarked/accuracy",
        "gate/marked/known/left/accuracy",
        "gate/marked/unknown/right/accuracy",
    ]
    result: dict[str, float] = {}
    for key in priority_metrics:
        if key in all_metrics:
            result[f"@/{key}"] = all_metrics.pop(key)
    result.update(all_metrics)
    return result
