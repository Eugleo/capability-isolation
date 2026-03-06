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

    correct_all = 0
    total_all = 0
    correct_known_marked = 0
    total_known_marked = 0
    correct_known_left = 0
    total_known_left = 0
    correct_known_right = 0
    total_known_right = 0
    correct_unknown_marked = 0
    total_unknown_marked = 0
    correct_unknown_left = 0
    total_unknown_left = 0
    correct_unknown_right = 0
    total_unknown_right = 0
    correct_unmarked = 0
    total_unmarked = 0
    correct_unmarked_known = 0
    total_unmarked_known = 0
    correct_unmarked_unknown = 0
    total_unmarked_unknown = 0

    bc_loss_sum = 0.0
    bc_loss_count = 0

    tp_total = 0
    fp_total = 0
    fn_total = 0

    with torch.no_grad():
        for batch in dataset:
            images_BCHW, _, kinds, kind_labels = batch
            images_BCHW = images_BCHW.to(device)
            batch_size = len(images_BCHW)

            gate_out_B1 = gate(images_BCHW)
            pred_marked_B = (gate_out_B1 >= 0.5).squeeze(1)

            targets_B1 = torch.tensor(
                [1.0 if k != "unmarked" else 0.0 for k in kinds],
                device=device,
                dtype=gate_out_B1.dtype,
            ).unsqueeze(1)
            bc_loss_sum += criterion(gate_out_B1, targets_B1).item() * batch_size
            bc_loss_count += batch_size

            is_marked_B = torch.tensor([k != "unmarked" for k in kinds], device=device)
            is_unmarked_B = ~is_marked_B
            is_known_marked_B = torch.tensor(
                [
                    k != "unmarked" and kl != "unknown"
                    for k, kl in zip(kinds, kind_labels)
                ],
                device=device,
            )
            is_known_left_B = torch.tensor(
                [k == "left" and kl == "left" for k, kl in zip(kinds, kind_labels)],
                device=device,
            )
            is_known_right_B = torch.tensor(
                [k == "right" and kl == "right" for k, kl in zip(kinds, kind_labels)],
                device=device,
            )
            is_unknown_marked_B = torch.tensor(
                [
                    k != "unmarked" and kl == "unknown"
                    for k, kl in zip(kinds, kind_labels)
                ],
                device=device,
            )
            is_unknown_left_B = torch.tensor(
                [k == "left" and kl == "unknown" for k, kl in zip(kinds, kind_labels)],
                device=device,
            )
            is_unknown_right_B = torch.tensor(
                [k == "right" and kl == "unknown" for k, kl in zip(kinds, kind_labels)],
                device=device,
            )
            is_unmarked_known_B = torch.tensor(
                [
                    k == "unmarked" and kl == "unmarked"
                    for k, kl in zip(kinds, kind_labels)
                ],
                device=device,
            )
            is_unmarked_unknown_B = torch.tensor(
                [
                    k == "unmarked" and kl == "unknown"
                    for k, kl in zip(kinds, kind_labels)
                ],
                device=device,
            )

            pred_marked_f_B = pred_marked_B.float()
            correct_all += (pred_marked_B == is_marked_B).sum().item()
            total_all += batch_size

            tp_total += (pred_marked_B & is_marked_B).sum().item()
            fp_total += (pred_marked_B & ~is_marked_B).sum().item()
            fn_total += ((~pred_marked_B) & is_marked_B).sum().item()

            correct_known_marked += (
                (pred_marked_f_B * is_known_marked_B.float()).sum().item()
            )
            total_known_marked += is_known_marked_B.sum().item()
            correct_known_left += (
                (pred_marked_f_B * is_known_left_B.float()).sum().item()
            )
            total_known_left += is_known_left_B.sum().item()
            correct_known_right += (
                (pred_marked_f_B * is_known_right_B.float()).sum().item()
            )
            total_known_right += is_known_right_B.sum().item()
            correct_unknown_marked += (
                (pred_marked_f_B * is_unknown_marked_B.float()).sum().item()
            )
            total_unknown_marked += is_unknown_marked_B.sum().item()
            correct_unknown_left += (
                (pred_marked_f_B * is_unknown_left_B.float()).sum().item()
            )
            total_unknown_left += is_unknown_left_B.sum().item()
            correct_unknown_right += (
                (pred_marked_f_B * is_unknown_right_B.float()).sum().item()
            )
            total_unknown_right += is_unknown_right_B.sum().item()

            pred_unmarked_B = ~pred_marked_B
            correct_unmarked += (pred_unmarked_B & is_unmarked_B).sum().item()
            total_unmarked += is_unmarked_B.sum().item()
            correct_unmarked_known += (
                (pred_unmarked_B.float() * is_unmarked_known_B.float()).sum().item()
            )
            total_unmarked_known += is_unmarked_known_B.sum().item()
            correct_unmarked_unknown += (
                (pred_unmarked_B.float() * is_unmarked_unknown_B.float()).sum().item()
            )
            total_unmarked_unknown += is_unmarked_unknown_B.sum().item()

    bc_loss = bc_loss_sum / bc_loss_count if bc_loss_count > 0 else math.nan

    tp = tp_total
    fp = fp_total
    fn = fn_total
    precision = tp / (tp + fp) if (tp + fp) > 0 else math.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else math.nan

    def _acc(c: int, t: int) -> float:
        return c / t if t > 0 else math.nan

    return {
        "gate/all/accuracy": _acc(correct_all, total_all),
        "gate/marked/precision": precision,
        "gate/marked/recall": recall,
        "gate/unmarked/accuracy": _acc(correct_unmarked, total_unmarked),
        "gate/unmarked/known/accuracy": _acc(
            correct_unmarked_known, total_unmarked_known
        ),
        "gate/unmarked/unknown/accuracy": _acc(
            correct_unmarked_unknown, total_unmarked_unknown
        ),
        "gate/marked/known/accuracy": _acc(correct_known_marked, total_known_marked),
        "gate/marked/known/left/accuracy": _acc(correct_known_left, total_known_left),
        "gate/marked/known/right/accuracy": _acc(
            correct_known_right, total_known_right
        ),
        "gate/marked/unknown/accuracy": _acc(
            correct_unknown_marked, total_unknown_marked
        ),
        "gate/marked/unknown/left/accuracy": _acc(
            correct_unknown_left, total_unknown_left
        ),
        "gate/marked/unknown/right/accuracy": _acc(
            correct_unknown_right, total_unknown_right
        ),
        "gate/bc_loss": bc_loss,
    }
