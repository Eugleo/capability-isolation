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

    bc_loss_sum = 0.0
    bc_loss_count = 0

    tp_total = 0
    fp_total = 0
    fn_total = 0

    with torch.no_grad():
        for batch in dataset:
            images, _, kinds, kind_labels = batch
            images = images.to(device)
            n = len(images)

            gate_out = gate(images)
            pred = (gate_out >= 0.5).squeeze(1)

            # Batch BCE loss instead of per-sample
            targets = torch.tensor(
                [1.0 if k != "unmarked" else 0.0 for k in kinds],
                device=device,
                dtype=gate_out.dtype,
            ).unsqueeze(1)
            bc_loss_sum += criterion(gate_out, targets).item() * n
            bc_loss_count += n

            # Vectorized masks from kinds/kind_labels
            marked = torch.tensor([k != "unmarked" for k in kinds], device=device)
            known_marked = torch.tensor(
                [
                    k != "unmarked" and kl != "unknown"
                    for k, kl in zip(kinds, kind_labels)
                ],
                device=device,
            )
            known_left = torch.tensor(
                [k == "left" and kl == "left" for k, kl in zip(kinds, kind_labels)],
                device=device,
            )
            known_right = torch.tensor(
                [k == "right" and kl == "right" for k, kl in zip(kinds, kind_labels)],
                device=device,
            )
            unknown_marked = torch.tensor(
                [
                    k != "unmarked" and kl == "unknown"
                    for k, kl in zip(kinds, kind_labels)
                ],
                device=device,
            )
            unknown_left = torch.tensor(
                [k == "left" and kl == "unknown" for k, kl in zip(kinds, kind_labels)],
                device=device,
            )
            unknown_right = torch.tensor(
                [k == "right" and kl == "unknown" for k, kl in zip(kinds, kind_labels)],
                device=device,
            )

            pred_f = pred.float()
            correct_all += (pred == marked).sum().item()
            total_all += n

            tp_total += (pred & marked).sum().item()
            fp_total += (pred & ~marked).sum().item()
            fn_total += ((~pred) & marked).sum().item()

            correct_known_marked += (pred_f * known_marked.float()).sum().item()
            total_known_marked += known_marked.sum().item()
            correct_known_left += (pred_f * known_left.float()).sum().item()
            total_known_left += known_left.sum().item()
            correct_known_right += (pred_f * known_right.float()).sum().item()
            total_known_right += known_right.sum().item()
            correct_unknown_marked += (pred_f * unknown_marked.float()).sum().item()
            total_unknown_marked += unknown_marked.sum().item()
            correct_unknown_left += (pred_f * unknown_left.float()).sum().item()
            total_unknown_left += unknown_left.sum().item()
            correct_unknown_right += (pred_f * unknown_right.float()).sum().item()
            total_unknown_right += unknown_right.sum().item()

    bc_loss = bc_loss_sum / bc_loss_count if bc_loss_count > 0 else 0.0

    tp = tp_total
    fp = fp_total
    fn = fn_total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def _acc(c: int, t: int) -> float:
        return c / t if t > 0 else 0.0

    return {
        "gate/all/accuracy": _acc(correct_all, total_all),
        "gate/marked/precision": precision,
        "gate/marked/recall": recall,
        "gate/marked/known/accuracy": _acc(correct_known_marked, total_known_marked),
        "gate/marked/left/accuracy": _acc(correct_known_left, total_known_left),
        "gate/marked/right/accuracy": _acc(correct_known_right, total_known_right),
        "gate/marked/unknown/accuracy": _acc(
            correct_unknown_marked, total_unknown_marked
        ),
        "gate/left/unknown/accuracy": _acc(correct_unknown_left, total_unknown_left),
        "gate/right/unknown/accuracy": _acc(correct_unknown_right, total_unknown_right),
        "gate/bc_loss": bc_loss,
    }
