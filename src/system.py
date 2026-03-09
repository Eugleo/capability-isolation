import math
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.classifier import Classifier
from src.config import Config
from src.data import get_dataloaders
from src.gate import Gate
from src.utils import format_metric_value, get_device, set_seed


class GatedSystem(nn.Module):
    def __init__(self, gate: Gate, model_safe: Classifier, model_unsafe: Classifier):
        super().__init__()
        self.gate = gate
        self.model_safe = model_safe
        self.model_unsafe = model_unsafe

    def save(
        self,
        path: Path | str,
        *,
        gate_metrics: Optional[dict[str, float]] = None,
        system_metrics: Optional[dict[str, float]] = None,
    ) -> None:
        """Save the full system to a checkpoint file."""
        path = Path(path)
        data = {
            "gate_state_dict": self.gate.state_dict(),
            "model_safe_state_dict": self.model_safe.state_dict(),
            "model_unsafe_state_dict": self.model_unsafe.state_dict(),
        }
        if gate_metrics is not None:
            data["gate_metrics"] = gate_metrics
        if system_metrics is not None:
            data["system_metrics"] = system_metrics
        torch.save(data, path)

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        device: Optional[torch.device] = None,
    ) -> "GatedSystem":
        """Load a GatedSystem from a checkpoint file."""
        path = Path(path)
        map_location = device if device is not None else "cpu"
        data = torch.load(path, map_location=map_location, weights_only=True)
        gate = Gate()
        gate.load_state_dict(data["gate_state_dict"])
        model_safe = Classifier()
        model_safe.load_state_dict(data["model_safe_state_dict"])
        model_unsafe = Classifier()
        model_unsafe.load_state_dict(data["model_unsafe_state_dict"])
        if device is not None:
            gate = gate.to(device)
            model_safe = model_safe.to(device)
            model_unsafe = model_unsafe.to(device)
        return cls(gate=gate, model_safe=model_safe, model_unsafe=model_unsafe)

    def forward(
        self,
        images_BCHW: torch.Tensor,
        kind_labels: Optional[
            list[Literal["unknown", "unmarked", "left", "right"]]
        ] = None,
        is_unsafe_allowed: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_safe_BC = self.model_safe(images_BCHW)
        logits_unsafe_BC = self.model_unsafe(images_BCHW)
        probs_safe_BC = torch.softmax(logits_safe_BC, dim=-1)
        probs_unsafe_BC = torch.softmax(logits_unsafe_BC, dim=-1)

        gate_computed_B1 = self.gate(images_BCHW)

        if self.training and kind_labels is not None:
            # Override based on kind_label: unmarked -> 0, marked (left/right) -> 1, unknown -> keep
            device = images_BCHW.device
            dtype = images_BCHW.dtype
            is_unmarked_B1 = torch.tensor(
                [kl == "unmarked" for kl in kind_labels], device=device
            ).unsqueeze(1)
            is_marked_B1 = torch.tensor(
                [kl in ("left", "right") for kl in kind_labels], device=device
            ).unsqueeze(1)
            is_unknown_B1 = ~(is_unmarked_B1 | is_marked_B1)
            gate_used_B1 = (
                is_unmarked_B1.to(dtype) * 0.0
                + is_marked_B1.to(dtype) * 1.0
                + is_unknown_B1.to(dtype) * gate_computed_B1
            )
        else:
            # Eval mode: threshold at 0.5
            gate_used_B1 = (gate_computed_B1 > 0.5).to(images_BCHW.dtype)

        if not is_unsafe_allowed:
            gate_used_B1 = torch.zeros_like(gate_used_B1)

        probs_BC = (1 - gate_used_B1) * probs_safe_BC + gate_used_B1 * probs_unsafe_BC

        return probs_BC, gate_computed_B1, gate_used_B1


def evaluate_gated_system(
    system: GatedSystem,
    dataset: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    system.eval()

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
    safe_only_counts = {k: (0.0, 0.0) for k in counts}

    with torch.no_grad():
        for batch in dataset:
            images_BCHW, labels_B, kinds, kind_labels = batch
            images_BCHW = images_BCHW.to(device)
            labels_B = labels_B.to(device)

            probs_BC, _, _ = system(images_BCHW, is_unsafe_allowed=True)
            probs_safe_only_BC, _, _ = system(images_BCHW, is_unsafe_allowed=False)
            pred_B = probs_BC.argmax(dim=-1)
            pred_safe_only_B = probs_safe_only_BC.argmax(dim=-1)
            correct_B = (pred_B == labels_B).float()
            correct_safe_only_B = (pred_safe_only_B == labels_B).float()

            is_unmarked_B = torch.tensor(
                [k == "unmarked" for k in kinds], device=device
            ).float()
            is_marked_B = 1.0 - is_unmarked_B
            is_known_unmarked_B = torch.tensor(
                [
                    k == "unmarked" and kl == "unmarked"
                    for k, kl in zip(kinds, kind_labels)
                ],
                device=device,
            ).float()
            is_unknown_unmarked_B = torch.tensor(
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

            def _update(
                c_dict: dict,
                correct: torch.Tensor,
                masks: list[tuple[str, torch.Tensor]],
            ) -> None:
                for name, mask_B in masks:
                    old_c, old_t = c_dict[name]
                    c_dict[name] = (
                        old_c + (correct * mask_B).sum().item(),
                        old_t + mask_B.sum().item(),
                    )

            masks_all = [
                ("all", torch.ones_like(correct_B)),
                ("unmarked", is_unmarked_B),
                ("unmarked/known", is_known_unmarked_B),
                ("unmarked/unknown", is_unknown_unmarked_B),
                ("marked", is_marked_B),
                ("marked/known", is_known_marked_B),
                ("marked/known/left", is_known_left_B),
                ("marked/known/right", is_known_right_B),
                ("marked/unknown", is_unknown_marked_B),
                ("marked/unknown/left", is_unknown_left_B),
                ("marked/unknown/right", is_unknown_right_B),
            ]

            _update(counts, correct_B, masks_all)
            _update(safe_only_counts, correct_safe_only_B, masks_all)

    priority_metrics = ["system/all", "safe_only/unmarked", "safe_only/marked"]

    all_metrics: dict[str, float] = {}
    for name in counts:
        c, t = counts[name]
        all_metrics[f"system/{name}/accuracy"] = _acc(c, t)
    for name in safe_only_counts:
        c, t = safe_only_counts[name]
        all_metrics[f"safe_only/{name}/accuracy"] = _acc(c, t)

    result: dict[str, float] = {}
    for base in priority_metrics:
        key = f"{base}/accuracy"
        if key in all_metrics:
            result[f"@/{key}"] = all_metrics.pop(key)
    result.update(all_metrics)
    return result


def main() -> None:
    config = Config()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    checkpoint_dir = Path(config.checkpoint_dir)
    system_path = checkpoint_dir / "system.pt"
    if system_path.exists():
        system = GatedSystem.load(system_path, device=device)
    else:
        gate = Gate().to(device)
        gate.load_state_dict(
            torch.load(
                checkpoint_dir / "gate_known.pt",
                map_location=device,
                weights_only=True,
            )["model_state_dict"]
        )
        model_safe = Classifier.load(checkpoint_dir / "classifier_pos=u_neg=m", device=device)
        model_unsafe = Classifier.load(checkpoint_dir / "classifier_all", device=device)
        system = GatedSystem(
            gate=gate, model_safe=model_safe, model_unsafe=model_unsafe
        )

    _, test_loader = get_dataloaders(
        known_kind_fraction=config.known_kind_fraction,
        unknown_kind_fraction=config.unknown_kind_fraction,
        seed=config.seed,
        batch_size=config.classifier_batch_size,
    )

    print("\n" + "=" * 60)
    print("GatedSystem evaluation")
    print("=" * 60)
    metrics = evaluate_gated_system(system, test_loader, device)
    for key, value in metrics.items():
        print(f"  {key}: {format_metric_value(key, value)}")


if __name__ == "__main__":
    main()
