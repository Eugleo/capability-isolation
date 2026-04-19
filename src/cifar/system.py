from pathlib import Path
from typing import Optional, TypedDict

import torch
import torch.nn as nn

from src.cifar.train_resnet import build_cifar_resnet18
from src.cifar.train_safety_classifier import build_binary_cifar_resnet18


class NaiveSystemOutput(TypedDict):
    prediction: torch.Tensor
    safe_logits: torch.Tensor
    dangerous_logits: torch.Tensor
    safe_probs: torch.Tensor
    dangerous_probs: torch.Tensor
    computed_gate: torch.Tensor
    used_gate: torch.Tensor


class NaiveSystem(nn.Module):
    def __init__(
        self,
        gate: nn.Module,
        model_safe: nn.Module,
        model_dangerous: nn.Module,
        num_classes: int,
    ):
        super().__init__()
        self.gate = gate
        self.model_safe = model_safe
        self.model_dangerous = model_dangerous
        self.num_classes = int(num_classes)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        data = {
            "num_classes": self.num_classes,
            "gate_state_dict": self.gate.state_dict(),
            "model_safe_state_dict": self.model_safe.state_dict(),
            "model_dangerous_state_dict": self.model_dangerous.state_dict(),
        }
        torch.save(data, path)

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        device: Optional[torch.device] = None,
    ) -> "NaiveSystem":
        path = Path(path)
        map_location = device if device is not None else "cpu"
        data = torch.load(path, map_location=map_location, weights_only=True)
        num_classes = int(data["num_classes"])
        gate = build_binary_cifar_resnet18()
        gate.load_state_dict(data["gate_state_dict"])
        model_safe = build_cifar_resnet18(num_classes=num_classes)
        model_safe.load_state_dict(data["model_safe_state_dict"])
        model_dangerous = build_cifar_resnet18(num_classes=num_classes)
        model_dangerous.load_state_dict(data["model_dangerous_state_dict"])
        if device is not None:
            gate = gate.to(device)
            model_safe = model_safe.to(device)
            model_dangerous = model_dangerous.to(device)
        return cls(
            gate=gate,
            model_safe=model_safe,
            model_dangerous=model_dangerous,
            num_classes=num_classes,
        )

    def forward(
        self,
        images_BCHW: torch.Tensor,
        *,
        force_gate_to_zero_B: Optional[torch.Tensor] = None,
        force_gate_to_one_B: Optional[torch.Tensor] = None,
    ) -> NaiveSystemOutput:
        safe_logits_BC = self.model_safe(images_BCHW)
        dangerous_logits_BC = self.model_dangerous(images_BCHW)

        gate_logits_B2 = self.gate(images_BCHW)
        computed_gate_B = torch.softmax(gate_logits_B2, dim=-1)[:, 1]

        used_gate_B = computed_gate_B
        if force_gate_to_zero_B is not None:
            used_gate_B = torch.where(
                force_gate_to_zero_B.to(device=used_gate_B.device, dtype=torch.bool),
                torch.zeros_like(used_gate_B),
                used_gate_B,
            )
        if force_gate_to_one_B is not None:
            used_gate_B = torch.where(
                force_gate_to_one_B.to(device=used_gate_B.device, dtype=torch.bool),
                torch.ones_like(used_gate_B),
                used_gate_B,
            )

        probs_safe_BC = torch.softmax(safe_logits_BC, dim=-1)
        probs_dangerous_BC = torch.softmax(dangerous_logits_BC, dim=-1)
        used_gate_B1 = used_gate_B.unsqueeze(1)
        prediction_BC = (
            (1 - used_gate_B1) * probs_safe_BC + used_gate_B1 * probs_dangerous_BC
        )

        return {
            "prediction": prediction_BC,
            "safe_logits": safe_logits_BC,
            "dangerous_logits": dangerous_logits_BC,
            "safe_probs": probs_safe_BC,
            "dangerous_probs": probs_dangerous_BC,
            "computed_gate": computed_gate_B,
            "used_gate": used_gate_B,
        }
