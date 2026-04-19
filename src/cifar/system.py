from pathlib import Path
from typing import Optional, TypedDict

import torch
import torch.nn as nn

from src.cifar.train_resnet import build_cifar_resnet18
from src.cifar.train_safety_classifier import build_binary_cifar_resnet18


class NaiveSystemOutput(TypedDict):
    system_output: torch.Tensor
    system_output_detached: torch.Tensor
    safe_model_output: torch.Tensor
    dangerous_model_output: torch.Tensor
    safe_logits: torch.Tensor
    dangerous_logits: torch.Tensor
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

    def forward(self, images_BCHW: torch.Tensor) -> NaiveSystemOutput:
        safe_logits_BC = self.model_safe(images_BCHW)
        dangerous_logits_BC = self.model_dangerous(images_BCHW)
        safe_probs_BC = torch.softmax(safe_logits_BC, dim=-1)
        dangerous_probs_BC = torch.softmax(dangerous_logits_BC, dim=-1)

        gate_logits_B2 = self.gate(images_BCHW)
        computed_gate_B = torch.softmax(gate_logits_B2, dim=-1)[:, 1]
        used_gate_B = computed_gate_B
        used_gate_B1 = used_gate_B.unsqueeze(1)

        system_output_BC = (
            (1 - used_gate_B1) * safe_probs_BC + used_gate_B1 * dangerous_probs_BC
        )
        # Same mixture, but with the two models' contributions detached so that
        # any loss computed from this output only propagates gradients into the
        # gate.
        system_output_detached_BC = (
            (1 - used_gate_B1) * safe_probs_BC.detach()
            + used_gate_B1 * dangerous_probs_BC.detach()
        )

        return {
            "system_output": system_output_BC,
            "system_output_detached": system_output_detached_BC,
            "safe_model_output": safe_probs_BC,
            "dangerous_model_output": dangerous_probs_BC,
            "safe_logits": safe_logits_BC,
            "dangerous_logits": dangerous_logits_BC,
            "computed_gate": computed_gate_B,
            "used_gate": used_gate_B,
        }
