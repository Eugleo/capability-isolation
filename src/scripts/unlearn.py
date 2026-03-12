import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.classifier import Classifier, evaluate_classifier
from src.config import Config
from src.data import MarkedMNIST
from src.gate import Gate
from src.system import GatedSystem, evaluate_gated_system
from src.train_classifier import build_eval_dataframe, plot_classifier_evaluation
from src.utils import format_metric_value, get_device, set_seed

WEIGHT_MARKED_LOSS = 5e-5
WEIGHT_DIVERGENCE = 0.0

BASELINE_SYSTEM_PATH = Path("experiments/2026-03-11_15-54-45_0ccf2fb5/system.pt")

STRATEGIES = [
    {
        "name": "retain_safe__forget_unsafe",
        "display_name": "Baseline (retain safe, forget unsafe)",
        "positive_categories": ("safe",),
        "negative_categories": ("unsafe",),
        "use_known_only": True,
    },
    {
        "name": "retain_safe_unk__forget_unsafe",
        "display_name": "Baseline (retain safe+unk, forget unsafe)",
        "positive_categories": ("safe", "unknown"),
        "negative_categories": ("unsafe",),
        "use_known_only": False,
    },
    {
        "name": "retain_safe__forget_unsafe_unk",
        "display_name": "Baseline (retain safe, forget unsafe+unk)",
        "positive_categories": ("safe",),
        "negative_categories": ("unsafe", "unknown"),
        "use_known_only": False,
    },
]

UNLEARN_MARKERS = ["o", "s", "^"]


def _sample_category(mark: str, is_known: bool) -> str:
    """Classify a sample: unknown trumps mark, then none→safe, else unsafe."""
    if not is_known:
        return "unknown"
    return "safe" if mark == "none" else "unsafe"


def l2_weight_divergence(model: nn.Module, frozen: nn.Module) -> torch.Tensor:
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for p, p_frozen in zip(model.parameters(), frozen.parameters()):
        total = total + ((p - p_frozen) ** 2).sum()
    return total


def train_entanglement_unlearn(
    model: Classifier,
    frozen: Classifier,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-3,
    *,
    positive_categories: tuple[str, ...] = ("safe",),
    negative_categories: tuple[str, ...] = ("unsafe",),
) -> tuple[Classifier, list[dict[str, float]]]:
    """Unlearn negative capability while preserving positive. Returns (model, epoch_history)."""
    model.train()
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            marks = list(batch["mark"])
            is_known_list = batch["is_known"].tolist()

            categories = [_sample_category(m, k) for m, k in zip(marks, is_known_list)]
            positive_mask = torch.tensor(
                [c in positive_categories for c in categories], device=device
            )
            negative_mask = torch.tensor(
                [c in negative_categories for c in categories], device=device
            )

            optimizer.zero_grad()
            logits = model(images)

            if positive_mask.any():
                loss_positive = criterion(logits[positive_mask], labels[positive_mask])
            else:
                loss_positive = torch.tensor(0.0, device=device)

            if negative_mask.any():
                loss_negative = criterion(logits[negative_mask], labels[negative_mask])
            else:
                loss_negative = torch.tensor(0.0, device=device)

            divergence = l2_weight_divergence(model, frozen)

            loss = (
                loss_positive
                - WEIGHT_MARKED_LOSS * loss_negative
                - WEIGHT_DIVERGENCE * divergence
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        model.eval()
        metrics = evaluate_classifier(model, test_loader, device)
        model.train()

        epoch_entry: dict[str, float] = {"epoch": float(epoch + 1), **metrics}
        history.append(epoch_entry)

        metric_str = "".join(
            f"\n  {k}: {format_metric_value(k, v)}" for k, v in metrics.items()
        )
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}{metric_str}")

    return model, history


def plot_pareto(
    results: list[dict],
    save_path: Path | str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for result in results:
        is_ours = result.get("ours", False)
        ax.scatter(
            result["performance"] * 100,
            result["safety"],
            color="#DAA520" if is_ours else "black",
            marker="*" if is_ours else result["marker"],
            s=300 if is_ours else 100,
            label=result["display_name"],
            zorder=4 if is_ours else 3,
            edgecolors="black",
            linewidths=0.8,
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Performance \u2191 (System Accuracy %)", fontsize=11)
    ax.set_ylabel("Safety \u2191 (100 \u2212 Safe Acc. on Unsafe Data %)", fontsize=11)
    ax.set_title("Performance vs Safety", fontsize=13)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _create_experiment_dir() -> Path:
    experiments_root = Path("experiments")
    experiments_root.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    short_id = uuid.uuid4().hex[:8]
    exp_dir = experiments_root / f"{timestamp}_{short_id}"
    exp_dir.mkdir(parents=True)
    return exp_dir


def main() -> None:
    config = Config()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    print("Weight of loss on positive: 1.0")
    print(f"Weight of (negative) loss on negative: {WEIGHT_MARKED_LOSS}")
    print(f"Weight of L2-divergence: {WEIGHT_DIVERGENCE}")

    checkpoint_dir = Path(config.checkpoint_dir)
    experiment_dir = _create_experiment_dir()
    print(f"Experiment dir: {experiment_dir}")
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    base_model = Classifier.load(checkpoint_dir / "classifier_all", device=device)

    train_dataset = MarkedMNIST(
        train=True, kind_fraction=config.kind_fraction, seed=config.seed
    )
    test_dataset = MarkedMNIST(
        train=False, kind_fraction=config.kind_fraction, seed=config.seed + 1
    )
    train_dataset.print_summary("Train dataset")
    test_dataset.print_summary("Test dataset")

    known_train_indices = [
        i for i in range(len(train_dataset)) if train_dataset.is_known_arr[i]
    ]
    train_known_only = Subset(train_dataset, known_train_indices)

    train_loader_known = DataLoader(
        train_known_only,
        batch_size=config.classifier_batch_size,
        shuffle=True,
        num_workers=0,
    )
    train_loader_full = DataLoader(
        train_dataset,
        batch_size=config.classifier_batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.classifier_batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Known-only train: {len(known_train_indices)} samples")
    print(f"Full train: {len(train_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")

    # Initial evaluation
    print("\nInitial metrics (classifier_all):")
    base_model.eval()
    for k, v in evaluate_classifier(base_model, test_loader, device).items():
        print(f"  {k}: {format_metric_value(k, v)}")

    model_histories: dict[str, list[dict[str, float]]] = {}
    trained_models: dict[str, Classifier] = {}

    for strategy in STRATEGIES:
        name = strategy["name"]
        display_name = strategy["display_name"]
        positive = strategy["positive_categories"]
        negative = strategy["negative_categories"]
        loader = train_loader_known if strategy["use_known_only"] else train_loader_full

        model = Classifier().to(device)
        model.load_state_dict(base_model.state_dict())
        frozen = Classifier().to(device)
        frozen.load_state_dict(base_model.state_dict())

        print(f"\n{'=' * 60}")
        print(f"Training: {display_name}")
        print(f"  positive={positive}, negative={negative}")
        print("=" * 60)

        model, history = train_entanglement_unlearn(
            model,
            frozen,
            loader,
            test_loader,
            device,
            epochs=3,
            lr=config.classifier_lr,
            positive_categories=positive,
            negative_categories=negative,
        )

        save_dir = experiment_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save(save_dir / "model.pt")
        with open(save_dir / "config.json", "w") as f:
            json.dump(
                {**asdict(config), "strategy": strategy},
                f,
                indent=2,
            )
        print(f"Saved to {save_dir / 'model.pt'}")

        model_histories[display_name] = history
        trained_models[name] = model

    # Classifier evaluation plot (accuracy per kind over epochs)
    if model_histories:
        eval_df = build_eval_dataframe(model_histories)
        metrics_csv_path = experiment_dir / "classifier_unlearn_metrics.csv"
        eval_df.write_csv(metrics_csv_path)
        plot_classifier_evaluation(
            eval_df,
            experiment_dir / "classifier_unlearn_evaluation.png",
            use_palette=True,
            single_legend=True,
        )
        print(f"\nSaved classifier metrics to {metrics_csv_path}")
        print(
            f"Saved evaluation plot to"
            f" {experiment_dir / 'classifier_unlearn_evaluation.png'}"
        )

    # --- Pareto comparison ---
    # Build systems: gate_known + unlearned safe model + classifier_all as unsafe
    print(f"\n{'=' * 60}")
    print("Building systems for Pareto comparison")
    print("=" * 60)

    gate_known = Gate().to(device)
    gate_known.load_state_dict(
        torch.load(
            checkpoint_dir / "gate_known.pt",
            map_location=device,
            weights_only=True,
        )["model_state_dict"]
    )
    gate_known.eval()

    model_unsafe = Classifier.load(checkpoint_dir / "classifier_all", device=device)
    model_unsafe.eval()

    pareto_results: list[dict] = []

    for i, strategy in enumerate(STRATEGIES):
        name = strategy["name"]
        display_name = strategy["display_name"]
        model_safe = trained_models[name]
        model_safe.eval()

        system = GatedSystem(
            gate=gate_known, model_safe=model_safe, model_unsafe=model_unsafe
        )
        system_metrics = evaluate_gated_system(system, test_loader, device)

        performance = system_metrics["system/all/accuracy"]
        safe_acc_on_unsafe = system_metrics["system_safe/marked/accuracy"]
        safety = 100.0 - safe_acc_on_unsafe * 100.0

        print(f"\n{display_name}:")
        print(f"  Performance (system accuracy): {performance:.2%}")
        print(f"  Safe model acc on unsafe data: {safe_acc_on_unsafe:.2%}")
        print(f"  Safety: {safety:.1f}")

        pareto_results.append(
            {
                "name": name,
                "display_name": display_name,
                "performance": performance,
                "safety": safety,
                "marker": UNLEARN_MARKERS[i % len(UNLEARN_MARKERS)],
            }
        )

    # Jointly trained system (ours)
    if BASELINE_SYSTEM_PATH.exists():
        print(f"\nLoading jointly trained system from {BASELINE_SYSTEM_PATH}")
        ours_system = GatedSystem.load(BASELINE_SYSTEM_PATH, device=device)
        ours_metrics = evaluate_gated_system(ours_system, test_loader, device)

        performance = ours_metrics["system/all/accuracy"]
        safe_acc_on_unsafe = ours_metrics["system_safe/marked/accuracy"]
        safety = 100.0 - safe_acc_on_unsafe * 100.0

        print("Jointly trained (ours):")
        print(f"  Performance (system accuracy): {performance:.2%}")
        print(f"  Safe model acc on unsafe data: {safe_acc_on_unsafe:.2%}")
        print(f"  Safety: {safety:.1f}")

        pareto_results.append(
            {
                "name": "jointly_trained",
                "display_name": "Ours (jointly trained)",
                "performance": performance,
                "safety": safety,
                "ours": True,
            }
        )
    else:
        print(f"\nWarning: jointly trained system not found at {BASELINE_SYSTEM_PATH}")

    plot_pareto(pareto_results, experiment_dir / "pareto_performance_safety.png")
    print(f"\nSaved Pareto plot to {experiment_dir / 'pareto_performance_safety.png'}")

    pareto_df = pl.DataFrame(pareto_results)
    pareto_df.write_csv(experiment_dir / "pareto_results.csv")
    print(f"Saved Pareto results to {experiment_dir / 'pareto_results.csv'}")

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to {experiment_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
