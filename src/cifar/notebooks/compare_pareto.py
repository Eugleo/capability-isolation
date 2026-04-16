# %%
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

# %%
# safe dang
# experiment_dirs: list[str] = [
#     "experiments/2026-04-14_21-26-34_safe_dang_1p",
#     "experiments/2026-04-14_21-30-00_safe_dang_5p",
#     "experiments/2026-04-14_21-33-25_safe_dang_10p",
#     "experiments/2026-04-14_21-36-50_safe_dang_25p",
#     "experiments/2026-04-14_21-40-14_safe_dang_50p",
# ]

# safe+unk dang
# experiment_dirs: list[str] = [
#     "experiments/2026-04-14_21-43-39_safe+unk_dang_1p",
#     "experiments/2026-04-14_21-47-03_safe+unk_dang_5p",
#     "experiments/2026-04-14_21-50-26_safe+unk_dang_10p",
#     "experiments/2026-04-14_21-53-49_safe+unk_dang_25p",
#     "experiments/2026-04-14_21-57-14_safe+unk_dang_50p",
# ]

# safe dang+unk
experiment_dirs: list[str] = [
    "experiments/2026-04-14_22-00-40_safe_dang+unk_1p",
    "experiments/2026-04-14_22-04-04_safe_dang+unk_5p",
    "experiments/2026-04-14_22-07-29_safe_dang+unk_10p",
    "experiments/2026-04-14_22-10-54_safe_dang+unk_25p",
    "experiments/2026-04-14_22-14-19_safe_dang+unk_50p",
]

# %%
def _load_cifar_unlearn_metrics(exp_path: Path) -> pl.DataFrame:
    """Return long metrics with columns step, class_name, metric, value."""
    metrics_path = exp_path / "metrics.csv"
    if metrics_path.exists():
        df = pl.read_csv(metrics_path)
        need = {"step", "class_name", "metric", "value"}
        if not need.issubset(df.columns):
            raise ValueError(f"metrics.csv missing columns; need {need}, got {df.columns}")
        return df

    legacy_wide = exp_path / "per_class_metrics.csv"
    if legacy_wide.exists():
        wide = pl.read_csv(legacy_wide)
        metric_cols = [
            "count",
            "top1_correct",
            "top5_correct",
            "loss_sum",
            "top1_acc",
            "top5_acc",
            "loss",
        ]
        idx = ["step", "class_idx", "class_name"]
        if not set(idx + metric_cols).issubset(wide.columns):
            raise ValueError(f"legacy wide CSV missing columns; got {wide.columns}")
        return wide.unpivot(
            index=idx,
            on=metric_cols,
            variable_name="metric",
            value_name="value",
        )

    raise FileNotFoundError(
        f"No metrics.csv or per_class_metrics.csv under {exp_path}"
    )


# %%
experiments: list[dict] = []

for exp_path_str in experiment_dirs:
    exp_path = Path(exp_path_str)
    label = exp_path.name
    try:
        with open(exp_path / "config.json") as f:
            config = json.load(f)
        df = _load_cifar_unlearn_metrics(exp_path)
    except Exception as e:
        warnings.warn(f"Skipping {label}: {e}")
        continue

    name = config.get("name") or label
    experiments.append({"path": exp_path, "label": name, "config": config, "df": df})

print(f"Loaded {len(experiments)}/{len(experiment_dirs)} experiments")

# %%
if len(experiments) >= 2:
    ref = experiments[0]["config"]
    for exp in experiments[1:]:
        cfg = exp["config"]
        if cfg.get("dangerous_classes") != ref.get("dangerous_classes"):
            warnings.warn(
                f"dangerous_classes mismatch: {exp['label']} has "
                f"'{cfg.get('dangerous_classes')}' vs reference '{ref.get('dangerous_classes')}'"
            )
        mask_key = (
            ("known", tuple(cfg.get("known_classes", [])))
            if "known_classes" in cfg
            else ("unknown", tuple(cfg.get("unknown_classes", [])))
        )
        ref_mask_key = (
            ("known", tuple(ref.get("known_classes", [])))
            if "known_classes" in ref
            else ("unknown", tuple(ref.get("unknown_classes", [])))
        )
        if mask_key != ref_mask_key:
            warnings.warn(
                f"known/unknown class mask mismatch: {exp['label']} has "
                f"{mask_key!r} vs reference {ref_mask_key!r}"
            )

# %%
dangerous_classes = experiments[0]["config"]["dangerous_classes"] if experiments else ["cat"]
dangerous_label = "+".join(dangerous_classes)

fig, ax = plt.subplots(figsize=(8, 8))
cmap = plt.get_cmap("tab10")

for i, exp in enumerate(experiments):
    df = exp["df"]
    color = cmap(i % 10)

    acc = df.filter(pl.col("metric") == "top1_acc")
    steps = sorted(acc["step"].unique().to_list())
    other_acc_pct: list[float] = []
    forget_quality_pct: list[float] = []

    for step in steps:
        step_df = acc.filter(pl.col("step") == step)
        dang = step_df.filter(pl.col("class_name").is_in(dangerous_classes))[
            "value"
        ].mean()
        others = step_df.filter(~pl.col("class_name").is_in(dangerous_classes))[
            "value"
        ].mean()
        other_acc_pct.append(float(others) * 100)
        forget_quality_pct.append((1.0 - float(dang)) * 100)

    ax.plot(
        other_acc_pct,
        forget_quality_pct,
        color=color,
        linewidth=1.5,
        label=exp["label"],
        zorder=1,
    )
    ax.scatter(
        other_acc_pct,
        forget_quality_pct,
        color=color,
        s=40,
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )

ax.set_xlabel("\u2191 Safe Utility (acc %)", fontsize=12)
ax.set_ylabel(
    f"\u2191 Dangerous Unlearning (1 \u2212 {dangerous_label} acc %)",
    fontsize=12,
)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_title("Unlearning Pareto: Retain Utility vs Forget Quality", fontsize=13)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()

# %%
