# %%
"""Compute directed class-to-class distances via per-class unlearning on CIFAR-100.

For each class X in CIFAR-100 we run NegGrad unlearning with:
  - X as the sole known danger (k-dang)
  - every other class as unknown safe (u-safe); there is no retain objective
  - max_steps unlearning steps, evaluating every eval_every_n_steps on all classes

Directed distance D[X -> Y] = first eval step at which the model's top-1 accuracy
on class Y drops below `DISTANCE_THRESHOLD` during the unlearning of X.

All per-class experiment directories are written under EXPERIMENTS_ROOT. After the
sweep we load metrics, compute distances, and produce:
  - reordered directed heatmap (seriation on the symmetric distance)
  - scatter of D[X -> Y] vs D[Y -> X] with class-pair labels
  - UMAP of classes on the symmetric distance

Cells (# %%) can be re-run independently once the sweep has produced metrics.csv
files. Run `uv sync` once to install the scipy + umap-learn dependencies.
"""

# %%
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.cifar.data import CIFAR100_CLASSES
from src.cifar.unlearn import UnlearnConfig, main as run_unlearn

# %%
# --- Config ---------------------------------------------------------------
EXPERIMENTS_ROOT = Path("experiments/2026-04-17_class_distances")
MAX_STEPS = 10_000
EVAL_EVERY_N_STEPS = 200
DISTANCE_THRESHOLD = 0.02  # top-1 accuracy below this = "class has been unlearned"
SEED = 42
SKIP_IF_METRICS_EXIST = True  # reuse existing per-class runs if present
DISTANCES_CSV = EXPERIMENTS_ROOT / "distances.csv"

# %%
# --- Sweep: one unlearning run per CIFAR-100 class ------------------------
def _find_existing_exp_dir(root: Path, class_name: str) -> Path | None:
    if not root.exists():
        return None
    for d in sorted(root.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        cfg_path = d / "config.json"
        metrics_path = d / "metrics.csv"
        if not (cfg_path.exists() and metrics_path.exists()):
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        if list(cfg.get("dangerous_classes", [])) == [class_name]:
            return d
    return None


def _build_config(class_name: str) -> UnlearnConfig:
    return UnlearnConfig(
        name=class_name,
        dataset="cifar100",
        seed=SEED,
        max_steps=MAX_STEPS,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        dangerous_classes=(class_name,),
        known_classes=(class_name,),  # only this class is known -> k-dang; rest are u-safe
        unlearning_strategy="ignore-unknown",
        eval_split="train",
        eval_class_groups={"all": CIFAR100_CLASSES},
        experiments_root=str(EXPERIMENTS_ROOT),
    )


EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)

for i, class_name in enumerate(CIFAR100_CLASSES, start=1):
    print(f"\n{'=' * 60}")
    print(f"[{i}/{len(CIFAR100_CLASSES)}] source class: {class_name}")
    print(f"{'=' * 60}")
    existing = _find_existing_exp_dir(EXPERIMENTS_ROOT, class_name)
    if SKIP_IF_METRICS_EXIST and existing is not None:
        print(f"Skipping (already done): {existing}")
        continue
    run_unlearn(_build_config(class_name))

# %%
# --- Load metrics and compute the directed distance matrix ----------------
def _first_step_below(steps: np.ndarray, values: np.ndarray, threshold: float) -> float:
    order = np.argsort(steps)
    steps = steps[order]
    values = values[order]
    below = values < threshold
    if not below.any():
        return float("nan")
    return float(steps[np.argmax(below)])


def _distances_from_exp_dir(exp_dir: Path, threshold: float) -> dict[str, float]:
    metrics = pl.read_csv(exp_dir / "metrics.csv")
    top1 = metrics.filter(pl.col("metric") == "top1_acc")
    out: dict[str, float] = {}
    for target in CIFAR100_CLASSES:
        sub = top1.filter(pl.col("class_name") == target)
        if sub.is_empty():
            out[target] = float("nan")
            continue
        steps = sub["step"].to_numpy()
        values = sub["value"].to_numpy()
        out[target] = _first_step_below(steps, values, threshold)
    return out


class_to_idx = {c: i for i, c in enumerate(CIFAR100_CLASSES)}
N = len(CIFAR100_CLASSES)
D = np.full((N, N), np.nan, dtype=float)

missing_sources: list[str] = []
for source in CIFAR100_CLASSES:
    exp_dir = _find_existing_exp_dir(EXPERIMENTS_ROOT, source)
    if exp_dir is None:
        missing_sources.append(source)
        continue
    per_target = _distances_from_exp_dir(exp_dir, DISTANCE_THRESHOLD)
    s_idx = class_to_idx[source]
    for target, dist in per_target.items():
        D[s_idx, class_to_idx[target]] = dist

if missing_sources:
    print(f"WARNING: missing experiments for {len(missing_sources)} classes: {missing_sources[:5]}...")

n_never_crossed = int(np.isnan(D).sum())
print(f"D shape: {D.shape}, NaNs (never crossed {DISTANCE_THRESHOLD:.0%}): {n_never_crossed}")

# %%
# --- Save distances as tidy long CSV (source, target, distance) -----------
rows: list[dict] = []
for si, source in enumerate(CIFAR100_CLASSES):
    for ti, target in enumerate(CIFAR100_CLASSES):
        rows.append({
            "source_idx": si,
            "source": source,
            "target_idx": ti,
            "target": target,
            "distance": float(D[si, ti]) if not np.isnan(D[si, ti]) else None,
        })
pl.DataFrame(rows).write_csv(DISTANCES_CSV)
print(f"Saved distances to {DISTANCES_CSV}")

# %%
# --- Symmetric distance + hierarchical ordering ---------------------------
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

D_sym = 0.5 * (D + D.T)
D_sym_clean = D_sym.copy()
np.fill_diagonal(D_sym_clean, 0.0)
# squareform wants finite, symmetric, zero-diag. Replace any residual NaN with the
# max finite value so seriation still runs.
if np.isnan(D_sym_clean).any():
    max_finite = np.nanmax(D_sym_clean)
    D_sym_clean = np.where(np.isnan(D_sym_clean), max_finite, D_sym_clean)
    D_sym_clean = 0.5 * (D_sym_clean + D_sym_clean.T)
    np.fill_diagonal(D_sym_clean, 0.0)

condensed = squareform(D_sym_clean, checks=False)
Z = linkage(condensed, method="average")
order = leaves_list(Z)
names_ordered = [CIFAR100_CLASSES[i] for i in order]
print(f"Seriation order (first 10): {names_ordered[:10]}")

# %%
# --- Reordered directed heatmap -------------------------------------------
D_reord = D[np.ix_(order, order)]

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(D_reord, aspect="equal", cmap="viridis", interpolation="nearest")
ax.set_xticks(range(N))
ax.set_xticklabels(names_ordered, rotation=90, fontsize=5)
ax.set_yticks(range(N))
ax.set_yticklabels(names_ordered, fontsize=5)
ax.set_xlabel("target class Y")
ax.set_ylabel("source class X (unlearned)")
ax.set_title(
    f"Directed unlearning distance D[X → Y]: steps until top-1(Y) < {DISTANCE_THRESHOLD:.0%}\n"
    f"(rows/cols reordered by average-linkage seriation on the symmetric distance)"
)
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("steps")
fig.tight_layout()
plt.show()

# %%
# --- Scatter of D[X, Y] vs D[Y, X] (one point per unordered pair) ---------
iu_i, iu_j = np.triu_indices(N, k=1)
xs = D[iu_i, iu_j]
ys = D[iu_j, iu_i]

fig, ax = plt.subplots(figsize=(11, 11))
finite = np.isfinite(xs) & np.isfinite(ys)
ax.scatter(xs[finite], ys[finite], s=6, alpha=0.35, color="#1f77b4")

mx = float(np.nanmax([xs, ys])) if finite.any() else 1.0
ax.plot([0, mx], [0, mx], color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

for i, j, x, y in zip(iu_i, iu_j, xs, ys):
    if not (np.isfinite(x) and np.isfinite(y)):
        continue
    label = f"{CIFAR100_CLASSES[i]}↔{CIFAR100_CLASSES[j]}"
    ax.annotate(label, (x, y), fontsize=3.0, alpha=0.55, color="#333333")

ax.set_xlabel("D[X → Y]  (steps)")
ax.set_ylabel("D[Y → X]  (steps)")
ax.set_title(
    "Symmetry of unlearning distance\n"
    "one point per unordered class pair; dashed line is y = x"
)
ax.set_xlim(0, mx * 1.02)
ax.set_ylim(0, mx * 1.02)
ax.set_aspect("equal")
ax.grid(True, alpha=0.25)

corr = float(np.corrcoef(xs[finite], ys[finite])[0, 1]) if finite.sum() > 1 else float("nan")
ax.text(
    0.02, 0.98, f"Pearson r = {corr:.3f}  (n={int(finite.sum())} pairs)",
    transform=ax.transAxes, fontsize=10, va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none"),
)
fig.tight_layout()
plt.show()

# %%
# --- UMAP on the symmetric distance ---------------------------------------
import umap

reducer = umap.UMAP(
    metric="precomputed",
    n_neighbors=15,
    min_dist=0.1,
    random_state=SEED,
)
embedding_N2 = reducer.fit_transform(D_sym_clean)

fig, ax = plt.subplots(figsize=(14, 12))
ax.scatter(embedding_N2[:, 0], embedding_N2[:, 1], s=24, color="#1f77b4", alpha=0.8)
for i, name in enumerate(CIFAR100_CLASSES):
    ax.annotate(
        name,
        (embedding_N2[i, 0], embedding_N2[i, 1]),
        fontsize=7,
        xytext=(3, 3),
        textcoords="offset points",
        alpha=0.9,
    )
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.set_title(
    "UMAP of CIFAR-100 classes on symmetric unlearning distance\n"
    f"metric='precomputed', n_neighbors={reducer.n_neighbors}, min_dist={reducer.min_dist}"
)
ax.grid(True, alpha=0.2)
fig.tight_layout()
plt.show()

# %%
