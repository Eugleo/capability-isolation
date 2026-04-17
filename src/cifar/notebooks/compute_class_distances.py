# %%
"""Compute directed class-to-class distances via per-class unlearning on CIFAR-100.

For each source class X in `CLASSES` we run NegGrad unlearning with:
  - X as the sole known danger (k-dang)
  - every other class as unknown safe (u-safe); there is no retain objective
  - `MAX_STEPS` unlearning steps, evaluating every `EVAL_EVERY_N_STEPS`

Directed distance D[X -> Y] = top-1 accuracy of the model on class Y at step
`DISTANCE_STEP` of the unlearning of X. Smaller values => Y was more affected by
unlearning X => X and Y are "closer" in representation space.

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
MAX_STEPS = 500
EVAL_EVERY_N_STEPS = 50
DISTANCE_STEP = 500  # D[X, Y] = top-1 accuracy of Y at this step of X's unlearning
SEED = 42
SKIP_IF_METRICS_EXIST = True  # reuse existing per-class runs if present
DISTANCES_CSV = EXPERIMENTS_ROOT / "distances.csv"
# Source+target classes for the directed distance matrix. Each listed class gets
# one unlearning run; the matrix is len(CLASSES) x len(CLASSES). Use
# CIFAR100_CLASSES to sweep all 100.
CLASSES: tuple[str, ...] = tuple(sorted(set(CIFAR100_CLASSES) - {"apple"}))

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
        known_classes=(class_name, "apple"),  # only this class is known -> k-dang; apple is k-safe; rest are u-safe
        unlearning_strategy="ignore-unknown",
        eval_split="train",
        eval_class_groups={"all": CIFAR100_CLASSES},
        experiments_root=str(EXPERIMENTS_ROOT),
    )


EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)

for i, class_name in enumerate(CLASSES, start=1):
    print(f"\n{'=' * 60}")
    print(f"[{i}/{len(CLASSES)}] source class: {class_name}")
    print(f"{'=' * 60}")
    existing = _find_existing_exp_dir(EXPERIMENTS_ROOT, class_name)
    if SKIP_IF_METRICS_EXIST and existing is not None:
        print(f"Skipping (already done): {existing}")
        continue
    run_unlearn(_build_config(class_name))

# %%
# --- Load metrics and compute the directed distance matrix ----------------
def _accuracies_at_step(exp_dir: Path, step: int, targets: tuple[str, ...]) -> dict[str, float]:
    metrics = pl.read_csv(exp_dir / "metrics.csv")
    top1 = metrics.filter((pl.col("metric") == "top1_acc") & (pl.col("step") == step))
    value_by_class = dict(zip(top1["class_name"].to_list(), top1["value"].to_list()))
    return {target: float(value_by_class.get(target, float("nan"))) for target in targets}


class_to_idx = {c: i for i, c in enumerate(CLASSES)}
N = len(CLASSES)
D = np.full((N, N), np.nan, dtype=float)

missing_sources: list[str] = []
for source in CLASSES:
    exp_dir = _find_existing_exp_dir(EXPERIMENTS_ROOT, source)
    if exp_dir is None:
        missing_sources.append(source)
        continue
    per_target = _accuracies_at_step(exp_dir, DISTANCE_STEP, CLASSES)
    s_idx = class_to_idx[source]
    for target, acc in per_target.items():
        D[s_idx, class_to_idx[target]] = acc

if missing_sources:
    print(f"WARNING: missing experiments for {len(missing_sources)} classes: {missing_sources}")

n_missing = int(np.isnan(D).sum())
print(f"D shape: {D.shape}, NaNs (no eval at step {DISTANCE_STEP}): {n_missing}")

# %%
# --- Save distances as tidy long CSV (source, target, distance) -----------
rows: list[dict] = []
for si, source in enumerate(CLASSES):
    for ti, target in enumerate(CLASSES):
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
names_ordered = [CLASSES[i] for i in order]
print(f"Seriation order: {names_ordered}")

# %%
# --- Reordered directed heatmap -------------------------------------------
D_reord = D[np.ix_(order, order)]

label_fontsize = max(5, min(11, int(300 / max(N, 1))))

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(D_reord, aspect="equal", cmap="viridis_r", interpolation="nearest", vmin=0, vmax=1)
ax.set_xticks(range(N))
ax.set_xticklabels(names_ordered, rotation=90, fontsize=label_fontsize)
ax.set_yticks(range(N))
ax.set_yticklabels(names_ordered, fontsize=label_fontsize)
ax.set_xlabel("target class Y")
ax.set_ylabel("source class X (unlearned)")
ax.set_title(
    f"Directed unlearning distance D[X → Y] = top-1 acc(Y) at step {DISTANCE_STEP}\n"
    f"(rows/cols reordered by average-linkage seriation on the symmetric distance; "
    f"darker = Y more affected by unlearning X = closer)"
)
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label(f"top-1 accuracy of Y at step {DISTANCE_STEP}")
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

pair_label_fontsize = 7 if len(iu_i) <= 50 else (5 if len(iu_i) <= 500 else 3)
for i, j, x, y in zip(iu_i, iu_j, xs, ys):
    if not (np.isfinite(x) and np.isfinite(y)):
        continue
    label = f"{CLASSES[i]}↔{CLASSES[j]}"
    ax.annotate(label, (x, y), fontsize=pair_label_fontsize, alpha=0.7, color="#333333")

ax.set_xlabel(f"D[X → Y]  (top-1 acc of Y at step {DISTANCE_STEP})")
ax.set_ylabel(f"D[Y → X]  (top-1 acc of X at step {DISTANCE_STEP})")
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

if N < 4:
    print(f"Skipping UMAP: need >= 4 classes, got {N}")
else:
    n_neighbors = min(15, N - 1)
    reducer = umap.UMAP(
        metric="precomputed",
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=SEED,
    )
    embedding_N2 = reducer.fit_transform(D_sym_clean)

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.scatter(embedding_N2[:, 0], embedding_N2[:, 1], s=24, color="#1f77b4", alpha=0.8)
    for i, name in enumerate(CLASSES):
        ax.annotate(
            name,
            (embedding_N2[i, 0], embedding_N2[i, 1]),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
            alpha=0.9,
        )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(
        "UMAP of classes on symmetric unlearning distance\n"
        f"metric='precomputed', n_neighbors={reducer.n_neighbors}, min_dist={reducer.min_dist}"
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.show()

# %%
