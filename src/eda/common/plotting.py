"""Shared matplotlib plotting helpers for the EDA pipeline (Phase 5).

All functions save a PNG to the given path and return it - no plt.show(),
these run headless from scripts. Every function uses only matplotlib
(already in requirements.txt) - no new dependencies.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def _save(fig, save_path: Path) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_class_distribution(counts: pd.Series, save_path: Path, title: str) -> Path:
    counts = counts.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, len(counts) * 0.9), 5))
    ax.bar([str(v) for v in counts.index], counts.values, color="#4C72B0")
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    return _save(fig, save_path)


def plot_categorical_bar(series: pd.Series, save_path: Path, title: str, xlabel: str) -> Path:
    counts = series.value_counts(dropna=False)
    counts.index = [str(v) for v in counts.index]
    fig, ax = plt.subplots(figsize=(max(5, len(counts) * 0.7), 4.5))
    ax.bar(counts.index, counts.values, color="#55A868")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    return _save(fig, save_path)


def plot_histogram(series: pd.Series, save_path: Path, title: str, xlabel: str, bins: int = 30) -> Path:
    data = series.dropna()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(data, bins=bins, color="#C44E52", edgecolor="white")
    ax.set_title(f"{title} (n={len(data)}, {series.isna().mean()*100:.1f}% missing)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    return _save(fig, save_path)


def plot_panel(
    df: pd.DataFrame,
    columns: list[str],
    kinds: dict[str, str],
    save_path: Path,
    suptitle: str,
    ncols: int = 3,
) -> Path:
    """Composite multi-panel figure. `kinds[col]` is 'hist' or 'bar'."""
    n = len(columns)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.8))
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        kind = kinds.get(col, "bar")
        if kind == "hist":
            data = df[col].dropna()
            ax.hist(data, bins=20, color="#C44E52", edgecolor="white")
        else:
            counts = df[col].value_counts(dropna=False)
            counts.index = [str(v) for v in counts.index]
            ax.bar(counts.index, counts.values, color="#55A868")
            ax.tick_params(axis="x", rotation=45)
            for tick in ax.get_xticklabels():
                tick.set_ha("right")
        missing_pct = df[col].isna().mean() * 100
        ax.set_title(f"{col} ({missing_pct:.1f}% missing)", fontsize=10)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, save_path)


def plot_missingness(df: pd.DataFrame, columns: list[str], save_path: Path, title: str) -> Path:
    missing_pct = df[columns].isna().mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(max(6, len(columns) * 0.5), 5))
    ax.barh(missing_pct.index[::-1], missing_pct.values[::-1], color="#8172B2")
    ax.set_title(title)
    ax.set_xlabel("% missing")
    for i, v in enumerate(missing_pct.values[::-1]):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8)
    return _save(fig, save_path)


def plot_image_dimensions(stats_df: pd.DataFrame, save_path: Path, title: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].scatter(stats_df["width"], stats_df["height"], alpha=0.4, s=15, color="#4C72B0")
    axes[0].set_xlabel("width (px)")
    axes[0].set_ylabel("height (px)")
    axes[0].set_title("Width vs. height")

    axes[1].hist(stats_df["aspect_ratio"], bins=20, color="#55A868", edgecolor="white")
    axes[1].set_xlabel("aspect ratio (w/h)")
    axes[1].set_title("Aspect ratio distribution")

    axes[2].hist(stats_df["size_kb"], bins=20, color="#C44E52", edgecolor="white")
    axes[2].set_xlabel("file size (KB)")
    axes[2].set_title("File size distribution")

    fig.suptitle(f"{title} (n={len(stats_df)} sampled)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return _save(fig, save_path)


def plot_sample_image_grid(
    df: pd.DataFrame,
    image_path_col: str,
    class_col: str,
    save_path: Path,
    n_per_class: int,
    seed: int,
    title: str,
) -> Path:
    classes = sorted(df[class_col].dropna().unique())
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(n_per_class * 2.2, len(classes) * 2.2))
    axes = np.atleast_2d(axes)

    rng = np.random.default_rng(seed)
    for row, cls in enumerate(classes):
        subset = df[df[class_col] == cls]
        n = min(n_per_class, len(subset))
        sampled = subset.sample(n=n, random_state=rng.integers(0, 1_000_000))
        paths = list(sampled[image_path_col])
        for col in range(n_per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(paths):
                try:
                    with Image.open(paths[col]) as im:
                        ax.imshow(im)
                except (FileNotFoundError, OSError):
                    ax.text(0.5, 0.5, "missing", ha="center", va="center")
            if col == 0:
                ax.set_ylabel(cls, fontsize=8)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, save_path)


def plot_split_balance(counts_by_split: dict[str, pd.Series], save_path: Path, title: str) -> Path:
    all_classes = sorted(set().union(*[s.index for s in counts_by_split.values()]))
    splits = list(counts_by_split.keys())
    x = np.arange(len(all_classes))
    width = 0.8 / len(splits)

    fig, ax = plt.subplots(figsize=(max(7, len(all_classes) * 1.1), 5))
    for i, split in enumerate(splits):
        proportions = counts_by_split[split].reindex(all_classes, fill_value=0)
        proportions = proportions / proportions.sum() * 100
        ax.bar(x + i * width, proportions.values, width, label=split)

    ax.set_xticks(x + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(all_classes, rotation=45, ha="right")
    ax.set_ylabel("% of split")
    ax.set_title(title)
    ax.legend()
    return _save(fig, save_path)


def plot_cross_dataset_label_comparison(
    counts_by_dataset: dict[str, pd.Series],
    save_path: Path,
    title: str,
    caption: str,
) -> Path:
    all_classes = sorted(set().union(*[s.index for s in counts_by_dataset.values()]))
    datasets = list(counts_by_dataset.keys())
    x = np.arange(len(all_classes))
    width = 0.8 / len(datasets)

    fig, ax = plt.subplots(figsize=(max(9, len(all_classes) * 1.2), 6))
    for i, ds in enumerate(datasets):
        counts = counts_by_dataset[ds].reindex(all_classes, fill_value=0)
        ax.bar(x + i * width, counts.values, width, label=ds)

    ax.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(all_classes, rotation=45, ha="right")
    ax.set_ylabel("train-set count")
    ax.set_title(title)
    ax.legend()
    fig.text(0.5, -0.05, caption, ha="center", fontsize=8, wrap=True)
    return _save(fig, save_path)
