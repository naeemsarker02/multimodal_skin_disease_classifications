"""Module 10: Class distribution.

Value counts and percentages for the diagnostic (dx) column, plus an
imbalance ratio and a bar chart. HAM10000's imbalance (nv dominates)
is expected to be more severe than PAD-UFES-20's.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data_audit.config import (
    HAM10000_FIGURES_DIR,
    HAM10000_METADATA_FILE,
    HAM10000_REPORTS_DIR,
)
from src.data_audit.common.io_utils import ensure_dir, save_csv


def run(logger) -> pd.DataFrame:
    metadata = pd.read_csv(HAM10000_METADATA_FILE)

    counts = metadata["dx"].value_counts()
    pct = (100 * counts / counts.sum()).round(2)

    df = pd.DataFrame(
        {"dx": counts.index, "count": counts.values, "pct": pct.values}
    ).sort_values("count", ascending=False).reset_index(drop=True)

    imbalance_ratio = round(counts.max() / counts.min(), 2)

    out_path = HAM10000_REPORTS_DIR / "09_class_distribution.csv"
    save_csv(df, out_path)

    ensure_dir(HAM10000_FIGURES_DIR)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["dx"], df["count"], color="teal")
    ax.set_title("HAM10000 diagnostic (dx) class distribution")
    ax.set_xlabel("dx")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig_path = HAM10000_FIGURES_DIR / "class_distribution.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    logger.info("Class distribution (%d classes): %s", len(df), dict(zip(df["dx"], df["count"])))
    logger.info("Imbalance ratio (majority:minority): %.2f", imbalance_ratio)
    logger.info("Saved -> %s", out_path)
    logger.info("Saved figure -> %s", fig_path)

    return df
