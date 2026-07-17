"""Module 10: Class distribution.

Value counts and percentages for the diagnostic (target) column, plus
an imbalance ratio and a bar chart - informs loss weighting / sampling
strategy decisions in later modeling phases.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data_audit.config import (
    PAD_UFES20_FIGURES_DIR,
    PAD_UFES20_METADATA_FILE,
    PAD_UFES20_REPORTS_DIR,
)
from src.data_audit.common.io_utils import ensure_dir, save_csv


def run(logger) -> pd.DataFrame:
    metadata = pd.read_csv(PAD_UFES20_METADATA_FILE)

    counts = metadata["diagnostic"].value_counts()
    pct = (100 * counts / counts.sum()).round(2)

    df = pd.DataFrame(
        {"diagnostic": counts.index, "count": counts.values, "pct": pct.values}
    ).sort_values("count", ascending=False).reset_index(drop=True)

    imbalance_ratio = round(counts.max() / counts.min(), 2)

    out_path = PAD_UFES20_REPORTS_DIR / "09_class_distribution.csv"
    save_csv(df, out_path)

    ensure_dir(PAD_UFES20_FIGURES_DIR)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["diagnostic"], df["count"], color="teal")
    ax.set_title("PAD-UFES-20 diagnostic class distribution")
    ax.set_xlabel("diagnostic")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig_path = PAD_UFES20_FIGURES_DIR / "class_distribution.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    logger.info("Class distribution (%d classes): %s", len(df), dict(zip(df["diagnostic"], df["count"])))
    logger.info("Imbalance ratio (majority:minority): %.2f", imbalance_ratio)
    logger.info("Saved -> %s", out_path)
    logger.info("Saved figure -> %s", fig_path)

    return df
