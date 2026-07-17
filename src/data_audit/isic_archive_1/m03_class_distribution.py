"""Module 3: Class distribution.

Value counts and percentages per class_label, broken down by the
Train/Test split that ships with this archive, plus a bar chart.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data_audit.config import ISIC1_FIGURES_DIR, ISIC1_REPORTS_DIR
from src.data_audit.common.io_utils import ensure_dir, save_csv


def run(logger, inventory_df: pd.DataFrame) -> pd.DataFrame:
    counts = inventory_df.groupby(["split", "class_label"]).size().reset_index(name="count")
    totals_per_split = inventory_df.groupby("split").size()
    counts["pct_of_split"] = counts.apply(lambda r: round(100 * r["count"] / totals_per_split[r["split"]], 2), axis=1)

    save_csv(counts, ISIC1_REPORTS_DIR / "05_class_distribution.csv")

    overall = inventory_df["class_label"].value_counts()
    imbalance_ratio = round(overall.max() / overall.min(), 2)

    ensure_dir(ISIC1_FIGURES_DIR)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(overall.index, overall.values, color="teal")
    ax.set_title("ISIC Archive 1 class distribution (Train+Test combined)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig_path = ISIC1_FIGURES_DIR / "class_distribution.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    logger.info("Classes: %d | Overall counts: %s", overall.nunique(), overall.to_dict())
    logger.info("Imbalance ratio (majority:minority): %.2f", imbalance_ratio)
    logger.info("Saved -> %s", ISIC1_REPORTS_DIR / "05_class_distribution.csv")
    logger.info("Saved figure -> %s", fig_path)

    return counts
