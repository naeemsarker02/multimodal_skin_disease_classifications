"""Module 6: Class distribution.

Reports both the coarse (diagnosis_1: Benign/Malignant/Indeterminate)
and fine-grained (diagnosis_3: specific disease name) class breakdown.
diagnosis_3 is the level used as the primary class label for this
archive (see column_description.csv), since it is populated for nearly
every row and names concrete diseases comparable to PAD-UFES-20/HAM10000/
ISIC Archive 1's class labels.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data_audit.config import ISIC2_FIGURES_DIR, ISIC2_METADATA_FILE, ISIC2_REPORTS_DIR
from src.data_audit.common.io_utils import ensure_dir, save_csv


def run(logger) -> pd.DataFrame:
    metadata = pd.read_csv(ISIC2_METADATA_FILE)

    coarse_counts = metadata["diagnosis_1"].value_counts(dropna=False)
    coarse_df = pd.DataFrame({"diagnosis_1": coarse_counts.index.astype(str), "count": coarse_counts.values})
    save_csv(coarse_df, ISIC2_REPORTS_DIR / "08_class_distribution_coarse.csv")

    fine_counts = metadata["diagnosis_3"].value_counts(dropna=False)
    fine_pct = (100 * fine_counts / len(metadata)).round(2)
    fine_df = pd.DataFrame(
        {"diagnosis_3": fine_counts.index.astype(str), "count": fine_counts.values, "pct": fine_pct.values}
    )
    save_csv(fine_df, ISIC2_REPORTS_DIR / "08_class_distribution_fine.csv")

    n_missing_fine = metadata["diagnosis_3"].isna().sum()
    imbalance_ratio = round(fine_counts.iloc[0] / fine_counts[fine_counts.index.notna()].min(), 2) if fine_counts.index.notna().any() else None

    ensure_dir(ISIC2_FIGURES_DIR)
    plot_counts = fine_counts[fine_counts.index.notna()]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(plot_counts.index.astype(str), plot_counts.values, color="teal")
    ax.set_title("ISIC Archive 2 class distribution (diagnosis_3)")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    fig_path = ISIC2_FIGURES_DIR / "class_distribution.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    logger.info("Coarse (diagnosis_1): %s", coarse_counts.to_dict())
    logger.info("Fine (diagnosis_3), %d classes + %d missing: %s", len(plot_counts), n_missing_fine, plot_counts.to_dict())
    logger.info("Imbalance ratio (majority:minority) over fine classes: %s", imbalance_ratio)
    logger.info("Saved -> %s", ISIC2_REPORTS_DIR / "08_class_distribution_fine.csv")
    logger.info("Saved figure -> %s", fig_path)

    return fine_df
