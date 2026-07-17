"""Module 9: Missing value analysis.

Per-column missing count and percentage, sorted descending, so it is
clear which fields are structurally sparse in this dataset versus fully
populated - this directly informs the "mark as unavailable, never
invent data" rule for the later standardization phase.
"""

import pandas as pd

from src.data_audit.config import PAD_UFES20_METADATA_FILE, PAD_UFES20_REPORTS_DIR
from src.data_audit.common.io_utils import save_csv


def run(logger) -> pd.DataFrame:
    metadata = pd.read_csv(PAD_UFES20_METADATA_FILE)
    n_rows = len(metadata)

    missing_count = metadata.isna().sum()
    missing_pct = (100 * missing_count / n_rows).round(2)

    df = pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count": missing_count.values,
            "missing_pct": missing_pct.values,
            "dtype": [str(metadata[c].dtype) for c in missing_count.index],
        }
    ).sort_values("missing_pct", ascending=False).reset_index(drop=True)

    out_path = PAD_UFES20_REPORTS_DIR / "08_missing_value_report.csv"
    save_csv(df, out_path)

    fully_populated = df[df["missing_count"] == 0]["column"].tolist()
    has_missing = df[df["missing_count"] > 0]

    logger.info("Columns with zero missing values: %d -> %s", len(fully_populated), fully_populated)
    logger.info("Columns with missing values: %d", len(has_missing))
    for _, row in has_missing.iterrows():
        logger.info("  %s: %d missing (%.2f%%)", row["column"], row["missing_count"], row["missing_pct"])
    logger.info("Saved -> %s", out_path)

    return df
