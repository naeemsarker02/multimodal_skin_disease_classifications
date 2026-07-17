"""Module 5: Missing value analysis via isna(). No hidden 'unknown'
string markers were found in this metadata (unlike HAM10000/PAD-UFES-20),
so isna() alone is a reliable missingness measure here.
"""

import pandas as pd

from src.data_audit.config import ISIC2_METADATA_FILE, ISIC2_REPORTS_DIR
from src.data_audit.common.io_utils import save_csv


def run(logger) -> pd.DataFrame:
    metadata = pd.read_csv(ISIC2_METADATA_FILE)
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

    out_path = ISIC2_REPORTS_DIR / "07_missing_value_report.csv"
    save_csv(df, out_path)

    logger.info("Missing values (isna()), highest first:")
    for _, row in df[df["missing_count"] > 0].iterrows():
        logger.info("  %s: %d missing (%.2f%%)", row["column"], row["missing_count"], row["missing_pct"])
    logger.info("Saved -> %s", out_path)

    return df
