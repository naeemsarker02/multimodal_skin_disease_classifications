"""Module 9: Missing value analysis.

Per-column missing count and percentage via isna(), sorted descending
- the same convention used for PAD-UFES-20's audit. In addition, and
informed directly by the PAD-UFES-20 experience (where isna() missed
the 'UNK' string used in several columns), this module also reports a
literal-'unknown'-string count per text column so that HAM10000's
equivalent hidden missingness (sex, localization) is visible at audit
time rather than only discovered during cleaning.
"""

import pandas as pd

from src.data_audit.config import HAM10000_METADATA_FILE, HAM10000_REPORTS_DIR
from src.data_audit.common.io_utils import save_csv


def run(logger) -> pd.DataFrame:
    metadata = pd.read_csv(HAM10000_METADATA_FILE)
    n_rows = len(metadata)

    missing_count = metadata.isna().sum()
    missing_pct = (100 * missing_count / n_rows).round(2)

    literal_unknown_count = {}
    for col in metadata.columns:
        if metadata[col].dtype == object or pd.api.types.is_string_dtype(metadata[col]):
            literal_unknown_count[col] = int((metadata[col] == "unknown").sum())
        else:
            literal_unknown_count[col] = 0

    df = pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count_isna": missing_count.values,
            "missing_pct_isna": missing_pct.values,
            "literal_unknown_string_count": [literal_unknown_count[c] for c in missing_count.index],
            "dtype": [str(metadata[c].dtype) for c in missing_count.index],
        }
    ).sort_values("missing_pct_isna", ascending=False).reset_index(drop=True)

    out_path = HAM10000_REPORTS_DIR / "08_missing_value_report.csv"
    save_csv(df, out_path)

    logger.info("Missing values (isna()):")
    for _, row in df[df["missing_count_isna"] > 0].iterrows():
        logger.info("  %s: %d missing (%.2f%%)", row["column"], row["missing_count_isna"], row["missing_pct_isna"])

    hidden = df[df["literal_unknown_string_count"] > 0]
    if not hidden.empty:
        logger.warning(
            "Literal 'unknown' string values found (NOT counted by isna() above):"
        )
        for _, row in hidden.iterrows():
            logger.warning("  %s: %d rows have the literal string 'unknown'", row["column"], row["literal_unknown_string_count"])

    logger.info("Saved -> %s", out_path)

    return df
