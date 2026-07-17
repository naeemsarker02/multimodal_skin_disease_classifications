"""Step 2: Value validation.

Flags implausible values in a report. Nothing is altered - anomalies
are the researcher's call.
"""

import pandas as pd

from src.data_audit.common.io_utils import save_csv
from src.data_cleaning.config import ISIC2_INTERIM_DIR

VALID_SEX_VALUES = {"MALE", "FEMALE"}
PLAUSIBLE_AGE_RANGE = (0, 110)
PLAUSIBLE_DIAMETER_RANGE_MM = (0, 200)


def run(logger, df: pd.DataFrame) -> pd.DataFrame:
    flags = []

    def flag(mask, column, reason):
        for idx in df[mask].index:
            flags.append(
                {
                    "row_index": idx,
                    "image_id": df.loc[idx, "image_id"],
                    "column": column,
                    "value": df.loc[idx, column],
                    "reason": reason,
                }
            )

    age_mask = df["age_approx"].notna() & ~df["age_approx"].between(*PLAUSIBLE_AGE_RANGE)
    flag(age_mask, "age_approx", f"outside plausible range {PLAUSIBLE_AGE_RANGE}")

    sex_mask = df["sex"].notna() & ~df["sex"].isin(VALID_SEX_VALUES)
    flag(sex_mask, "sex", f"not in expected set {VALID_SEX_VALUES}")

    diam_mask = df["clin_size_long_diam_mm"].notna() & ~df["clin_size_long_diam_mm"].between(*PLAUSIBLE_DIAMETER_RANGE_MM)
    flag(diam_mask, "clin_size_long_diam_mm", f"outside plausible diameter range (mm) {PLAUSIBLE_DIAMETER_RANGE_MM}")

    report_df = pd.DataFrame(flags)
    out_path = ISIC2_INTERIM_DIR / "value_validation_report.csv"
    save_csv(report_df, out_path)

    if report_df.empty:
        logger.info("Value validation: no anomalies flagged")
    else:
        logger.warning("Value validation: %d anomalies flagged (see report)", len(report_df))
        for col, count in report_df["column"].value_counts().items():
            logger.warning("  %s: %d flagged values", col, count)
    logger.info("Saved -> %s", out_path)

    return report_df
