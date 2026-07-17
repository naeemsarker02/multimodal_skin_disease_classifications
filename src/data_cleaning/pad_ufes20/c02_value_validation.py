"""Step 2: Value validation.

Flags implausible values (e.g. out-of-range age, non-standard sex
categories) in a report. Nothing is altered here - anomalies are the
researcher's call, not something a cleaning script should silently
"fix".
"""

import pandas as pd

from src.data_audit.common.io_utils import save_csv
from src.data_cleaning.config import PAD_UFES20_INTERIM_DIR

VALID_SEX_VALUES = {"MALE", "FEMALE"}
VALID_FITZPATRICK_RANGE = (1, 6)
PLAUSIBLE_AGE_RANGE = (0, 110)
PLAUSIBLE_DIAMETER_RANGE_MM = (0, 100)


def run(logger, df: pd.DataFrame) -> pd.DataFrame:
    flags = []

    def flag(mask, column, reason):
        for idx in df[mask].index:
            flags.append(
                {
                    "row_index": idx,
                    "patient_id": df.loc[idx, "patient_id"],
                    "image_id": df.loc[idx, "image_id"],
                    "column": column,
                    "value": df.loc[idx, column],
                    "reason": reason,
                }
            )

    age_mask = df["age"].notna() & ~df["age"].between(*PLAUSIBLE_AGE_RANGE)
    flag(age_mask, "age", f"outside plausible range {PLAUSIBLE_AGE_RANGE}")

    sex_mask = df["sex"].notna() & ~df["sex"].isin(VALID_SEX_VALUES)
    flag(sex_mask, "sex", f"not in expected set {VALID_SEX_VALUES}")

    fitz_mask = df["fitspatrick"].notna() & ~df["fitspatrick"].between(*VALID_FITZPATRICK_RANGE)
    flag(fitz_mask, "fitspatrick", f"outside Fitzpatrick scale {VALID_FITZPATRICK_RANGE}")

    for col in ["diameter_1", "diameter_2"]:
        d_mask = df[col].notna() & ~df[col].between(*PLAUSIBLE_DIAMETER_RANGE_MM)
        flag(d_mask, col, f"outside plausible diameter range (mm) {PLAUSIBLE_DIAMETER_RANGE_MM}")

    report_df = pd.DataFrame(flags)
    out_path = PAD_UFES20_INTERIM_DIR / "value_validation_report.csv"
    save_csv(report_df, out_path)

    if report_df.empty:
        logger.info("Value validation: no anomalies flagged")
    else:
        logger.warning("Value validation: %d anomalies flagged (see report)", len(report_df))
        for col, count in report_df["column"].value_counts().items():
            logger.warning("  %s: %d flagged values", col, count)
    logger.info("Saved -> %s", out_path)

    return report_df
