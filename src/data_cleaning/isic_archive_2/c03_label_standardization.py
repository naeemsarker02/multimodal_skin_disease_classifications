"""Step 3: Label standardization.

Maps diagnosis_3 (the finest level populated for nearly all rows) to
the shared ISIC canonical disease taxonomy. Rows with no diagnosis_3
value (255 rows, 1.01% - see audit) have no usable fine-grained label
and are excluded from the labeled/split dataset, since diagnosis_1/2
alone are too coarse for the project's multi-class target; they are
kept in the standardized interim file but dropped before splitting.
"""

import pandas as pd

from src.data_audit.common.io_utils import save_csv
from src.data_cleaning.common_label_mapping import ISIC_LABEL_MAPPING
from src.data_cleaning.config import ISIC2_INTERIM_DIR, ISIC2_PROCESSED_DIR


def run(logger, df: pd.DataFrame) -> pd.DataFrame:
    present_codes = set(df["diagnosis_3"].dropna().str.lower())
    unknown = sorted(present_codes - set(ISIC_LABEL_MAPPING))
    if unknown:
        logger.error("Unmapped diagnosis_3 values found: %s", unknown)
        raise ValueError(f"Unmapped diagnosis_3 values: {unknown}")

    df["disease_label"] = df["diagnosis_3"].apply(
        lambda c: ISIC_LABEL_MAPPING[c.lower()][0] if pd.notna(c) else pd.NA
    )

    n_unlabeled = df["disease_label"].isna().sum()
    logger.warning("%d rows have no diagnosis_3 value and therefore no disease_label (excluded from splits, kept in interim file)", n_unlabeled)

    mapping_rows = [
        {"original_label": code, "standardized_label": name, "reason": reason}
        for code, (name, reason) in ISIC_LABEL_MAPPING.items()
        if code in present_codes
    ]
    save_csv(pd.DataFrame(mapping_rows), ISIC2_PROCESSED_DIR / "label_mapping.csv")

    out_path = ISIC2_INTERIM_DIR / "metadata_standardized.csv"
    save_csv(df, out_path)

    logger.info("Label standardization applied: %d codes -> %d standardized labels", len(present_codes), df["disease_label"].nunique())
    logger.info("Saved -> %s", ISIC2_PROCESSED_DIR / "label_mapping.csv")

    return df
