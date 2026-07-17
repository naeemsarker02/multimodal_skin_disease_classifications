"""Step 2: Label standardization.

Maps the folder-name class labels to the shared ISIC canonical disease
taxonomy (src/data_cleaning/common_label_mapping.py), used identically
by ISIC Archive 2 so overlapping diseases get identical strings.
"""

import pandas as pd

from src.data_audit.common.io_utils import save_csv
from src.data_cleaning.common_label_mapping import ISIC_LABEL_MAPPING
from src.data_cleaning.config import ISIC1_INTERIM_DIR, ISIC1_PROCESSED_DIR


def run(logger, df: pd.DataFrame) -> pd.DataFrame:
    unknown = sorted(set(df["class_label"].str.lower()) - set(ISIC_LABEL_MAPPING))
    if unknown:
        logger.error("Unmapped class labels found: %s", unknown)
        raise ValueError(f"Unmapped class labels: {unknown}")

    df["disease_label"] = df["class_label"].str.lower().map(lambda c: ISIC_LABEL_MAPPING[c][0])

    used_codes = sorted(set(df["class_label"].str.lower()))
    mapping_rows = [
        {"original_label": code, "standardized_label": ISIC_LABEL_MAPPING[code][0], "reason": ISIC_LABEL_MAPPING[code][1]}
        for code in used_codes
    ]
    save_csv(pd.DataFrame(mapping_rows), ISIC1_PROCESSED_DIR / "label_mapping.csv")

    out_path = ISIC1_INTERIM_DIR / "metadata_standardized.csv"
    save_csv(df, out_path)

    logger.info("Label standardization applied: %d codes -> %d standardized labels", len(used_codes), df["disease_label"].nunique())
    logger.info("Saved -> %s", ISIC1_PROCESSED_DIR / "label_mapping.csv")

    return df
