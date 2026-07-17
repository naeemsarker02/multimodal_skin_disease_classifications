"""Step 3: Label standardization.

Maps PAD-UFES-20's short diagnostic codes to full standardized disease
names, and writes label_mapping.csv documenting the original code,
the standardized label, and the reason for the mapping - required so
the mapping is defensible and traceable in the thesis.
"""

import pandas as pd

from src.data_audit.common.io_utils import save_csv
from src.data_cleaning.config import PAD_UFES20_INTERIM_DIR, PAD_UFES20_PROCESSED_DIR

LABEL_MAPPING = {
    "BCC": ("Basal Cell Carcinoma", "Standard dermatological full name for code BCC."),
    "SCC": ("Squamous Cell Carcinoma", "Standard dermatological full name for code SCC."),
    "ACK": ("Actinic Keratosis", "Standard dermatological full name for code ACK."),
    "NEV": ("Nevus", "Standard dermatological full name for code NEV (common mole)."),
    "SEK": ("Seborrheic Keratosis", "Standard dermatological full name for code SEK."),
    "MEL": ("Melanoma", "Standard dermatological full name for code MEL."),
}


def run(logger, df: pd.DataFrame) -> pd.DataFrame:
    unknown_codes = sorted(set(df["diagnostic_code"].dropna()) - set(LABEL_MAPPING))
    if unknown_codes:
        logger.error("Unmapped diagnostic codes found: %s", unknown_codes)
        raise ValueError(f"Unmapped diagnostic codes: {unknown_codes}")

    df["disease_label"] = df["diagnostic_code"].map(lambda c: LABEL_MAPPING[c][0])

    mapping_rows = [
        {"original_label": code, "standardized_label": name, "reason": reason}
        for code, (name, reason) in LABEL_MAPPING.items()
    ]
    mapping_df = pd.DataFrame(mapping_rows)
    save_csv(mapping_df, PAD_UFES20_PROCESSED_DIR / "label_mapping.csv")

    out_path = PAD_UFES20_INTERIM_DIR / "metadata_standardized.csv"
    save_csv(df, out_path)

    logger.info("Label standardization applied: %d codes -> %d standardized labels", len(LABEL_MAPPING), df["disease_label"].nunique())
    logger.info("Saved -> %s", PAD_UFES20_PROCESSED_DIR / "label_mapping.csv")
    logger.info("Updated -> %s", out_path)

    return df
