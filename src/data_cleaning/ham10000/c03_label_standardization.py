"""Step 3: Label standardization.

Maps HAM10000's short dx codes to full standardized disease names, and
writes label_mapping.csv documenting the original code, the
standardized label, and the reason for the mapping.

Labels that denote the same underlying disease as PAD-UFES-20 use the
identical standardized name (Basal Cell Carcinoma, Melanoma, Nevus) so
the two datasets share a common taxonomy for cross-dataset experiments.
HAM10000-specific categories (bkl, akiec, df, vasc) are given their own
distinct standardized names rather than being force-mapped onto the
closest PAD-UFES-20 code, since they cover different or broader
clinical groupings.
"""

import pandas as pd

from src.data_audit.common.io_utils import save_csv
from src.data_cleaning.config import HAM10000_INTERIM_DIR, HAM10000_PROCESSED_DIR

LABEL_MAPPING = {
    "bcc": ("Basal Cell Carcinoma", "Same disease as PAD-UFES-20's BCC code; standardized name shared across datasets."),
    "mel": ("Melanoma", "Same disease as PAD-UFES-20's MEL code; standardized name shared across datasets."),
    "nv": ("Nevus", "Melanocytic nevus; same disease as PAD-UFES-20's NEV code; standardized name shared across datasets."),
    "akiec": ("Actinic Keratosis / Intraepithelial Carcinoma", "HAM10000's akiec groups actinic keratoses with Bowen's disease (SCC in situ); kept distinct from PAD-UFES-20's narrower ACK code."),
    "bkl": ("Benign Keratosis-like Lesion", "HAM10000's bkl groups solar lentigines, seborrheic keratoses, and lichen-planus-like keratoses; broader than PAD-UFES-20's SEK code, so kept distinct."),
    "df": ("Dermatofibroma", "HAM10000-specific class, not present in PAD-UFES-20."),
    "vasc": ("Vascular Lesion", "HAM10000-specific class, not present in PAD-UFES-20."),
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
    save_csv(mapping_df, HAM10000_PROCESSED_DIR / "label_mapping.csv")

    out_path = HAM10000_INTERIM_DIR / "metadata_standardized.csv"
    save_csv(df, out_path)

    logger.info("Label standardization applied: %d codes -> %d standardized labels", len(LABEL_MAPPING), df["disease_label"].nunique())
    logger.info("Saved -> %s", HAM10000_PROCESSED_DIR / "label_mapping.csv")
    logger.info("Updated -> %s", out_path)

    return df
