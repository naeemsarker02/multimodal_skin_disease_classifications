"""Step 1: Column standardization.

Renames a handful of columns to the unified cross-dataset schema,
fixes dtypes (proper nullable boolean for the True/False/blank
columns), normalizes categorical text, resolves image_path back into
data/raw/ (no image bytes are copied), and adds dataset_source.

No rows are dropped and no values are imputed - missing values found
in the audit remain NaN. This script only changes *representation*,
never *content*.
"""

import pandas as pd

from src.data_audit.common.io_utils import save_csv
from src.data_cleaning.config import (
    DATASET_SOURCE_NAME,
    PAD_UFES20_IMAGE_SUBDIRS,
    PAD_UFES20_INTERIM_DIR,
    PAD_UFES20_METADATA_FILE,
    PAD_UFES20_RAW_DIR,
    PROJECT_ROOT,
)

# Original column -> canonical column name.
RENAME_MAP = {
    "gender": "sex",
    "region": "anatomical_site",
    "img_id": "image_id",
    "diagnostic": "diagnostic_code",
}

# True/False/NaN already as native Python bool/NaN in the raw CSV.
PLAIN_BOOLEAN_COLUMNS = [
    "smoke",
    "drink",
    "pesticide",
    "skin_cancer_history",
    "cancer_history",
    "has_piped_water",
    "has_sewage_system",
    "biopsed",
]

# Stored as the strings "TRUE"/"FALSE"/"UNK" in the raw CSV. "UNK" means
# "not assessed" - it is real missingness that pandas' isna() does NOT
# detect (it's a non-null string), so the dataset audit's "0 missing"
# finding for these columns undercounted. Mapping UNK -> NA here makes
# that missingness visible without inventing any information: "unknown"
# is being represented the same way as every other missing value, not
# guessed at.
UNKNOWN_CODED_BOOLEAN_COLUMNS = ["itch", "grew", "hurt", "changed", "bleed", "elevation"]

STRING_COLUMNS = [
    "background_father",
    "background_mother",
    "sex",
    "anatomical_site",
    "diagnostic_code",
]

# Conceptual grouping documented for the thesis (resolves the
# symptoms / clinical_features ambiguity between README and
# Dataset_Strategy.md without collapsing the underlying columns).
SYMPTOM_COLUMNS = ["itch", "grew", "hurt", "changed", "bleed", "elevation"]
CLINICAL_FEATURE_COLUMNS = [
    "smoke",
    "drink",
    "pesticide",
    "skin_cancer_history",
    "cancer_history",
    "has_piped_water",
    "has_sewage_system",
    "fitspatrick",
    "background_father",
    "background_mother",
    "diameter_1",
    "diameter_2",
    "biopsed",
]


def _find_image_path(image_id: str) -> str:
    """Resolve an image_id to its actual location under data/raw/."""
    for subdir in PAD_UFES20_IMAGE_SUBDIRS:
        candidate = PAD_UFES20_RAW_DIR / subdir / image_id
        if candidate.exists():
            return str(candidate.relative_to(PROJECT_ROOT)).replace("\\", "/")
    return ""  # left blank + flagged by caller if unresolved


def run(logger) -> pd.DataFrame:
    df = pd.read_csv(PAD_UFES20_METADATA_FILE)
    logger.info("Loaded raw metadata: %d rows x %d columns", *df.shape)

    df = df.rename(columns=RENAME_MAP)
    logger.info("Renamed columns: %s", RENAME_MAP)

    for col in PLAIN_BOOLEAN_COLUMNS:
        before_na = df[col].isna().sum()
        df[col] = df[col].astype("boolean")  # pandas nullable boolean, keeps NA
        after_na = df[col].isna().sum()
        assert before_na == after_na, f"Missingness changed while casting {col}"
    logger.info("Cast %d plain boolean columns to nullable boolean dtype (NA preserved)", len(PLAIN_BOOLEAN_COLUMNS))

    for col in UNKNOWN_CODED_BOOLEAN_COLUMNS:
        n_unk = (df[col] == "UNK").sum()
        df[col] = df[col].map({"TRUE": True, "FALSE": False, "UNK": pd.NA})
        df[col] = df[col].astype("boolean")
        logger.info("  %s: %d 'UNK' values mapped to NA (hidden missingness not caught by the audit's isna() check)", col, n_unk)
    logger.info("Cast %d UNK-coded columns to nullable boolean dtype (UNK -> NA)", len(UNKNOWN_CODED_BOOLEAN_COLUMNS))

    for col in STRING_COLUMNS:
        df[col] = df[col].astype("string").str.strip().str.upper()
        df[col] = df[col].replace({"": pd.NA, "NAN": pd.NA})
    logger.info("Normalized %d categorical text columns (trim + uppercase)", len(STRING_COLUMNS))

    df["dataset_source"] = DATASET_SOURCE_NAME
    df["image_path"] = df["image_id"].apply(_find_image_path)

    unresolved = df[df["image_path"] == ""]
    if not unresolved.empty:
        logger.warning("%d image_id values could not be resolved to a raw file", len(unresolved))
    else:
        logger.info("All %d image_id values resolved to a file under data/raw/", len(df))

    out_path = PAD_UFES20_INTERIM_DIR / "metadata_standardized.csv"
    save_csv(df, out_path)
    logger.info("Saved -> %s", out_path)

    logger.info("Symptom columns (conceptual group): %s", SYMPTOM_COLUMNS)
    logger.info("Clinical feature columns (conceptual group): %s", CLINICAL_FEATURE_COLUMNS)

    return df
