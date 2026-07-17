"""Step 1: Column standardization.

Renames columns to the unified cross-dataset schema, fixes dtypes,
resolves image_path back into data/raw/ (no image bytes copied), and
adds dataset_source. HAM10000 has no patient identifier - lesion_id is
the only grouping key available (see audit note 11).

No rows are dropped and no values are imputed - missing values found
in the audit remain NaN. This script only changes *representation*,
never *content*.
"""

import pandas as pd

from src.data_audit.common.io_utils import save_csv
from src.data_cleaning.config import (
    HAM10000_DATASET_SOURCE_NAME,
    HAM10000_IMAGE_SUBDIRS,
    HAM10000_INTERIM_DIR,
    HAM10000_METADATA_FILE,
    HAM10000_RAW_DIR,
    PROJECT_ROOT,
)

RENAME_MAP = {
    "dx": "diagnostic_code",
    "dx_type": "diagnosis_confirm_type",
    "localization": "anatomical_site",
}

STRING_COLUMNS = ["sex", "anatomical_site", "diagnostic_code", "diagnosis_confirm_type"]

# "unknown" is a literal string marker for real missingness in the raw
# CSV (see audit section 6-9) - isna() alone does not catch it. Mapping
# it to NA here makes that missingness visible without inventing any
# information, matching the same standardization decision applied to
# PAD-UFES-20's "UNK" strings.
UNKNOWN_STRING_COLUMNS = ["sex", "anatomical_site"]


def _find_image_path(image_id: str) -> str:
    for subdir in HAM10000_IMAGE_SUBDIRS:
        candidate = HAM10000_RAW_DIR / subdir / f"{image_id}.jpg"
        if candidate.exists():
            return str(candidate.relative_to(PROJECT_ROOT)).replace("\\", "/")
    return ""


def run(logger) -> pd.DataFrame:
    df = pd.read_csv(HAM10000_METADATA_FILE)
    logger.info("Loaded raw metadata: %d rows x %d columns", *df.shape)

    df = df.rename(columns=RENAME_MAP)
    logger.info("Renamed columns: %s", RENAME_MAP)

    for col in STRING_COLUMNS:
        df[col] = df[col].astype("string").str.strip().str.lower()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA})

    for col in UNKNOWN_STRING_COLUMNS:
        n_unknown = (df[col] == "unknown").sum()
        df[col] = df[col].replace({"unknown": pd.NA})
        logger.info("  %s: %d 'unknown' values mapped to NA (hidden missingness not caught by the audit's isna() check)", col, n_unknown)

    df["sex"] = df["sex"].str.upper()

    df["dataset_source"] = HAM10000_DATASET_SOURCE_NAME
    df["image_path"] = df["image_id"].apply(_find_image_path)

    unresolved = df[df["image_path"] == ""]
    if not unresolved.empty:
        logger.warning("%d image_id values could not be resolved to a raw file", len(unresolved))
    else:
        logger.info("All %d image_id values resolved to a file under data/raw/", len(df))

    out_path = HAM10000_INTERIM_DIR / "metadata_standardized.csv"
    save_csv(df, out_path)
    logger.info("Saved -> %s", out_path)

    return df
