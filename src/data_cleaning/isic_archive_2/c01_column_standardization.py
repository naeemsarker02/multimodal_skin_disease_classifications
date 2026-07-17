"""Step 1: Column standardization.

Renames a handful of columns to the unified cross-dataset schema,
resolves image_path back into data/raw/ (no image bytes copied), and
adds dataset_source. No rows are dropped and no values are imputed.
"""

import pandas as pd

from src.data_audit.common.io_utils import save_csv
from src.data_cleaning.config import (
    ISIC2_DATASET_SOURCE_NAME,
    ISIC2_IMAGE_SUBDIRS,
    ISIC2_INTERIM_DIR,
    ISIC2_METADATA_FILE,
    ISIC2_RAW_DIR,
)
from src.data_cleaning.config import PROJECT_ROOT

RENAME_MAP = {"isic_id": "image_id"}

STRING_COLUMNS = ["sex", "diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_confirm_type"]


def _find_image_path(image_id: str) -> str:
    for subdir in ISIC2_IMAGE_SUBDIRS:
        candidate = ISIC2_RAW_DIR / subdir / f"{image_id}.jpg"
        if candidate.exists():
            return str(candidate.relative_to(PROJECT_ROOT)).replace("\\", "/")
    return ""


def run(logger) -> pd.DataFrame:
    df = pd.read_csv(ISIC2_METADATA_FILE)
    logger.info("Loaded raw metadata: %d rows x %d columns", *df.shape)

    df = df.rename(columns=RENAME_MAP)
    logger.info("Renamed columns: %s", RENAME_MAP)

    for col in STRING_COLUMNS:
        df[col] = df[col].astype("string").str.strip()

    df["sex"] = df["sex"].str.upper()

    df["dataset_source"] = ISIC2_DATASET_SOURCE_NAME
    df["image_path"] = df["image_id"].apply(_find_image_path)

    unresolved = df[df["image_path"] == ""]
    if not unresolved.empty:
        logger.warning("%d image_id values could not be resolved to a raw file", len(unresolved))
    else:
        logger.info("All %d image_id values resolved to a file under data/raw/", len(df))

    out_path = ISIC2_INTERIM_DIR / "metadata_standardized.csv"
    save_csv(df, out_path)
    logger.info("Saved -> %s", out_path)

    return df
