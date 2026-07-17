"""Module 2: Image inventory.

Builds the single source-of-truth list of every image file found under
the two HAM10000_images_part_* folders. Every later image-related
module reads from this inventory instead of re-walking the filesystem.
"""

import pandas as pd

from src.data_audit.config import (
    HAM10000_IMAGE_SUBDIRS,
    HAM10000_RAW_DIR,
    HAM10000_REPORTS_DIR,
    VALID_IMAGE_EXTENSIONS,
)
from src.data_audit.common.io_utils import list_files, save_csv


def run(logger) -> pd.DataFrame:
    logger.info("Building image inventory from %s", HAM10000_IMAGE_SUBDIRS)

    rows = []
    for subdir in HAM10000_IMAGE_SUBDIRS:
        folder = HAM10000_RAW_DIR / subdir
        files = list_files(folder, VALID_IMAGE_EXTENSIONS)
        logger.info("  %s -> %d image files", subdir, len(files))
        for fpath in files:
            rows.append(
                {
                    "source_folder": subdir,
                    "filename": fpath.name,
                    "extension": fpath.suffix.lower(),
                    "size_bytes": fpath.stat().st_size,
                    "relative_path": str(fpath.relative_to(HAM10000_RAW_DIR)),
                }
            )

    df = pd.DataFrame(rows)

    duplicate_filenames = df[df.duplicated("filename", keep=False)]
    if not duplicate_filenames.empty:
        logger.warning(
            "Found %d filenames appearing in more than one location",
            duplicate_filenames["filename"].nunique(),
        )

    out_path = HAM10000_REPORTS_DIR / "02_image_inventory.csv"
    save_csv(df, out_path)

    logger.info("Total images inventoried: %d", len(df))
    logger.info("Saved -> %s", out_path)

    return df
