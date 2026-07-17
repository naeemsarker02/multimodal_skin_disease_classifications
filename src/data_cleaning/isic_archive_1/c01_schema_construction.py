"""Step 1: Schema construction.

ISIC Archive 1 has no metadata.csv - the folder structure (Train|Test /
class_label / filename.jpg) IS the metadata. This step walks that
structure once and builds the standardized metadata dataframe.

The audit (reports/ISIC_Archive_1/02_duplicate_filename_label_conflicts.csv)
found 155 images filed under two conflicting class labels - a genuine
data-quality defect, not just a repeated file. Those images are
excluded here rather than arbitrarily kept under one label, since their
true class is ambiguous and keeping either guess would silently
introduce label noise.
"""

import pandas as pd

from src.data_audit.common.io_utils import list_files, save_csv
from src.data_cleaning.config import (
    ISIC1_DATASET_SOURCE_NAME,
    ISIC1_INTERIM_DIR,
    ISIC1_RAW_DIR,
    ISIC1_SPLIT_SUBDIRS,
)
from src.data_audit.config import VALID_IMAGE_EXTENSIONS, PROJECT_ROOT


def run(logger) -> tuple[pd.DataFrame, int]:
    rows = []
    for split in ISIC1_SPLIT_SUBDIRS:
        split_dir = ISIC1_RAW_DIR / split
        class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir()) if split_dir.exists() else []
        for class_dir in class_dirs:
            for fpath in list_files(class_dir, VALID_IMAGE_EXTENSIONS):
                rows.append(
                    {
                        "image_id": fpath.stem,
                        "filename": fpath.name,
                        "provided_split": split,
                        "class_label": class_dir.name,
                        "image_path": str(fpath.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    }
                )
    df = pd.DataFrame(rows)
    logger.info("Built metadata from folder structure: %d rows x %d columns", *df.shape)

    conflicting = df.groupby("filename")["class_label"].apply(lambda s: len(set(s)) > 1)
    conflicting_filenames = set(conflicting[conflicting].index)
    n_before = len(df)
    df = df[~df["filename"].isin(conflicting_filenames)].reset_index(drop=True)
    n_excluded = n_before - len(df)
    logger.warning(
        "Excluded %d rows (%d distinct filenames) with conflicting class labels across locations",
        n_excluded, len(conflicting_filenames),
    )

    df["dataset_source"] = ISIC1_DATASET_SOURCE_NAME

    out_path = ISIC1_INTERIM_DIR / "metadata_standardized.csv"
    save_csv(df, out_path)
    logger.info("Saved -> %s", out_path)

    return df, n_excluded
