"""Module 1: Folder structure + image inventory.

ISIC Archive 1 has no metadata.csv - it ships pre-split into Train/Test,
each containing one subfolder per class, and the folder name itself IS
the label (e.g. Train/melanoma/ISIC_0024310.jpg). This module walks
that structure once and builds the single source-of-truth inventory
(split, class_label, filename, relative_path) that every later module
in this audit reads from, combining what would otherwise be separate
"folder structure" and "image inventory" modules since here they are
the same walk.
"""

import pandas as pd

from src.data_audit.config import ISIC1_RAW_DIR, ISIC1_REPORTS_DIR, ISIC1_SPLIT_SUBDIRS, VALID_IMAGE_EXTENSIONS
from src.data_audit.common.io_utils import list_files, save_csv


def run(logger) -> pd.DataFrame:
    logger.info("Building folder structure + image inventory from %s", ISIC1_RAW_DIR)

    rows = []
    folder_rows = []
    for split in ISIC1_SPLIT_SUBDIRS:
        split_dir = ISIC1_RAW_DIR / split
        class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir()) if split_dir.exists() else []
        for class_dir in class_dirs:
            files = list_files(class_dir, VALID_IMAGE_EXTENSIONS)
            folder_rows.append({"split": split, "class_label": class_dir.name, "num_files": len(files)})
            for fpath in files:
                rows.append(
                    {
                        "split": split,
                        "class_label": class_dir.name,
                        "filename": fpath.name,
                        "extension": fpath.suffix.lower(),
                        "size_bytes": fpath.stat().st_size,
                        "relative_path": str(fpath.relative_to(ISIC1_RAW_DIR)),
                    }
                )

    inventory_df = pd.DataFrame(rows)
    folder_df = pd.DataFrame(folder_rows)

    save_csv(folder_df, ISIC1_REPORTS_DIR / "01_folder_structure.csv")
    save_csv(inventory_df, ISIC1_REPORTS_DIR / "02_image_inventory.csv")

    logger.info("Splits: %s | Classes per split: %s", ISIC1_SPLIT_SUBDIRS, sorted(inventory_df["class_label"].unique()))
    for _, row in folder_df.iterrows():
        logger.info("  %s/%s -> %d files", row["split"], row["class_label"], row["num_files"])
    logger.info("Total images inventoried: %d", len(inventory_df))

    duplicate_filenames = inventory_df[inventory_df.duplicated("filename", keep=False)]
    if not duplicate_filenames.empty:
        logger.warning(
            "Found %d filenames appearing in more than one location (possible Train/Test or cross-class duplication)",
            duplicate_filenames["filename"].nunique(),
        )
        class_pairs = duplicate_filenames.groupby("filename")["class_label"].apply(lambda s: tuple(sorted(set(s))))
        conflicting = class_pairs[class_pairs.apply(len) > 1]
        if not conflicting.empty:
            logger.warning(
                "  %d of those are the SAME image filed under two DIFFERENT class labels "
                "(a label conflict, not just a split leak): %s",
                len(conflicting),
                conflicting.value_counts().to_dict(),
            )
    dup_path = ISIC1_REPORTS_DIR / "02_duplicate_filenames.csv"
    save_csv(duplicate_filenames, dup_path)

    logger.info("Saved -> %s", ISIC1_REPORTS_DIR / "01_folder_structure.csv")
    logger.info("Saved -> %s", ISIC1_REPORTS_DIR / "02_image_inventory.csv")
    logger.info("Saved -> %s", dup_path)

    return inventory_df
