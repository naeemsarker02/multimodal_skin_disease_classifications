"""Module 1: Folder structure analysis.

Walks data/raw/HAM10000/ (read-only) and records, per folder, how many
files of each extension it contains. This also surfaces the four
hmnist_*.csv files present in the raw folder (Kaggle-provided
precomputed pixel matrices) so their presence is documented rather
than silently ignored - they are out of scope for this pipeline, which
works from the original .jpg images and HAM10000_metadata.csv only.
"""

import os

import pandas as pd

from src.data_audit.config import HAM10000_RAW_DIR, HAM10000_REPORTS_DIR
from src.data_audit.common.io_utils import save_csv

OUT_OF_SCOPE_FILES = {
    "hmnist_28_28_L.csv",
    "hmnist_28_28_RGB.csv",
    "hmnist_8_8_L.csv",
    "hmnist_8_8_RGB.csv",
}


def run(logger) -> pd.DataFrame:
    logger.info("Starting folder structure analysis of %s", HAM10000_RAW_DIR)

    rows = []
    for dirpath, _dirnames, filenames in os.walk(HAM10000_RAW_DIR):
        if not filenames:
            continue
        ext_counts: dict[str, int] = {}
        total_size = 0
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            ext = os.path.splitext(fname)[1].lower() or "<no_ext>"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            total_size += os.path.getsize(fpath)

        rel_folder = os.path.relpath(dirpath, HAM10000_RAW_DIR)
        rows.append(
            {
                "folder": rel_folder,
                "num_files": len(filenames),
                "extensions_present": ", ".join(sorted(ext_counts.keys())),
                "extension_breakdown": ", ".join(
                    f"{ext}:{count}" for ext, count in sorted(ext_counts.items())
                ),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
            }
        )

    df = pd.DataFrame(rows).sort_values("folder").reset_index(drop=True)

    out_path = HAM10000_REPORTS_DIR / "01_folder_structure.csv"
    save_csv(df, out_path)

    logger.info("Folder structure report: %d folders analyzed", len(df))
    for _, row in df.iterrows():
        logger.info(
            "  %s -> %d files (%s)",
            row["folder"],
            row["num_files"],
            row["extension_breakdown"],
        )

    found_out_of_scope = [f for f in OUT_OF_SCOPE_FILES if (HAM10000_RAW_DIR / f).exists()]
    if found_out_of_scope:
        logger.info(
            "Out-of-scope files present (Kaggle precomputed pixel matrices, "
            "not used by this pipeline): %s",
            sorted(found_out_of_scope),
        )

    logger.info("Saved -> %s", out_path)

    return df
