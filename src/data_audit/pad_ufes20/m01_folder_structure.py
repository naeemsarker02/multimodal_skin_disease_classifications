"""Module 1: Folder structure analysis.

Walks data/raw/PAD_UFES20/ (read-only) and records, per folder, how many
files of each extension it contains. This establishes that the raw
layout matches what the rest of the pipeline assumes before any script
relies on it.
"""

import os

import pandas as pd

from src.data_audit.config import PAD_UFES20_RAW_DIR, PAD_UFES20_REPORTS_DIR
from src.data_audit.common.io_utils import save_csv


def run(logger) -> pd.DataFrame:
    logger.info("Starting folder structure analysis of %s", PAD_UFES20_RAW_DIR)

    rows = []
    for dirpath, _dirnames, filenames in os.walk(PAD_UFES20_RAW_DIR):
        if not filenames:
            continue
        ext_counts: dict[str, int] = {}
        total_size = 0
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            ext = os.path.splitext(fname)[1].lower() or "<no_ext>"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            total_size += os.path.getsize(fpath)

        rel_folder = os.path.relpath(dirpath, PAD_UFES20_RAW_DIR)
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

    out_path = PAD_UFES20_REPORTS_DIR / "01_folder_structure.csv"
    save_csv(df, out_path)

    logger.info("Folder structure report: %d folders analyzed", len(df))
    for _, row in df.iterrows():
        logger.info(
            "  %s -> %d files (%s)",
            row["folder"],
            row["num_files"],
            row["extension_breakdown"],
        )
    logger.info("Saved -> %s", out_path)

    return df
