"""Step 4: Split quality report.

Verifies no filename appears in more than one of train/val/test, and
reports per-split size and class balance. Filename overlap is the
relevant leakage check here since no lesion/patient identifier exists.
"""

import pandas as pd

from src.data_audit.common.io_utils import save_csv
from src.data_cleaning.config import ISIC1_PROCESSED_DIR


def run(logger, splits: dict) -> pd.DataFrame:
    filename_sets = {name: set(df["filename"]) for name, df in splits.items()}

    overlaps = []
    names = list(filename_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = filename_sets[names[i]] & filename_sets[names[j]]
            if shared:
                overlaps.append((names[i], names[j], len(shared)))

    if overlaps:
        logger.error("Filename leakage detected across splits: %s", overlaps)
        raise ValueError(f"Filename leakage detected: {overlaps}")
    logger.info("No filename overlap between train/val/test - split is leakage-free")

    rows = []
    total_images = sum(len(df) for df in splits.values())
    for split_name, split_df in splits.items():
        for label, count in split_df["disease_label"].value_counts().items():
            rows.append(
                {
                    "split": split_name,
                    "disease_label": label,
                    "count": count,
                    "pct_of_split": round(100 * count / len(split_df), 2),
                    "pct_of_total_images": round(100 * len(split_df) / total_images, 2),
                }
            )

    report_df = pd.DataFrame(rows)
    out_path = ISIC1_PROCESSED_DIR / "split_quality_report.csv"
    save_csv(report_df, out_path)

    for split_name, split_df in splits.items():
        logger.info("%s: %d images (%.1f%% of total)", split_name, len(split_df), 100 * len(split_df) / total_images)
    logger.info("Saved -> %s", out_path)

    return report_df
