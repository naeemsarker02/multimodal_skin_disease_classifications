"""Step 5: Split quality report.

Verifies the split actually did what it claims: zero lesion_id overlap
between train/val/test, and reports per-split size and class balance.
"""

import pandas as pd

from src.data_audit.common.io_utils import save_csv
from src.data_cleaning.config import HAM10000_PROCESSED_DIR


def run(logger, splits: dict) -> pd.DataFrame:
    lesion_sets = {name: set(df["lesion_id"]) for name, df in splits.items()}

    overlaps = []
    names = list(lesion_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = lesion_sets[names[i]] & lesion_sets[names[j]]
            if shared:
                overlaps.append((names[i], names[j], len(shared)))

    if overlaps:
        logger.error("Lesion leakage detected across splits: %s", overlaps)
        raise ValueError(f"Lesion leakage detected: {overlaps}")
    logger.info("No lesion_id overlap between train/val/test - split is leakage-free")

    rows = []
    total_images = sum(len(df) for df in splits.values())
    for split_name, split_df in splits.items():
        class_counts = split_df["disease_label"].value_counts()
        for label, count in class_counts.items():
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
    out_path = HAM10000_PROCESSED_DIR / "split_quality_report.csv"
    save_csv(report_df, out_path)

    for split_name, split_df in splits.items():
        logger.info(
            "%s: %d images (%.1f%% of total), %d lesions",
            split_name,
            len(split_df),
            100 * len(split_df) / total_images,
            split_df["lesion_id"].nunique(),
        )
    logger.info("Saved -> %s", out_path)

    return report_df
