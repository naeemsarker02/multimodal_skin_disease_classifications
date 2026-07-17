"""Step 5: Split quality report.

Verifies zero group_id (lesion_id, or image_id as fallback singleton
group) overlap between train/val/test, and reports per-split size and
class balance.
"""

import pandas as pd

from src.data_audit.common.io_utils import save_csv
from src.data_cleaning.config import ISIC2_PROCESSED_DIR


def _group_ids(df: pd.DataFrame) -> set:
    return set(df["lesion_id"].fillna(df["image_id"]))


def run(logger, splits: dict) -> pd.DataFrame:
    group_sets = {name: _group_ids(df) for name, df in splits.items()}

    overlaps = []
    names = list(group_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = group_sets[names[i]] & group_sets[names[j]]
            if shared:
                overlaps.append((names[i], names[j], len(shared)))

    if overlaps:
        logger.error("Group leakage detected across splits: %s", overlaps)
        raise ValueError(f"Group leakage detected: {overlaps}")
    logger.info("No lesion/image group overlap between train/val/test - split is leakage-free")

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
    out_path = ISIC2_PROCESSED_DIR / "split_quality_report.csv"
    save_csv(report_df, out_path)

    for split_name, split_df in splits.items():
        logger.info("%s: %d images (%.1f%% of total)", split_name, len(split_df), 100 * len(split_df) / total_images)
    logger.info("Saved -> %s", out_path)

    return report_df
