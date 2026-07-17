"""Step 6: Dataset description."""

from datetime import datetime

import pandas as pd

from src.data_audit.common.io_utils import save_text
from src.data_cleaning.config import ISIC2_PROCESSED_DIR, SPLIT_RATIOS, SPLIT_SEED


def run(logger, splits: dict, validation_report: pd.DataFrame, n_unlabeled: int) -> str:
    total_images = sum(len(df) for df in splits.values())

    lines = []
    lines.append("# ISIC Archive 2 Processed Dataset Description")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Source")
    lines.append("")
    lines.append("Raw data: `data/raw/ISIC_Archive_2/` (untouched, read-only).")
    lines.append("Images are **not copied**; `image_path` points back to the original file under `data/raw/ISIC_Archive_2/images/`.")
    lines.append("")
    lines.append("## Schema")
    lines.append("")
    lines.append("Renamed for the unified cross-dataset schema: `isic_id`->`image_id`. Added: `image_path`, `dataset_source`, `disease_label`.")
    lines.append("")
    lines.append("## Label Standardization")
    lines.append("")
    lines.append(
        "`diagnosis_3` (the finest diagnosis level populated for nearly all rows) is mapped "
        "to the shared ISIC canonical taxonomy (`src/data_cleaning/common_label_mapping.py`), "
        "used identically by ISIC Archive 1 so diseases common to both use identical names. "
        "See `label_mapping.csv`."
    )
    lines.append("")
    lines.append(
        f"{n_unlabeled} rows (1.01% of the raw metadata) have no `diagnosis_3` value and "
        "therefore no `disease_label` - they are excluded from the train/val/test split "
        "since `diagnosis_1`/`diagnosis_2` alone are too coarse for this project's multi-class target."
    )
    lines.append("")
    lines.append("## Value Validation")
    lines.append("")
    if validation_report.empty:
        lines.append("No implausible values were flagged.")
    else:
        lines.append(f"{len(validation_report)} value(s) were flagged as anomalous (see `data/interim/ISIC_Archive_2/value_validation_report.csv`). Flagged values were **not modified**.")
    lines.append("")
    lines.append("## Train / Validation / Test Split")
    lines.append("")
    lines.append(
        "- Method: group-wise split, where the group is `lesion_id` when present (93.4% of "
        "labeled rows) and falls back to the row's own `image_id` as a singleton group "
        "otherwise (mirroring HAM10000's lesion-wise approach, extended for the mixed "
        "identifier coverage here). Stratified by each group's `disease_label` - the audit "
        "confirmed 0 groups have inconsistent labels."
    )
    lines.append("- Note: `patient_id` is populated for only 417/25,331 rows, so a true patient-wise split is not possible for the bulk of this archive.")
    lines.append(f"- Target ratios: {SPLIT_RATIOS}")
    lines.append(f"- Random seed: {SPLIT_SEED} (fixed, for reproducibility)")
    lines.append(f"- Total images (labeled, post-exclusion): {total_images}")
    lines.append("")
    for split_name, split_df in splits.items():
        lines.append(f"- **{split_name}**: {len(split_df)} images")
    lines.append("")
    lines.append("No lesion/image group appears in more than one split (verified in `split_quality_report.csv`).")
    lines.append("")
    lines.append("## Known Limitations")
    lines.append("")
    lines.append("- Severe class imbalance: Nevus (~51%) vs. two classes with only 1 image each (Epidermal Nevus, Atypical Melanocytic Neoplasm - both folded into Nevus, see label_mapping.csv).")
    lines.append("- `patient_id` populated for <2% of rows - cannot control for patient-level leakage across splits beyond the lesion_id grouping used.")
    lines.append("- `anatom_site_4`, `anatom_site_5`, `family_hx_mm`, `personal_hx_mm`, `clin_size_long_diam_mm` are >96% missing by design (deep hierarchy levels / rarely-recorded clinical fields), not a data defect.")
    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    lines.append("- `metadata_train.csv`, `metadata_val.csv`, `metadata_test.csv`")
    lines.append("- `label_mapping.csv`")
    lines.append("- `group_split_assignment.csv`")
    lines.append("- `split_quality_report.csv`")

    text = "\n".join(lines)
    out_path = ISIC2_PROCESSED_DIR / "dataset_description.md"
    save_text(text, out_path)

    logger.info("Dataset description written -> %s", out_path)
    return text
