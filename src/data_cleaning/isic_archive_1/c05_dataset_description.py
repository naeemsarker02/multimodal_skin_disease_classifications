"""Step 5: Dataset description."""

from datetime import datetime

import pandas as pd

from src.data_audit.common.io_utils import save_text
from src.data_cleaning.config import ISIC1_PROCESSED_DIR, SPLIT_SEED
from src.data_cleaning.isic_archive_1.c03_val_split import VAL_FRACTION_OF_TRAIN


def run(logger, splits: dict, n_excluded_conflicts: int) -> str:
    total_images = sum(len(df) for df in splits.values())

    lines = []
    lines.append("# ISIC Archive 1 Processed Dataset Description")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Source")
    lines.append("")
    lines.append("Raw data: `data/raw/ISIC_Archive_1/` (untouched, read-only).")
    lines.append(
        "No metadata.csv ships with this archive - class labels come from the folder "
        "structure (`Train|Test/<class_label>/*.jpg`). Images are **not copied**; "
        "`image_path` points back to the original file."
    )
    lines.append("")
    lines.append("## Data-Quality Exclusion")
    lines.append("")
    lines.append(
        f"The audit (`reports/ISIC_Archive_1/02_duplicate_filename_label_conflicts.csv`) found "
        f"{n_excluded_conflicts} images filed under two different, conflicting class labels "
        "(the same filename appearing under e.g. both `melanoma` and `seborrheic keratosis`, "
        "systematically - 78 and 77 occurrences respectively for the two conflicting pairs). "
        "These images were **excluded** from the cleaned dataset rather than arbitrarily "
        "assigned to one label, since their true class cannot be determined from this archive alone."
    )
    lines.append("")
    lines.append("## Label Standardization")
    lines.append("")
    lines.append(
        "See `label_mapping.csv`. Uses the taxonomy shared with ISIC Archive 2 "
        "(`src/data_cleaning/common_label_mapping.py`) so diseases common to both "
        "archives (and to PAD-UFES-20/HAM10000 where applicable) use identical names."
    )
    lines.append("")
    lines.append("## Train / Validation / Test Split")
    lines.append("")
    lines.append(
        "- No lesion or patient identifier exists in this archive - each image is treated "
        "as independent."
    )
    lines.append(
        f"- The archive's own **Test** split is kept exactly as provided (final held-out set, "
        "untouched). A validation set is carved out of the archive's **Train** split only "
        f"({VAL_FRACTION_OF_TRAIN:.0%} of Train, stratified by disease_label)."
    )
    lines.append(f"- Random seed: {SPLIT_SEED} (fixed, for reproducibility)")
    lines.append(f"- Total images: {total_images}")
    lines.append("")
    for split_name, split_df in splits.items():
        lines.append(f"- **{split_name}**: {len(split_df)} images")
    lines.append("")
    lines.append("No filename appears in more than one split (verified in `split_quality_report.csv`).")
    lines.append("")
    lines.append("## Known Limitations")
    lines.append("")
    lines.append("- Test set is small and imbalanced (e.g. only 3 seborrheic keratosis and 3 vascular lesion images).")
    lines.append("- No clinical/demographic metadata (unlike PAD-UFES-20) - image + class label only.")
    lines.append("- No patient identifier - cannot rule out the same patient/lesion contributing images to both Train and Test.")
    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    lines.append("- `metadata_train.csv`, `metadata_val.csv`, `metadata_test.csv`")
    lines.append("- `label_mapping.csv`")
    lines.append("- `split_quality_report.csv`")

    text = "\n".join(lines)
    out_path = ISIC1_PROCESSED_DIR / "dataset_description.md"
    save_text(text, out_path)

    logger.info("Dataset description written -> %s", out_path)
    return text
