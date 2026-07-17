"""Step 6: Dataset description.

Writes the final markdown document describing the processed HAM10000
dataset: schema, split methodology, seed, and known limitations. This
is the citable artifact for the thesis methodology section.
"""

from datetime import datetime

import pandas as pd

from src.data_audit.common.io_utils import save_text
from src.data_cleaning.config import HAM10000_PROCESSED_DIR, SPLIT_RATIOS, SPLIT_SEED


def run(logger, splits: dict, validation_report: pd.DataFrame) -> str:
    total_images = sum(len(df) for df in splits.values())
    total_lesions = len(set().union(*(set(df["lesion_id"]) for df in splits.values())))
    all_rows = pd.concat(splits.values())

    unknown_missing = {
        col: round(100 * all_rows[col].isna().sum() / total_images, 2)
        for col in ["sex", "anatomical_site"]
    }

    lines = []
    lines.append("# HAM10000 Processed Dataset Description")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Source")
    lines.append("")
    lines.append("Raw data: `data/raw/HAM10000/` (untouched, read-only).")
    lines.append(
        "Images are **not copied**. `image_path` in the metadata CSVs points "
        "back to the original file under `data/raw/HAM10000/HAM10000_images_part_*/` "
        "(disk space constraint - consistent with PAD-UFES-20). The Kaggle-provided "
        "`hmnist_*.csv` pixel-matrix files are out of scope and untouched."
    )
    lines.append("")
    lines.append("## Schema")
    lines.append("")
    lines.append(
        "All original HAM10000 columns are retained. Renamed for the unified "
        "cross-dataset schema: `dx`->`diagnostic_code`, `dx_type`->`diagnosis_confirm_type`, "
        "`localization`->`anatomical_site`. Added: `image_path`, `dataset_source`, `disease_label`."
    )
    lines.append("")
    lines.append(
        "**No patient identifier exists in this dataset** - `lesion_id` is the only "
        "grouping key, and it is not 1:1 with images (7,470 unique lesions across "
        "10,015 images)."
    )
    lines.append("")
    lines.append("## Label Standardization")
    lines.append("")
    lines.append(
        "See `label_mapping.csv` for the dx code -> disease_label mapping. Labels "
        "for diseases shared with PAD-UFES-20 (Basal Cell Carcinoma, Melanoma, "
        "Nevus) use identical standardized names for cross-dataset consistency; "
        "HAM10000-specific categories (Actinic Keratosis / Intraepithelial "
        "Carcinoma, Benign Keratosis-like Lesion, Dermatofibroma, Vascular "
        "Lesion) are kept distinct rather than force-mapped onto PAD-UFES-20's "
        "narrower codes."
    )
    lines.append("")
    lines.append("## Missing Values")
    lines.append("")
    lines.append(
        "No values were imputed or invented. For most columns, missingness "
        "matches the dataset audit exactly "
        "(see `reports/HAM10000/08_missing_value_report.csv`)."
    )
    lines.append("")
    lines.append(
        "One exception: the audit measured missingness with `isna()`, which "
        "could not detect that the raw `sex` and `localization` columns encode "
        "\"not assessed\" as the literal string `unknown` rather than a blank "
        "value. During cleaning, `unknown` was mapped to a proper missing value "
        "(no information was invented). Actual missingness for these columns:"
    )
    lines.append("")
    for col, pct in unknown_missing.items():
        lines.append(f"- `{col}`: {pct}% missing")
    lines.append("")
    lines.append("## Value Validation")
    lines.append("")
    if validation_report.empty:
        lines.append("No implausible values were flagged.")
    else:
        lines.append(
            f"{len(validation_report)} value(s) were flagged as anomalous "
            "(see `data/interim/HAM10000/value_validation_report.csv`). "
            "Flagged values were **not modified**."
        )
    lines.append("")
    lines.append("## Train / Validation / Test Split")
    lines.append("")
    lines.append(
        "- Method: **lesion-wise** split (not patient-wise - no patient "
        "identifier exists), stratified by each lesion's `disease_label`. The "
        "audit confirmed 0 lesions have inconsistent dx labels across their "
        "images, so stratification is exact (no dominant-label approximation "
        "needed, unlike PAD-UFES-20)."
    )
    lines.append(f"- Target ratios: {SPLIT_RATIOS}")
    lines.append(f"- Random seed: {SPLIT_SEED} (fixed, for reproducibility)")
    lines.append(f"- Total images: {total_images} | Total lesions: {total_lesions}")
    lines.append("")
    for split_name, split_df in splits.items():
        lines.append(
            f"- **{split_name}**: {len(split_df)} images, {split_df['lesion_id'].nunique()} lesions"
        )
    lines.append("")
    lines.append("No lesion_id appears in more than one split (verified in `split_quality_report.csv`).")
    lines.append("")
    lines.append("## Known Limitations")
    lines.append("")
    lines.append("- Class imbalance: `nv` (Nevus) is ~67% of the dataset; `df` (Dermatofibroma) is ~1.15%, a ~58:1 majority:minority ratio.")
    lines.append("- No patient identifier - cannot rule out the same patient contributing lesions to more than one split (only lesion-level leakage is controlled for).")
    lines.append("- `sex` missing for 57 rows, `anatomical_site` missing for 234 rows (both encoded as `unknown` in the raw data).")
    lines.append("- No clinical/lifestyle metadata (unlike PAD-UFES-20) - only age, sex, anatomical_site, and diagnosis confirmation method are available.")
    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    lines.append("- `metadata_train.csv`, `metadata_val.csv`, `metadata_test.csv`")
    lines.append("- `label_mapping.csv`")
    lines.append("- `lesion_split_assignment.csv`")
    lines.append("- `split_quality_report.csv`")

    text = "\n".join(lines)
    out_path = HAM10000_PROCESSED_DIR / "dataset_description.md"
    save_text(text, out_path)

    logger.info("Dataset description written -> %s", out_path)
    return text
