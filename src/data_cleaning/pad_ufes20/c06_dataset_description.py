"""Step 6: Dataset description.

Writes the final markdown document describing the processed PAD-UFES-20
dataset: schema, split methodology, seed, and known limitations. This
is the citable artifact for the thesis methodology section.
"""

from datetime import datetime

import pandas as pd

from src.data_audit.common.io_utils import save_text
from src.data_cleaning.config import PAD_UFES20_PROCESSED_DIR, SPLIT_RATIOS, SPLIT_SEED
from src.data_cleaning.pad_ufes20.c01_column_standardization import (
    CLINICAL_FEATURE_COLUMNS,
    PLAIN_BOOLEAN_COLUMNS,
    SYMPTOM_COLUMNS,
    UNKNOWN_CODED_BOOLEAN_COLUMNS,
)


def run(logger, splits: dict, validation_report: pd.DataFrame) -> str:
    total_images = sum(len(df) for df in splits.values())
    total_patients = len(set().union(*(set(df["patient_id"]) for df in splits.values())))
    all_rows = pd.concat(splits.values())

    symptom_missing = {
        col: round(100 * all_rows[col].isna().sum() / total_images, 2)
        for col in UNKNOWN_CODED_BOOLEAN_COLUMNS
    }

    lines = []
    lines.append("# PAD-UFES-20 Processed Dataset Description")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Source")
    lines.append("")
    lines.append("Raw data: `data/raw/PAD_UFES20/` (untouched, read-only).")
    lines.append(
        "Images are **not copied**. `image_path` in the metadata CSVs points "
        "back to the original file under `data/raw/PAD_UFES20/imgs_part_*/` "
        "(disk space constraint - see project decision log)."
    )
    lines.append("")
    lines.append("## Schema")
    lines.append("")
    lines.append(
        "All original PAD-UFES-20 columns are retained. Renamed for the "
        "unified cross-dataset schema: `gender`->`sex`, `region`->`anatomical_site`, "
        "`img_id`->`image_id`, `diagnostic`->`diagnostic_code`. Added: `image_path`, "
        "`dataset_source`, `disease_label`."
    )
    lines.append("")
    lines.append(
        "The README/Dataset_Strategy conceptual groupings map to these concrete columns:"
    )
    lines.append(f"- **symptoms**: {', '.join(SYMPTOM_COLUMNS)}")
    lines.append(f"- **clinical_features**: {', '.join(CLINICAL_FEATURE_COLUMNS)}")
    lines.append("")
    lines.append("## Label Standardization")
    lines.append("")
    lines.append("See `label_mapping.csv` for the diagnostic code -> disease_label mapping.")
    lines.append("")
    lines.append("## Missing Values")
    lines.append("")
    lines.append(
        "No values were imputed or invented. For most columns, missingness "
        "percentages match the dataset audit exactly "
        "(see `reports/PAD_UFES20/08_missing_value_report.csv`)."
    )
    lines.append("")
    lines.append(
        "One exception: the audit measured missingness with `isna()`, which "
        "could not detect that the raw symptom columns "
        f"({', '.join(UNKNOWN_CODED_BOOLEAN_COLUMNS)}) encode \"not assessed\" "
        "as the literal string `UNK` rather than a blank value. During "
        "cleaning, `UNK` was mapped to a proper missing value (no information "
        "was invented - an existing \"unknown\" marker was simply "
        "standardized). Actual missingness for these columns is therefore "
        "higher than the audit reported:"
    )
    lines.append("")
    for col, pct in symptom_missing.items():
        lines.append(f"- `{col}`: {pct}% missing")
    lines.append("")
    lines.append("## Value Validation")
    lines.append("")
    if validation_report.empty:
        lines.append("No implausible values were flagged.")
    else:
        lines.append(
            f"{len(validation_report)} value(s) were flagged as anomalous "
            "(see `data/interim/PAD_UFES20/value_validation_report.csv`). "
            "Flagged values were **not modified**."
        )
    lines.append("")
    lines.append("## Train / Validation / Test Split")
    lines.append("")
    lines.append(f"- Method: patient-wise split, stratified by each patient's dominant `disease_label`.")
    lines.append(f"- Target ratios: {SPLIT_RATIOS}")
    lines.append(f"- Random seed: {SPLIT_SEED} (fixed, for reproducibility)")
    lines.append(f"- Total images: {total_images} | Total patients: {total_patients}")
    lines.append("")
    for split_name, split_df in splits.items():
        lines.append(
            f"- **{split_name}**: {len(split_df)} images, {split_df['patient_id'].nunique()} patients"
        )
    lines.append("")
    lines.append("No patient appears in more than one split (verified in `split_quality_report.csv`).")
    lines.append("")
    lines.append("## `lesion_id` Uniqueness")
    lines.append("")
    lines.append(
        "`lesion_id` is unique **per patient only**, not globally - two "
        "different patients can share the same `lesion_id` value. Always "
        "group or join on `(patient_id, lesion_id)` together, never on "
        "`lesion_id` alone."
    )
    lines.append("")
    lines.append("## Recommended Metadata Loading Strategy")
    lines.append("")
    lines.append(
        "Boolean columns are written to CSV as `True` / `False` / empty. "
        "`pandas.read_csv` parses these as plain Python `bool`/`NaN` objects "
        "but does not assign the nullable `boolean` dtype automatically, and "
        "which columns come back as `bool` vs `object` can vary by file "
        "depending on whether that file happens to contain any missing "
        "values. Downstream code should cast explicitly after loading:"
    )
    lines.append("")
    lines.append("```python")
    lines.append("boolean_columns = [")
    for col in PLAIN_BOOLEAN_COLUMNS + UNKNOWN_CODED_BOOLEAN_COLUMNS:
        lines.append(f'    "{col}",')
    lines.append("]")
    lines.append("df[boolean_columns] = df[boolean_columns].astype(\"boolean\")")
    lines.append("```")
    lines.append("")
    lines.append("## Known Limitations")
    lines.append("")
    lines.append("- Class imbalance: BCC/ACK are far more frequent than MEL (~16:1 majority:minority in the full dataset).")
    lines.append("- ~35% missingness in lifestyle/socioeconomic/measurement fields (smoke, drink, pesticide, cancer history, piped water, sewage system, diameters, Fitzpatrick).")
    lines.append("- 179 patients have images spanning more than one diagnosis; the split uses each patient's dominant diagnosis for stratification, so minority diagnoses for those patients may land in a different split than their dominant one.")
    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    lines.append("- `metadata_train.csv`, `metadata_val.csv`, `metadata_test.csv`")
    lines.append("- `label_mapping.csv`")
    lines.append("- `patient_split_assignment.csv`")
    lines.append("- `split_quality_report.csv`")

    text = "\n".join(lines)
    out_path = PAD_UFES20_PROCESSED_DIR / "dataset_description.md"
    save_text(text, out_path)

    logger.info("Dataset description written -> %s", out_path)
    return text
