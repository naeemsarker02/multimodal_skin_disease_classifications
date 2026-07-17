"""Module 8: Dataset summary."""

from datetime import datetime

import pandas as pd

from src.data_audit.config import ISIC2_REPORTS_DIR
from src.data_audit.common.io_utils import save_text


def run(logger, results: dict) -> str:
    inventory_df: pd.DataFrame = results["inventory"]
    verification_df: pd.DataFrame = results["verification"]
    missing_stats: dict = results["missing_image_stats"]
    size_stats_df: pd.DataFrame = results["size_stats"]
    class_dist_fine_df: pd.DataFrame = results["class_distribution"]
    lesion_stats: dict = results["lesion_stats"]

    n_ok = (verification_df["status"] == "OK").sum()
    n_corrupted = (verification_df["status"] == "CORRUPTED").sum()
    width_row = size_stats_df[size_stats_df["metric"] == "width"].iloc[0]
    height_row = size_stats_df[size_stats_df["metric"] == "height"].iloc[0]

    lines = []
    lines.append("# ISIC Archive 2 Dataset Audit Summary")
    lines.append("")
    lines.append(f"Audit date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 1. Folder Structure & Image Inventory")
    lines.append("")
    lines.append("- Flat `images/` folder, single `metadata.csv` (27 columns), no pre-existing split.")
    lines.append(f"- Total images found on disk: {len(inventory_df)}")
    lines.append("- See: `01_folder_structure.csv`, `02_image_inventory.csv`")
    lines.append("")
    lines.append("## 2. Image Verification & Corrupted Images")
    lines.append("")
    lines.append(f"- Images successfully decoded (OK): {n_ok}")
    lines.append(f"- Corrupted / unreadable images: {n_corrupted}")
    lines.append("- See: `03_image_verification.csv`, `03_corrupted_images.csv`")
    lines.append("")
    lines.append("## 3. Missing / Orphan Images")
    lines.append("")
    lines.append(f"- Metadata rows: {missing_stats['metadata_rows']}")
    lines.append(f"- Inventory files: {missing_stats['inventory_files']}")
    lines.append(f"- Missing images (in metadata, not on disk): {missing_stats['missing_count']}")
    lines.append(f"- Orphan images (on disk, not in metadata): {missing_stats['orphan_count']}")
    lines.append(f"- Duplicate isic_id entries in metadata: {missing_stats['duplicate_isic_id_count']}")
    lines.append("- See: `05_missing_images.csv`, `05_orphan_images.csv`, `05_duplicate_isic_id.csv`")
    lines.append("")
    lines.append("## 4. Image Size Statistics")
    lines.append("")
    lines.append(
        f"- Width: min={width_row['min']:.0f} max={width_row['max']:.0f} mean={width_row['mean']:.1f} median={width_row['median']:.0f} px"
    )
    lines.append(
        f"- Height: min={height_row['min']:.0f} max={height_row['max']:.0f} mean={height_row['mean']:.1f} median={height_row['median']:.0f} px"
    )
    lines.append("- See: `04_image_size_stats.csv`, `04_resolution_frequency.csv`, `figures/image_size_distribution.png`")
    lines.append("")
    lines.append("## 5-6. Metadata Analysis, Column Descriptions, Missing Values")
    lines.append("")
    lines.append(
        "- 27 columns, several forming a specificity hierarchy (`diagnosis_1`->`diagnosis_5`, "
        "`anatom_site_1`->`anatom_site_5`) that gets sparser at deeper levels by design, not by defect."
    )
    lines.append("- See: `06_column_description.csv`, `06_metadata_numeric_describe.csv`, `07_missing_value_report.csv`")
    lines.append("")
    lines.append("## 7. Class Distribution")
    lines.append("")
    lines.append(
        "`diagnosis_3` is used as the primary class label (populated for nearly every row, "
        "and names concrete diseases comparable to the other three datasets in this project)."
    )
    lines.append("")
    lines.append("| diagnosis_3 | count | pct |")
    lines.append("|---|---|---|")
    for _, row in class_dist_fine_df.sort_values("count", ascending=False).iterrows():
        lines.append(f"| {row['diagnosis_3']} | {row['count']} | {row.get('pct', '')}% |")
    lines.append("")
    lines.append("- See: `08_class_distribution_coarse.csv`, `08_class_distribution_fine.csv`, `figures/class_distribution.png`")
    lines.append("")
    lines.append("## 8. Lesion / Patient Statistics")
    lines.append("")
    lines.append(
        f"- `lesion_id` populated for {lesion_stats['n_with_lesion_id']}/{lesion_stats['n_rows']} rows "
        f"({lesion_stats['n_unique_lesions']} unique lesions); `patient_id` populated for only "
        f"{lesion_stats['n_with_patient_id']}/{lesion_stats['n_rows']} rows "
        f"({lesion_stats['n_unique_patients']} unique patients)."
    )
    lines.append(
        "- Unlike PAD-UFES-20, patient-wise splitting is not possible for the bulk of this "
        "archive; the cleaning phase must split by `lesion_id` where available, and treat rows "
        "with neither identifier as their own singleton group (as with HAM10000's lesion-wise approach)."
    )
    lines.append(f"- Lesions with inconsistent `diagnosis_3` labels: {lesion_stats['lesions_with_multiple_diagnoses']}")
    lines.append("- See: `09_lesion_patient_statistics.csv`, `09_lesions_multiple_diagnoses.csv`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This audit is READ-ONLY. `data/raw/` was not modified.")
    lines.append(
        "- This archive overlaps substantially in disease taxonomy with ISIC Archive 1's "
        "class-folder labels and with PAD-UFES-20/HAM10000; label harmonization across all "
        "four sources happens in the cleaning phase (`label_mapping.csv`)."
    )

    text = "\n".join(lines)
    out_path = ISIC2_REPORTS_DIR / "10_dataset_audit_summary.md"
    save_text(text, out_path)

    logger.info("Dataset summary written -> %s", out_path)
    return text
