"""Module 12: Dataset summary.

Rolls up the headline numbers from every prior module into a single
human-readable markdown report - the primary artifact to cite/attach in
the thesis, with pointers to every detailed sub-report.
"""

from datetime import datetime

import pandas as pd

from src.data_audit.config import PAD_UFES20_REPORTS_DIR
from src.data_audit.common.io_utils import save_text


def run(logger, results: dict) -> str:
    folder_df: pd.DataFrame = results["folder_structure"]
    inventory_df: pd.DataFrame = results["inventory"]
    verification_df: pd.DataFrame = results["verification"]
    missing_stats: dict = results["missing_image_stats"]
    size_stats_df: pd.DataFrame = results["size_stats"]
    class_dist_df: pd.DataFrame = results["class_distribution"]
    patient_stats: dict = results["patient_stats"]

    n_ok = (verification_df["status"] == "OK").sum()
    n_corrupted = (verification_df["status"] == "CORRUPTED").sum()

    width_row = size_stats_df[size_stats_df["metric"] == "width"].iloc[0]
    height_row = size_stats_df[size_stats_df["metric"] == "height"].iloc[0]

    lines = []
    lines.append("# PAD-UFES-20 Dataset Audit Summary")
    lines.append("")
    lines.append(f"Audit date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 1. Folder Structure")
    lines.append("")
    lines.append(f"- Folders analyzed: {len(folder_df)}")
    lines.append(f"- See: `01_folder_structure.csv`")
    lines.append("")
    lines.append("## 2. Image Inventory")
    lines.append("")
    lines.append(f"- Total images found on disk: {len(inventory_df)}")
    lines.append(f"- See: `02_image_inventory.csv`")
    lines.append("")
    lines.append("## 3. Image Verification & Corrupted Images")
    lines.append("")
    lines.append(f"- Images successfully decoded (OK): {n_ok}")
    lines.append(f"- Corrupted / unreadable images: {n_corrupted}")
    lines.append(f"- See: `03_image_verification.csv`, `03_corrupted_images.csv`")
    lines.append("")
    lines.append("## 4. Missing / Orphan Images")
    lines.append("")
    lines.append(f"- Metadata rows: {missing_stats['metadata_rows']}")
    lines.append(f"- Inventory files: {missing_stats['inventory_files']}")
    lines.append(f"- Missing images (in metadata, not on disk): {missing_stats['missing_count']}")
    lines.append(f"- Orphan images (on disk, not in metadata): {missing_stats['orphan_count']}")
    lines.append(f"- Duplicate img_id entries in metadata: {missing_stats['duplicate_img_id_count']}")
    lines.append(f"- See: `04_missing_images.csv`, `04_orphan_images.csv`, `04_duplicate_img_id.csv`")
    lines.append("")
    lines.append("## 5. Image Size Statistics")
    lines.append("")
    lines.append(
        f"- Width: min={width_row['min']:.0f} max={width_row['max']:.0f} "
        f"mean={width_row['mean']:.1f} median={width_row['median']:.0f} px"
    )
    lines.append(
        f"- Height: min={height_row['min']:.0f} max={height_row['max']:.0f} "
        f"mean={height_row['mean']:.1f} median={height_row['median']:.0f} px"
    )
    lines.append(f"- See: `05_image_size_stats.csv`, `05_resolution_frequency.csv`, `figures/image_size_distribution.png`")
    lines.append("")
    lines.append("## 6-9. Metadata Analysis, Column Descriptions, Missing Values")
    lines.append("")
    lines.append(f"- See: `06_metadata_overview.csv`, `07_column_description.csv`, `08_missing_value_report.csv`")
    lines.append("")
    lines.append("## 10. Class Distribution (diagnostic)")
    lines.append("")
    lines.append("| diagnostic | count | pct |")
    lines.append("|---|---|---|")
    for _, row in class_dist_df.iterrows():
        lines.append(f"| {row['diagnostic']} | {row['count']} | {row['pct']}% |")
    lines.append("")
    lines.append(f"- See: `09_class_distribution.csv`, `figures/class_distribution.png`")
    lines.append("")
    lines.append("## 11. Patient Statistics")
    lines.append("")
    lines.append(f"- Unique patients: {patient_stats['n_unique_patients']}")
    lines.append(f"- Unique lesions: {patient_stats['n_unique_lesions']}")
    lines.append(f"- Patients with multiple lesions: {patient_stats['patients_with_multiple_lesions']}")
    lines.append(f"- Patients with multiple distinct diagnoses: {patient_stats['patients_with_multiple_diagnoses']}")
    lines.append(f"- See: `10_patient_statistics.csv`, `10_patients_multiple_diagnoses.csv`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This audit is READ-ONLY. `data/raw/` was not modified.")
    lines.append("- No cleaning or processed dataset creation was performed in this phase.")
    lines.append("- Findings above should inform the upcoming cleaning and standardization phase.")

    text = "\n".join(lines)
    out_path = PAD_UFES20_REPORTS_DIR / "11_dataset_audit_summary.md"
    save_text(text, out_path)

    logger.info("Dataset summary written -> %s", out_path)
    return text
