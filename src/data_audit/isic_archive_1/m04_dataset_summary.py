"""Module 4: Dataset summary.

Rolls up the headline numbers from every prior module into a single
human-readable markdown report.
"""

from datetime import datetime

import pandas as pd

from src.data_audit.config import ISIC1_REPORTS_DIR
from src.data_audit.common.io_utils import save_csv, save_text


def run(logger, results: dict) -> str:
    inventory_df: pd.DataFrame = results["inventory"]
    verification_df: pd.DataFrame = results["verification"]
    class_dist_df: pd.DataFrame = results["class_distribution"]

    n_ok = (verification_df["status"] == "OK").sum()
    n_corrupted = (verification_df["status"] == "CORRUPTED").sum()

    duplicate_filenames = inventory_df[inventory_df.duplicated("filename", keep=False)]
    class_pairs = duplicate_filenames.groupby("filename")["class_label"].apply(lambda s: tuple(sorted(set(s))))
    conflicting = class_pairs[class_pairs.apply(len) > 1]
    conflict_rows_df = inventory_df[inventory_df["filename"].isin(conflicting.index)].sort_values("filename")
    save_csv(conflict_rows_df, ISIC1_REPORTS_DIR / "02_duplicate_filename_label_conflicts.csv")

    lines = []
    lines.append("# ISIC Archive 1 Dataset Audit Summary")
    lines.append("")
    lines.append(f"Audit date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Structure")
    lines.append("")
    lines.append(
        "No metadata.csv ships with this archive - it is pre-split into "
        "`Train/` and `Test/`, each with one subfolder per class; the "
        "folder name itself is the label. Folder structure + image "
        "inventory were built in one pass (`01_folder_structure.csv`, "
        "`02_image_inventory.csv`)."
    )
    lines.append(f"- Total images found on disk: {len(inventory_df)}")
    lines.append(f"- Classes: {sorted(inventory_df['class_label'].unique())}")
    lines.append("")
    lines.append("## Duplicate Filenames Across Locations")
    lines.append("")
    lines.append(
        f"- {duplicate_filenames['filename'].nunique()} filenames appear in more than one "
        "location. See `02_duplicate_filenames.csv`."
    )
    if not conflicting.empty:
        lines.append(
            f"- **Data-quality finding:** {len(conflicting)} of those are the *same image* "
            "filed under two *different* class labels - a genuine label conflict, not merely "
            "a Train/Test split repeat. The conflicts are systematic, not random noise:"
        )
        lines.append("")
        for pair, count in conflicting.value_counts().items():
            lines.append(f"  - `{pair[0]}` <-> `{pair[1]}`: {count} images")
        lines.append("")
        lines.append(
            "  See `02_duplicate_filename_label_conflicts.csv` for the full row-level detail. "
            "These images should be excluded or resolved (not silently kept under either "
            "label) during the cleaning phase, since their true class is ambiguous."
        )
    lines.append("")
    lines.append("## Image Verification & Corrupted Images")
    lines.append("")
    lines.append(f"- Images successfully decoded (OK): {n_ok}")
    lines.append(f"- Corrupted / unreadable images: {n_corrupted}")
    lines.append("- See: `03_image_verification.csv`, `03_corrupted_images.csv`")
    lines.append("")
    lines.append("## Image Size Statistics")
    lines.append("")
    lines.append("- See: `04_image_size_stats.csv`, `04_resolution_frequency.csv`, `figures/image_size_distribution.png`")
    lines.append("")
    lines.append("## Class Distribution")
    lines.append("")
    lines.append("| split | class_label | count | pct_of_split |")
    lines.append("|---|---|---|---|")
    for _, row in class_dist_df.sort_values(["split", "count"], ascending=[True, False]).iterrows():
        lines.append(f"| {row['split']} | {row['class_label']} | {row['count']} | {row['pct_of_split']}% |")
    lines.append("")
    lines.append("- See: `05_class_distribution.csv`, `figures/class_distribution.png`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This audit is READ-ONLY. `data/raw/` was not modified.")
    lines.append(
        "- No patient or lesion identifier exists in this archive - only "
        "a bare `ISIC_xxxxxxx` filename per image, so no patient/lesion "
        "leakage check is possible here. The provided Train/Test split "
        "is used as-is; the cleaning phase further carves a validation "
        "set out of Train."
    )
    lines.append(
        "- Class names overlap heavily with ISIC Archive 2's `diagnosis_3` "
        "field and with PAD-UFES-20/HAM10000's disease taxonomy - label "
        "harmonization across all four sources is handled in the cleaning "
        "phase (`label_mapping.csv`)."
    )

    text = "\n".join(lines)
    out_path = ISIC1_REPORTS_DIR / "06_dataset_audit_summary.md"
    save_text(text, out_path)

    logger.info("Dataset summary written -> %s", out_path)
    return text
