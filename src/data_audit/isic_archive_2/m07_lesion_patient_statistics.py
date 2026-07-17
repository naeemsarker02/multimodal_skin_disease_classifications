"""Module 7: Lesion / patient statistics.

lesion_id is populated for most rows (23,664/25,331) but patient_id is
populated for only a small subset (417/25,331) - unlike PAD-UFES-20,
patient-wise splitting is not possible for the bulk of this archive, so
the cleaning phase must split by lesion_id where available and treat
rows with neither identifier as singleton "lesions" (their own group),
mirroring the HAM10000 approach.
"""

import pandas as pd

from src.data_audit.config import ISIC2_METADATA_FILE, ISIC2_REPORTS_DIR
from src.data_audit.common.io_utils import save_csv


def run(logger) -> dict:
    metadata = pd.read_csv(ISIC2_METADATA_FILE)
    n_rows = len(metadata)

    n_with_lesion_id = metadata["lesion_id"].notna().sum()
    n_with_patient_id = metadata["patient_id"].notna().sum()
    n_unique_lesions = metadata["lesion_id"].nunique(dropna=True)
    n_unique_patients = metadata["patient_id"].nunique(dropna=True)

    images_per_lesion = metadata[metadata["lesion_id"].notna()].groupby("lesion_id").size()
    diagnoses_per_lesion = metadata[metadata["lesion_id"].notna()].groupby("lesion_id")["diagnosis_3"].nunique()
    multi_dx_lesions = diagnoses_per_lesion[diagnoses_per_lesion > 1]

    summary_df = pd.DataFrame(
        [
            {"metric": "n_rows", "value": n_rows},
            {"metric": "n_rows_with_lesion_id", "value": n_with_lesion_id},
            {"metric": "n_rows_with_patient_id", "value": n_with_patient_id},
            {"metric": "n_unique_lesions", "value": n_unique_lesions},
            {"metric": "n_unique_patients", "value": n_unique_patients},
            {"metric": "images_per_lesion_max", "value": images_per_lesion.max() if not images_per_lesion.empty else 0},
            {"metric": "lesions_with_multiple_images", "value": int((images_per_lesion > 1).sum())},
            {"metric": "lesions_with_inconsistent_diagnosis_3", "value": len(multi_dx_lesions)},
        ]
    )
    save_csv(summary_df, ISIC2_REPORTS_DIR / "09_lesion_patient_statistics.csv")

    multi_dx_df = metadata[metadata["lesion_id"].isin(multi_dx_lesions.index)][
        ["lesion_id", "isic_id", "diagnosis_3"]
    ].sort_values("lesion_id")
    save_csv(multi_dx_df, ISIC2_REPORTS_DIR / "09_lesions_multiple_diagnoses.csv")

    logger.info("Rows with lesion_id: %d/%d | Rows with patient_id: %d/%d", n_with_lesion_id, n_rows, n_with_patient_id, n_rows)
    logger.info("Unique lesions: %d | Unique patients: %d", n_unique_lesions, n_unique_patients)
    logger.info("Lesions with multiple images: %d", int((images_per_lesion > 1).sum()))
    logger.info("Lesions with inconsistent diagnosis_3: %d", len(multi_dx_lesions))
    logger.info("Saved -> %s", ISIC2_REPORTS_DIR / "09_lesion_patient_statistics.csv")

    return {
        "n_rows": n_rows,
        "n_with_lesion_id": int(n_with_lesion_id),
        "n_with_patient_id": int(n_with_patient_id),
        "n_unique_lesions": int(n_unique_lesions),
        "n_unique_patients": int(n_unique_patients),
        "lesions_with_multiple_diagnoses": len(multi_dx_lesions),
    }
