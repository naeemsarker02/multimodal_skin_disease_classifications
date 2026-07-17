"""Module 11: Patient statistics.

Unique patient count, images/lesions per patient, and patients with
multiple lesions or conflicting diagnoses. This is required *before*
the later train/val/test split can be built, since that split must be
patient-wise (no patient's images may appear in more than one split) to
avoid data leakage.
"""

import pandas as pd

from src.data_audit.config import PAD_UFES20_METADATA_FILE, PAD_UFES20_REPORTS_DIR
from src.data_audit.common.io_utils import save_csv


def run(logger) -> dict:
    metadata = pd.read_csv(PAD_UFES20_METADATA_FILE)

    n_unique_patients = metadata["patient_id"].nunique()
    n_unique_lesions = metadata["lesion_id"].nunique()

    images_per_patient = metadata.groupby("patient_id").size()
    lesions_per_patient = metadata.groupby("patient_id")["lesion_id"].nunique()

    summary_df = pd.DataFrame(
        [
            {"metric": "n_unique_patients", "value": n_unique_patients},
            {"metric": "n_unique_lesions", "value": n_unique_lesions},
            {"metric": "n_total_images", "value": len(metadata)},
            {"metric": "images_per_patient_min", "value": images_per_patient.min()},
            {"metric": "images_per_patient_max", "value": images_per_patient.max()},
            {"metric": "images_per_patient_mean", "value": round(images_per_patient.mean(), 2)},
            {"metric": "images_per_patient_median", "value": images_per_patient.median()},
            {"metric": "lesions_per_patient_min", "value": lesions_per_patient.min()},
            {"metric": "lesions_per_patient_max", "value": lesions_per_patient.max()},
            {"metric": "lesions_per_patient_mean", "value": round(lesions_per_patient.mean(), 2)},
            {"metric": "patients_with_multiple_lesions", "value": (lesions_per_patient > 1).sum()},
        ]
    )
    save_csv(summary_df, PAD_UFES20_REPORTS_DIR / "10_patient_statistics.csv")

    # Patients whose lesions carry more than one distinct diagnostic label -
    # a data-integrity flag worth knowing about before splitting/stratifying.
    diagnoses_per_patient = metadata.groupby("patient_id")["diagnostic"].nunique()
    multi_diag_patients = diagnoses_per_patient[diagnoses_per_patient > 1]
    multi_diag_df = metadata[metadata["patient_id"].isin(multi_diag_patients.index)][
        ["patient_id", "lesion_id", "diagnostic", "img_id"]
    ].sort_values("patient_id")
    save_csv(multi_diag_df, PAD_UFES20_REPORTS_DIR / "10_patients_multiple_diagnoses.csv")

    logger.info("Unique patients: %d | Unique lesions: %d | Total images: %d",
                n_unique_patients, n_unique_lesions, len(metadata))
    logger.info(
        "Images per patient: min=%d max=%d mean=%.2f median=%.1f",
        images_per_patient.min(), images_per_patient.max(),
        images_per_patient.mean(), images_per_patient.median(),
    )
    logger.info("Patients with multiple lesions: %d", (lesions_per_patient > 1).sum())
    logger.info("Patients with multiple distinct diagnoses: %d", len(multi_diag_patients))
    logger.info("Saved -> %s", PAD_UFES20_REPORTS_DIR / "10_patient_statistics.csv")

    return {
        "n_unique_patients": n_unique_patients,
        "n_unique_lesions": n_unique_lesions,
        "patients_with_multiple_lesions": int((lesions_per_patient > 1).sum()),
        "patients_with_multiple_diagnoses": len(multi_diag_patients),
    }
