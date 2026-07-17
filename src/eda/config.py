"""Central paths and constants for the EDA pipeline (Phase 5).

Mirrors src/data_audit/config.py's conventions: paths resolved relative to
this file, so scripts work regardless of the launch directory. EDA never
writes to data/raw/ or to any existing data/processed/ split file - outputs
go only to reports/eda/ (new) and logs/<Dataset>/ (existing).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = PROJECT_ROOT / "data" / "raw"

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
PAD_UFES20_PROCESSED_DIR = PROCESSED_ROOT / "PAD_UFES20"
HAM10000_PROCESSED_DIR = PROCESSED_ROOT / "HAM10000"
ISIC1_PROCESSED_DIR = PROCESSED_ROOT / "ISIC_Archive_1"
ISIC2_PROCESSED_DIR = PROCESSED_ROOT / "ISIC_Archive_2"

REPORTS_EDA_ROOT = PROJECT_ROOT / "reports" / "eda"
PAD_UFES20_EDA_DIR = REPORTS_EDA_ROOT / "PAD_UFES20"
PAD_UFES20_EDA_FIGURES_DIR = PAD_UFES20_EDA_DIR / "figures"

HAM10000_EDA_DIR = REPORTS_EDA_ROOT / "HAM10000"
HAM10000_EDA_FIGURES_DIR = HAM10000_EDA_DIR / "figures"

ISIC1_EDA_DIR = REPORTS_EDA_ROOT / "ISIC_Archive_1"
ISIC1_EDA_FIGURES_DIR = ISIC1_EDA_DIR / "figures"

ISIC2_EDA_DIR = REPORTS_EDA_ROOT / "ISIC_Archive_2"
ISIC2_EDA_FIGURES_DIR = ISIC2_EDA_DIR / "figures"

CROSS_DATASET_EDA_DIR = REPORTS_EDA_ROOT / "cross_dataset"
CROSS_DATASET_EDA_FIGURES_DIR = CROSS_DATASET_EDA_DIR / "figures"

LOGS_ROOT = PROJECT_ROOT / "logs"
PAD_UFES20_LOGS_DIR = LOGS_ROOT / "PAD_UFES20"
HAM10000_LOGS_DIR = LOGS_ROOT / "HAM10000"
ISIC1_LOGS_DIR = LOGS_ROOT / "ISIC_Archive_1"
ISIC2_LOGS_DIR = LOGS_ROOT / "ISIC_Archive_2"
CROSS_DATASET_LOGS_DIR = LOGS_ROOT / "cross_dataset"

# Image-dimension sampling (Phase 5 addition, approved 2026-07-08)
IMAGE_DIM_SAMPLE_SIZE = 250
IMAGE_DIM_SAMPLE_SEED = 42

# Sample image grid
IMAGES_PER_CLASS_IN_GRID = 4
GRID_SAMPLE_SEED = 42
