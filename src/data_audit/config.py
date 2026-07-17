"""Central paths and constants for the dataset audit pipeline.

All paths are resolved relative to this file's location so the scripts
work regardless of the current working directory they are launched from.
"""

from pathlib import Path

# src/data_audit/config.py -> project root is two levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Raw data (READ-ONLY, never write here) ---------------------------------
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PAD_UFES20_RAW_DIR = RAW_ROOT / "PAD_UFES20"
PAD_UFES20_METADATA_FILE = PAD_UFES20_RAW_DIR / "metadata.csv"
PAD_UFES20_IMAGE_SUBDIRS = ["imgs_part_1", "imgs_part_2", "imgs_part_3"]

HAM10000_RAW_DIR = RAW_ROOT / "HAM10000"
HAM10000_METADATA_FILE = HAM10000_RAW_DIR / "HAM10000_metadata.csv"
HAM10000_IMAGE_SUBDIRS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]

# ISIC Archive 1: pre-split into Train/Test, one subfolder per class, no
# metadata CSV - the folder name IS the label.
ISIC1_RAW_DIR = RAW_ROOT / "ISIC_Archive_1"
ISIC1_SPLIT_SUBDIRS = ["Train", "Test"]

# ISIC Archive 2: flat images/ folder + a single rich metadata.csv.
ISIC2_RAW_DIR = RAW_ROOT / "ISIC_Archive_2"
ISIC2_METADATA_FILE = ISIC2_RAW_DIR / "metadata.csv"
ISIC2_IMAGE_SUBDIRS = ["images"]

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# --- Outputs -----------------------------------------------------------------
REPORTS_ROOT = PROJECT_ROOT / "reports"
PAD_UFES20_REPORTS_DIR = REPORTS_ROOT / "PAD_UFES20"
PAD_UFES20_FIGURES_DIR = PAD_UFES20_REPORTS_DIR / "figures"

HAM10000_REPORTS_DIR = REPORTS_ROOT / "HAM10000"
HAM10000_FIGURES_DIR = HAM10000_REPORTS_DIR / "figures"

ISIC1_REPORTS_DIR = REPORTS_ROOT / "ISIC_Archive_1"
ISIC1_FIGURES_DIR = ISIC1_REPORTS_DIR / "figures"

ISIC2_REPORTS_DIR = REPORTS_ROOT / "ISIC_Archive_2"
ISIC2_FIGURES_DIR = ISIC2_REPORTS_DIR / "figures"

LOGS_ROOT = PROJECT_ROOT / "logs"
PAD_UFES20_LOGS_DIR = LOGS_ROOT / "PAD_UFES20"
HAM10000_LOGS_DIR = LOGS_ROOT / "HAM10000"
ISIC1_LOGS_DIR = LOGS_ROOT / "ISIC_Archive_1"
ISIC2_LOGS_DIR = LOGS_ROOT / "ISIC_Archive_2"
