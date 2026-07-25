"""Central paths and constants for Phase 6 Stage 1 baseline models.

Dataset-parameterized (extended 2026-07-13 for HAM10000; originally
PAD-UFES-20-only). Every dataset shares the exact same recipe (same
architectures, hyperparameters, class-weighted loss mechanism, seeds) -
only paths, class list, and metadata feature lists differ per dataset.
This is intentional: per Project_Tracking.md's "Sequencing Decision -
HAM10000 Baseline Before Phase 7" entry, keeping the recipe identical
across datasets is required so any later cross-dataset performance
difference (Phase 8) can be attributed to the dataset itself, not to
per-dataset tuning.

Environment-aware (added 2026-07-09, moving Stage 1 training to Kaggle
since this machine has no GPU): detects whether it's running locally or
on Kaggle and resolves raw-image, processed-metadata, and
output(logs/checkpoints/reports) roots accordingly. The CSVs themselves
are never edited - image_path values stay exactly as written
("data/raw/<Dataset>/...") on every environment; only the code that
resolves that string into an actual filesystem Path changes.
"""

from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    # __file__ is undefined when this module's code runs as a pasted/exec'd
    # notebook cell rather than an imported .py file (e.g. Kaggle cell body
    # run directly instead of via %%writefile + import). Fall back to cwd,
    # which on Kaggle is /kaggle/working and is only used for the
    # (unused-on-Kaggle) local RAW_ROOT default anyway.
    PROJECT_ROOT = Path.cwd()

IS_KAGGLE = Path("/kaggle/input").exists()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_DATASETS_ROOT = KAGGLE_INPUT_ROOT / "datasets"
KAGGLE_WORKING_ROOT = Path("/kaggle/working")

# --- Raw images -------------------------------------------------------
# image_path in metadata_{train,val,test}.csv always looks like
# "data/raw/<Dataset>/..." - locally that's a real path under
# PROJECT_ROOT. On Kaggle there is no data/raw/ at all; each dataset you
# "Add Data" gets mounted under
# /kaggle/input/datasets/<owner>/<dataset-slug>/ (confirmed via
# folder-verification cells - Kaggle nests datasets one level deeper
# than a flat /kaggle/input/<slug>/, under an owner-username folder),
# and the internal layout depends on how that specific Kaggle dataset
# was packaged.
KAGGLE_DATASET_SLUGS = {
    # mahdavi1202/skin-cancer - verified raw mirror of PAD-UFES-20
    # (imgs_part_1/2/3 + metadata.csv, 2,298 PNGs, matches our data/raw
    # exactly). Mounted at
    # /kaggle/input/datasets/mahdavi1202/skin-cancer/.
    "PAD_UFES20": ("mahdavi1202", "skin-cancer"),
    # kmader/skin-cancer-mnist-ham10000 - verified raw mirror of HAM10000
    # (HAM10000_images_part_1/2 + HAM10000_metadata.csv, 10,015 images,
    # matches our data/raw/HAM10000 exactly; hmnist_*.csv pixel-matrix
    # files in this mirror are ignored/unused). Folder names
    # (HAM10000_images_part_1/2) match our own data/raw/HAM10000 layout,
    # unlike PAD-UFES-20's imgs_part_N mirror - still verified via a
    # folder-verification notebook cell before trusting this, rather than
    # assumed. Mounted at
    # /kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/.
    "HAM10000": ("kmader", "skin-cancer-mnist-ham10000"),
}
KAGGLE_DATASET_SUBPATH = {
    "PAD_UFES20": "",
    "HAM10000": "",
}

RAW_ROOT = PROJECT_ROOT / "data" / "raw"  # local only; unused on Kaggle


def resolve_image_path(image_path: str) -> Path:
    """Turn a CSV image_path value into a real filesystem path for the
    current environment, without ever touching the CSV itself.
    """
    rel = Path(image_path)  # "data/raw/<Dataset>/<rest...>"
    parts = rel.parts
    if len(parts) < 3 or parts[0] != "data" or parts[1] != "raw":
        raise ValueError(f"Unexpected image_path format: {image_path!r}")
    dataset_dir = parts[2]  # e.g. "PAD_UFES20"
    rest = Path(*parts[3:])  # e.g. "imgs_part_3/xxx.png"

    if not IS_KAGGLE:
        return PROJECT_ROOT / rel

    owner_slug = KAGGLE_DATASET_SLUGS.get(dataset_dir)
    if owner_slug is None or owner_slug[0].startswith("REPLACE_WITH"):
        raise RuntimeError(
            f"No Kaggle dataset slug configured for {dataset_dir!r}. "
            f"Run `!ls /kaggle/input/datasets` in the notebook, then set "
            f"KAGGLE_DATASET_SLUGS[{dataset_dir!r}] in src/models/config.py."
        )
    owner, slug = owner_slug
    subpath = KAGGLE_DATASET_SUBPATH.get(dataset_dir, "")
    dataset_root = KAGGLE_DATASETS_ROOT / owner / slug / subpath

    # PAD-UFES-20's Kaggle mirror double-nests imgs_part_N/ folders:
    # verified via direct listing that imgs_part_1, imgs_part_2, and
    # imgs_part_3 are ALL doubled the same way (each contains a
    # subfolder of the identical name). Still checked with .exists()
    # per-call rather than hardcoded, so the fix keeps working even if a
    # future dataset version changes the packaging for only some parts,
    # and so it's a no-op (never matches) for datasets without
    # imgs_part_N/ folders, e.g. HAM10000.
    rest_parts = rest.parts
    if rest_parts and rest_parts[0].startswith("imgs_part_"):
        top_dir = rest_parts[0]
        doubled = dataset_root / top_dir / top_dir / Path(*rest_parts[1:])
        if doubled.exists():
            return doubled

    return dataset_root / rest


# --- Our processed metadata (train/val/test CSVs, feature_whitelist.md) -
# These are our own split/label artifacts, not raw images - uploaded as a
# separate private Kaggle dataset per source dataset. Mounted at
# /kaggle/input/datasets/<owner>/<slug>/.
KAGGLE_PROCESSED_SLUGS = {
    # naeemsarkertracer/pad-ufes20-processed - verified 2026-07-09/13.
    "PAD_UFES20": ("naeemsarkertracer", "pad-ufes20-processed"),
    # naeemsarkertracer/ham10000-processed - uploaded and published
    # 2026-07-15 (https://www.kaggle.com/datasets/naeemsarkertracer/ham10000-processed).
    "HAM10000": ("naeemsarkertracer", "ham10000-processed"),
}
# Whether that Kaggle dataset was zipped from the dataset folder itself
# (True -> mounted root wraps everything in an extra "<Dataset>/"
# subfolder, e.g. PAD-UFES-20's) or from the folder's contents (False ->
# mounted root already contains metadata_train.csv etc. directly). Set
# per-dataset once the HAM10000 processed dataset is actually uploaded.
KAGGLE_PROCESSED_WRAPPED = {
    "PAD_UFES20": True,
    "HAM10000": True,
}


def _processed_dir(dataset: str) -> Path:
    if not IS_KAGGLE:
        return PROJECT_ROOT / "data" / "processed" / dataset

    owner_slug = KAGGLE_PROCESSED_SLUGS.get(dataset)
    if owner_slug is None or owner_slug[0].startswith("REPLACE_WITH"):
        raise RuntimeError(
            f"Set KAGGLE_PROCESSED_SLUGS[{dataset!r}] in "
            f"src/models/config.py to the owner/slug of the private Kaggle "
            f"dataset holding data/processed/{dataset}/."
        )
    owner, slug = owner_slug
    root = KAGGLE_DATASETS_ROOT / owner / slug

    # Auto-detect wrapping rather than trusting KAGGLE_PROCESSED_WRAPPED
    # alone: if a dataset/<Dataset>/metadata_train.csv exists, the zip was
    # made from the folder itself (wrapped); if root/metadata_train.csv
    # exists directly, it was zipped from the folder's contents
    # (unwrapped). KAGGLE_PROCESSED_WRAPPED is only the fallback for the
    # rare case neither path exists yet (e.g. this exact assertion running
    # before the dataset is mounted).
    wrapped_candidate = root / dataset
    if (wrapped_candidate / "metadata_train.csv").exists():
        return wrapped_candidate
    if (root / "metadata_train.csv").exists():
        return root
    return wrapped_candidate if KAGGLE_PROCESSED_WRAPPED.get(dataset, True) else root


# --- Phase 7 Stage 1 fusion warm-start: Stage 1 baseline checkpoints --
# Locally these already sit under OUTPUT_ROOT/logs/<Dataset>/checkpoints/
# (written by Stage 1 train.py runs). On Kaggle, /kaggle/working is wiped
# fresh per session, so the Stage 1 checkpoints must be uploaded as their
# own private Kaggle "Add Data" input (zip data/logs/<Dataset>/checkpoints/
# from this machine) before running fusion training there.
KAGGLE_STAGE1_CHECKPOINT_SLUGS = {
    # naeemsarkertracer/pad-ufes20-stage1-checkpoints - published 2026-07-16
    # (https://www.kaggle.com/datasets/naeemsarkertracer/pad-ufes20-stage1-checkpoints).
    "PAD_UFES20": ("naeemsarkertracer", "pad-ufes20-stage1-checkpoints"),
}
KAGGLE_STAGE1_CHECKPOINT_WRAPPED = {
    "PAD_UFES20": True,
}


def _stage1_checkpoints_dir(dataset: str) -> Path:
    if not IS_KAGGLE:
        return OUTPUT_ROOT / "logs" / dataset / "checkpoints"

    owner_slug = KAGGLE_STAGE1_CHECKPOINT_SLUGS.get(dataset)
    if owner_slug is None or owner_slug[0].startswith("REPLACE_WITH"):
        raise RuntimeError(
            f"Set KAGGLE_STAGE1_CHECKPOINT_SLUGS[{dataset!r}] in "
            f"src/models/config.py to the owner/slug of the private Kaggle "
            f"dataset holding the Stage 1 checkpoints, and attach it as an "
            f"Add Data source."
        )
    owner, slug = owner_slug
    root = KAGGLE_DATASETS_ROOT / owner / slug

    # Auto-detect wrapping, same approach as _processed_dir.
    wrapped_candidate = root / "checkpoints"
    if (wrapped_candidate / "image_seed0_best.pt").exists():
        return wrapped_candidate
    if (root / "image_seed0_best.pt").exists():
        return root
    return (
        wrapped_candidate
        if KAGGLE_STAGE1_CHECKPOINT_WRAPPED.get(dataset, True)
        else root
    )


# --- Outputs (checkpoints, training logs, evaluation reports) ---------
# /kaggle/input is read-only, so on Kaggle these must live under
# /kaggle/working (which Kaggle preserves as the notebook's Output when
# run via "Save & Run All (Commit)").
OUTPUT_ROOT = KAGGLE_WORKING_ROOT if IS_KAGGLE else PROJECT_ROOT

IMAGE_INPUT_SIZE = 224  # matches EfficientNet-B0 pretrained expectation

SEEDS = [0, 1, 2]  # 3 seeds per branch, per Project_Tracking.md decision (4)

BATCH_SIZE = 32
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7  # epochs without val macro-F1 improvement
LEARNING_RATE_IMAGE = 1e-4
LEARNING_RATE_METADATA = 1e-3
# Phase 7 Stage 1 (late fusion): warm-started from Stage 1 checkpoints
# and fine-tuned end-to-end, unfrozen, at a lower LR than either branch's
# own Stage 1 LR - both branches are already converged, so a
# Stage-1-scale LR risks catastrophically forgetting the warm-started
# weights in early fusion epochs.
LEARNING_RATE_FUSION = 1e-5
# Phase 7 Stage 2 (cross-attention fusion): same warm-start-then-fine-tune
# discipline as Stage 1's late fusion, at the same conservative LR - both
# embedders are Stage 1-converged and only the new cross-attention/head
# parameters are randomly initialized, so a low LR protects the warm-started
# weights the same way it did for Stage 1's fusion head.
LEARNING_RATE_CROSS_ATTENTION = 1e-5
WEIGHT_DECAY = 1e-4


# --- Phase 8: reduced-feature schema for PAD-UFES-20 -> HAM10000 -------
# cross-dataset generalization. HAM10000's own metadata whitelist has only
# 3 columns (age, sex, anatomical_site) - a PAD-UFES-20 metadata/fusion/
# cross-attention checkpoint trained on the full 21-column whitelist
# cannot run on HAM10000 data at all (18 columns are simply absent).
# REDUCED_* restricts PAD-UFES-20 training to just the 3 columns
# HAM10000 also has, so a schema-matched model can be evaluated on both
# datasets. See docs/Phase8_Anatomical_Site_Mapping.csv for the full
# per-category mapping review (approved 2026-07-18).
REDUCED_NUMERIC_FEATURES = ["age"]
REDUCED_CATEGORICAL_FEATURES = ["sex", "anatomical_site"]

# PAD-UFES-20 anatomical_site (uppercase, finer-grained) -> HAM10000
# anatomical_site (lowercase, coarser) - approved 2026-07-18, per
# docs/Phase8_Anatomical_Site_Mapping.csv. 9 clean (casing-only), 3
# lossy/coarsened (ARM+FOREARM collide into "upper extremity"; THIGH ->
# "lower extremity"), 2 deliberately absent (LIP, NOSE - no HAM10000
# equivalent, approved to fall through to the "__MISSING__" bucket rather
# than being force-mapped to "face").
ANATOMICAL_SITE_CROSS_DATASET_MAP = {
    "ABDOMEN": "abdomen",
    "BACK": "back",
    "CHEST": "chest",
    "EAR": "ear",
    "FACE": "face",
    "FOOT": "foot",
    "HAND": "hand",
    "NECK": "neck",
    "SCALP": "scalp",
    "ARM": "upper extremity",
    "FOREARM": "upper extremity",
    "THIGH": "lower extremity",
    # LIP, NOSE intentionally absent - normalize_anatomical_site_for_cross_dataset()
    # falls through to "__MISSING__" for these, same treatment as a
    # genuinely missing value.
}


def normalize_anatomical_site_for_cross_dataset(raw_value) -> str:
    """PAD-UFES-20's anatomical_site value -> HAM10000-vocabulary string,
    for the reduced-feature cross-dataset models only. HAM10000's own
    anatomical_site values pass through MetadataPreprocessor unchanged
    (already in the target vocabulary) - this normalization only needs to
    run on the PAD-UFES-20 side.
    """
    import pandas as pd

    if pd.isna(raw_value):
        return "__MISSING__"
    return ANATOMICAL_SITE_CROSS_DATASET_MAP.get(str(raw_value).strip(), "__MISSING__")


class DatasetConfig:
    """Everything Stage 1 code needs for one dataset. Class order is
    fixed (alphabetical by standardized label) so label-encoding is
    identical and reproducible across every script/run, per-dataset.
    """

    def __init__(self, name: str, class_names: list, numeric_features: list,
                 categorical_features: list):
        self.name = name
        self.class_names = class_names
        self.label_to_idx = {n: i for i, n in enumerate(class_names)}
        self.num_classes = len(class_names)
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        processed_dir = _processed_dir(name)
        self.train_csv = processed_dir / "metadata_train.csv"
        self.val_csv = processed_dir / "metadata_val.csv"
        self.test_csv = processed_dir / "metadata_test.csv"

        self.logs_dir = OUTPUT_ROOT / "logs" / name
        self.checkpoints_dir = self.logs_dir / "checkpoints"
        self.baseline_reports_dir = OUTPUT_ROOT / "reports" / name / "baseline"
        self.fusion_reports_dir = OUTPUT_ROOT / "reports" / name / "fusion"
        self._stage1_checkpoints_dir = None  # lazy - see property below

    @property
    def stage1_checkpoints_dir(self) -> Path:
        """Phase 7 Stage 1 warm-start source (Stage 1 baseline checkpoints -
        distinct from checkpoints_dir, which is where fusion checkpoints
        get *written*; on Kaggle these live in different mounted dirs).

        Computed lazily, not in __init__: only fusion-eligible datasets
        (PAD_UFES20) have a KAGGLE_STAGE1_CHECKPOINT_SLUGS entry.
        _stage1_checkpoints_dir() raises for datasets without one - fine
        for a property nothing calls unless fusion code actually needs
        it, but fatal if run eagerly for every DATASETS entry at import
        time (as happened when HAM10000 - never used with fusion - hit
        this line merely by being constructed alongside PAD_UFES20).
        """
        if self._stage1_checkpoints_dir is None:
            self._stage1_checkpoints_dir = _stage1_checkpoints_dir(self.name)
        return self._stage1_checkpoints_dir

    def with_features(self, numeric_features: list, categorical_features: list) -> "DatasetConfig":
        """Shallow copy overriding only the metadata feature lists - all
        paths/class lists/checkpoint dirs stay identical. Used for Phase 8's
        reduced-feature PAD-UFES-20 variants (REDUCED_NUMERIC_FEATURES /
        REDUCED_CATEGORICAL_FEATURES above), so the schema-matched training
        runs reuse the same DatasetConfig plumbing without a second,
        near-duplicate PAD_UFES20 entry in DATASETS.
        """
        import copy

        clone = copy.copy(self)
        clone.numeric_features = numeric_features
        clone.categorical_features = categorical_features
        return clone


# data/processed/PAD_UFES20/feature_whitelist.md - allowed model-input
# columns only.
PAD_UFES20 = DatasetConfig(
    name="PAD_UFES20",
    class_names=[
        "Actinic Keratosis",
        "Basal Cell Carcinoma",
        "Melanoma",
        "Nevus",
        "Seborrheic Keratosis",
        "Squamous Cell Carcinoma",
    ],
    numeric_features=["age", "diameter_1", "diameter_2"],
    categorical_features=[
        "smoke", "drink", "background_father", "background_mother",
        "pesticide", "sex", "skin_cancer_history", "cancer_history",
        "has_piped_water", "has_sewage_system", "fitspatrick",
        "anatomical_site", "itch", "grew", "hurt", "changed", "bleed",
        "elevation",
    ],
)

# data/processed/HAM10000/feature_whitelist.md - 3 allowed columns only
# (age, sex, anatomical_site) - verified 2026-07-08 and re-verified
# 2026-07-13 (see Project_Tracking.md's HAM10000 leakage-audit entries).
HAM10000 = DatasetConfig(
    name="HAM10000",
    class_names=[
        "Actinic Keratosis / Intraepithelial Carcinoma",
        "Basal Cell Carcinoma",
        "Benign Keratosis-like Lesion",
        "Dermatofibroma",
        "Melanoma",
        "Nevus",
        "Vascular Lesion",
    ],
    numeric_features=["age"],
    categorical_features=["sex", "anatomical_site"],
)

DATASETS = {
    "PAD_UFES20": PAD_UFES20,
    "HAM10000": HAM10000,
}


def get_dataset(name: str) -> DatasetConfig:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choices: {list(DATASETS)}")
    return DATASETS[name]
