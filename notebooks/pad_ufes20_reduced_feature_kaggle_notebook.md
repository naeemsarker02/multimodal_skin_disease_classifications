# PAD-UFES-20 Kaggle Notebook — Phase 8 Reduced-Feature Models (Cross-Dataset Generalization Prep)

> **WARNING - read before pasting any cell into Kaggle:** for every `%%writefile` cell below, `%%writefile <path>` MUST be the exact first line of the Kaggle cell, with absolutely nothing above it. **After pasting each `%%writefile` cell, re-open it and visually confirm `%%writefile` is line 1 before running it.**

Structured for "Save & Run All (Commit)" from the start. Trains **9 new schema-matched models** on PAD-UFES-20 (metadata_reduced, fusion_reduced, cross_attention_reduced x 3 seeds each), restricted to the 3 metadata columns HAM10000 also has (age, sex, anatomical_site — see `docs/Phase8_Anatomical_Site_Mapping.csv` for the approved anatomical_site normalization mapping), purpose-built for the Phase 8 PAD-UFES-20 → HAM10000 cross-dataset generalization experiment. **Training order matters:** metadata_reduced (seeds 0/1/2) must complete first — fusion_reduced/cross_attention_reduced warm-start their metadata side from its checkpoints (their image side warm-starts from the *existing* Stage 1 image checkpoints, already in the Stage 1 checkpoints dataset).

Requires the same **three** Kaggle "Add Data" sources as the Phase 7 notebooks (no new upload needed):

- `mahdavi1202/skin-cancer` (raw PAD-UFES-20 image mirror)
- `naeemsarkertracer/pad-ufes20-processed` (our processed metadata CSVs)
- `naeemsarkertracer/pad-ufes20-stage1-checkpoints` (Stage 1 `image_seed{0,1,2}_best.pt` — warm-start source for the image side of fusion_reduced/cross_attention_reduced)

Paste each cell below into a separate Kaggle notebook cell, in order.

---

## Cell 1 — Folder verification (raw mirror + processed nesting + checkpoint dataset check)

```python
import os

def show(path, label):
    print(f"--- {label}: {path} ---")
    if not os.path.isdir(path):
        print("  !! NOT FOUND")
        return
    for entry in sorted(os.listdir(path)):
        full = os.path.join(path, entry)
        kind = "dir" if os.path.isdir(full) else "file"
        print(f"  [{kind}] {entry}")

raw_root = "/kaggle/input/datasets/mahdavi1202/skin-cancer"
show("/kaggle/input/datasets", "datasets root")
show(raw_root, "raw PAD-UFES-20 mirror")

processed_root = "/kaggle/input/datasets/naeemsarkertracer/pad-ufes20-processed"
show(processed_root, "processed PAD-UFES-20 (root)")
wrapped = os.path.join(processed_root, "PAD_UFES20")
show(wrapped, "processed PAD-UFES-20 (wrapped candidate)")
assert os.path.isfile(os.path.join(wrapped, "metadata_train.csv")), (
    "Expected wrapped metadata_train.csv not found - check processed dataset packaging"
)

checkpoint_root = "/kaggle/input/datasets/naeemsarkertracer/pad-ufes20-stage1-checkpoints"
show(checkpoint_root, "Stage 1 checkpoints (root)")
wrapped_ckpt = os.path.join(checkpoint_root, "checkpoints")
show(wrapped_ckpt, "Stage 1 checkpoints (wrapped candidate: root/checkpoints/)")

expected = [f"image_seed{s}_best.pt" for s in (0, 1, 2)]

def all_present(base):
    return all(os.path.isfile(os.path.join(base, f)) for f in expected)

wrapped_ok = all_present(wrapped_ckpt)
root_ok = all_present(checkpoint_root)
print(f"\nAll 3 image checkpoints found at wrapped candidate ({wrapped_ckpt}): {wrapped_ok}")
print(f"All 3 image checkpoints found at root ({checkpoint_root}): {root_ok}")

if wrapped_ok:
    resolved_checkpoint_dir = wrapped_ckpt
elif root_ok:
    resolved_checkpoint_dir = checkpoint_root
else:
    resolved_checkpoint_dir = None

assert resolved_checkpoint_dir is not None, (
    "Stage 1 image checkpoints not found in either expected layout - "
    "fusion_reduced/cross_attention_reduced warm-start cannot proceed."
)

print("\nOK: raw images present, processed dataset wrapped as expected, "
      f"Stage 1 image checkpoints found at: {resolved_checkpoint_dir}")
```

---

## Cell 2 — Setup / os.makedirs

```python
import os, sys

os.makedirs("/kaggle/working/src/models", exist_ok=True)
os.makedirs("/kaggle/working/src/evaluation", exist_ok=True)
open("/kaggle/working/src/__init__.py", "w").close()
open("/kaggle/working/src/models/__init__.py", "w").close()
open("/kaggle/working/src/evaluation/__init__.py", "w").close()

sys.path.insert(0, "/kaggle/working")
print("setup OK, sys.path[0]:", sys.path[0])
```

---

## Cell 3 — `%%writefile /kaggle/working/src/models/config.py`

```python
%%writefile /kaggle/working/src/models/config.py
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
```

---

## Cell 4 — `%%writefile /kaggle/working/src/models/dataset.py`

```python
%%writefile /kaggle/working/src/models/dataset.py
"""Shared PyTorch Datasets for Phase 6 Stage 1 baselines.

Dataset-parameterized (extended 2026-07-13 for HAM10000). Reads
metadata_{train,val,test}.csv (never modifies them). image_path in those
CSVs already points into data/raw/<Dataset>/... - loaded directly, never
copied, per PROJECT_PLAN.md's no-image-copying rule.
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.models.config import IMAGE_INPUT_SIZE, DatasetConfig, resolve_image_path

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResizePad:
    """Aspect-ratio-preserving resize to a square canvas.

    Scales the longer side to `size`, then pads the shorter side with
    zeros (black) to reach size x size - avoids the distortion a naive
    stretch-to-square would introduce, given the documented image-size
    heterogeneity (see Project_Tracking.md decision 3).
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        img = TF.resize(img, [new_h, new_w])
        pad_left = (self.size - new_w) // 2
        pad_right = self.size - new_w - pad_left
        pad_top = (self.size - new_h) // 2
        pad_bottom = self.size - new_h - pad_top
        return TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0)


def build_image_transform(train: bool) -> transforms.Compose:
    ops = [ResizePad(IMAGE_INPUT_SIZE)]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


class ImageDataset(Dataset):
    """Image-only branch: returns (image_tensor, label_idx)."""

    def __init__(self, csv_path: Path, dataset_config: DatasetConfig, train: bool):
        self.df = pd.read_csv(csv_path)
        self.label_to_idx = dataset_config.label_to_idx
        self.transform = build_image_transform(train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = resolve_image_path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = self.label_to_idx[row["disease_label"]]
        return image, label


class MetadataPreprocessor:
    """Fits standardization/encoding on the train split only, applies it
    identically to val/test - prevents any val/test statistic (mean,
    std, category set) from leaking into the transform.

    column_transforms (optional): {column_name: callable(raw_value) -> str}
    applied to a categorical column's raw value before the standard
    one-hot logic, for both fit() and transform_row(). Used by Phase 8's
    reduced-feature PAD-UFES-20 models to normalize anatomical_site into
    HAM10000's vocabulary (config.normalize_anatomical_site_for_cross_dataset)
    before fitting/encoding - HAM10000's own values pass through unchanged
    since they're already in the target vocabulary and have no transform
    registered.
    """

    def __init__(self, dataset_config: DatasetConfig, column_transforms: dict = None):
        self.numeric_features = dataset_config.numeric_features
        self.categorical_features = dataset_config.categorical_features
        self.column_transforms = column_transforms or {}
        self.numeric_means = {}
        self.numeric_stds = {}
        self.categorical_values = {}  # col -> sorted list of seen categories

    def _categorical_value(self, col: str, raw_value) -> str:
        if col in self.column_transforms:
            return self.column_transforms[col](raw_value)
        return "__MISSING__" if pd.isna(raw_value) else str(raw_value)

    def fit(self, df: pd.DataFrame) -> "MetadataPreprocessor":
        for col in self.numeric_features:
            values = pd.to_numeric(df[col], errors="coerce")
            self.numeric_means[col] = values.mean()
            std = values.std()
            self.numeric_stds[col] = std if std and std > 0 else 1.0
        for col in self.categorical_features:
            values = df[col].apply(lambda v, c=col: self._categorical_value(c, v))
            self.categorical_values[col] = sorted(values.unique().tolist())
        return self

    def without_transforms(self) -> "MetadataPreprocessor":
        """Shallow copy with column_transforms cleared, keeping the fitted
        numeric_means/stds/categorical_values as-is. Used at Phase 8
        cross-dataset evaluation time: the preprocessor is fit on
        PAD-UFES-20's train split with anatomical_site normalized into
        HAM10000's vocabulary (config.normalize_anatomical_site_for_cross_dataset),
        but HAM10000's own anatomical_site/sex values are already in that
        target vocabulary - re-applying the transform to them would
        incorrectly try to re-map already-correct strings (e.g. the
        transform's dict is keyed on PAD-UFES-20's uppercase site names,
        so a HAM10000 value like "abdomen" wouldn't match and would
        wrongly fall to "__MISSING__").
        """
        import copy

        clone = copy.copy(self)
        clone.column_transforms = {}
        return clone

    @property
    def output_dim(self) -> int:
        numeric_dim = len(self.numeric_features)
        categorical_dim = sum(len(v) for v in self.categorical_values.values())
        return numeric_dim + categorical_dim

    def transform_row(self, row: pd.Series) -> torch.Tensor:
        parts = []
        for col in self.numeric_features:
            raw = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.isna(raw):
                raw = self.numeric_means[col]
            parts.append((raw - self.numeric_means[col]) / self.numeric_stds[col])
        for col in self.categorical_features:
            value = self._categorical_value(col, row[col])
            categories = self.categorical_values[col]
            one_hot = [1.0 if value == cat else 0.0 for cat in categories]
            if value not in categories:
                # unseen category at val/test time (should not happen if
                # fit on train, but guard rather than crash)
                one_hot = [0.0] * len(categories)
            parts.extend(one_hot)
        return torch.tensor(parts, dtype=torch.float32)


class MetadataDataset(Dataset):
    """Metadata-only branch: returns (feature_tensor, label_idx)."""

    def __init__(self, csv_path: Path, dataset_config: DatasetConfig,
                 preprocessor: MetadataPreprocessor):
        self.df = pd.read_csv(csv_path)
        self.label_to_idx = dataset_config.label_to_idx
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        features = self.preprocessor.transform_row(row)
        label = self.label_to_idx[row["disease_label"]]
        return features, label


class FusionDataset(Dataset):
    """Phase 7 late-fusion branch: returns (image_tensor, feature_tensor,
    label_idx) for the same row - same image transform as ImageDataset,
    same preprocessor contract as MetadataDataset.
    """

    def __init__(self, csv_path: Path, dataset_config: DatasetConfig,
                 preprocessor: MetadataPreprocessor, train: bool):
        self.df = pd.read_csv(csv_path)
        self.label_to_idx = dataset_config.label_to_idx
        self.transform = build_image_transform(train)
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = resolve_image_path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        features = self.preprocessor.transform_row(row)
        label = self.label_to_idx[row["disease_label"]]
        return image, features, label
```

---

## Cell 5 — `%%writefile /kaggle/working/src/models/image_model.py`

```python
%%writefile /kaggle/working/src/models/image_model.py
"""EfficientNet-B0 wrapper for the PAD-UFES-20 image-only branch.

Chosen over ResNet-50 for this dataset size (~2,298 images) - see
Project_Tracking.md decision (4): far fewer parameters (~5.3M vs
~25.6M), lower overfitting risk, comparable ImageNet accuracy, smaller
memory/compute footprint for free-tier GPU training.
"""

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_efficientnet_b0(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
```

---

## Cell 6 — `%%writefile /kaggle/working/src/models/metadata_model.py`

```python
%%writefile /kaggle/working/src/models/metadata_model.py
"""MLP for the PAD-UFES-20 metadata-only branch.

Establishes the metadata-alone performance floor for Phase 6 - not
intended to be competitive with the image branch alone (that comparison,
plus fusion, is Phase 7).
"""

import torch.nn as nn


class MetadataMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)
```

---

## Cell 7 — `%%writefile /kaggle/working/src/models/fusion_model.py`

```python
%%writefile /kaggle/working/src/models/fusion_model.py
"""Late-fusion model for Phase 7 Stage 1 (PAD-UFES-20 only).

Concatenates the penultimate-layer embeddings from each Stage 1 branch -
EfficientNet-B0's 1280-d classifier[-1] input (image_model.py) and
MetadataMLP's 64-d pre-final-layer output (metadata_model.py) - then
feeds the 1344-d joint vector through its own small classifier head.

Each embedder wraps the *full* Stage 1 architecture (not a re-keyed
subset) so a Stage 1 checkpoint's state_dict loads with strict=True -
no manual key remapping to get wrong. The embedder's forward pass simply
stops short of the final Linear that Stage 1 used for its own
single-branch prediction.

Deliberate limitation, logged in Project_Tracking.md rather than treated
as a bug: 1280:64 is a large dimensionality imbalance, so the image
branch will likely dominate this concatenated representation numerically
even with a deeper joint head. That's acceptable for a late-fusion
baseline - it's expected motivation for Phase 7 Stage 2 (cross-attention
fusion), not something to fix here.
"""

import torch
import torch.nn as nn

from src.models.image_model import build_efficientnet_b0
from src.models.metadata_model import MetadataMLP


class ImageEmbedder(nn.Module):
    """Wraps a full build_efficientnet_b0() model; forward returns the
    1280-d vector that Stage 1's classifier[-1] consumed, instead of
    that layer's output.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = build_efficientnet_b0(num_classes=num_classes)
        self.embed_dim = self.backbone.classifier[-1].in_features

    def load_stage1(self, checkpoint_path, device: torch.device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.backbone.load_state_dict(checkpoint["model_state_dict"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier[0](x)  # dropout only; identity at eval
        return x


class MetadataEmbedder(nn.Module):
    """Wraps a full MetadataMLP; forward returns the 64-d vector that
    Stage 1's final Linear(64, num_classes) consumed, instead of that
    layer's output.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.backbone = MetadataMLP(input_dim=input_dim, num_classes=num_classes)
        # Same module objects as self.backbone.net[:-1] - loading
        # self.backbone's state_dict updates these parameters in place.
        self.embedder_net = nn.Sequential(*list(self.backbone.net.children())[:-1])
        self.embed_dim = 64

    def load_stage1(self, checkpoint_path, device: torch.device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.backbone.load_state_dict(checkpoint["model_state_dict"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedder_net(x)


class FusionModel(nn.Module):
    """Concatenates image + metadata embeddings, classifies with a joint
    head one hidden layer deep (128-d) rather than a single Linear, so
    the head has room to learn a real weighting between the 1280-d and
    64-d branches instead of the image branch dominating by dimension
    count alone.
    """

    def __init__(self, metadata_input_dim: int, num_classes: int):
        super().__init__()
        self.image_embedder = ImageEmbedder(num_classes)
        self.metadata_embedder = MetadataEmbedder(metadata_input_dim, num_classes)
        joint_dim = self.image_embedder.embed_dim + self.metadata_embedder.embed_dim
        self.head = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def load_stage1_checkpoints(
        self, image_checkpoint_path, metadata_checkpoint_path, device: torch.device
    ) -> None:
        self.image_embedder.load_stage1(image_checkpoint_path, device)
        self.metadata_embedder.load_stage1(metadata_checkpoint_path, device)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        image_features = self.image_embedder(image)
        metadata_features = self.metadata_embedder(metadata)
        joint = torch.cat([image_features, metadata_features], dim=1)
        return self.head(joint)
```

---

## Cell 8 — `%%writefile /kaggle/working/src/models/cross_attention_fusion_model.py`

```python
%%writefile /kaggle/working/src/models/cross_attention_fusion_model.py
"""Cross-attention fusion model for Phase 7 Stage 2 (PAD-UFES-20 only).

Confirmed 2026-07-18 (Project_Tracking.md, "MetaBlock Mechanism Confirmed;
Phase 7 Stage 2 Proposal") that this is NOT a reproduction of Pacheco &
Krohling's MetaBlock - MetaBlock is a channel-wise gated affine transform
(sigmoid(tanh(V*t1) + t2)), uniform across spatial positions within a
channel. This module instead computes genuine per-spatial-location
attention weights: metadata queries EfficientNet-B0's 49 spatial tokens
(the 7x7 pre-pool feature map) via standard multi-head scaled dot-product
attention, so different image regions can be weighted differently
depending on metadata - something channel-wise gating cannot do. Framed
as "cross-attention, contrasted with MetaBlock's channel-gating approach,"
never "MetaBlock-inspired."

Directly addresses Phase 7 Stage 1's diagnosed limitation: late fusion's
1280:64 raw-dimension concatenation let the image branch numerically
dominate. Here, both modalities are projected into a shared d_model before
any interaction, so raw dimension counts no longer mechanically bias the
result.

Reuses MetadataEmbedder from fusion_model.py unchanged (same 64-d Stage 1
metadata embedding). Adds SpatialImageEmbedder (new: stops at the
pre-avgpool feature map instead of the pooled vector) and
CrossAttentionFusionModel (new: metadata-as-query cross-attention + joint
head), alongside - not replacing - Stage 1's ImageEmbedder/FusionModel, so
Stage 1's late-fusion results and checkpoints stay reproducible.
"""

import torch
import torch.nn as nn

from src.models.fusion_model import MetadataEmbedder
from src.models.image_model import build_efficientnet_b0


class SpatialImageEmbedder(nn.Module):
    """Wraps a full build_efficientnet_b0() model; forward returns the
    49 (7x7) spatial tokens of 1280-d each from the pre-avgpool feature
    map, instead of the pooled 1280-d vector ImageEmbedder returns.

    Same full-architecture-wrapping approach as ImageEmbedder (not a
    re-keyed subset), so a Stage 1 image checkpoint's state_dict loads
    with strict=True.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = build_efficientnet_b0(num_classes=num_classes)
        self.embed_dim = self.backbone.classifier[-1].in_features  # 1280

    def load_stage1(self, checkpoint_path, device: torch.device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.backbone.load_state_dict(checkpoint["model_state_dict"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)  # [B, 1280, 7, 7]
        b, c, h, w = x.shape
        x = x.flatten(2)  # [B, 1280, 49]
        x = x.transpose(1, 2)  # [B, 49, 1280] - 49 spatial tokens
        return x


class MetadataChannelGate(nn.Module):
    """Optional dual-mechanism add-on (Suresh et al. TG-CAVNet-inspired,
    per Project_Tracking.md's "Future Improvements" - channel-wise gating
    + cross-attention). A metadata-conditioned sigmoid gate over the 1280
    image channels, applied before cross-attention, so metadata reweights
    channels *and* spatially attends rather than either alone.

    TG-CAVNet itself remains only partially captured in
    Literature_Review.md (row #2) - kept as a secondary, optional
    mechanism (can be disabled via use_channel_gate=False), not a primary
    design input pending its own full-text read.
    """

    def __init__(self, metadata_dim: int, num_channels: int):
        super().__init__()
        self.gate = nn.Linear(metadata_dim, num_channels)

    def forward(self, image_tokens: torch.Tensor, metadata_embedding: torch.Tensor) -> torch.Tensor:
        # image_tokens: [B, 49, C], metadata_embedding: [B, metadata_dim]
        channel_scale = torch.sigmoid(self.gate(metadata_embedding))  # [B, C]
        return image_tokens * channel_scale.unsqueeze(1)  # broadcast over 49 tokens


class CrossAttentionFusionModel(nn.Module):
    """Metadata (Query) cross-attends over EfficientNet-B0's 49 spatial
    image tokens (Key/Value) via standard multi-head scaled dot-product
    attention. Both modalities are projected into a shared d_model before
    interaction, so the 1280:64 raw-dimension imbalance that let Stage 1's
    concatenation-based fusion numerically favor the image branch no
    longer applies here.

    use_channel_gate=True (default) enables the optional TG-CAVNet-style
    channel gate ahead of attention (see MetadataChannelGate).
    """

    def __init__(
        self,
        metadata_input_dim: int,
        num_classes: int,
        d_model: int = 256,
        num_heads: int = 8,
        use_channel_gate: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.image_embedder = SpatialImageEmbedder(num_classes)
        self.metadata_embedder = MetadataEmbedder(metadata_input_dim, num_classes)

        self.use_channel_gate = use_channel_gate
        if use_channel_gate:
            self.channel_gate = MetadataChannelGate(
                metadata_dim=self.metadata_embedder.embed_dim,
                num_channels=self.image_embedder.embed_dim,
            )

        self.query_proj = nn.Linear(self.metadata_embedder.embed_dim, d_model)
        self.kv_proj = nn.Linear(self.image_embedder.embed_dim, d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=0.1, batch_first=True
        )

        joint_dim = d_model + self.metadata_embedder.embed_dim
        self.head = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def load_stage1_checkpoints(
        self, image_checkpoint_path, metadata_checkpoint_path, device: torch.device
    ) -> None:
        self.image_embedder.load_stage1(image_checkpoint_path, device)
        self.metadata_embedder.load_stage1(metadata_checkpoint_path, device)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        image_tokens = self.image_embedder(image)  # [B, 49, 1280]
        metadata_embedding = self.metadata_embedder(metadata)  # [B, 64]

        if self.use_channel_gate:
            image_tokens = self.channel_gate(image_tokens, metadata_embedding)

        query = self.query_proj(metadata_embedding).unsqueeze(1)  # [B, 1, d_model]
        key_value = self.kv_proj(image_tokens)  # [B, 49, d_model]
        attended, _ = self.attention(query, key_value, key_value)  # [B, 1, d_model]
        attended = attended.squeeze(1)  # [B, d_model]

        joint = torch.cat([attended, metadata_embedding], dim=1)
        return self.head(joint)
```

---

## Cell 9 — `%%writefile /kaggle/working/src/models/train.py`

```python
%%writefile /kaggle/working/src/models/train.py
"""Phase 6 Stage 1 training entrypoint - dataset-parameterized baselines.

Usage:
    python -m src.models.train --dataset PAD_UFES20 --branch image --seed 0
    python -m src.models.train --dataset HAM10000 --branch metadata --seed 0

Trains exactly one dataset/branch/seed combination per invocation. Uses
the train split for gradient updates and the val split for model
selection (early stopping, checkpoint picking). The test split is never
loaded here - it is read only by src/evaluation/evaluate.py, in a
separate, later, final run. See Project_Tracking.md decision (4) for the
full val/test discipline rationale.

Every dataset uses the identical recipe (architecture, hyperparameters,
class-weighted loss mechanism, seeds) - only the dataset's own paths,
class list, and metadata feature list differ (src/models/config.py).
This is intentional, per Project_Tracking.md's "Sequencing Decision -
HAM10000 Baseline Before Phase 7" entry: keeping the recipe fixed across
datasets means any later cross-dataset performance difference (Phase 8)
can be attributed to the dataset itself, not to per-dataset tuning.
"""

import argparse
import csv
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.models.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE_IMAGE,
    LEARNING_RATE_METADATA,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    get_dataset,
)
from src.models.dataset import ImageDataset, MetadataDataset, MetadataPreprocessor
from src.models.image_model import build_efficientnet_b0
from src.models.metadata_model import MetadataMLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(train_csv, class_names) -> torch.Tensor:
    """Inverse class frequency, computed from the train split only."""
    df = pd.read_csv(train_csv)
    counts = df["disease_label"].value_counts()
    freqs = np.array([counts.get(name, 0) for name in class_names], dtype=np.float64)
    weights = freqs.sum() / (len(class_names) * freqs)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def train_one_run(dataset_name: str, branch: str, seed: int) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)

    if branch == "image":
        train_ds = ImageDataset(ds_config.train_csv, ds_config, train=True)
        val_ds = ImageDataset(ds_config.val_csv, ds_config, train=False)
        model = build_efficientnet_b0(num_classes=ds_config.num_classes).to(device)
        lr = LEARNING_RATE_IMAGE
    elif branch == "metadata":
        preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))
        train_ds = MetadataDataset(ds_config.train_csv, ds_config, preprocessor)
        val_ds = MetadataDataset(ds_config.val_csv, ds_config, preprocessor)
        model = MetadataMLP(
            input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes
        ).to(device)
        lr = LEARNING_RATE_METADATA
    else:
        raise ValueError(f"Unknown branch: {branch}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(
        ds_config.train_csv, ds_config.class_names
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    ds_config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ds_config.logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{branch}_seed{seed}"
    checkpoint_path = ds_config.checkpoints_dir / f"{run_name}_best.pt"
    metrics_csv_path = ds_config.logs_dir / f"train_{run_name}.csv"

    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_macro_f1", "val_loss", "val_macro_f1"]
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            start = time.time()
            train_loss, train_macro_f1 = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_macro_f1 = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )
            writer.writerow([epoch, train_loss, train_macro_f1, val_loss, val_macro_f1])
            f.flush()
            elapsed = time.time() - start
            print(
                f"[{dataset_name}/{run_name}] epoch {epoch:02d} "
                f"train_loss={train_loss:.4f} train_macroF1={train_macro_f1:.4f} "
                f"val_loss={val_loss:.4f} val_macroF1={val_macro_f1:.4f} "
                f"({elapsed:.1f}s)"
            )

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "dataset": dataset_name,
                        "branch": branch,
                        "seed": seed,
                        "epoch": epoch,
                        "val_macro_f1": val_macro_f1,
                        "num_classes": ds_config.num_classes,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"[{dataset_name}/{run_name}] early stopping at epoch {epoch}")
                    break

    summary = {
        "dataset": dataset_name,
        "branch": branch,
        "seed": seed,
        "best_val_macro_f1": best_val_macro_f1,
        "checkpoint_path": str(checkpoint_path),
    }
    summary_path = ds_config.logs_dir / f"train_{run_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[{dataset_name}/{run_name}] best val macro-F1: "
        f"{best_val_macro_f1:.4f} -> {checkpoint_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Stage 1 training")
    parser.add_argument("--dataset", choices=["PAD_UFES20", "HAM10000"], required=True)
    parser.add_argument("--branch", choices=["image", "metadata"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train_one_run(args.dataset, args.branch, args.seed)


if __name__ == "__main__":
    main()
```

---

## Cell 10 — `%%writefile /kaggle/working/src/models/train_fusion.py`

```python
%%writefile /kaggle/working/src/models/train_fusion.py
"""Phase 7 Stage 1 training entrypoint - PAD-UFES-20 late fusion only.

Usage:
    python -m src.models.train_fusion --dataset PAD_UFES20 --seed 0

Warm-starts both branches from their Stage 1 checkpoints
(ds_config.stage1_checkpoints_dir / f"{branch}_seed{seed}_best.pt" -
seed-matched, e.g. fusion seed 0 loads image_seed0_best.pt and
metadata_seed0_best.pt), then fine-tunes the whole model end-to-end
(nothing frozen) at LEARNING_RATE_FUSION - lower than either branch's own
Stage 1 LR, since both are already converged.

Same loss/metric/seed/split discipline as Stage 1 (train.py): class-
weighted CrossEntropyLoss, macro-F1 for model selection and early
stopping, SEEDS = [0, 1, 2], identical train/val split files. The test
split is never loaded here - only src/evaluation/evaluate.py reads it,
later.
"""

import argparse
import csv
import json
import time

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.models.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE_FUSION,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    get_dataset,
)
from src.models.dataset import FusionDataset, MetadataPreprocessor
from src.models.fusion_model import FusionModel
from src.models.train import compute_class_weights, set_seed


def run_epoch_fusion(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, metadata, labels in loader:
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def train_one_run(dataset_name: str, seed: int) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)

    preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))
    train_ds = FusionDataset(ds_config.train_csv, ds_config, preprocessor, train=True)
    val_ds = FusionDataset(ds_config.val_csv, ds_config, preprocessor, train=False)

    model = FusionModel(
        metadata_input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes
    ).to(device)

    image_checkpoint = ds_config.stage1_checkpoints_dir / f"image_seed{seed}_best.pt"
    metadata_checkpoint = ds_config.stage1_checkpoints_dir / f"metadata_seed{seed}_best.pt"
    for path in (image_checkpoint, metadata_checkpoint):
        if not path.exists():
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found: {path} - fusion warm-start "
                f"requires both branches' Stage 1 checkpoints to exist first."
            )
    model.load_stage1_checkpoints(image_checkpoint, metadata_checkpoint, device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(
        ds_config.train_csv, ds_config.class_names
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE_FUSION, weight_decay=WEIGHT_DECAY
    )

    ds_config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ds_config.logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"fusion_seed{seed}"
    checkpoint_path = ds_config.checkpoints_dir / f"{run_name}_best.pt"
    metrics_csv_path = ds_config.logs_dir / f"train_{run_name}.csv"

    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_macro_f1", "val_loss", "val_macro_f1"]
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            start = time.time()
            train_loss, train_macro_f1 = run_epoch_fusion(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_macro_f1 = run_epoch_fusion(
                model, val_loader, criterion, optimizer, device, train=False
            )
            writer.writerow([epoch, train_loss, train_macro_f1, val_loss, val_macro_f1])
            f.flush()
            elapsed = time.time() - start
            print(
                f"[{dataset_name}/{run_name}] epoch {epoch:02d} "
                f"train_loss={train_loss:.4f} train_macroF1={train_macro_f1:.4f} "
                f"val_loss={val_loss:.4f} val_macroF1={val_macro_f1:.4f} "
                f"({elapsed:.1f}s)"
            )

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "dataset": dataset_name,
                        "branch": "fusion",
                        "seed": seed,
                        "epoch": epoch,
                        "val_macro_f1": val_macro_f1,
                        "num_classes": ds_config.num_classes,
                        "metadata_input_dim": preprocessor.output_dim,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"[{dataset_name}/{run_name}] early stopping at epoch {epoch}")
                    break

    summary = {
        "dataset": dataset_name,
        "branch": "fusion",
        "seed": seed,
        "best_val_macro_f1": best_val_macro_f1,
        "checkpoint_path": str(checkpoint_path),
        "warm_start_image_checkpoint": str(image_checkpoint),
        "warm_start_metadata_checkpoint": str(metadata_checkpoint),
    }
    summary_path = ds_config.logs_dir / f"train_{run_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[{dataset_name}/{run_name}] best val macro-F1: "
        f"{best_val_macro_f1:.4f} -> {checkpoint_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 7 Stage 1 fusion training")
    parser.add_argument("--dataset", choices=["PAD_UFES20"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train_one_run(args.dataset, args.seed)


if __name__ == "__main__":
    main()
```

---

## Cell 11 — `%%writefile /kaggle/working/src/models/train_cross_attention_fusion.py`

```python
%%writefile /kaggle/working/src/models/train_cross_attention_fusion.py
"""Phase 7 Stage 2 training entrypoint - PAD-UFES-20 cross-attention fusion only.

Usage:
    python -m src.models.train_cross_attention_fusion --dataset PAD_UFES20 --seed 0

Same warm-start-then-fine-tune discipline as Stage 1's train_fusion.py:
warm-starts both embedders from their Stage 1 checkpoints
(ds_config.stage1_checkpoints_dir / f"{branch}_seed{seed}_best.pt" -
seed-matched), then fine-tunes the whole model end-to-end (nothing
frozen, including the new cross-attention/head parameters) at
LEARNING_RATE_CROSS_ATTENTION.

Same loss/metric/seed/split discipline as every prior stage: class-
weighted CrossEntropyLoss, macro-F1 for model selection and early
stopping, SEEDS = [0, 1, 2], identical train/val split files. The test
split is never loaded here - only src/evaluation/evaluate.py reads it,
later.
"""

import argparse
import csv
import json
import time

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.models.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE_CROSS_ATTENTION,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    get_dataset,
)
from src.models.cross_attention_fusion_model import CrossAttentionFusionModel
from src.models.dataset import FusionDataset, MetadataPreprocessor
from src.models.train import compute_class_weights, set_seed


def run_epoch_cross_attention(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, metadata, labels in loader:
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def train_one_run(dataset_name: str, seed: int) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)

    preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))
    train_ds = FusionDataset(ds_config.train_csv, ds_config, preprocessor, train=True)
    val_ds = FusionDataset(ds_config.val_csv, ds_config, preprocessor, train=False)

    model = CrossAttentionFusionModel(
        metadata_input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes
    ).to(device)

    image_checkpoint = ds_config.stage1_checkpoints_dir / f"image_seed{seed}_best.pt"
    metadata_checkpoint = ds_config.stage1_checkpoints_dir / f"metadata_seed{seed}_best.pt"
    for path in (image_checkpoint, metadata_checkpoint):
        if not path.exists():
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found: {path} - cross-attention "
                f"warm-start requires both branches' Stage 1 checkpoints to "
                f"exist first."
            )
    model.load_stage1_checkpoints(image_checkpoint, metadata_checkpoint, device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(
        ds_config.train_csv, ds_config.class_names
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE_CROSS_ATTENTION, weight_decay=WEIGHT_DECAY
    )

    ds_config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ds_config.logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"cross_attention_seed{seed}"
    checkpoint_path = ds_config.checkpoints_dir / f"{run_name}_best.pt"
    metrics_csv_path = ds_config.logs_dir / f"train_{run_name}.csv"

    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_macro_f1", "val_loss", "val_macro_f1"]
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            start = time.time()
            train_loss, train_macro_f1 = run_epoch_cross_attention(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_macro_f1 = run_epoch_cross_attention(
                model, val_loader, criterion, optimizer, device, train=False
            )
            writer.writerow([epoch, train_loss, train_macro_f1, val_loss, val_macro_f1])
            f.flush()
            elapsed = time.time() - start
            print(
                f"[{dataset_name}/{run_name}] epoch {epoch:02d} "
                f"train_loss={train_loss:.4f} train_macroF1={train_macro_f1:.4f} "
                f"val_loss={val_loss:.4f} val_macroF1={val_macro_f1:.4f} "
                f"({elapsed:.1f}s)"
            )

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "dataset": dataset_name,
                        "branch": "cross_attention",
                        "seed": seed,
                        "epoch": epoch,
                        "val_macro_f1": val_macro_f1,
                        "num_classes": ds_config.num_classes,
                        "metadata_input_dim": preprocessor.output_dim,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"[{dataset_name}/{run_name}] early stopping at epoch {epoch}")
                    break

    summary = {
        "dataset": dataset_name,
        "branch": "cross_attention",
        "seed": seed,
        "best_val_macro_f1": best_val_macro_f1,
        "checkpoint_path": str(checkpoint_path),
        "warm_start_image_checkpoint": str(image_checkpoint),
        "warm_start_metadata_checkpoint": str(metadata_checkpoint),
    }
    summary_path = ds_config.logs_dir / f"train_{run_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[{dataset_name}/{run_name}] best val macro-F1: "
        f"{best_val_macro_f1:.4f} -> {checkpoint_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 7 Stage 2 cross-attention fusion training")
    parser.add_argument("--dataset", choices=["PAD_UFES20"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train_one_run(args.dataset, args.seed)


if __name__ == "__main__":
    main()
```

---

## Cell 12 — `%%writefile /kaggle/working/src/models/train_metadata_reduced.py`

```python
%%writefile /kaggle/working/src/models/train_metadata_reduced.py
"""Phase 8 reduced-feature metadata training - PAD-UFES-20 only.

Usage:
    python -m src.models.train_metadata_reduced --dataset PAD_UFES20 --seed 0

Trains a metadata-only model on PAD-UFES-20 restricted to the 3 columns
HAM10000 also has (age, sex, anatomical_site - config.REDUCED_NUMERIC_FEATURES/
REDUCED_CATEGORICAL_FEATURES), with anatomical_site normalized into
HAM10000's vocabulary (config.normalize_anatomical_site_for_cross_dataset,
per docs/Phase8_Anatomical_Site_Mapping.csv, approved 2026-07-18).

Purpose-built for Phase 8's PAD-UFES-20 -> HAM10000 cross-dataset
generalization experiment - the existing rich-feature (21-column)
metadata_seed{N}_best.pt checkpoints cannot run on HAM10000 data at all
(18 of PAD-UFES-20's columns don't exist there), so this schema-matched
variant is a separate, additional model, not a replacement. Saved as
metadata_reduced_seed{N}_best.pt, alongside (not overwriting) the existing
metadata_seed{N}_best.pt used for every already-reported PAD-UFES-20-
internal result.

Mirrors train.py's metadata branch exactly (same architecture,
hyperparameters, class-weighted loss mechanism, seeds) - only the
DatasetConfig's feature lists and the preprocessor's anatomical_site
normalization differ. Same loss/metric/seed/split discipline as every
prior stage: class-weighted CrossEntropyLoss, macro-F1 for model
selection/early stopping, SEEDS = [0, 1, 2], identical train/val split
files. Test split untouched.
"""

import argparse
import csv
import json
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE_METADATA,
    NUM_EPOCHS,
    REDUCED_CATEGORICAL_FEATURES,
    REDUCED_NUMERIC_FEATURES,
    WEIGHT_DECAY,
    get_dataset,
    normalize_anatomical_site_for_cross_dataset,
)
from src.models.dataset import MetadataDataset, MetadataPreprocessor
from src.models.metadata_model import MetadataMLP
from src.models.train import compute_class_weights, run_epoch, set_seed


def build_reduced_preprocessor(ds_config, train_df: pd.DataFrame):
    reduced_config = ds_config.with_features(REDUCED_NUMERIC_FEATURES, REDUCED_CATEGORICAL_FEATURES)
    preprocessor = MetadataPreprocessor(
        reduced_config,
        column_transforms={"anatomical_site": normalize_anatomical_site_for_cross_dataset},
    ).fit(train_df)
    return reduced_config, preprocessor


def train_one_run(dataset_name: str, seed: int) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)

    train_df = pd.read_csv(ds_config.train_csv)
    reduced_config, preprocessor = build_reduced_preprocessor(ds_config, train_df)

    train_ds = MetadataDataset(reduced_config.train_csv, reduced_config, preprocessor)
    val_ds = MetadataDataset(reduced_config.val_csv, reduced_config, preprocessor)
    model = MetadataMLP(input_dim=preprocessor.output_dim, num_classes=reduced_config.num_classes).to(device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(
        reduced_config.train_csv, reduced_config.class_names
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE_METADATA, weight_decay=WEIGHT_DECAY
    )

    ds_config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ds_config.logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"metadata_reduced_seed{seed}"
    checkpoint_path = ds_config.checkpoints_dir / f"{run_name}_best.pt"
    metrics_csv_path = ds_config.logs_dir / f"train_{run_name}.csv"

    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_macro_f1", "val_loss", "val_macro_f1"]
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            start = time.time()
            train_loss, train_macro_f1 = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_macro_f1 = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )
            writer.writerow([epoch, train_loss, train_macro_f1, val_loss, val_macro_f1])
            f.flush()
            elapsed = time.time() - start
            print(
                f"[{dataset_name}/{run_name}] epoch {epoch:02d} "
                f"train_loss={train_loss:.4f} train_macroF1={train_macro_f1:.4f} "
                f"val_loss={val_loss:.4f} val_macroF1={val_macro_f1:.4f} "
                f"({elapsed:.1f}s)"
            )

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "dataset": dataset_name,
                        "branch": "metadata_reduced",
                        "seed": seed,
                        "epoch": epoch,
                        "val_macro_f1": val_macro_f1,
                        "num_classes": reduced_config.num_classes,
                        "metadata_input_dim": preprocessor.output_dim,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"[{dataset_name}/{run_name}] early stopping at epoch {epoch}")
                    break

    summary = {
        "dataset": dataset_name,
        "branch": "metadata_reduced",
        "seed": seed,
        "best_val_macro_f1": best_val_macro_f1,
        "checkpoint_path": str(checkpoint_path),
        "numeric_features": REDUCED_NUMERIC_FEATURES,
        "categorical_features": REDUCED_CATEGORICAL_FEATURES,
        "metadata_input_dim": preprocessor.output_dim,
    }
    summary_path = ds_config.logs_dir / f"train_{run_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[{dataset_name}/{run_name}] best val macro-F1: "
        f"{best_val_macro_f1:.4f} -> {checkpoint_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 8 reduced-feature metadata training")
    parser.add_argument("--dataset", choices=["PAD_UFES20"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train_one_run(args.dataset, args.seed)


if __name__ == "__main__":
    main()
```

---

## Cell 13 — `%%writefile /kaggle/working/src/models/train_fusion_reduced.py`

```python
%%writefile /kaggle/working/src/models/train_fusion_reduced.py
"""Phase 8 reduced-feature late-fusion training - PAD-UFES-20 only.

Usage:
    python -m src.models.train_fusion_reduced --dataset PAD_UFES20 --seed 0

Schema-matched counterpart to train_fusion.py for the PAD-UFES-20 ->
HAM10000 cross-dataset generalization experiment: metadata is restricted
to the 3 columns HAM10000 also has (age, sex, anatomical_site), with
anatomical_site normalized into HAM10000's vocabulary (per
docs/Phase8_Anatomical_Site_Mapping.csv, approved 2026-07-18). The image
branch has no schema mismatch, so it warm-starts from the *existing,
unchanged* image_seed{N}_best.pt (Phase 6 Stage 1); only the metadata side
warm-starts from the new metadata_reduced_seed{N}_best.pt
(train_metadata_reduced.py) instead of the rich-feature
metadata_seed{N}_best.pt.

Saved as fusion_reduced_seed{N}_best.pt, alongside (not overwriting) Phase
7 Stage 1's fusion_seed{N}_best.pt, which remains the checkpoint used for
every already-reported PAD-UFES-20-internal late-fusion result.

Same loss/metric/seed/split discipline as every prior stage.
"""

import argparse
import csv
import json
import time

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.models.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE_FUSION,
    NUM_EPOCHS,
    REDUCED_CATEGORICAL_FEATURES,
    REDUCED_NUMERIC_FEATURES,
    WEIGHT_DECAY,
    get_dataset,
    normalize_anatomical_site_for_cross_dataset,
)
from src.models.dataset import FusionDataset, MetadataPreprocessor
from src.models.fusion_model import FusionModel
from src.models.train import compute_class_weights, set_seed
from src.models.train_fusion import run_epoch_fusion


def train_one_run(dataset_name: str, seed: int) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)
    reduced_config = ds_config.with_features(REDUCED_NUMERIC_FEATURES, REDUCED_CATEGORICAL_FEATURES)

    train_df = pd.read_csv(reduced_config.train_csv)
    preprocessor = MetadataPreprocessor(
        reduced_config,
        column_transforms={"anatomical_site": normalize_anatomical_site_for_cross_dataset},
    ).fit(train_df)

    train_ds = FusionDataset(reduced_config.train_csv, reduced_config, preprocessor, train=True)
    val_ds = FusionDataset(reduced_config.val_csv, reduced_config, preprocessor, train=False)

    model = FusionModel(
        metadata_input_dim=preprocessor.output_dim, num_classes=reduced_config.num_classes
    ).to(device)

    image_checkpoint = ds_config.stage1_checkpoints_dir / f"image_seed{seed}_best.pt"
    metadata_checkpoint = ds_config.checkpoints_dir / f"metadata_reduced_seed{seed}_best.pt"
    for path in (image_checkpoint, metadata_checkpoint):
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path} - reduced-feature fusion warm-start "
                f"requires the existing Stage 1 image checkpoint and the new "
                f"metadata_reduced checkpoint (train_metadata_reduced.py) to exist first."
            )
    model.load_stage1_checkpoints(image_checkpoint, metadata_checkpoint, device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(
        reduced_config.train_csv, reduced_config.class_names
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE_FUSION, weight_decay=WEIGHT_DECAY
    )

    ds_config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ds_config.logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"fusion_reduced_seed{seed}"
    checkpoint_path = ds_config.checkpoints_dir / f"{run_name}_best.pt"
    metrics_csv_path = ds_config.logs_dir / f"train_{run_name}.csv"

    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_macro_f1", "val_loss", "val_macro_f1"]
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            start = time.time()
            train_loss, train_macro_f1 = run_epoch_fusion(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_macro_f1 = run_epoch_fusion(
                model, val_loader, criterion, optimizer, device, train=False
            )
            writer.writerow([epoch, train_loss, train_macro_f1, val_loss, val_macro_f1])
            f.flush()
            elapsed = time.time() - start
            print(
                f"[{dataset_name}/{run_name}] epoch {epoch:02d} "
                f"train_loss={train_loss:.4f} train_macroF1={train_macro_f1:.4f} "
                f"val_loss={val_loss:.4f} val_macroF1={val_macro_f1:.4f} "
                f"({elapsed:.1f}s)"
            )

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "dataset": dataset_name,
                        "branch": "fusion_reduced",
                        "seed": seed,
                        "epoch": epoch,
                        "val_macro_f1": val_macro_f1,
                        "num_classes": reduced_config.num_classes,
                        "metadata_input_dim": preprocessor.output_dim,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"[{dataset_name}/{run_name}] early stopping at epoch {epoch}")
                    break

    summary = {
        "dataset": dataset_name,
        "branch": "fusion_reduced",
        "seed": seed,
        "best_val_macro_f1": best_val_macro_f1,
        "checkpoint_path": str(checkpoint_path),
        "warm_start_image_checkpoint": str(image_checkpoint),
        "warm_start_metadata_checkpoint": str(metadata_checkpoint),
    }
    summary_path = ds_config.logs_dir / f"train_{run_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[{dataset_name}/{run_name}] best val macro-F1: "
        f"{best_val_macro_f1:.4f} -> {checkpoint_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 8 reduced-feature late-fusion training")
    parser.add_argument("--dataset", choices=["PAD_UFES20"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train_one_run(args.dataset, args.seed)


if __name__ == "__main__":
    main()
```

---

## Cell 14 — `%%writefile /kaggle/working/src/models/train_cross_attention_fusion_reduced.py`

```python
%%writefile /kaggle/working/src/models/train_cross_attention_fusion_reduced.py
"""Phase 8 reduced-feature cross-attention fusion training - PAD-UFES-20 only.

Usage:
    python -m src.models.train_cross_attention_fusion_reduced --dataset PAD_UFES20 --seed 0

Schema-matched counterpart to train_cross_attention_fusion.py for the
PAD-UFES-20 -> HAM10000 cross-dataset generalization experiment - same
reduced-feature/site-normalization rationale as train_fusion_reduced.py.
Warm-starts the image embedder from the existing, unchanged
image_seed{N}_best.pt and the metadata embedder from the new
metadata_reduced_seed{N}_best.pt.

Saved as cross_attention_reduced_seed{N}_best.pt, alongside (not
overwriting) Phase 7 Stage 2's cross_attention_seed{N}_best.pt, which
remains the checkpoint used for every already-reported PAD-UFES-20-
internal cross-attention result.
"""

import argparse
import csv
import json
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE_CROSS_ATTENTION,
    NUM_EPOCHS,
    REDUCED_CATEGORICAL_FEATURES,
    REDUCED_NUMERIC_FEATURES,
    WEIGHT_DECAY,
    get_dataset,
    normalize_anatomical_site_for_cross_dataset,
)
from src.models.cross_attention_fusion_model import CrossAttentionFusionModel
from src.models.dataset import FusionDataset, MetadataPreprocessor
from src.models.train import compute_class_weights, set_seed
from src.models.train_cross_attention_fusion import run_epoch_cross_attention


def train_one_run(dataset_name: str, seed: int) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)
    reduced_config = ds_config.with_features(REDUCED_NUMERIC_FEATURES, REDUCED_CATEGORICAL_FEATURES)

    train_df = pd.read_csv(reduced_config.train_csv)
    preprocessor = MetadataPreprocessor(
        reduced_config,
        column_transforms={"anatomical_site": normalize_anatomical_site_for_cross_dataset},
    ).fit(train_df)

    train_ds = FusionDataset(reduced_config.train_csv, reduced_config, preprocessor, train=True)
    val_ds = FusionDataset(reduced_config.val_csv, reduced_config, preprocessor, train=False)

    model = CrossAttentionFusionModel(
        metadata_input_dim=preprocessor.output_dim, num_classes=reduced_config.num_classes
    ).to(device)

    image_checkpoint = ds_config.stage1_checkpoints_dir / f"image_seed{seed}_best.pt"
    metadata_checkpoint = ds_config.checkpoints_dir / f"metadata_reduced_seed{seed}_best.pt"
    for path in (image_checkpoint, metadata_checkpoint):
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path} - reduced-feature cross-attention "
                f"warm-start requires the existing Stage 1 image checkpoint and "
                f"the new metadata_reduced checkpoint (train_metadata_reduced.py) "
                f"to exist first."
            )
    model.load_stage1_checkpoints(image_checkpoint, metadata_checkpoint, device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(
        reduced_config.train_csv, reduced_config.class_names
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE_CROSS_ATTENTION, weight_decay=WEIGHT_DECAY
    )

    ds_config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ds_config.logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"cross_attention_reduced_seed{seed}"
    checkpoint_path = ds_config.checkpoints_dir / f"{run_name}_best.pt"
    metrics_csv_path = ds_config.logs_dir / f"train_{run_name}.csv"

    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_macro_f1", "val_loss", "val_macro_f1"]
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            start = time.time()
            train_loss, train_macro_f1 = run_epoch_cross_attention(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_macro_f1 = run_epoch_cross_attention(
                model, val_loader, criterion, optimizer, device, train=False
            )
            writer.writerow([epoch, train_loss, train_macro_f1, val_loss, val_macro_f1])
            f.flush()
            elapsed = time.time() - start
            print(
                f"[{dataset_name}/{run_name}] epoch {epoch:02d} "
                f"train_loss={train_loss:.4f} train_macroF1={train_macro_f1:.4f} "
                f"val_loss={val_loss:.4f} val_macroF1={val_macro_f1:.4f} "
                f"({elapsed:.1f}s)"
            )

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "dataset": dataset_name,
                        "branch": "cross_attention_reduced",
                        "seed": seed,
                        "epoch": epoch,
                        "val_macro_f1": val_macro_f1,
                        "num_classes": reduced_config.num_classes,
                        "metadata_input_dim": preprocessor.output_dim,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"[{dataset_name}/{run_name}] early stopping at epoch {epoch}")
                    break

    summary = {
        "dataset": dataset_name,
        "branch": "cross_attention_reduced",
        "seed": seed,
        "best_val_macro_f1": best_val_macro_f1,
        "checkpoint_path": str(checkpoint_path),
        "warm_start_image_checkpoint": str(image_checkpoint),
        "warm_start_metadata_checkpoint": str(metadata_checkpoint),
    }
    summary_path = ds_config.logs_dir / f"train_{run_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[{dataset_name}/{run_name}] best val macro-F1: "
        f"{best_val_macro_f1:.4f} -> {checkpoint_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8 reduced-feature cross-attention fusion training"
    )
    parser.add_argument("--dataset", choices=["PAD_UFES20"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train_one_run(args.dataset, args.seed)


if __name__ == "__main__":
    main()
```

---

## Cell 15 — Sanity check (reduced-feature preprocessor + normalization)

```python
import sys
for mod in list(sys.modules):
    if mod.startswith("src."):
        del sys.modules[mod]

import pandas as pd
from src.models.config import (
    get_dataset, REDUCED_NUMERIC_FEATURES, REDUCED_CATEGORICAL_FEATURES,
    normalize_anatomical_site_for_cross_dataset,
)
from src.models.dataset import MetadataPreprocessor

ds_config = get_dataset("PAD_UFES20")
assert ds_config.train_csv.exists(), "metadata_train.csv not found"

for site in ("CHEST", "FOREARM", "ARM", "THIGH", "NOSE", "LIP"):
    print(site, "->", normalize_anatomical_site_for_cross_dataset(site))

reduced_config = ds_config.with_features(REDUCED_NUMERIC_FEATURES, REDUCED_CATEGORICAL_FEATURES)
train_df = pd.read_csv(reduced_config.train_csv)
preprocessor = MetadataPreprocessor(
    reduced_config, column_transforms={"anatomical_site": normalize_anatomical_site_for_cross_dataset}
).fit(train_df)
print("\nreduced output_dim:", preprocessor.output_dim)
print("anatomical_site categories:", preprocessor.categorical_values["anatomical_site"])
assert "upper extremity" in preprocessor.categorical_values["anatomical_site"]
assert "__MISSING__" in preprocessor.categorical_values["anatomical_site"]

for path in (ds_config.stage1_checkpoints_dir / f"image_seed{s}_best.pt" for s in (0, 1, 2)):
    assert path.exists(), f"Missing Stage 1 image checkpoint: {path}"
print("\nSANITY CHECK PASSED")
```

---

## Cell 16 — Train: metadata_reduced, seed 0

```python
!python -m src.models.train_metadata_reduced --dataset PAD_UFES20 --seed 0
```

---

## Cell 17 — Train: metadata_reduced, seed 1

```python
!python -m src.models.train_metadata_reduced --dataset PAD_UFES20 --seed 1
```

---

## Cell 18 — Train: metadata_reduced, seed 2

```python
!python -m src.models.train_metadata_reduced --dataset PAD_UFES20 --seed 2
```

---

## Cell 19 — Train: fusion_reduced, seed 0

```python
!python -m src.models.train_fusion_reduced --dataset PAD_UFES20 --seed 0
```

---

## Cell 20 — Train: fusion_reduced, seed 1

```python
!python -m src.models.train_fusion_reduced --dataset PAD_UFES20 --seed 1
```

---

## Cell 21 — Train: fusion_reduced, seed 2

```python
!python -m src.models.train_fusion_reduced --dataset PAD_UFES20 --seed 2
```

---

## Cell 22 — Train: cross_attention_reduced, seed 0

```python
!python -m src.models.train_cross_attention_fusion_reduced --dataset PAD_UFES20 --seed 0
```

---

## Cell 23 — Train: cross_attention_reduced, seed 1

```python
!python -m src.models.train_cross_attention_fusion_reduced --dataset PAD_UFES20 --seed 1
```

---

## Cell 24 — Train: cross_attention_reduced, seed 2

```python
!python -m src.models.train_cross_attention_fusion_reduced --dataset PAD_UFES20 --seed 2
```

