# PAD-UFES-20 Kaggle Notebook — Phase 7 Stage 1 (Late Fusion)

> **WARNING - read before pasting any cell into Kaggle:** for every `%%writefile` cell below, `%%writefile <path>` MUST be the exact first line of the Kaggle cell, with absolutely nothing above it - no blank line, no comment, no stray character. Kaggle (like Jupyter) only treats a line as a cell magic if it is the first line of the cell; anything preceding it turns `%%writefile` into a plain, non-magic line and the file silently never gets written, which is exactly what broke the last run. **After pasting each `%%writefile` cell, re-open it and visually confirm `%%writefile` is line 1 before running it.**

Structured for "Save & Run All (Commit)" from the start, same process as
the Stage 1 baseline notebooks. Requires **three** Kaggle "Add Data"
sources attached to the notebook before running:

- `mahdavi1202/skin-cancer` (raw PAD-UFES-20 image mirror)
- `naeemsarkertracer/pad-ufes20-processed` (our processed metadata CSVs)
- `naeemsarkertracer/pad-ufes20-stage1-checkpoints` (published —
  https://www.kaggle.com/datasets/naeemsarkertracer/pad-ufes20-stage1-checkpoints),
  containing the Stage 1 checkpoints (`image_seed{0,1,2}_best.pt`,
  `metadata_seed{0,1,2}_best.pt`). Fusion training warm-starts from
  these; it cannot proceed without them. Like the processed-metadata
  dataset, this was almost certainly zipped from the `checkpoints/`
  folder itself, so it likely mounts with an extra nested
  `checkpoints/` subfolder rather than the 6 `.pt` files directly at
  the mount root — Cell 1 checks both candidate layouts explicitly
  and reports which one is real, rather than assuming either way.

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

# --- Stage 1 checkpoint dataset check ------------------------------------
# Wrapping is uncertain (same ambiguity the processed-metadata dataset
# had) - check both candidate layouts explicitly rather than assuming.
# Priority order mirrors src/models/config.py's _stage1_checkpoints_dir():
# wrapped candidate (root/checkpoints/) checked first, root second.
checkpoint_root = "/kaggle/input/datasets/naeemsarkertracer/pad-ufes20-stage1-checkpoints"
show(checkpoint_root, "Stage 1 checkpoints (root)")
wrapped_ckpt = os.path.join(checkpoint_root, "checkpoints")
show(wrapped_ckpt, "Stage 1 checkpoints (wrapped candidate: root/checkpoints/)")

expected = [f"{b}_seed{s}_best.pt" for b in ("image", "metadata") for s in (0, 1, 2)]

def all_present(base):
    return all(os.path.isfile(os.path.join(base, f)) for f in expected)

wrapped_ok = all_present(wrapped_ckpt)
root_ok = all_present(checkpoint_root)
print(f"\nAll 6 expected checkpoints found at wrapped candidate ({wrapped_ckpt}): {wrapped_ok}")
print(f"All 6 expected checkpoints found at root ({checkpoint_root}): {root_ok}")

if wrapped_ok:
    resolved_checkpoint_dir = wrapped_ckpt
    print(f"\n-> Wrapped layout confirmed (root/checkpoints/) - matches what "
          f"_stage1_checkpoints_dir() will resolve to at train time.")
elif root_ok:
    resolved_checkpoint_dir = checkpoint_root
    print(f"\n-> Unwrapped layout confirmed (files directly under root) - matches "
          f"what _stage1_checkpoints_dir() will resolve to at train time.")
else:
    resolved_checkpoint_dir = None

assert resolved_checkpoint_dir is not None, (
    "Stage 1 checkpoints not found in either expected layout (wrapped under "
    "checkpoints/, or directly at root) - fusion warm-start cannot proceed. "
    "Re-check the uploaded dataset's contents/packaging."
)

print("\nOK: raw images present, processed dataset wrapped as expected, "
      f"all 6 Stage 1 checkpoints found at: {resolved_checkpoint_dir}")
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
WEIGHT_DECAY = 1e-4


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
    """

    def __init__(self, dataset_config: DatasetConfig):
        self.numeric_features = dataset_config.numeric_features
        self.categorical_features = dataset_config.categorical_features
        self.numeric_means = {}
        self.numeric_stds = {}
        self.categorical_values = {}  # col -> sorted list of seen categories

    def fit(self, df: pd.DataFrame) -> "MetadataPreprocessor":
        for col in self.numeric_features:
            values = pd.to_numeric(df[col], errors="coerce")
            self.numeric_means[col] = values.mean()
            std = values.std()
            self.numeric_stds[col] = std if std and std > 0 else 1.0
        for col in self.categorical_features:
            values = df[col].astype("string").fillna("__MISSING__")
            self.categorical_values[col] = sorted(values.unique().tolist())
        return self

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
            value = row[col]
            value = "__MISSING__" if pd.isna(value) else str(value)
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

## Cell 8 — `%%writefile /kaggle/working/src/models/train.py`

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

## Cell 9 — `%%writefile /kaggle/working/src/models/train_fusion.py`

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

## Cell 10 — Sanity check (config load + Stage 1 checkpoint resolution + real image path resolution)

```python
import sys
for mod in list(sys.modules):
    if mod.startswith("src."):
        del sys.modules[mod]

import pandas as pd
from src.models.config import get_dataset, resolve_image_path

ds_config = get_dataset("PAD_UFES20")
print("num_classes:", ds_config.num_classes)
print("train_csv:", ds_config.train_csv, "exists:", ds_config.train_csv.exists())
print("val_csv:  ", ds_config.val_csv, "exists:", ds_config.val_csv.exists())
print("test_csv: ", ds_config.test_csv, "exists:", ds_config.test_csv.exists())
assert ds_config.train_csv.exists(), "metadata_train.csv not found - check processed dataset nesting"

print("\nstage1_checkpoints_dir:", ds_config.stage1_checkpoints_dir)
for branch in ("image", "metadata"):
    for seed in (0, 1, 2):
        p = ds_config.stage1_checkpoints_dir / f"{branch}_seed{seed}_best.pt"
        assert p.exists(), f"Missing Stage 1 checkpoint: {p}"
print("all 6 Stage 1 checkpoints resolved OK")

df = pd.read_csv(ds_config.train_csv)
sample_image_path = df.iloc[0]["image_path"]
resolved = resolve_image_path(sample_image_path)
print("\nsample image_path (from CSV):", sample_image_path)
print("resolved filesystem path:    ", resolved)
print("resolved path exists:        ", resolved.exists())
assert resolved.exists(), f"Resolved image path does not exist: {resolved}"

import random
random.seed(0)
sample_rows = df.sample(n=min(20, len(df)), random_state=0)
missing = []
for _, row in sample_rows.iterrows():
    p = resolve_image_path(row["image_path"])
    if not p.exists():
        missing.append((row["image_path"], str(p)))
print(f"\nChecked {len(sample_rows)} random rows, missing: {len(missing)}")
if missing:
    print("First few missing:", missing[:5])
assert not missing, "Some resolved image paths do not exist - check folder-verification cell output"

print("\nSANITY CHECK PASSED")
```

---

## Cell 11 — Full model/GPU/dependency check (fusion model, warm-start, one real batch)

```python
import torch, sys
print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

from src.models.config import get_dataset
from src.models.dataset import FusionDataset, MetadataPreprocessor
from src.models.fusion_model import FusionModel
import pandas as pd

ds_config = get_dataset("PAD_UFES20")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))
model = FusionModel(metadata_input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes)

image_ckpt = ds_config.stage1_checkpoints_dir / "image_seed0_best.pt"
metadata_ckpt = ds_config.stage1_checkpoints_dir / "metadata_seed0_best.pt"
model.load_stage1_checkpoints(image_ckpt, metadata_ckpt, device)
model.to(device)
model.eval()  # single-sample batch below - BatchNorm1d needs eval mode, not train mode, to run on batch size 1
print("Stage 1 checkpoints (seed 0) loaded into FusionModel OK")

train_ds = FusionDataset(ds_config.train_csv, ds_config, preprocessor, train=True)
image, metadata, label = train_ds[0]
assert image.shape == (3, 224, 224)
assert metadata.shape == (preprocessor.output_dim,)

with torch.no_grad():
    out = model(image.unsqueeze(0).to(device), metadata.unsqueeze(0).to(device))
assert out.shape == (1, ds_config.num_classes)
print("fusion model forward pass OK, output shape:", out.shape)

print("\nALL CHECKS PASSED - ready to train")
```

---

## Cell 12 — Train: fusion, seed 0

```python
!python -m src.models.train_fusion --dataset PAD_UFES20 --seed 0
```

---

## Cell 13 — Train: fusion, seed 1

```python
!python -m src.models.train_fusion --dataset PAD_UFES20 --seed 1
```

---

## Cell 14 — Train: fusion, seed 2

```python
!python -m src.models.train_fusion --dataset PAD_UFES20 --seed 2
```
