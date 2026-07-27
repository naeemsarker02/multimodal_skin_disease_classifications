# External ISIC Evaluation Kaggle Notebook — Phase 8 (HAM10000 → ISIC Archive 1/2, zero-shot, Protocol A)

> **WARNING - read before pasting any cell into Kaggle:** for every `%%writefile` cell below, `%%writefile <path>` MUST be the exact first line of the Kaggle cell, with absolutely nothing above it - no blank line, no comment, no stray character. Kaggle (like Jupyter) only treats a line as a cell magic if it is the first line of the cell; anything preceding it turns `%%writefile` into a plain, non-magic line and the file silently never gets written. **After pasting each `%%writefile` cell, re-open it and visually confirm `%%writefile` is line 1 before running it.**

Evaluation-only notebook: no training happens here. HAM10000's already-finalized Stage 1 baseline checkpoints (image + metadata, seeds 0-2) are run zero-shot against ISIC Archive 1 and ISIC Archive 2, per `src/evaluation/evaluate_external_isic.py`'s Protocol A (native, unmodified 7-class argmax; scoring restricted to exact-string-match shared classes). Structured for "Save & Run All (Commit)" so all 9 evaluation runs and their `reports/HAM10000/external_isic/` outputs are preserved as the notebook's Kaggle Output.

Requires **six** Kaggle "Add Data" sources attached before running, all previously verified via `scripts/isic_full_verification_cell.py` and `scripts/isic_mirror_verification_cell.py` (2026-07-27):

- `nodoubttome/skin-cancer9-classesisic` (ISIC Archive 1 raw image mirror - double-nested under "Skin cancer ISIC The International Skin Imaging Collaboration/")
- `andrewmvd/isic-2019` (ISIC Archive 2 raw image mirror - flat)
- `naeemsarkertracer/isic-archive1-processed` (our processed metadata CSVs - flat layout)
- `naeemsarkertracer/isic-archive2-processed` (our processed metadata CSVs - flat layout)
- `naeemsarkertracer/ham10000-processed` (HAM10000's own processed metadata CSVs - needed to fit the metadata preprocessor for the Archive 2 metadata-variant runs; wrapped layout)
- `naeemsarkertracer/ham10000-stage1-checkpoints` (published — https://www.kaggle.com/datasets/naeemsarkertracer/ham10000-stage1-checkpoints), the actual `image_seed{0,1,2}_best.pt` / `metadata_seed{0,1,2}_best.pt` weights being evaluated - flat layout)

Paste each cell below into a separate Kaggle notebook cell, in order.

---

## Cell 1 — Folder verification (all 6 sources, confirmed layouts)

```python
import os, glob

def show(path, label):
    print(f"--- {label}: {path} ---")
    if not os.path.isdir(path):
        print("  !! NOT FOUND")
        return
    for entry in sorted(os.listdir(path)):
        full = os.path.join(path, entry)
        kind = "dir" if os.path.isdir(full) else "file"
        print(f"  [{kind}] {entry}")

show("/kaggle/input/datasets", "datasets root")

# --- 1. ISIC Archive 1 raw mirror (double-nested, confirmed 2026-07-27) --
archive1_root = "/kaggle/input/datasets/nodoubttome/skin-cancer9-classesisic"
show(archive1_root, "ISIC Archive 1 raw (root)")
archive1_nested = os.path.join(archive1_root, "Skin cancer ISIC The International Skin Imaging Collaboration")
show(archive1_nested, "ISIC Archive 1 raw (double-nested candidate)")
assert os.path.isdir(os.path.join(archive1_nested, "Train")) and os.path.isdir(os.path.join(archive1_nested, "Test")), (
    "Expected double-nested Train/Test not found - check archive1_nested path above"
)

# --- 2. ISIC Archive 2 raw mirror (flat, confirmed 2026-07-27) -----------
archive2_root = "/kaggle/input/datasets/andrewmvd/isic-2019"
show(archive2_root, "ISIC Archive 2 raw (root)")
assert os.path.isdir(os.path.join(archive2_root, "ISIC_2019_Training_Input")), (
    "ISIC_2019_Training_Input/ not found under Archive 2 raw root"
)

# --- 3. ISIC Archive 1 processed (flat, confirmed 2026-07-27) -----------
isic1_proc_root = "/kaggle/input/datasets/naeemsarkertracer/isic-archive1-processed"
show(isic1_proc_root, "ISIC Archive 1 processed (root)")
assert os.path.isfile(os.path.join(isic1_proc_root, "metadata_train.csv")), (
    "Expected flat metadata_train.csv not found at ISIC Archive 1 processed root"
)

# --- 4. ISIC Archive 2 processed (flat, confirmed 2026-07-27) -----------
isic2_proc_root = "/kaggle/input/datasets/naeemsarkertracer/isic-archive2-processed"
show(isic2_proc_root, "ISIC Archive 2 processed (root)")
assert os.path.isfile(os.path.join(isic2_proc_root, "metadata_train.csv")), (
    "Expected flat metadata_train.csv not found at ISIC Archive 2 processed root"
)

# --- 5. HAM10000 processed (wrapped, published 2026-07-15) --------------
ham_proc_root = "/kaggle/input/datasets/naeemsarkertracer/ham10000-processed"
show(ham_proc_root, "HAM10000 processed (root)")
ham_proc_wrapped = os.path.join(ham_proc_root, "HAM10000")
show(ham_proc_wrapped, "HAM10000 processed (wrapped candidate)")
assert (
    os.path.isfile(os.path.join(ham_proc_wrapped, "metadata_train.csv"))
    or os.path.isfile(os.path.join(ham_proc_root, "metadata_train.csv"))
), "metadata_train.csv not found at either HAM10000 processed candidate"

# --- 6. HAM10000 Stage 1 checkpoints (flat, confirmed 2026-07-27) -------
ham_ckpt_root = "/kaggle/input/datasets/naeemsarkertracer/ham10000-stage1-checkpoints"
show(ham_ckpt_root, "HAM10000 Stage 1 checkpoints (root)")
expected_ckpts = [f"{b}_seed{s}_best.pt" for b in ("image", "metadata") for s in (0, 1, 2)]
missing_ckpts = [f for f in expected_ckpts if not os.path.isfile(os.path.join(ham_ckpt_root, f))]
assert not missing_ckpts, f"Missing HAM10000 Stage 1 checkpoints at root: {missing_ckpts}"

print("\nOK: all 6 Kaggle datasets found with their confirmed layouts.")
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
    # nodoubttome/skin-cancer9-classesisic - verified raw mirror of
    # ISIC Archive 1 (2,357 images, all 18 per-class Train/Test counts
    # match exactly). Mounted at
    # /kaggle/input/datasets/nodoubttome/skin-cancer9-classesisic/, but
    # double-nested one level deeper under "Skin cancer ISIC The
    # International Skin Imaging Collaboration/" before reaching
    # Train/Test - see KAGGLE_DATASET_SUBPATH below.
    "ISIC_Archive_1": ("nodoubttome", "skin-cancer9-classesisic"),
    # andrewmvd/isic-2019 - verified raw mirror of ISIC Archive 2
    # (25,331 images match exactly; ISIC_2019_Training_Metadata.csv also
    # present, 25,331 rows, columns image/age_approx/anatom_site_general/
    # lesion_id/sex - no attribution column, which is expected since our
    # own processed CSV already carries that from Phase 4's audit).
    # Mounted at /kaggle/input/datasets/andrewmvd/isic-2019/.
    "ISIC_Archive_2": ("andrewmvd", "isic-2019"),
}
KAGGLE_DATASET_SUBPATH = {
    "PAD_UFES20": "",
    "HAM10000": "",
    "ISIC_Archive_1": "Skin cancer ISIC The International Skin Imaging Collaboration",
    "ISIC_Archive_2": "",
}

# Per-dataset top-level rest-folder rename, Kaggle-only: our own local
# data/raw/<Dataset>/ layout is never touched (image_path values in the
# CSVs keep saying "images/..." everywhere) - this only maps that first
# path component to whatever this *specific* Kaggle mirror actually
# named the folder, when the packaging disagrees with our local naming.
# Distinct from KAGGLE_DATASET_SUBPATH (which shifts the whole dataset
# root deeper) and from the imgs_part_N doubling check below (which
# handles a folder being nested inside an identically-named folder, not
# a rename) - this is a third, independent kind of Kaggle packaging
# mismatch.
KAGGLE_REST_FOLDER_RENAME = {
    # andrewmvd/isic-2019 packages images under "ISIC_2019_Training_Input/",
    # not "images/" like our own local data/raw/ISIC_Archive_2/images/ -
    # confirmed via folder-verification cell 2026-07-27 (top-level
    # contents: ISIC_2019_Training_GroundTruth.csv, ISIC_2019_Training_Input,
    # ISIC_2019_Training_Metadata.csv). Filenames themselves are unaffected
    # (ISIC_0000000.jpg-style IDs match on both sides) - only the
    # containing folder name differs.
    "ISIC_Archive_2": {"images": "ISIC_2019_Training_Input"},
}

# Per-dataset filename-suffix fallback, Kaggle-only: some mirrors rename
# individual files when repackaging (e.g. downsampling oversized
# originals) rather than keeping the original filename - a per-file
# naming difference affecting only a subset of files, distinct from
# KAGGLE_REST_FOLDER_RENAME (whole containing folder renamed) and the
# imgs_part_N/renamed-folder doubling check (nesting, not renaming)
# above. Checked via .exists() per-call, same defensive pattern as every
# other fallback here - never assumed to apply to every file.
KAGGLE_FILENAME_FALLBACK_SUFFIX = {
    # andrewmvd/isic-2019 renames ~2,074 of our 25,076 processed IDs
    # (8.3%) with a "_downsampled" suffix before the extension (e.g.
    # ISIC_0016058.jpg only exists as ISIC_0016058_downsampled.jpg) -
    # root cause confirmed via scripts/isic_archive2_id_comparison_cell.py
    # 2026-07-27 (see Project_Tracking.md's ISIC Archive 2 Kaggle mirror
    # quirks entry). Presumably the mirror uploader's own downsizing of
    # oversized originals; undocumented upstream, so treated as this
    # mirror's own idiosyncrasy - never assumed for other datasets.
    "ISIC_Archive_2": "_downsampled",
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

    rest_parts = rest.parts

    # Apply this dataset's rest-folder rename (if any) before any
    # existence checks below, so the doubling check operates on the
    # renamed name too.
    rename_map = KAGGLE_REST_FOLDER_RENAME.get(dataset_dir, {})
    if rest_parts and rest_parts[0] in rename_map:
        rest_parts = (rename_map[rest_parts[0]],) + rest_parts[1:]

    # PAD-UFES-20's Kaggle mirror double-nests imgs_part_N/ folders:
    # verified via direct listing that imgs_part_1, imgs_part_2, and
    # imgs_part_3 are ALL doubled the same way (each contains a
    # subfolder of the identical name). Still checked with .exists()
    # per-call rather than hardcoded, so the fix keeps working even if a
    # future dataset version changes the packaging for only some parts,
    # and so it's a no-op (never matches) for datasets without
    # imgs_part_N/ folders, e.g. HAM10000. Generalized to also cover a
    # freshly-renamed folder (e.g. ISIC_2019_Training_Input/) rather than
    # just the imgs_part_ prefix - nesting-depth was never independently
    # confirmed for that renamed folder (only that it exists at the top
    # level), so this checks rather than assumes a flat layout.
    candidates = []
    if rest_parts:
        top_dir = rest_parts[0]
        renamed_targets = set(rename_map.values())
        if top_dir.startswith("imgs_part_") or top_dir in renamed_targets:
            candidates.append(dataset_root / top_dir / top_dir / Path(*rest_parts[1:]))
    candidates.append(dataset_root / Path(*rest_parts))

    # For each candidate location (doubled first if applicable, then
    # flat), try the plain filename, then this dataset's filename-suffix
    # fallback (if any) at that same location - e.g. ISIC Archive 2's
    # "_downsampled" quirk. Returns the first that actually exists;
    # falls back to the primary (first) candidate, unresolved, if none
    # do, so the caller's eventual FileNotFoundError still names a
    # sensible expected path rather than a silently-wrong guess.
    filename_suffix = KAGGLE_FILENAME_FALLBACK_SUFFIX.get(dataset_dir)
    for candidate in candidates:
        if candidate.exists():
            return candidate
        if filename_suffix:
            suffixed = candidate.with_name(f"{candidate.stem}{filename_suffix}{candidate.suffix}")
            if suffixed.exists():
                return suffixed

    return candidates[0]


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
    # naeemsarkertracer/isic-archive1-processed - uploaded and published
    # 2026-07-27 (https://www.kaggle.com/datasets/naeemsarkertracer/isic-archive1-processed).
    "ISIC_Archive_1": ("naeemsarkertracer", "isic-archive1-processed"),
    # naeemsarkertracer/isic-archive2-processed - uploaded and published
    # 2026-07-27 (https://www.kaggle.com/datasets/naeemsarkertracer/isic-archive2-processed).
    "ISIC_Archive_2": ("naeemsarkertracer", "isic-archive2-processed"),
}
# Whether that Kaggle dataset was zipped from the dataset folder itself
# (True -> mounted root wraps everything in an extra "<Dataset>/"
# subfolder, e.g. PAD-UFES-20's) or from the folder's contents (False ->
# mounted root already contains metadata_train.csv etc. directly). Set
# per-dataset once the HAM10000 processed dataset is actually uploaded.
# Not yet confirmed for the two ISIC archives - _processed_dir()
# auto-detects the actual layout at runtime regardless, so this fallback
# value only matters if that detection can't find either candidate path.
KAGGLE_PROCESSED_WRAPPED = {
    "PAD_UFES20": True,
    "HAM10000": True,
    # Confirmed FLAT via folder-verification cell 2026-07-27 (metadata_train.csv
    # found directly at dataset root, not under an ISIC_Archive_1/ subfolder).
    "ISIC_Archive_1": False,
    # Confirmed FLAT via folder-verification cell 2026-07-27 (metadata_train.csv
    # found directly at dataset root, not under an ISIC_Archive_2/ subfolder).
    "ISIC_Archive_2": False,
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
    # naeemsarkertracer/ham10000-stage1-checkpoints - published 2026-07-27
    # (https://www.kaggle.com/datasets/naeemsarkertracer/ham10000-stage1-checkpoints).
    # Deliberately a separate dataset from ham10000-processed (KAGGLE_PROCESSED_SLUGS
    # above) - that one holds the train/val/test split CSVs, this one holds
    # the Stage 1 image/metadata checkpoints for fusion warm-start.
    "HAM10000": ("naeemsarkertracer", "ham10000-stage1-checkpoints"),
}
KAGGLE_STAGE1_CHECKPOINT_WRAPPED = {
    "PAD_UFES20": True,
    # Confirmed FLAT via folder-verification cell 2026-07-27 (all 6
    # checkpoint .pt files found directly at dataset root, not under a
    # checkpoints/ subfolder; byte sizes match local logs/HAM10000/checkpoints/
    # exactly - image: 16,367,949 bytes each, metadata: 54,523 bytes each).
    "HAM10000": False,
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


# --- ISIC Archive 2 -> HAM10000 anatomical-site mapping (external
# validation, Gap resolution approved 2026-07-25) -----------------------
# Source field is anatom_site_general (Archive 2's finer of its two
# location fields - see docs/Phase8_ISIC_Archive2_Anatomical_Site_Mapping.csv
# for the full per-category review and approval). 4 clean (2 casing-only,
# 2 dermatology-standard synonyms: palms/soles -> acral, posterior torso
# -> back), 2 lossy/coarsened (anterior torso + lateral torso -> HAM10000's
# own generic "trunk" catch-all - same legitimate-coarsening precedent as
# PAD-UFES-20's ARM/FOREARM -> upper extremity), 2 deliberately absent
# (head/neck, oral/genital - no single HAM10000 category covers either
# bundle without guessing which sub-part - falls through to
# "__MISSING__", same treatment as PAD-UFES-20's LIP/NOSE).
ISIC_ARCHIVE2_ANATOMICAL_SITE_CROSS_DATASET_MAP = {
    "lower extremity": "lower extremity",
    "upper extremity": "upper extremity",
    "palms/soles": "acral",
    "posterior torso": "back",
    "anterior torso": "trunk",
    "lateral torso": "trunk",
    # head/neck, oral/genital intentionally absent - fall through to
    # "__MISSING__", same as PAD-UFES-20's LIP/NOSE.
}


def normalize_isic_archive2_anatomical_site_for_ham10000(raw_value) -> str:
    """ISIC Archive 2's anatom_site_general value -> HAM10000-vocabulary
    string, for the HAM10000->ISIC external validation metadata branch
    only. Mirrors normalize_anatomical_site_for_cross_dataset()'s shape
    but uses the Archive-2-specific map above.
    """
    import pandas as pd

    if pd.isna(raw_value):
        return "__MISSING__"
    return ISIC_ARCHIVE2_ANATOMICAL_SITE_CROSS_DATASET_MAP.get(str(raw_value).strip(), "__MISSING__")


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

## Cell 7 — `%%writefile /kaggle/working/src/evaluation/evaluate_external_isic.py`

```python
%%writefile /kaggle/working/src/evaluation/evaluate_external_isic.py
"""Phase 8 - HAM10000 -> ISIC external validation (scope approved
2026-07-25, Project_Tracking.md "Proposed ISIC External Validation
Scope").

Usage:
    python -m src.evaluation.evaluate_external_isic --archive ISIC_Archive_1 --variant image --seed 0
    python -m src.evaluation.evaluate_external_isic --archive ISIC_Archive_2 --variant metadata --seed 0

Zero-shot transfer, same protocol as evaluate_cross_dataset.py's PAD->HAM
experiment (Protocol A): a HAM10000-trained model's native, UNMODIFIED
classifier (full argmax over its own 7-class output, never masked) is run
on the target ISIC archive; scoring is restricted to the classes that
exist in both taxonomies (exact-string match only - AK and the
keratosis-family naming/granularity mismatches are deliberately NOT
merged, per the 2026-07-25 scope decision - conservative, no assumed
clinical equivalence, consistent with Protocol A).

Archive 1 has 0 usable metadata columns (image-only by necessity) - only
the image branch runs there. Archive 2 has usable metadata but its
anatom_site_general vocabulary needs mapping into HAM10000's
anatomical_site vocabulary first - see
docs/Phase8_ISIC_Archive2_Anatomical_Site_Mapping.csv (approved
2026-07-25) and
config.normalize_isic_archive2_anatomical_site_for_ham10000().

Exclusions applied per archive, both required before scoring:
    data/processed/<archive>/external_validation_exclusions.csv
        (2026-07-08 - drops images already seen by the HAM10000-trained
        model, i.e. overlapping with HAM10000 itself)
    data/processed/<archive>/label_conflict_exclusions.csv
        (2026-07-25 - drops the 3 images with disagreeing ground truth
        between ISIC Archive 1 and ISIC Archive 2)

Not gated by test_split_guard.py: this evaluates HAM10000's already-
finalized *training* checkpoints against ISIC's own data - it is not a
second read of HAM10000's own test split, and neither ISIC archive's own
train/val/test split was ever used for ISIC-internal model training in
this project (no ISIC-trained models exist).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from src.models.config import (
    BATCH_SIZE,
    get_dataset,
    normalize_isic_archive2_anatomical_site_for_ham10000,
    resolve_image_path,
)
from src.models.config import _processed_dir as _config_processed_dir
from src.models.dataset import MetadataPreprocessor, build_image_transform
from src.models.image_model import build_efficientnet_b0
from src.models.metadata_model import MetadataMLP

ARCHIVE_SHARED_CLASSES = {
    # Exact-string-match only with HAM10000's taxonomy (Protocol A
    # precedent) - AK and keratosis-family naming/granularity mismatches
    # deliberately deferred, not merged (2026-07-25 scope decision).
    "ISIC_Archive_1": ["Basal Cell Carcinoma", "Dermatofibroma", "Melanoma", "Nevus", "Vascular Lesion"],
    "ISIC_Archive_2": ["Basal Cell Carcinoma", "Dermatofibroma", "Melanoma", "Nevus"],
}

ARCHIVE_HAS_METADATA = {
    "ISIC_Archive_1": False,
    "ISIC_Archive_2": True,
}

ARCHIVE_METADATA_COLUMNS = {
    # archive column name -> HAM10000-schema column name. anatom_site_general
    # additionally passes through the ISIC Archive 2 -> HAM10000 mapping
    # (see normalize_isic_archive2_anatomical_site_for_ham10000).
    "ISIC_Archive_2": {"age_approx": "age", "sex": "sex", "anatom_site_general": "anatomical_site"},
}


def _archive_processed_dir(archive: str) -> Path:
    """Delegates to config.py's Kaggle-aware resolver (KAGGLE_PROCESSED_SLUGS
    -> flat-vs-wrapped auto-detection) instead of hardcoding a local-only
    "data/processed/<archive>" path - this script must also run inside the
    Kaggle notebook, where that local path does not exist.
    """
    return _config_processed_dir(archive)


def load_archive_metadata(archive: str) -> pd.DataFrame:
    processed_dir = _archive_processed_dir(archive)
    dfs = [pd.read_csv(processed_dir / f"metadata_{split}.csv") for split in ("train", "val", "test")]
    return pd.concat(dfs, ignore_index=True)


def apply_exclusions(df: pd.DataFrame, archive: str) -> pd.DataFrame:
    processed_dir = _archive_processed_dir(archive)
    ham_overlap = set(pd.read_csv(processed_dir / "external_validation_exclusions.csv")["image_id"])
    label_conflicts = set(pd.read_csv(processed_dir / "label_conflict_exclusions.csv")["image_id"])
    excluded = ham_overlap | label_conflicts
    return df[~df["image_id"].isin(excluded)].reset_index(drop=True)


class ExternalIsicEvalDataset(Dataset):
    """ISIC archive rows, filtered to that archive's shared-class list and
    with both exclusion lists applied. Labels are encoded in HAM10000's
    label_to_idx space (the model's own output space) - the shared class
    name strings are identical across HAM10000 and both ISIC archives'
    disease_label vocabularies (verified during dataset preparation, same
    harmonized taxonomy used project-wide).
    """

    def __init__(self, archive: str, ham_label_to_idx: dict, need_metadata: bool,
                 eval_preprocessor: MetadataPreprocessor = None):
        df = load_archive_metadata(archive)
        df = apply_exclusions(df, archive)
        shared_classes = ARCHIVE_SHARED_CLASSES[archive]
        self.df = df[df["disease_label"].isin(shared_classes)].reset_index(drop=True)
        self.label_to_idx = ham_label_to_idx
        self.transform = build_image_transform(train=False)
        self.need_metadata = need_metadata
        self.preprocessor = eval_preprocessor
        self.archive = archive
        if need_metadata and eval_preprocessor is None:
            raise ValueError("eval_preprocessor required when need_metadata=True")

    def __len__(self) -> int:
        return len(self.df)

    def _adapted_row(self, row: pd.Series) -> pd.Series:
        """Archive-native column names/vocab -> HAM10000-schema row, for
        MetadataPreprocessor.transform_row(). Archive 1 never reaches here
        (no metadata branch); only Archive 2 is handled.
        """
        col_map = ARCHIVE_METADATA_COLUMNS[self.archive]
        adapted = {}
        for archive_col, ham_col in col_map.items():
            value = row[archive_col]
            if ham_col == "anatomical_site":
                value = normalize_isic_archive2_anatomical_site_for_ham10000(value)
            adapted[ham_col] = value
        return pd.Series(adapted)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = resolve_image_path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = self.label_to_idx[row["disease_label"]]
        if self.need_metadata:
            metadata = self.preprocessor.transform_row(self._adapted_row(row))
            return image, metadata, label
        return image, label


def build_ham_eval_preprocessor(ham_ds_config) -> MetadataPreprocessor:
    """Fits on HAM10000's own train split, in HAM10000's own schema
    (age/sex/anatomical_site) - no column_transforms needed here, since
    the ISIC Archive 2 -> HAM10000 normalization is applied to Archive
    2's raw values in ExternalIsicEvalDataset._adapted_row() before this
    preprocessor ever sees them, not via a transform hook on the fit
    side (unlike the PAD->HAM10000 experiment, HAM10000 and Archive 2
    don't share raw column names, so there's no single df both could be
    fit/transformed from).
    """
    train_df = pd.read_csv(ham_ds_config.train_csv)
    return MetadataPreprocessor(ham_ds_config).fit(train_df)


def load_model_for_variant(variant: str, seed: int, ham_ds_config, num_classes: int,
                            metadata_input_dim: int, device) -> torch.nn.Module:
    # stage1_checkpoints_dir (KAGGLE_STAGE1_CHECKPOINT_SLUGS-resolved), NOT
    # checkpoints_dir (OUTPUT_ROOT write-target for newly-trained output) -
    # the two only happened to coincide locally because OUTPUT_ROOT ==
    # PROJECT_ROOT off-Kaggle; on Kaggle checkpoints_dir is an empty
    # /kaggle/working path and this eval script never writes there.
    ckpt_path = ham_ds_config.stage1_checkpoints_dir / f"{variant}_seed{seed}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if variant == "image":
        model = build_efficientnet_b0(num_classes=num_classes)
    elif variant == "metadata":
        model = MetadataMLP(input_dim=metadata_input_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate(archive: str, variant: str, seed: int) -> dict:
    if variant == "metadata" and not ARCHIVE_HAS_METADATA[archive]:
        raise ValueError(f"{archive} has no usable metadata columns - only --variant image is valid here.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ham_ds_config = get_dataset("HAM10000")

    need_metadata = variant == "metadata"
    eval_preprocessor = None
    metadata_input_dim = None
    if need_metadata:
        eval_preprocessor = build_ham_eval_preprocessor(ham_ds_config)
        metadata_input_dim = eval_preprocessor.output_dim

    dataset = ExternalIsicEvalDataset(archive, ham_ds_config.label_to_idx, need_metadata, eval_preprocessor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model_for_variant(variant, seed, ham_ds_config, ham_ds_config.num_classes, metadata_input_dim, device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if need_metadata:
                _, metadata, labels = batch
                outputs = model(metadata.to(device))
            else:
                images, labels = batch
                outputs = model(images.to(device))
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    class_names = ham_ds_config.class_names
    shared_classes = ARCHIVE_SHARED_CLASSES[archive]
    shared_indices = [ham_ds_config.label_to_idx[c] for c in shared_classes]

    macro_f1 = f1_score(all_labels, all_preds, labels=shared_indices, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, labels=shared_indices, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    spillover_count = sum(1 for p in all_preds if p not in shared_indices)

    result = {
        "experiment": "phase8_ham_to_isic_external_validation",
        "protocol": "A - full native argmax, restricted to exact-string-match shared classes for scoring",
        "archive": archive,
        "variant": variant,
        "seed": seed,
        "n_eval_rows": len(dataset),
        "shared_classes": shared_classes,
        "macro_f1_shared_classes": macro_f1,
        "per_class_f1_shared_classes": dict(zip(shared_classes, per_class_f1.tolist())),
        "spillover_count": spillover_count,
        "spillover_rate": spillover_count / len(all_labels) if all_labels else 0.0,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_class_order": class_names,
    }

    predictions_df = dataset.df[["image_id", "image_path"]].copy()
    predictions_df["true_label_idx"] = all_labels
    predictions_df["true_label_name"] = [class_names[i] for i in all_labels]
    predictions_df["pred_label_idx"] = all_preds
    predictions_df["pred_label_name"] = [class_names[i] for i in all_preds]

    return result, predictions_df


def main():
    parser = argparse.ArgumentParser(description="Phase 8: HAM10000 -> ISIC external validation")
    parser.add_argument("--archive", choices=["ISIC_Archive_1", "ISIC_Archive_2"], required=True)
    parser.add_argument("--variant", choices=["image", "metadata"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    result, predictions_df = evaluate(args.archive, args.variant, args.seed)

    reports_dir = Path("reports") / "HAM10000" / "external_isic"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"eval_{args.archive}_{args.variant}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    predictions_path = reports_dir / f"predictions_{args.archive}_{args.variant}_seed{args.seed}.csv"
    predictions_df.to_csv(predictions_path, index=False)

    print(f"n_eval_rows: {result['n_eval_rows']}")
    print(f"macro-F1 (shared classes): {result['macro_f1_shared_classes']:.4f}")
    print("per-class F1 (shared classes):")
    for name, score in result["per_class_f1_shared_classes"].items():
        print(f"  {name}: {score:.4f}")
    print(f"spillover rate (predicted a non-shared HAM10000 class): {result['spillover_rate']:.4f}")
    print(f"written -> {out_path}")
    print(f"written -> {predictions_path}")


if __name__ == "__main__":
    main()
```

---

## Cell 8 — Sanity check (config load + both archives' image paths + HAM10000 Stage 1 checkpoints + a real ExternalIsicEvalDataset row)

```python
import sys
for mod in list(sys.modules):
    if mod.startswith("src."):
        del sys.modules[mod]

import pandas as pd
from src.models.config import get_dataset, resolve_image_path
from src.evaluation.evaluate_external_isic import (
    ARCHIVE_SHARED_CLASSES,
    ExternalIsicEvalDataset,
    build_ham_eval_preprocessor,
    load_archive_metadata,
    apply_exclusions,
)

ham_ds_config = get_dataset("HAM10000")
print("HAM10000 num_classes:", ham_ds_config.num_classes)
print("HAM10000 train_csv:", ham_ds_config.train_csv, "exists:", ham_ds_config.train_csv.exists())
assert ham_ds_config.train_csv.exists(), "HAM10000 metadata_train.csv not found - check processed dataset nesting"

print("\nHAM10000 stage1_checkpoints_dir:", ham_ds_config.stage1_checkpoints_dir)
for branch in ("image", "metadata"):
    for seed in (0, 1, 2):
        p = ham_ds_config.stage1_checkpoints_dir / f"{branch}_seed{seed}_best.pt"
        assert p.exists(), f"Missing HAM10000 Stage 1 checkpoint: {p}"
print("all 6 HAM10000 Stage 1 checkpoints resolved OK")

# --- both ISIC archives' processed metadata + image path resolution -----
for archive in ("ISIC_Archive_1", "ISIC_Archive_2"):
    df = load_archive_metadata(archive)
    df = apply_exclusions(df, archive)
    print(f"\n{archive}: {len(df)} rows after exclusions")
    sample_path = df.iloc[0]["image_path"]
    resolved = resolve_image_path(sample_path)
    print(f"  sample image_path: {sample_path}")
    print(f"  resolved:           {resolved}")
    print(f"  exists:             {resolved.exists()}")
    assert resolved.exists(), f"Resolved image path does not exist: {resolved}"

    import random
    random.seed(0)
    sample_rows = df.sample(n=min(20, len(df)), random_state=0)
    missing = [row["image_path"] for _, row in sample_rows.iterrows() if not resolve_image_path(row["image_path"]).exists()]
    print(f"  checked {len(sample_rows)} random rows, missing: {len(missing)}")
    assert not missing, f"Some resolved image paths do not exist for {archive}: {missing[:5]}"

# --- one real ExternalIsicEvalDataset row per shared-class configuration -
eval_preprocessor = build_ham_eval_preprocessor(ham_ds_config)
print(f"\nHAM10000 metadata preprocessor output_dim: {eval_preprocessor.output_dim}")

ds_a1_image = ExternalIsicEvalDataset("ISIC_Archive_1", ham_ds_config.label_to_idx, need_metadata=False)
print(f"ISIC_Archive_1 image-eval dataset: {len(ds_a1_image)} rows, shared classes {ARCHIVE_SHARED_CLASSES['ISIC_Archive_1']}")
image, label = ds_a1_image[0]
assert image.shape == (3, 224, 224)

ds_a2_metadata = ExternalIsicEvalDataset("ISIC_Archive_2", ham_ds_config.label_to_idx, need_metadata=True, eval_preprocessor=eval_preprocessor)
print(f"ISIC_Archive_2 metadata-eval dataset: {len(ds_a2_metadata)} rows, shared classes {ARCHIVE_SHARED_CLASSES['ISIC_Archive_2']}")
image, metadata, label = ds_a2_metadata[0]
assert metadata.shape == (eval_preprocessor.output_dim,)

print("\nSANITY CHECK PASSED")
```

---

## Cell 9 — Full model/GPU/dependency check (one real forward pass per branch type, both HAM10000-checkpoint-loaded models)

```python
import torch, sys
print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

from src.models.config import get_dataset
from src.evaluation.evaluate_external_isic import (
    ExternalIsicEvalDataset,
    build_ham_eval_preprocessor,
    load_model_for_variant,
)

ham_ds_config = get_dataset("HAM10000")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image branch, seed 0, against ISIC Archive 1
image_model = load_model_for_variant("image", 0, ham_ds_config, ham_ds_config.num_classes, None, device)
ds_a1_image = ExternalIsicEvalDataset("ISIC_Archive_1", ham_ds_config.label_to_idx, need_metadata=False)
image, label = ds_a1_image[0]
with torch.no_grad():
    out = image_model(image.unsqueeze(0).to(device))
assert out.shape == (1, ham_ds_config.num_classes)
print("image-branch (seed 0) forward pass on ISIC Archive 1 OK, output shape:", out.shape)

# metadata branch, seed 0, against ISIC Archive 2
eval_preprocessor = build_ham_eval_preprocessor(ham_ds_config)
metadata_model = load_model_for_variant("metadata", 0, ham_ds_config, ham_ds_config.num_classes, eval_preprocessor.output_dim, device)
metadata_model.eval()  # single-sample batch below - BatchNorm1d needs eval mode on batch size 1
ds_a2_metadata = ExternalIsicEvalDataset("ISIC_Archive_2", ham_ds_config.label_to_idx, need_metadata=True, eval_preprocessor=eval_preprocessor)
image, metadata, label = ds_a2_metadata[0]
with torch.no_grad():
    out = metadata_model(metadata.unsqueeze(0).to(device))
assert out.shape == (1, ham_ds_config.num_classes)
print("metadata-branch (seed 0) forward pass on ISIC Archive 2 OK, output shape:", out.shape)

print("\nALL CHECKS PASSED - ready to evaluate")
```

---

## Cell 10 — Evaluate: ISIC_Archive_1, image branch, seed 0

```python
!python -m src.evaluation.evaluate_external_isic --archive ISIC_Archive_1 --variant image --seed 0
```

---

## Cell 11 — Evaluate: ISIC_Archive_1, image branch, seed 1

```python
!python -m src.evaluation.evaluate_external_isic --archive ISIC_Archive_1 --variant image --seed 1
```

---

## Cell 12 — Evaluate: ISIC_Archive_1, image branch, seed 2

```python
!python -m src.evaluation.evaluate_external_isic --archive ISIC_Archive_1 --variant image --seed 2
```

---

## Cell 13 — Evaluate: ISIC_Archive_2, image branch, seed 0

```python
!python -m src.evaluation.evaluate_external_isic --archive ISIC_Archive_2 --variant image --seed 0
```

---

## Cell 14 — Evaluate: ISIC_Archive_2, image branch, seed 1

```python
!python -m src.evaluation.evaluate_external_isic --archive ISIC_Archive_2 --variant image --seed 1
```

---

## Cell 15 — Evaluate: ISIC_Archive_2, image branch, seed 2

```python
!python -m src.evaluation.evaluate_external_isic --archive ISIC_Archive_2 --variant image --seed 2
```

---

## Cell 16 — Evaluate: ISIC_Archive_2, metadata branch, seed 0

```python
!python -m src.evaluation.evaluate_external_isic --archive ISIC_Archive_2 --variant metadata --seed 0
```

---

## Cell 17 — Evaluate: ISIC_Archive_2, metadata branch, seed 1

```python
!python -m src.evaluation.evaluate_external_isic --archive ISIC_Archive_2 --variant metadata --seed 1
```

---

## Cell 18 — Evaluate: ISIC_Archive_2, metadata branch, seed 2

```python
!python -m src.evaluation.evaluate_external_isic --archive ISIC_Archive_2 --variant metadata --seed 2
```

