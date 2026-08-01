# PAD-UFES-20-Expanded Kaggle Notebook — Phase 8B Step 3 (5-Backbone Comparison, Image Branch Only)

> **WARNING — read before pasting any cell into Kaggle:** for every `%%writefile` cell below, `%%writefile <path>` MUST be the exact first line of the Kaggle cell, with absolutely nothing above it — no blank line, no comment, no stray character. **After pasting each `%%writefile` cell, re-open it and visually confirm `%%writefile` is line 1 before running it.**

Structured for "Save & Run All (Commit)", same process as every prior notebook in this project. Requires **four** Kaggle "Add Data" sources attached before running:

1. `mahdavi1202/skin-cancer` (raw PAD-UFES-20 image mirror) — **already used by every prior notebook, should already exist.**
2. `naeemsarkertracer/derm12345-skin-lesion-images` (raw DERM12345) — uploaded and published 2026-07-29.
3. `naeemsarkertracer/med-node-skin-lesion-images` (raw MED-NODE) — uploaded and published 2026-07-29.
4. `naeemsarkertracer/pad-ufes20-expanded-processed` (processed PAD_UFES20_Expanded) — uploaded and published 2026-07-29.

This notebook's `%%writefile` cells and folder-verification cell (Cell 1) embed the real `(owner, slug)` values from `src/models/config.py`'s `KAGGLE_DATASET_SLUGS`/`KAGGLE_PROCESSED_SLUGS` as of generation time — re-run `scripts/generate_backbone_comparison_kaggle_notebook.py` any time those change to keep this notebook in sync.

15 runs total: 5 backbones (EfficientNet-B0, MobileNetV3-Large, DenseNet121, ResNet50, ConvNeXt-Tiny) × 3 seeds, `--dataset PAD_UFES20_Expanded --branch image` (the `image_branch_only=True` guard added 2026-07-29 makes any other `--branch` value fail loudly for this dataset — see `Project_Tracking.md`, "Pre-Step-3 Wiring Verification").

Paste each cell below into a separate Kaggle notebook cell, in order.

---

## Cell 1 — Folder verification (raw PAD-UFES-20 mirror + DERM12345 + MED-NODE + processed PAD_UFES20_Expanded)

```python
import os

def show(path, label):
    print(f"--- {label}: {path} ---")
    if not os.path.isdir(path):
        print("  !! NOT FOUND")
        return
    for entry in sorted(os.listdir(path))[:20]:
        full = os.path.join(path, entry)
        kind = "dir" if os.path.isdir(full) else "file"
        print(f"  [{kind}] {entry}")

raw_root = "/kaggle/input/datasets/mahdavi1202/skin-cancer"
show("/kaggle/input/datasets", "datasets root")
show(raw_root, "raw PAD-UFES-20 mirror")

derm_root = "/kaggle/input/datasets/naeemsarkertracer/derm12345-skin-lesion-images"
mednode_root = "/kaggle/input/datasets/naeemsarkertracer/med-node-skin-lesion-images"
processed_root = "/kaggle/input/datasets/naeemsarkertracer/pad-ufes20-expanded-processed"
show(derm_root, "DERM12345 raw")
show(mednode_root, "MED-NODE raw")
show(processed_root, "PAD_UFES20_Expanded processed")

assert os.path.isdir(raw_root), "raw PAD-UFES-20 mirror not found — check Add Data"
assert os.path.isdir(derm_root), "DERM12345 raw dataset not found — check Add Data"
assert os.path.isdir(mednode_root), "MED-NODE raw dataset not found — check Add Data"
assert os.path.isdir(processed_root), "PAD_UFES20_Expanded processed dataset not found — check Add Data"
print("\nOK — all 4 Add Data sources found")

# Sample-file resolution check (added 2026-07-30 after a MED-NODE
# FileNotFoundError mid-training — resolve_image_path()'s candidate
# order tried here directly, against one real CSV row per new
# dataset, so any Kaggle-mount nesting mismatch is caught here
# instead of after an epoch has already started).
def check_candidates(root, dataset_dir, sample_rel, strip_prefix=None):
    print(f"--- resolving sample for {dataset_dir}: {sample_rel} ---")
    top, _, rest = sample_rel.partition("/")
    candidates = [
        os.path.join(root, top, top, rest),
        os.path.join(root, dataset_dir, sample_rel),
        os.path.join(root, sample_rel),
    ]
    # Mirrors resolve_image_path()'s KAGGLE_REST_STRIP_PREFIX case: a
    # leading segment our image_path values expect (e.g.
    # "complete_mednode_dataset") that this specific mount omits
    # entirely - opposite of the doubled/wrapped candidates above.
    if strip_prefix and top == strip_prefix and rest:
        s_top, _, s_rest = rest.partition("/")
        candidates.append(os.path.join(root, s_top, s_top, s_rest))
        candidates.append(os.path.join(root, dataset_dir, rest))
        candidates.append(os.path.join(root, rest))
    found_any = False
    for c in candidates:
        exists = os.path.exists(c)
        found_any = found_any or exists
        print(f"  [{'FOUND' if exists else 'missing'}] {c}")
    if not found_any:
        print("  !! none of the expected candidate paths exist — "
              "resolve_image_path() will fail for this dataset")
    return found_any

mednode_ok = check_candidates(mednode_root, "MED-NODE", "complete_mednode_dataset/melanoma/136733.jpg", strip_prefix="complete_mednode_dataset")
derm_ok = check_candidates(derm_root, "DERM12345", "images/alm/DERM_767235.jpg")
assert mednode_ok, "No candidate path resolved a MED-NODE sample file — check Add Data / mount layout"
assert derm_ok, "No candidate path resolved a DERM12345 sample file — check Add Data / mount layout"
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
    # Phase 8C dataset expansion (2026-07-29) - unlike the above 4, no
    # existing public Kaggle mirror was verified for these, since they
    # were pulled directly from Harvard Dataverse / cs.rug.nl (see
    # Project_Tracking.md, "Step 2 - PAD-UFES-20-Expanded Built"). Must
    # be uploaded as your OWN private Kaggle dataset (zip
    # data/raw/DERM12345/ as-is) before any PAD_UFES20_Expanded image
    # branch training run on Kaggle - REPLACE_WITH_* is a deliberate
    # placeholder (resolve_image_path() raises a clear RuntimeError
    # rather than silently misresolving if this isn't set).
    # Uploaded and published 2026-07-29
    # (https://www.kaggle.com/datasets/naeemsarkertracer/derm12345-skin-lesion-images).
    "DERM12345": ("naeemsarkertracer", "derm12345-skin-lesion-images"),
    # Same as above - upload data/raw/MED-NODE/ (only the melanoma/
    # subset is used, but the naevus/ folder can be included harmlessly
    # since it's simply never referenced by any image_path). Uploaded and
    # published 2026-07-29
    # (https://www.kaggle.com/datasets/naeemsarkertracer/med-node-skin-lesion-images).
    "MED-NODE": ("naeemsarkertracer", "med-node-skin-lesion-images"),
}
KAGGLE_DATASET_SUBPATH = {
    "PAD_UFES20": "",
    "HAM10000": "",
    "ISIC_Archive_1": "Skin cancer ISIC The International Skin Imaging Collaboration",
    "ISIC_Archive_2": "",
    "DERM12345": "",
    "MED-NODE": "",
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

# Per-dataset leading-segment strip, Kaggle-only: the opposite quirk class
# from KAGGLE_REST_FOLDER_RENAME/the doubling check above (which both add
# an extra folder level) - here a folder level our own image_path values
# expect is simply ABSENT on this specific mount, because the uploader
# zipped from inside that folder rather than including it. Distinct from
# KAGGLE_DATASET_SUBPATH (which shifts the whole dataset root, applied
# before rest_parts is even computed) - this strips a segment out of
# rest_parts itself, dataset by dataset.
KAGGLE_REST_STRIP_PREFIX = {
    # naeemsarkertracer/med-node-skin-lesion-images - confirmed via Cell 1's
    # raw folder-listing diagnostic 2026-07-30: melanoma/ and naevus/ sit
    # directly at the mount root, with no complete_mednode_dataset/ folder
    # at all. Our own image_path values (and local data/raw/MED-NODE/
    # layout) still say ".../complete_mednode_dataset/melanoma/..." - only
    # this specific Kaggle mount is missing the segment, so it's stripped
    # here rather than in the CSV or locally.
    "MED-NODE": "complete_mednode_dataset",
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

    flat_candidate = dataset_root / Path(*rest_parts)

    # PAD-UFES-20's Kaggle mirror double-nests imgs_part_N/ folders:
    # verified via direct listing that imgs_part_1, imgs_part_2, and
    # imgs_part_3 are ALL doubled the same way (each contains a
    # subfolder of the identical name). Checked with .exists() per-call
    # rather than hardcoded, so the fix keeps working even if a future
    # dataset version changes the packaging for only some parts, and so
    # it's a no-op (never matches) for datasets without this quirk, e.g.
    # HAM10000. Generalized 2026-07-30 to try for ANY top_dir (not just
    # imgs_part_*/renamed_targets) after hitting a FileNotFoundError for
    # MED-NODE's complete_mednode_dataset/ folder that this narrower
    # condition didn't catch - safe to widen because each candidate is an
    # exact .exists() check, so there's no risk of a false match for
    # datasets that don't actually have this quirk.
    candidates = []
    if rest_parts:
        top_dir = rest_parts[0]
        candidates.append(dataset_root / top_dir / top_dir / Path(*rest_parts[1:]))

    # Whole-dataset wrapping: the mount wraps everything in an extra
    # "<dataset_dir>/" folder - same "wrapped vs flat" ambiguity
    # _processed_dir()/KAGGLE_PROCESSED_WRAPPED already handle for the
    # processed-metadata datasets (depends on whether the uploader zipped
    # the folder itself or just its contents), not previously handled
    # here for raw-image datasets. Added 2026-07-30 after a
    # FileNotFoundError for a MED-NODE row whose expected flat path
    # (dataset_root/complete_mednode_dataset/melanoma/...) matches our
    # own local data/raw/MED-NODE/ layout exactly (verified - not a CSV
    # bug), meaning the Kaggle-side mount must be nested differently.
    # Checked via .exists() like every other candidate here - never
    # assumed for MED-NODE or any other dataset.
    candidates.append(dataset_root / dataset_dir / Path(*rest_parts))

    candidates.append(flat_candidate)

    # Missing-wrapping-folder quirk (KAGGLE_REST_STRIP_PREFIX): rest_parts
    # still starts with a segment (e.g. "complete_mednode_dataset") that
    # isn't present on this particular Kaggle mount at all - the opposite
    # problem from the doubled/wrapped candidates above, which all ADD a
    # folder level. Appended in addition to, not instead of, every
    # candidate above, since this only fires when the configured prefix
    # actually matches, and every candidate here is still checked via
    # .exists() before being trusted - never assumed.
    strip_prefix = KAGGLE_REST_STRIP_PREFIX.get(dataset_dir)
    if strip_prefix and rest_parts and rest_parts[0] == strip_prefix:
        stripped_parts = rest_parts[1:]
        if stripped_parts:
            stripped_top = stripped_parts[0]
            candidates.append(
                dataset_root / stripped_top / stripped_top / Path(*stripped_parts[1:])
            )
            candidates.append(dataset_root / dataset_dir / Path(*stripped_parts))
            candidates.append(dataset_root / Path(*stripped_parts))

    # For each candidate location (doubled, dataset-wrapped, flat, then any
    # prefix-stripped variants), try the plain filename, then this
    # dataset's filename-suffix fallback (if any) at that same location -
    # e.g. ISIC Archive 2's
    # "_downsampled" quirk. Returns the first that actually exists; falls
    # back to the flat (un-doubled, un-wrapped) candidate, unresolved, if
    # none do, so the caller's eventual FileNotFoundError still names the
    # most legible expected path rather than a guessed nested one.
    filename_suffix = KAGGLE_FILENAME_FALLBACK_SUFFIX.get(dataset_dir)
    for candidate in candidates:
        if candidate.exists():
            return candidate
        if filename_suffix:
            suffixed = candidate.with_name(f"{candidate.stem}{filename_suffix}{candidate.suffix}")
            if suffixed.exists():
                return suffixed

    return flat_candidate


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
    # Phase 8C (2026-07-29) - data/processed/PAD_UFES20_Expanded/ uploaded
    # and published 2026-07-29 as its own private Kaggle dataset, same
    # pattern as the 4 above
    # (https://www.kaggle.com/datasets/naeemsarkertracer/pad-ufes20-expanded-processed).
    "PAD_UFES20_Expanded": ("naeemsarkertracer", "pad-ufes20-expanded-processed"),
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
    # Auto-detection handles this regardless (see _processed_dir below);
    # this fallback value only matters before the dataset is uploaded/mounted.
    "PAD_UFES20_Expanded": True,
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

# --- Step 3a (Phase 8B/8C plan): class-imbalance ablation (a)/(b) -------
# Classes targeted by the WeightedRandomSampler and class-targeted
# augmentation ablations - PAD-UFES-20's two worst-supported classes
# (Melanoma: 38 train images; Squamous Cell Carcinoma: 135). Both ablations
# are tested in isolation on EfficientNet-B0 first (train.py --sampler /
# --strong-augment flags) before being adopted or rejected for the
# 5-backbone comparison and fusion runs - see docs/Project_Tracking.md,
# "Phase 8B+8C — Master Plan Adopted" (2026-07-29), "isolate before
# stacking" (train_cross_attention_improved.py's earlier
# confounded-experiment lesson).
STRONG_AUGMENT_TARGET_CLASSES = ["Melanoma", "Squamous Cell Carcinoma"]


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
                 categorical_features: list, train_csv_name: str = "metadata_train.csv",
                 image_branch_only: bool = False):
        self.name = name
        self.class_names = class_names
        self.label_to_idx = {n: i for i, n in enumerate(class_names)}
        self.num_classes = len(class_names)
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        # Phase 8C (2026-07-29): True only for PAD_UFES20_Expanded. When
        # True, train.py/evaluate.py must refuse any branch other than
        # "image" - this dataset's train_csv (metadata_train_image_only.csv)
        # includes DERM12345/MED-NODE rows with no compatible clinical
        # metadata (all whitelist columns NaN), so metadata/fusion training
        # would silently train on garbage rather than erroring, which is
        # exactly the failure mode this flag exists to prevent. See
        # Project_Tracking.md, "Step 2 Integration Plan" (2026-07-29).
        self.image_branch_only = image_branch_only

        processed_dir = _processed_dir(name)
        self.train_csv = processed_dir / train_csv_name
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

# Phase 8C (2026-07-29) - PAD-UFES-20 + DERM12345 (Melanoma/SCC) +
# MED-NODE (Melanoma) for the Phase 8B 5-backbone comparison's image
# branch only. Same 6-class taxonomy and metadata feature lists as
# PAD_UFES20 (unused by image-only training, kept identical only for
# documentation consistency), same processed-dir resolution
# (_processed_dir("PAD_UFES20_Expanded")), but train_csv_name points at
# metadata_train_image_only.csv - not metadata_train.csv, which also
# exists in this dataset's processed dir (byte-identical to PAD_UFES20's)
# but is deliberately not what this DatasetConfig reads.
# image_branch_only=True hard-blocks metadata/fusion training in
# train.py/evaluate.py. val/test are untouched byte-for-byte copies of
# PAD_UFES20's - see dataset_description.md.
PAD_UFES20_EXPANDED = DatasetConfig(
    name="PAD_UFES20_Expanded",
    class_names=PAD_UFES20.class_names,
    numeric_features=PAD_UFES20.numeric_features,
    categorical_features=PAD_UFES20.categorical_features,
    train_csv_name="metadata_train_image_only.csv",
    image_branch_only=True,
)

DATASETS = {
    "PAD_UFES20": PAD_UFES20,
    "HAM10000": HAM10000,
    "PAD_UFES20_Expanded": PAD_UFES20_EXPANDED,
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

# Verified 2026-07-29 (not assumed): for all 5 Phase 8B backbones
# (EfficientNet_B0_Weights.IMAGENET1K_V1, MobileNet_V3_Large_Weights.IMAGENET1K_V2,
# DenseNet121_Weights.IMAGENET1K_V1, ResNet50_Weights.IMAGENET1K_V2,
# ConvNeXt_Tiny_Weights.IMAGENET1K_V1), weights.transforms().mean/std were
# queried directly and all resolve to this same mean/std. See
# docs/Project_Tracking.md, "Phase 8B backbone normalization verified".
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


def build_image_transform(train: bool, strong: bool = False) -> transforms.Compose:
    """strong=True (Step 3a ablation (b)): more aggressive rotation/crop/
    color-jitter, applied only to per-sample classes an ImageDataset caller
    opts into via strong_augment_classes - never the default path, so
    every existing call site (train=True, strong left at its default) is
    unaffected.
    """
    ops = [ResizePad(IMAGE_INPUT_SIZE)]
    if train:
        if strong:
            ops += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(35),
                transforms.RandomResizedCrop(IMAGE_INPUT_SIZE, scale=(0.75, 1.0)),
                transforms.ColorJitter(brightness=0.25, contrast=0.25),
            ]
        else:
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
    """Image-only branch: returns (image_tensor, label_idx).

    strong_augment_classes (optional, Step 3a ablation (b)): disease_label
    values that get build_image_transform(train, strong=True) instead of
    the normal train transform. Only consulted when train=True. Default
    None reproduces the exact pre-Step-3a behavior (single transform for
    every sample).
    """

    def __init__(self, csv_path: Path, dataset_config: DatasetConfig, train: bool,
                 strong_augment_classes: set = None):
        self.df = pd.read_csv(csv_path)
        self.label_to_idx = dataset_config.label_to_idx
        self.train = train
        self.transform = build_image_transform(train)
        self.strong_augment_classes = strong_augment_classes or set()
        self.strong_transform = (
            build_image_transform(train, strong=True) if self.strong_augment_classes else None
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = resolve_image_path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        label_name = row["disease_label"]
        if self.train and label_name in self.strong_augment_classes:
            image = self.strong_transform(image)
        else:
            image = self.transform(image)
        label = self.label_to_idx[label_name]
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

## Cell 5 — `%%writefile /kaggle/working/src/models/backbones.py`

```python
%%writefile /kaggle/working/src/models/backbones.py
"""Phase 8B backbone registry - 5 pretrained image backbones for the
image-only branch, plus a generic penultimate-embedding wrapper used by
backbone_fusion_model.py.

Same one-line-swap pattern as image_model.py's build_efficientnet_b0: load
ImageNet1K pretrained weights, replace only the final classification layer
with nn.Linear(in_features, num_classes). See docs/Project_Tracking.md,
"Phase 8B+8C — Master Plan Adopted" (2026-07-29) for why these 5 and not
others - spans ~5M-29M params so the comparison tests architecture capacity,
not just architecture-family trivia.
"""

import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    convnext_tiny,
    densenet121,
    efficientnet_b0,
    mobilenet_v3_large,
    resnet50,
)


def _build_efficientnet_b0(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _build_mobilenet_v3_large(num_classes: int) -> nn.Module:
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _build_densenet121(num_classes: int) -> nn.Module:
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def _build_resnet50(num_classes: int) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _build_convnext_tiny(num_classes: int) -> nn.Module:
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


# name -> (builder, final-linear locator). The locator finds the same
# nn.Linear the builder just replaced, given an already-built model -
# needed by BackboneEmbedder (backbone_fusion_model.py) to read
# embed_dim and hook the penultimate features, without duplicating each
# architecture's classifier layout in a second place.
BACKBONE_REGISTRY = {
    "efficientnet_b0": (_build_efficientnet_b0, lambda m: m.classifier[-1]),
    "mobilenet_v3_large": (_build_mobilenet_v3_large, lambda m: m.classifier[-1]),
    "densenet121": (_build_densenet121, lambda m: m.classifier),
    "resnet50": (_build_resnet50, lambda m: m.fc),
    "convnext_tiny": (_build_convnext_tiny, lambda m: m.classifier[-1]),
}

BACKBONE_NAMES = list(BACKBONE_REGISTRY)


def build_backbone(name: str, num_classes: int) -> nn.Module:
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone {name!r}; choices: {BACKBONE_NAMES}")
    builder, _ = BACKBONE_REGISTRY[name]
    return builder(num_classes)


def final_linear(name: str, model: nn.Module) -> nn.Linear:
    """Locates the same nn.Linear build_backbone(name, ...) just installed
    as the final classification layer, for an already-built model of
    architecture `name`."""
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone {name!r}; choices: {BACKBONE_NAMES}")
    _, locate = BACKBONE_REGISTRY[name]
    return locate(model)
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

## Cell 7 — `%%writefile /kaggle/working/src/models/train.py`

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
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.models.backbones import BACKBONE_NAMES, build_backbone
from src.models.config import (
    BATCH_SIZE,
    DATASETS,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE_IMAGE,
    LEARNING_RATE_METADATA,
    NUM_EPOCHS,
    STRONG_AUGMENT_TARGET_CLASSES,
    WEIGHT_DECAY,
    get_dataset,
)
from src.models.dataset import ImageDataset, MetadataDataset, MetadataPreprocessor
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


def _image_run_name(seed: int, backbone: str, sampler: str, strong_augment: str) -> str:
    """image_seed{N} for the pre-Phase-8B/Step-3a default (backbone=
    efficientnet_b0, sampler=shuffle, strong_augment=none), so every
    existing checkpoint/log filename (and Phase 7 fusion's warm-start path)
    keeps working unchanged. Non-default choices are appended so Phase 8B's
    5-backbone runs and Step 3a's imbalance-ablation runs never collide on
    disk.
    """
    parts = ["image"]
    if backbone != "efficientnet_b0":
        parts.append(backbone)
    if sampler != "shuffle":
        parts.append(sampler)
    if strong_augment != "none":
        parts.append(strong_augment)
    parts.append(f"seed{seed}")
    return "_".join(parts)


def train_one_run(
    dataset_name: str,
    branch: str,
    seed: int,
    backbone: str = "efficientnet_b0",
    sampler: str = "shuffle",
    strong_augment: str = "none",
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)

    if ds_config.image_branch_only and branch != "image":
        raise ValueError(
            f"{dataset_name!r} is image_branch_only=True (its train_csv includes "
            f"rows with no compatible clinical metadata - see "
            f"Project_Tracking.md, 'Step 2 Integration Plan', 2026-07-29). "
            f"--branch {branch!r} would silently train on all-NaN metadata for "
            f"those rows. Use --branch image, or --dataset PAD_UFES20 for "
            f"metadata/fusion training."
        )

    if branch == "image":
        strong_augment_classes = (
            set(STRONG_AUGMENT_TARGET_CLASSES) if strong_augment == "minority" else None
        )
        train_ds = ImageDataset(
            ds_config.train_csv, ds_config, train=True,
            strong_augment_classes=strong_augment_classes,
        )
        val_ds = ImageDataset(ds_config.val_csv, ds_config, train=False)
        model = build_backbone(backbone, num_classes=ds_config.num_classes).to(device)
        lr = LEARNING_RATE_IMAGE
        run_name = _image_run_name(seed, backbone, sampler, strong_augment)
    elif branch == "metadata":
        preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))
        train_ds = MetadataDataset(ds_config.train_csv, ds_config, preprocessor)
        val_ds = MetadataDataset(ds_config.val_csv, ds_config, preprocessor)
        model = MetadataMLP(
            input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes
        ).to(device)
        lr = LEARNING_RATE_METADATA
        run_name = f"{branch}_seed{seed}"
    else:
        raise ValueError(f"Unknown branch: {branch}")

    class_weights = compute_class_weights(
        ds_config.train_csv, ds_config.class_names
    ).to(device)

    if branch == "image" and sampler == "weighted":
        # Step 3a ablation (a): oversample rare classes via inverse
        # train-class-frequency weights (same numbers compute_class_weights
        # already derives for the loss) - reused here as per-sample sampling
        # weights instead. Mutually exclusive with shuffle=True per
        # DataLoader's own constraint.
        train_df = pd.read_csv(ds_config.train_csv)
        sample_weights = train_df["disease_label"].map(
            lambda name: class_weights[ds_config.label_to_idx[name]].item()
        ).to_numpy()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    ds_config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ds_config.logs_dir.mkdir(parents=True, exist_ok=True)
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
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "dataset": dataset_name,
                    "branch": branch,
                    "seed": seed,
                    "epoch": epoch,
                    "val_macro_f1": val_macro_f1,
                    "num_classes": ds_config.num_classes,
                }
                if branch == "image":
                    checkpoint["backbone"] = backbone
                    checkpoint["sampler"] = sampler
                    checkpoint["strong_augment"] = strong_augment
                torch.save(checkpoint, checkpoint_path)
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
    if branch == "image":
        summary["backbone"] = backbone
        summary["sampler"] = sampler
        summary["strong_augment"] = strong_augment
    summary_path = ds_config.logs_dir / f"train_{run_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[{dataset_name}/{run_name}] best val macro-F1: "
        f"{best_val_macro_f1:.4f} -> {checkpoint_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Stage 1 / Phase 8B / Step 3a training")
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--branch", choices=["image", "metadata"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--backbone", choices=BACKBONE_NAMES, default="efficientnet_b0",
        help="Phase 8B 5-backbone comparison (image branch only). Ignored for --branch metadata.",
    )
    parser.add_argument(
        "--sampler", choices=["shuffle", "weighted"], default="shuffle",
        help="Step 3a imbalance ablation (a): WeightedRandomSampler (image branch only).",
    )
    parser.add_argument(
        "--strong-augment", choices=["none", "minority"], default="none",
        dest="strong_augment",
        help="Step 3a imbalance ablation (b): stronger augmentation for "
             "STRONG_AUGMENT_TARGET_CLASSES only (image branch only).",
    )
    args = parser.parse_args()
    train_one_run(
        args.dataset, args.branch, args.seed,
        backbone=args.backbone, sampler=args.sampler, strong_augment=args.strong_augment,
    )


if __name__ == "__main__":
    main()
```

---

## Cell 8 — Sanity check (config load + real image path resolution, all 3 sources)

```python
import sys
for mod in list(sys.modules):
    if mod.startswith("src."):
        del sys.modules[mod]

import pandas as pd
from src.models.config import get_dataset, resolve_image_path

ds_config = get_dataset("PAD_UFES20_Expanded")
print("num_classes:", ds_config.num_classes)
print("image_branch_only:", ds_config.image_branch_only)
print("train_csv:", ds_config.train_csv, "exists:", ds_config.train_csv.exists())
print("val_csv:  ", ds_config.val_csv, "exists:", ds_config.val_csv.exists())
print("test_csv: ", ds_config.test_csv, "exists:", ds_config.test_csv.exists())
assert ds_config.train_csv.exists(), "metadata_train_image_only.csv not found — check processed dataset nesting"

df = pd.read_csv(ds_config.train_csv)
print("\ntrain rows:", len(df))
print(df["dataset_source"].value_counts())
print("\nclass counts:")
print(df["disease_label"].value_counts())

val_df = pd.read_csv(ds_config.val_csv)
print("\nval rows:", len(val_df), "| val dataset_source values:", val_df["dataset_source"].unique())
assert set(val_df["dataset_source"].unique()) == {"PAD_UFES20"}, (
    "val split contains a non-PAD_UFES20 row — the train-only-not-val rule is broken, STOP"
)

import random
random.seed(0)
sample_rows = df.sample(n=min(30, len(df)), random_state=0)
missing = []
for _, row in sample_rows.iterrows():
    p = resolve_image_path(row["image_path"])
    if not p.exists():
        missing.append((row["dataset_source"], row["image_path"], str(p)))
print(f"\nChecked {len(sample_rows)} random rows (mixed sources), missing: {len(missing)}")
if missing:
    print("First few missing:", missing[:5])
assert not missing, "Some resolved image paths do not exist — check folder-verification cell output"

print("\nSANITY CHECK PASSED")
```

---

## Cell 9 — Full model/GPU/dependency check (each backbone, one real batch, guard verification)

```python
import torch, sys
print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

from src.models.config import get_dataset
from src.models.dataset import ImageDataset
from src.models.backbones import BACKBONE_NAMES, build_backbone

ds_config = get_dataset("PAD_UFES20_Expanded")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = ImageDataset(ds_config.train_csv, ds_config, train=True)
image, label = train_ds[0]
assert image.shape == (3, 224, 224)

print("BACKBONE_NAMES:", BACKBONE_NAMES)
for name in BACKBONE_NAMES:
    model = build_backbone(name, num_classes=ds_config.num_classes).to(device)
    model.eval()
    with torch.no_grad():
        out = model(image.unsqueeze(0).to(device))
    assert out.shape == (1, ds_config.num_classes), f"{name}: unexpected output shape {out.shape}"
    print(f"{name}: forward pass OK, output shape {out.shape}")
    del model

# Confirm the image_branch_only guard actually fires for this dataset
from src.models.train import train_one_run
try:
    train_one_run("PAD_UFES20_Expanded", "metadata", seed=0)
    raise AssertionError("BUG: image_branch_only guard did not fire")
except ValueError as e:
    print("\nimage_branch_only guard fired correctly:", str(e)[:100], "...")

print("\nALL CHECKS PASSED — ready to train")
```

---

### Backbone: `efficientnet_b0`

## Cell 10 — Train: efficientnet_b0, seed 0

```python
!python -m src.models.train --dataset PAD_UFES20_Expanded --branch image --backbone efficientnet_b0 --seed 0
```

---

## Cell 11 — Train: efficientnet_b0, seed 1

```python
!python -m src.models.train --dataset PAD_UFES20_Expanded --branch image --backbone efficientnet_b0 --seed 1
```

---

## Cell 12 — Train: efficientnet_b0, seed 2

```python
!python -m src.models.train --dataset PAD_UFES20_Expanded --branch image --backbone efficientnet_b0 --seed 2
```

---

### Backbone: `mobilenet_v3_large`

## Cell 13 — Train: mobilenet_v3_large, seed 0

```python
!python -m src.models.train --dataset PAD_UFES20_Expanded --branch image --backbone mobilenet_v3_large --seed 0
```

---

## Cell 14 — Train: mobilenet_v3_large, seed 1

```python
!python -m src.models.train --dataset PAD_UFES20_Expanded --branch image --backbone mobilenet_v3_large --seed 1
```

---

## Cell 15 — Train: mobilenet_v3_large, seed 2

```python
!python -m src.models.train --dataset PAD_UFES20_Expanded --branch image --backbone mobilenet_v3_large --seed 2
```

---

### Backbone: `densenet121`

## Cell 16 — Train: densenet121, seed 0

```python
!python -m src.models.train --dataset PAD_UFES20_Expanded --branch image --backbone densenet121 --seed 0
```

---

## Cell 17 — Train: densenet121, seed 1

```python
!python -m src.models.train --dataset PAD_UFES20_Expanded --branch image --backbone densenet121 --seed 1
```

---

## Cell 18 — Train: densenet121, seed 2

```python
!python -m src.models.train --dataset PAD_UFES20_Expanded --branch image --backbone densenet121 --seed 2
```

---

### Backbone: `resnet50`

## Cell 19 — Train: resnet50, seed 0

```python
!python -m src.models.train --dataset PAD_UFES20_Expanded --branch image --backbone resnet50 --seed 0
```

---

## Cell 20 — Train: resnet50, seed 1

```python
!python -m src.models.train --dataset PAD_UFES20_Expanded --branch image --backbone resnet50 --seed 1
```

---

## Cell 21 — Train: resnet50, seed 2

```python
!python -m src.models.train --dataset PAD_UFES20_Expanded --branch image --backbone resnet50 --seed 2
```

---

### Backbone: `convnext_tiny`

## Cell 22 — Train: convnext_tiny, seed 0

```python
!python -m src.models.train --dataset PAD_UFES20_Expanded --branch image --backbone convnext_tiny --seed 0
```

---

## Cell 23 — Train: convnext_tiny, seed 1

```python
!python -m src.models.train --dataset PAD_UFES20_Expanded --branch image --backbone convnext_tiny --seed 1
```

---

## Cell 24 — Train: convnext_tiny, seed 2

```python
!python -m src.models.train --dataset PAD_UFES20_Expanded --branch image --backbone convnext_tiny --seed 2
```

---

