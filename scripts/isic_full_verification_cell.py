"""Standalone Kaggle folder-verification cell for all 5 Kaggle datasets
this fusion notebook depends on. Paste the contents of this file into
its own Kaggle notebook cell, after adding all 5 as "Add Data" sources:
  1. nodoubttome/skin-cancer9-classesisic  (ISIC Archive 1 raw mirror)
  2. andrewmvd/isic-2019                   (ISIC Archive 2 raw mirror)
  3. naeemsarkertracer/isic-archive1-processed
  4. naeemsarkertracer/isic-archive2-processed
  5. naeemsarkertracer/ham10000-stage1-checkpoints

Does not assert/crash on mismatch - everything is a printed
PASS/MISMATCH so a bad candidate can be inspected rather than blowing up
the notebook.

Datasets 1-2 were already verified (image counts + per-class counts /
metadata row count) via scripts/isic_mirror_verification_cell.py - this
cell just re-confirms the mount path is stable, it does not repeat the
full per-class check. Datasets 3-5 are new uploads whose zip layout
(wrapped in an extra folder vs. flat) has never been confirmed, matching
this project's rule that every Kaggle upload gets a folder-verification
cell before src/models/config.py's auto-detection is trusted blind.
"""

import os
import glob

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def find_dataset_dir(candidates):
    """Try a list of candidate mount paths, return the first that exists."""
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def count_images(root):
    total = 0
    for ext in IMAGE_EXTS:
        total += len(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        total += len(glob.glob(os.path.join(root, "**", f"*{ext.upper()}"), recursive=True))
    return total


def check(label, actual, expected):
    status = "PASS" if actual == expected else "MISMATCH"
    print(f"  [{status}] {label}: expected {expected}, got {actual}")
    return actual == expected


print("=" * 70)
print("Kaggle input root listing (in case mount path differs from expected)")
print("=" * 70)
for root_candidate in ("/kaggle/input/datasets", "/kaggle/input"):
    if os.path.isdir(root_candidate):
        print(f"\n{root_candidate}:")
        for entry in sorted(os.listdir(root_candidate)):
            print(f"  [dir] {entry}")


def report_layout(label, root, marker_relpath, wrapped_subdir):
    """Print whether `marker_relpath` exists directly under `root`
    (flat) or under `root/wrapped_subdir/` (wrapped), matching the same
    auto-detection order src/models/config.py's _processed_dir() /
    _stage1_checkpoints_dir() use. Prints both candidate paths either
    way so a human can eyeball anything unexpected (e.g. present in
    neither, or present in both).
    """
    print(f"\n  --- {label} layout check ---")
    wrapped_candidate = os.path.join(root, wrapped_subdir, marker_relpath)
    flat_candidate = os.path.join(root, marker_relpath)
    wrapped_exists = os.path.isfile(wrapped_candidate)
    flat_exists = os.path.isfile(flat_candidate)
    print(f"  wrapped candidate ({wrapped_subdir}/{marker_relpath}): {'FOUND' if wrapped_exists else 'not found'} -> {wrapped_candidate}")
    print(f"  flat candidate ({marker_relpath}):{' ' * max(1, len(wrapped_subdir) + 1)}{'FOUND' if flat_exists else 'not found'} -> {flat_candidate}")
    if wrapped_exists and flat_exists:
        print(f"  !! both candidates exist - inspect manually, config.py's auto-detect will pick the wrapped one first")
    elif wrapped_exists:
        print(f"  -> layout is WRAPPED (root/{wrapped_subdir}/...). KAGGLE_*_WRAPPED[...] = True is correct.")
    elif flat_exists:
        print(f"  -> layout is FLAT (root/...). KAGGLE_*_WRAPPED[...] = False would be correct (auto-detect handles either way regardless).")
    else:
        print(f"  !! neither candidate found - list root contents below and inspect")
        try:
            print(f"  root contents: {sorted(os.listdir(root))}")
        except OSError as e:
            print(f"  !! could not list root: {e}")


# --- 1. ISIC Archive 1 raw mirror: nodoubttome/skin-cancer9-classesisic -
print("\n" + "=" * 70)
print("1. ISIC Archive 1 RAW MIRROR: nodoubttome/skin-cancer9-classesisic")
print("   (already verified - re-confirming mount path is stable only)")
print("=" * 70)

archive1_candidates = [
    "/kaggle/input/datasets/nodoubttome/skin-cancer9-classesisic",
    "/kaggle/input/skin-cancer9-classesisic",
]
archive1_root = find_dataset_dir(archive1_candidates)

if archive1_root is None:
    print(f"  !! NOT FOUND at any of: {archive1_candidates}")
else:
    print(f"  Found at: {archive1_root}")
    print(f"  Top-level contents: {sorted(os.listdir(archive1_root))}")

    train_dir = os.path.join(archive1_root, "Train")
    if not os.path.isdir(train_dir):
        nested = [d for d in os.listdir(archive1_root) if os.path.isdir(os.path.join(archive1_root, d))]
        for d in nested:
            candidate = os.path.join(archive1_root, d, "Train")
            if os.path.isdir(candidate):
                print(f"  NOTE: double-nested layout confirmed under '{d}/' - matches KAGGLE_DATASET_SUBPATH['ISIC_Archive_1'] in config.py")
                archive1_root = os.path.join(archive1_root, d)
                train_dir = candidate
                break

    test_dir = os.path.join(archive1_root, "Test")
    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        total = count_images(archive1_root)
        check("total images (Train+Test)", total, 2357)
    else:
        print(f"  !! Train/Test folders not found under {archive1_root} - inspect top-level contents above")

# --- 2. ISIC Archive 2 raw mirror: andrewmvd/isic-2019 ------------------
print("\n" + "=" * 70)
print("2. ISIC Archive 2 RAW MIRROR: andrewmvd/isic-2019")
print("   (already verified - re-confirming mount path is stable only)")
print("=" * 70)

archive2_candidates = [
    "/kaggle/input/datasets/andrewmvd/isic-2019",
    "/kaggle/input/isic-2019",
]
archive2_root = find_dataset_dir(archive2_candidates)

if archive2_root is None:
    print(f"  !! NOT FOUND at any of: {archive2_candidates}")
else:
    print(f"  Found at: {archive2_root}")
    print(f"  Top-level contents: {sorted(os.listdir(archive2_root))}")
    total = count_images(archive2_root)
    check("total images", total, 25331)
    metadata_matches = glob.glob(os.path.join(archive2_root, "**", "*Metadata*.csv"), recursive=True)
    print(f"  *Metadata*.csv found: {bool(metadata_matches)} ({metadata_matches[0] if metadata_matches else 'n/a'})")

# --- 3. ISIC Archive 1 processed: naeemsarkertracer/isic-archive1-processed
print("\n" + "=" * 70)
print("3. ISIC Archive 1 PROCESSED: naeemsarkertracer/isic-archive1-processed")
print("   (NEW - layout never verified)")
print("=" * 70)

isic1_proc_candidates = [
    "/kaggle/input/datasets/naeemsarkertracer/isic-archive1-processed",
    "/kaggle/input/isic-archive1-processed",
]
isic1_proc_root = find_dataset_dir(isic1_proc_candidates)

if isic1_proc_root is None:
    print(f"  !! NOT FOUND at any of: {isic1_proc_candidates}")
else:
    print(f"  Found at: {isic1_proc_root}")
    print(f"  Top-level contents: {sorted(os.listdir(isic1_proc_root))}")
    report_layout("ISIC_Archive_1 processed", isic1_proc_root, "metadata_train.csv", "ISIC_Archive_1")

# --- 4. ISIC Archive 2 processed: naeemsarkertracer/isic-archive2-processed
print("\n" + "=" * 70)
print("4. ISIC Archive 2 PROCESSED: naeemsarkertracer/isic-archive2-processed")
print("   (NEW - layout never verified)")
print("=" * 70)

isic2_proc_candidates = [
    "/kaggle/input/datasets/naeemsarkertracer/isic-archive2-processed",
    "/kaggle/input/isic-archive2-processed",
]
isic2_proc_root = find_dataset_dir(isic2_proc_candidates)

if isic2_proc_root is None:
    print(f"  !! NOT FOUND at any of: {isic2_proc_candidates}")
else:
    print(f"  Found at: {isic2_proc_root}")
    print(f"  Top-level contents: {sorted(os.listdir(isic2_proc_root))}")
    report_layout("ISIC_Archive_2 processed", isic2_proc_root, "metadata_train.csv", "ISIC_Archive_2")

# --- 5. HAM10000 Stage 1 checkpoints: naeemsarkertracer/ham10000-stage1-checkpoints
print("\n" + "=" * 70)
print("5. HAM10000 STAGE 1 CHECKPOINTS: naeemsarkertracer/ham10000-stage1-checkpoints")
print("   (NEW - layout never verified)")
print("=" * 70)

ham_ckpt_candidates = [
    "/kaggle/input/datasets/naeemsarkertracer/ham10000-stage1-checkpoints",
    "/kaggle/input/ham10000-stage1-checkpoints",
]
ham_ckpt_root = find_dataset_dir(ham_ckpt_candidates)

if ham_ckpt_root is None:
    print(f"  !! NOT FOUND at any of: {ham_ckpt_candidates}")
else:
    print(f"  Found at: {ham_ckpt_root}")
    print(f"  Top-level contents: {sorted(os.listdir(ham_ckpt_root))}")
    report_layout("HAM10000 Stage 1 checkpoints", ham_ckpt_root, "image_seed0_best.pt", "checkpoints")

    # If found (either layout), also confirm all 6 expected files are
    # present, not just the one marker file, matching how thoroughly the
    # raw mirrors were checked above.
    expected_ckpt_files = [
        "image_seed0_best.pt", "image_seed1_best.pt", "image_seed2_best.pt",
        "metadata_seed0_best.pt", "metadata_seed1_best.pt", "metadata_seed2_best.pt",
    ]
    for base_candidate in (os.path.join(ham_ckpt_root, "checkpoints"), ham_ckpt_root):
        if os.path.isfile(os.path.join(base_candidate, expected_ckpt_files[0])):
            print(f"\n  Confirming all 6 expected checkpoint files under: {base_candidate}")
            for fname in expected_ckpt_files:
                fpath = os.path.join(base_candidate, fname)
                exists = os.path.isfile(fpath)
                size = os.path.getsize(fpath) if exists else 0
                status = "PASS" if exists and size > 0 else "MISMATCH"
                print(f"  [{status}] {fname}: {'found, ' + str(size) + ' bytes' if exists else 'NOT FOUND'}")
            break

print("\n" + "=" * 70)
print("Done. Review every [MISMATCH] / '!!' line above before trusting any of the 5 datasets.")
print("Once layouts for datasets 3-5 are confirmed, update KAGGLE_PROCESSED_WRAPPED /")
print("KAGGLE_STAGE1_CHECKPOINT_WRAPPED in src/models/config.py to match (informational -")
print("the runtime auto-detection in config.py does not require this, but keeping the")
print("fallback dict accurate avoids a wrong guess if the real layout is ever unreachable).")
print("=" * 70)
