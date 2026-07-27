"""Standalone Kaggle folder-verification cell for the 2 candidate ISIC
mirrors (Archive 1: nodoubttome/skin-cancer9-classesisic; Archive 2:
andrewmvd/isic-2019). Paste the contents of this file into its own
Kaggle notebook cell, after adding both as "Add Data" sources. Does not
assert/crash on mismatch - everything is a printed PASS/MISMATCH so a
bad candidate can be inspected rather than blowing up the notebook.

Expected numbers below were read directly from this project's local
data/raw/ISIC_Archive_1 and data/raw/ISIC_Archive_2 on 2026-07-25 - see
Project_Tracking.md.
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

# --- Archive 1: nodoubttome/skin-cancer9-classesisic --------------------
print("\n" + "=" * 70)
print("ARCHIVE 1 candidate: nodoubttome/skin-cancer9-classesisic")
print("=" * 70)

archive1_candidates = [
    "/kaggle/input/datasets/nodoubttome/skin-cancer9-classesisic",
    "/kaggle/input/skin-cancer9-classesisic",
]
archive1_root = find_dataset_dir(archive1_candidates)

if archive1_root is None:
    print(f"  !! NOT FOUND at any of: {archive1_candidates}")
    print("  -> check the exact mounted path above and adjust archive1_candidates")
else:
    print(f"  Found at: {archive1_root}")
    print(f"  Top-level contents: {sorted(os.listdir(archive1_root))}")

    # Handle possible double-nesting (e.g. root/skin-cancer9-classesisic/Train)
    # the same way this project's other Kaggle-mount checks do - don't assume
    # a clean unwrapped layout.
    train_dir = os.path.join(archive1_root, "Train")
    if not os.path.isdir(train_dir):
        nested = [d for d in os.listdir(archive1_root) if os.path.isdir(os.path.join(archive1_root, d))]
        for d in nested:
            candidate = os.path.join(archive1_root, d, "Train")
            if os.path.isdir(candidate):
                print(f"  NOTE: double-nested layout detected under '{d}/' - using that")
                archive1_root = os.path.join(archive1_root, d)
                train_dir = candidate
                break

    test_dir = os.path.join(archive1_root, "Test")

    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        total = count_images(archive1_root)
        check("total images (Train+Test)", total, 2357)

        expected_train = {
            "actinic keratosis": 114, "basal cell carcinoma": 376, "dermatofibroma": 95,
            "melanoma": 438, "nevus": 357, "pigmented benign keratosis": 462,
            "seborrheic keratosis": 77, "squamous cell carcinoma": 181, "vascular lesion": 139,
        }
        expected_test = {
            "actinic keratosis": 16, "basal cell carcinoma": 16, "dermatofibroma": 16,
            "melanoma": 16, "nevus": 16, "pigmented benign keratosis": 16,
            "seborrheic keratosis": 3, "squamous cell carcinoma": 16, "vascular lesion": 3,
        }

        print("\n  Train per-class counts:")
        actual_classes = {d.lower(): d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))}
        print(f"  Class folders found: {sorted(actual_classes.keys())}")
        for cls, expected_n in expected_train.items():
            folder = actual_classes.get(cls)
            actual_n = count_images(os.path.join(train_dir, folder)) if folder else 0
            check(f"Train/{cls}", actual_n, expected_n)

        print("\n  Test per-class counts:")
        actual_classes_test = {d.lower(): d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))}
        for cls, expected_n in expected_test.items():
            folder = actual_classes_test.get(cls)
            actual_n = count_images(os.path.join(test_dir, folder)) if folder else 0
            check(f"Test/{cls}", actual_n, expected_n)
    else:
        print(f"  !! Train/Test folders not found under {archive1_root} - inspect top-level contents above")

# --- Archive 2: andrewmvd/isic-2019 -------------------------------------
print("\n" + "=" * 70)
print("ARCHIVE 2 candidate: andrewmvd/isic-2019")
print("=" * 70)

archive2_candidates = [
    "/kaggle/input/datasets/andrewmvd/isic-2019",
    "/kaggle/input/isic-2019",
]
archive2_root = find_dataset_dir(archive2_candidates)

if archive2_root is None:
    print(f"  !! NOT FOUND at any of: {archive2_candidates}")
    print("  -> check the exact mounted path above and adjust archive2_candidates")
else:
    print(f"  Found at: {archive2_root}")
    print(f"  Top-level contents: {sorted(os.listdir(archive2_root))}")

    total = count_images(archive2_root)
    check("total images", total, 25331)

    # Confirm the expected sub-layout for this mirror rather than
    # assuming it: images under ISIC_2019_Training_Input/, labels in
    # ISIC_2019_Training_GroundTruth.csv, metadata in
    # ISIC_2019_Training_Metadata.csv (NOT "metadata.csv" - that was
    # wrong, see below).
    input_dirs = glob.glob(os.path.join(archive2_root, "**", "ISIC_2019_Training_Input"), recursive=True)
    groundtruth_matches = glob.glob(os.path.join(archive2_root, "**", "ISIC_2019_Training_GroundTruth.csv"), recursive=True)
    print(f"  ISIC_2019_Training_Input/ found: {bool(input_dirs)}")
    print(f"  ISIC_2019_Training_GroundTruth.csv found: {bool(groundtruth_matches)}")

    # The real filename is ISIC_2019_Training_Metadata.csv, not
    # metadata.csv - search broadly (any *Metadata*.csv) so this still
    # finds it if Kaggle's exact casing/naming drifts again.
    metadata_matches = glob.glob(os.path.join(archive2_root, "**", "*Metadata*.csv"), recursive=True)
    if not metadata_matches:
        print("  !! no *Metadata*.csv found anywhere under the mounted root")
    else:
        metadata_path = metadata_matches[0]
        print(f"  metadata file found at: {metadata_path}")
        import pandas as pd

        df = pd.read_csv(metadata_path)
        print(f"  columns present: {list(df.columns)}")
        # Row count: accept either 25331 (no header counted) or 25332
        # (if something upstream double-counted a header row) rather
        # than hard-failing on a one-off discrepancy.
        row_count = len(df)
        status = "PASS" if row_count in (25331, 25332) else "MISMATCH"
        print(f"  [{status}] metadata row count: expected 25331 or 25332, got {row_count}")

        # Do NOT assume a column called "attribution" exists here - the
        # official ISIC 2019 Training_Metadata.csv is documented to
        # contain only image/age_approx/anatom_site_general/lesion_id/sex,
        # with no per-row institution field. Search for any column whose
        # *name* looks attribution/institution-like instead of assuming
        # our local processed CSV's schema applies to this file.
        candidate_cols = [c for c in df.columns if any(k in c.lower() for k in ("attribution", "institution", "source", "center", "centre"))]
        if not candidate_cols:
            print("  !! no attribution/institution-like column found in this file's columns above")
            print("     (expected if this is the stock ISIC_2019_Training_Metadata.csv - that")
            print("      file does not carry per-row source-institution data upstream)")
        else:
            for col in candidate_cols:
                counts = df[col].value_counts(dropna=False)
                print(f"\n  '{col}' value counts (raw):")
                print(counts.to_string())

                # Match by substring rather than exact accented string, to
                # avoid a false MISMATCH from an encoding artifact rather
                # than a real content difference. NOTE: expected numbers
                # below are NOT hardcoded to old project figures (those
                # were post-dedup, 25,076-row totals) - report actuals and
                # let the human confirm rather than asserting a specific
                # split here.
                if hasattr(counts.index, "str"):
                    hospital_n = counts[counts.index.str.contains("Hospital", na=False)].sum()
                    vidir_n = counts[counts.index.str.contains("ViDIR", na=False)].sum()
                    anon_n = counts.get("Anonymous", 0)
                    print(f"  -> Hospital Clínic de Barcelona (substring match): {int(hospital_n)}")
                    print(f"  -> ViDIR Group / Medical University of Vienna (substring match): {int(vidir_n)}")
                    print(f"  -> Anonymous (exact match): {int(anon_n)}")
                    print(f"  -> sum of the three: {int(hospital_n) + int(vidir_n) + int(anon_n)} (compare to row_count={row_count})")

print("\n" + "=" * 70)
print("Done. Review every [MISMATCH] above before trusting either mirror.")
print("=" * 70)
