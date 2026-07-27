"""Standalone Kaggle diagnostic cell - ISIC Archive 2 mirror ID coverage.

Triggered by Cell 8 (sanity check) of external_isic_evaluation_kaggle_notebook.md
finding 3 of 20 random rows with neither the flat nor doubled
ISIC_2019_Training_Input/ path existing (ISIC_0016058, ISIC_0014570,
ISIC_0012151), despite the mirror's TOTAL image count matching our
expected 25,331 exactly during earlier verification. That combination
(matching total, missing specific IDs) means the mirror's actual file
*set* may differ slightly from our local data/raw/ISIC_Archive_2/ even
though the counts coincide - needs a real ID-level comparison, not
another count check.

Locally confirmed (2026-07-27, off-Kaggle): all 3 reported IDs exist in
data/raw/ISIC_Archive_2/images/ (25,331 files total) - so this is not a
bug in our own image_path/exclusion generation. Purely a question of
whether this specific Kaggle mirror is missing them.

ROOT CAUSE FOUND (2026-07-27): this mirror renames some large images
with a "_downsampled" suffix (e.g. ISIC_0016058.jpg only exists as
ISIC_0016058_downsampled.jpg) - affects ~2,074 of our 25,076 processed
IDs (8.3%), a systematic naming convention, not stray files. Fixed in
src/models/config.py's resolve_image_path() via
KAGGLE_FILENAME_FALLBACK_SUFFIX. STEP 3 below re-verifies that fix
against every ID this script finds missing, using the real (fixed)
resolve_image_path() - not just the original 3 reported IDs, and not
assumed to work without checking.

Paste into its own Kaggle cell AFTER Cell 1 (folder verification), Cell
2 (setup/sys.path), and Cell 3 (config.py %%writefile - must be the
UPDATED version with KAGGLE_FILENAME_FALLBACK_SUFFIX) have already run,
since Step 3 imports the live resolve_image_path() from
/kaggle/working/src/models/config.py. Report back the full printed
output, and separately download
/kaggle/working/isic_archive2_kaggle_mirror_ids.csv from the notebook's
Output after running/committing - that file lets the full 25,331-ID
local-raw-vs-mirror comparison happen back on this machine without
pasting a ~25k-row list through chat.
"""

import csv
import glob
import os

archive2_root = "/kaggle/input/datasets/andrewmvd/isic-2019"
input_root = os.path.join(archive2_root, "ISIC_2019_Training_Input")

print("=" * 70)
print("STEP 1: broad search for the 3 reported-missing IDs")
print("(case-insensitive substring match anywhere under archive2_root,")
print(" not just the two candidate paths resolve_image_path() already tried)")
print("=" * 70)

reported_missing = ["ISIC_0016058", "ISIC_0014570", "ISIC_0012151"]
all_files = glob.glob(os.path.join(archive2_root, "**", "*"), recursive=True)
print(f"\n(scanned {len(all_files)} total filesystem entries under archive2_root)")

for isic_id in reported_missing:
    hits = [f for f in all_files if isic_id.lower() in os.path.basename(f).lower()]
    print(f"\n{isic_id}: {len(hits)} match(es)")
    for h in hits:
        print(f"    {h}")
    if not hits:
        print("    (genuinely not found anywhere under this mirror's root)")

print("\n" + "=" * 70)
print("STEP 2a: mirror's full ID set vs our MOUNTED PROCESSED dataset")
print("(25,076 post-dedup IDs, naeemsarkertracer/isic-archive2-processed -")
print(" the operationally-relevant set evaluate_external_isic.py actually reads,")
print(" already mounted, so this needs no download round-trip)")
print("=" * 70)

jpg_files = (
    glob.glob(os.path.join(input_root, "**", "*.jpg"), recursive=True)
    + glob.glob(os.path.join(input_root, "**", "*.JPG"), recursive=True)
    + glob.glob(os.path.join(input_root, "**", "*.jpeg"), recursive=True)
    + glob.glob(os.path.join(input_root, "**", "*.JPEG"), recursive=True)
)
print(f"\nTotal .jpg/.jpeg files found (recursive, both case variants) under {input_root}: {len(jpg_files)}")

kaggle_ids = sorted(set(os.path.splitext(os.path.basename(f))[0] for f in jpg_files))
print(f"Unique IDs in mirror: {len(kaggle_ids)}")

import pandas as pd

proc_root = "/kaggle/input/datasets/naeemsarkertracer/isic-archive2-processed"
dfs = [pd.read_csv(os.path.join(proc_root, f"metadata_{s}.csv")) for s in ("train", "val", "test")]
processed_ids = set(pd.concat(dfs, ignore_index=True)["image_id"])
kaggle_id_set = set(kaggle_ids)

missing_from_kaggle = processed_ids - kaggle_id_set
extra_in_kaggle = kaggle_id_set - processed_ids

print(f"\nOur processed (post-dedup) CSVs: {len(processed_ids)} unique IDs")
print(f"IDs in our processed set but NOT found in this Kaggle mirror: {len(missing_from_kaggle)}")
print(f"IDs in this Kaggle mirror but NOT in our processed set: {len(extra_in_kaggle)}")

if missing_from_kaggle:
    missing_out = "/kaggle/working/isic_archive2_missing_from_kaggle_mirror.csv"
    pd.Series(sorted(missing_from_kaggle), name="image_id").to_csv(missing_out, index=False)
    print(f"\nWrote full missing-ID list -> {missing_out}")
    print("First 20 missing IDs:", sorted(missing_from_kaggle)[:20])

print("\n" + "=" * 70)
print("STEP 2b: writing mirror's complete ID inventory for the FULL raw-vs-mirror")
print("comparison (25,331 local raw files - broader than the 25,076 processed set)")
print("=" * 70)

out_path = "/kaggle/working/isic_archive2_kaggle_mirror_ids.csv"
with open(out_path, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["image_id"])
    for i in kaggle_ids:
        writer.writerow([i])
print(f"\nWrote full mirror ID list ({len(kaggle_ids)} rows) -> {out_path}")
print("Download this file from the notebook's Output (after running/committing)")
print("and share it back for the complete data/raw/ISIC_Archive_2/ comparison.")

print("\n" + "=" * 70)
print("STEP 3: verify the '_downsampled' fallback actually closes the gap")
print("(real resolve_image_path(), not another assumption - requires Cell 3's")
print(" config.py to already be the UPDATED version with")
print(" KAGGLE_FILENAME_FALLBACK_SUFFIX)")
print("=" * 70)

import sys

for mod in list(sys.modules):
    if mod.startswith("src."):
        del sys.modules[mod]

from src.models.config import resolve_image_path

id_to_path = dict(zip(pd.concat(dfs, ignore_index=True)["image_id"], pd.concat(dfs, ignore_index=True)["image_path"]))

still_missing = []
resolved_via_fallback = []
for isic_id in sorted(missing_from_kaggle):
    image_path = id_to_path[isic_id]
    resolved = resolve_image_path(image_path)
    if resolved.exists():
        resolved_via_fallback.append((isic_id, resolved))
    else:
        still_missing.append((isic_id, resolved))

print(f"\nOf {len(missing_from_kaggle)} previously-missing IDs:")
print(f"  now resolve via the _downsampled fallback: {len(resolved_via_fallback)}")
print(f"  STILL missing after the fix: {len(still_missing)}")

if resolved_via_fallback:
    print("\nSample of newly-resolved paths (first 5):")
    for isic_id, resolved in resolved_via_fallback[:5]:
        print(f"    {isic_id} -> {resolved}")

if still_missing:
    still_missing_out = "/kaggle/working/isic_archive2_still_missing_after_downsampled_fix.csv"
    pd.DataFrame(still_missing, columns=["image_id", "attempted_path"]).to_csv(still_missing_out, index=False)
    print(f"\nWrote remaining-missing list ({len(still_missing)} rows) -> {still_missing_out}")
    print("First 20 still-missing IDs:", [i for i, _ in still_missing[:20]])
else:
    print("\nAll previously-missing IDs now resolve. Gap fully closed.")

print("\n" + "=" * 70)
print("Done. Report the full output above (especially STEP 3's counts), plus the")
print("downloaded CSVs, before proceeding to Cell 9 / the commit.")
print("=" * 70)
