"""Generates notebooks/pad_ufes20_cross_attention_efficientnet_expanded_kaggle_notebook.md.

Reads the actual source files from src/models/ and embeds their real
content into %%writefile cells - never hand-typed, so the notebook
cannot silently drift from the source-of-truth .py files. Mirrors
generate_cross_attention_kaggle_notebook.py's structure (folder
verification -> setup -> %%writefile cells -> sanity check -> full
model/GPU check -> training cells), with one difference matching
generate_cross_attention_backbone_fusion_kaggle_notebook.py's pattern:
a 4th "Add Data" source is needed because the image embedder warm-starts
from PAD_UFES20_Expanded's EfficientNet-B0 checkpoint, not PAD_UFES20's
own Stage 1 image checkpoint.

Dataset-expansion-only ablation (approved 2026-08-01, Project_Tracking.md
"Pre-Registered Prediction"): trains the ORIGINAL CrossAttentionFusionModel
(EfficientNet-B0), warm-started from Step 3's PAD_UFES20_Expanded
image_seed{0,1,2}_best.pt (not the convnext_tiny/densenet121 ones Step 4
used) plus PAD_UFES20's own Stage 1 metadata_seed{N}_best.pt. Val-only -
no test-split code path exists in this notebook at all.

IMPORTANT - Kaggle dataset note: PAD_UFES20_Expanded/checkpoints/image_seed{0,1,2}_best.pt
(EfficientNet-B0, the *default* backbone naming - no backbone suffix) are
NOT the same 6 files already uploaded to the
"pad-ufes20-expanded-backbone-checkpoints" Kaggle dataset for Step 4
(those are image_convnext_tiny_seed{N}/image_densenet121_seed{N} only).
These 3 additional EfficientNet-B0 files must be added to that same
Kaggle dataset (recommended - one dataset, new version) or a new one,
before this notebook can run. EXPANDED_EFFICIENTNET_CHECKPOINTS_SLUG
below defaults to reusing the existing dataset slug; change it if a
separate dataset is preferred.

Run: python scripts/generate_cross_attention_efficientnet_expanded_kaggle_notebook.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src" / "models"
OUT_PATH = PROJECT_ROOT / "notebooks" / "pad_ufes20_cross_attention_efficientnet_expanded_kaggle_notebook.md"

# Reuses the same private Kaggle dataset Step 4 already uses, on the
# assumption the 3 additional image_seed{0,1,2}_best.pt (EfficientNet-B0)
# files get added to it as a new dataset version. Change this if a
# separate dataset is preferred instead.
EXPANDED_EFFICIENTNET_CHECKPOINTS_SLUG = "naeemsarkertracer/pad-ufes20-expanded-backbone-checkpoints"

SEEDS = (0, 1, 2)

# (source file, path to %%writefile on Kaggle)
FILES = [
    ("config.py", "/kaggle/working/src/models/config.py"),
    ("dataset.py", "/kaggle/working/src/models/dataset.py"),
    ("image_model.py", "/kaggle/working/src/models/image_model.py"),
    ("metadata_model.py", "/kaggle/working/src/models/metadata_model.py"),
    ("fusion_model.py", "/kaggle/working/src/models/fusion_model.py"),
    ("cross_attention_fusion_model.py", "/kaggle/working/src/models/cross_attention_fusion_model.py"),
    ("backbones.py", "/kaggle/working/src/models/backbones.py"),
    ("train.py", "/kaggle/working/src/models/train.py"),
    ("train_cross_attention_efficientnet_expanded.py", "/kaggle/working/src/models/train_cross_attention_efficientnet_expanded.py"),
]


def read_source(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8").rstrip("\n")


def writefile_cell(cell_num: int, source_name: str, kaggle_path: str) -> str:
    code = read_source(source_name)
    return (
        f"## Cell {cell_num} — `%%writefile {kaggle_path}`\n\n"
        f"```python\n%%writefile {kaggle_path}\n{code}\n```\n"
    )


def build_notebook() -> str:
    parts = []

    parts.append(
        "# PAD-UFES-20 Kaggle Notebook — Dataset-Expansion-Only Ablation "
        "(EfficientNet-B0 Cross-Attention, Expanded Warm-Start)\n\n"
        "> **WARNING - read before pasting any cell into Kaggle:** for every "
        "`%%writefile` cell below, `%%writefile <path>` MUST be the exact first "
        "line of the Kaggle cell, with absolutely nothing above it - no blank "
        "line, no comment, no stray character. Kaggle (like Jupyter) only "
        "treats a line as a cell magic if it is the first line of the cell; "
        "anything preceding it turns `%%writefile` into a plain, non-magic "
        "line and the file silently never gets written. **After pasting each "
        "`%%writefile` cell, re-open it and visually confirm `%%writefile` is "
        "line 1 before running it.**\n\n"
        "Trains the ORIGINAL Phase 7 Stage 2 architecture "
        "(`CrossAttentionFusionModel`, EfficientNet-B0 image embedder) x 3 "
        "seeds = 3 runs, all on PAD_UFES20 (never PAD_UFES20_Expanded - "
        "image_branch_only=True blocks metadata/fusion training there). "
        "Purpose: isolate the dataset-expansion effect from Step 4's "
        "backbone-architecture-change effect, holding architecture constant "
        "- see Project_Tracking.md's \"Pre-Registered Prediction - "
        "Dataset-Expansion-Only Ablation\" entry (predicted val macro-F1 "
        "range: 0.61-0.66, logged BEFORE this run). Warm-starts are "
        "cross-dataset, same pattern as Step 4: the image embedder loads "
        "Step 3's (Phase 8B) PAD_UFES20_Expanded EfficientNet-B0 checkpoint "
        "(`image_seed{N}_best.pt`, the *default*-backbone naming - not "
        "`image_convnext_tiny_...`/`image_densenet121_...`), the metadata "
        "embedder loads the existing Phase 7 Stage 1 PAD_UFES20 metadata "
        "checkpoint. Validation-only ablation: no test-split code path "
        "exists anywhere in this notebook, and `evaluate.py` additionally "
        "refuses `--split test` for this branch unconditionally, "
        "independent of PAD_UFES20's test_split_guard.py marker (already "
        "twice-consumed regardless).\n\n"
        "Requires **four** Kaggle \"Add Data\" sources attached before "
        "running:\n\n"
        "- `mahdavi1202/skin-cancer` (raw PAD-UFES-20 image mirror)\n"
        "- `naeemsarkertracer/pad-ufes20-processed` (our processed metadata CSVs)\n"
        "- `naeemsarkertracer/pad-ufes20-stage1-checkpoints` (Phase 7 Stage 1 "
        "checkpoints - only `metadata_seed{0,1,2}_best.pt` are used here, for "
        "the metadata embedder warm start)\n"
        f"- `{EXPANDED_EFFICIENTNET_CHECKPOINTS_SLUG}` - **needs "
        "`image_seed{0,1,2}_best.pt` (EfficientNet-B0, from this machine's "
        "`logs/PAD_UFES20_Expanded/checkpoints/`) added to it first** (as a "
        "new dataset version, if reusing Step 4's existing "
        "\"pad-ufes20-expanded-backbone-checkpoints\" dataset, which "
        "currently only holds the convnext_tiny/densenet121 files) - "
        "**this is a real, not-yet-done upload step, confirm it before "
        "running this notebook.**\n\n"
        "Paste each cell below into a separate Kaggle notebook cell, in order.\n\n"
        "---\n\n"
    )

    parts.append(
        "## Cell 1 — Folder verification (raw mirror + processed nesting + both checkpoint datasets)\n\n"
        "```python\n"
        "import os\n\n"
        "def show(path, label):\n"
        "    print(f\"--- {label}: {path} ---\")\n"
        "    if not os.path.isdir(path):\n"
        "        print(\"  !! NOT FOUND\")\n"
        "        return\n"
        "    for entry in sorted(os.listdir(path)):\n"
        "        full = os.path.join(path, entry)\n"
        "        kind = \"dir\" if os.path.isdir(full) else \"file\"\n"
        "        print(f\"  [{kind}] {entry}\")\n\n"
        "raw_root = \"/kaggle/input/datasets/mahdavi1202/skin-cancer\"\n"
        "show(\"/kaggle/input/datasets\", \"datasets root\")\n"
        "show(raw_root, \"raw PAD-UFES-20 mirror\")\n\n"
        "processed_root = \"/kaggle/input/datasets/naeemsarkertracer/pad-ufes20-processed\"\n"
        "show(processed_root, \"processed PAD-UFES-20 (root)\")\n"
        "wrapped = os.path.join(processed_root, \"PAD_UFES20\")\n"
        "show(wrapped, \"processed PAD-UFES-20 (wrapped candidate)\")\n"
        "assert os.path.isfile(os.path.join(wrapped, \"metadata_train.csv\")), (\n"
        "    \"Expected wrapped metadata_train.csv not found - check processed dataset packaging\"\n"
        ")\n\n"
        "# --- Stage 1 checkpoint dataset check (metadata_seed{N}_best.pt only) ----\n"
        "checkpoint_root = \"/kaggle/input/datasets/naeemsarkertracer/pad-ufes20-stage1-checkpoints\"\n"
        "show(checkpoint_root, \"Stage 1 checkpoints (root)\")\n"
        "wrapped_ckpt = os.path.join(checkpoint_root, \"checkpoints\")\n"
        "show(wrapped_ckpt, \"Stage 1 checkpoints (wrapped candidate: root/checkpoints/)\")\n\n"
        "expected_stage1 = [f\"metadata_seed{s}_best.pt\" for s in (0, 1, 2)]\n\n"
        "def all_present(base, names):\n"
        "    return all(os.path.isfile(os.path.join(base, n)) for n in names)\n\n"
        "wrapped_ok = all_present(wrapped_ckpt, expected_stage1)\n"
        "root_ok = all_present(checkpoint_root, expected_stage1)\n"
        "print(f\"\\nAll 3 metadata checkpoints found at wrapped candidate ({wrapped_ckpt}): {wrapped_ok}\")\n"
        "print(f\"All 3 metadata checkpoints found at root ({checkpoint_root}): {root_ok}\")\n\n"
        "if wrapped_ok:\n"
        "    resolved_stage1_dir = wrapped_ckpt\n"
        "elif root_ok:\n"
        "    resolved_stage1_dir = checkpoint_root\n"
        "else:\n"
        "    resolved_stage1_dir = None\n\n"
        "assert resolved_stage1_dir is not None, (\n"
        "    \"Stage 1 metadata checkpoints not found in either expected layout - \"\n"
        "    \"metadata embedder warm-start cannot proceed.\"\n"
        ")\n\n"
        "# --- Step 3/Phase 8B expanded EfficientNet-B0 checkpoint dataset check ---\n"
        "# Recursive search, not a fixed root/root-checkpoints check: this dataset's\n"
        "# actual packaging (folder names inside the uploaded zip) can vary between\n"
        "# dataset versions/uploads (e.g. a nested \"PAD-UFES20 Expanded Backbone\n"
        "# Checkpoints/\" folder, plus an unrelated \"expand_backup/\" folder observed\n"
        "# in practice) - walk the whole tree instead of guessing a fixed layout.\n"
        f"expanded_ckpt_root = \"/kaggle/input/datasets/{EXPANDED_EFFICIENTNET_CHECKPOINTS_SLUG}\"\n"
        "show(expanded_ckpt_root, \"Step 3 PAD_UFES20_Expanded EfficientNet-B0 checkpoints (root)\")\n\n"
        "expected_image = [f\"image_seed{s}_best.pt\" for s in (0, 1, 2)]\n\n"
        "def find_dirs_containing(root, filenames):\n"
        "    \"\"\"Returns {dirpath: set(matched filenames)} for every directory under\n"
        "    root (recursively) that contains at least one of the given filenames.\n"
        "    \"\"\"\n"
        "    matches = {}\n"
        "    for dirpath, _dirnames, files in os.walk(root):\n"
        "        found = set(files) & set(filenames)\n"
        "        if found:\n"
        "            matches[dirpath] = found\n"
        "    return matches\n\n"
        "candidate_dirs = find_dirs_containing(expanded_ckpt_root, expected_image)\n"
        "print(f\"\\nDirectories containing any of {expected_image}:\")\n"
        "for d, found in candidate_dirs.items():\n"
        "    print(f\"  {d}  ->  {sorted(found)}\")\n\n"
        "complete_dirs = [d for d, found in candidate_dirs.items() if found == set(expected_image)]\n"
        "if len(complete_dirs) == 1:\n"
        "    resolved_expanded_dir = complete_dirs[0]\n"
        "elif len(complete_dirs) > 1:\n"
        "    # Ambiguous (e.g. duplicated across a backup folder) - print all and\n"
        "    # pick the first deterministically rather than silently guessing;\n"
        "    # re-check the print-out above if this looks wrong.\n"
        "    print(f\"\\nWARNING: {len(complete_dirs)} directories all contain the complete \"\n"
        "          \"set - picking the first. Verify this is the intended one:\")\n"
        "    for d in complete_dirs:\n"
        "        print(f\"  candidate: {d}\")\n"
        "    resolved_expanded_dir = sorted(complete_dirs)[0]\n"
        "else:\n"
        "    resolved_expanded_dir = None\n\n"
        "assert resolved_expanded_dir is not None, (\n"
        "    \"Step 3 PAD_UFES20_Expanded EfficientNet-B0 checkpoints "
        "(image_seed{0,1,2}_best.pt) not found anywhere under the dataset root, "
        "in any single directory together - image embedder warm-start cannot "
        "proceed. These are DIFFERENT files from Step 4's "
        "image_convnext_tiny_.../image_densenet121_... checkpoints - confirm "
        "they were actually added to the uploaded dataset before running "
        "this notebook. See the directory listing printed above for what was "
        "actually found and where.\"\n"
        ")\n\n"
        "print(\"\\nOK: raw images present, processed dataset wrapped as expected, \"\n"
        "      f\"Stage 1 metadata checkpoints found at: {resolved_stage1_dir}, \"\n"
        "      f\"Step 3 EfficientNet-B0 checkpoints found at: {resolved_expanded_dir}\")\n"
        "```\n\n---\n\n"
    )

    parts.append(
        "## Cell 2 — Setup / os.makedirs / copy Step 3 EfficientNet-B0 checkpoints into place\n\n"
        "```python\n"
        "import os, sys, shutil\n\n"
        "os.makedirs(\"/kaggle/working/src/models\", exist_ok=True)\n"
        "os.makedirs(\"/kaggle/working/src/evaluation\", exist_ok=True)\n"
        "open(\"/kaggle/working/src/__init__.py\", \"w\").close()\n"
        "open(\"/kaggle/working/src/models/__init__.py\", \"w\").close()\n"
        "open(\"/kaggle/working/src/evaluation/__init__.py\", \"w\").close()\n\n"
        "sys.path.insert(0, \"/kaggle/working\")\n\n"
        "# PAD_UFES20_Expanded has no Kaggle-slug indirection in config.py (unlike\n"
        "# PAD_UFES20/HAM10000's stage1_checkpoints_dir) - its checkpoints_dir is a\n"
        "# plain /kaggle/working/logs/PAD_UFES20_Expanded/checkpoints path, so the\n"
        "# 3 Step 3 EfficientNet-B0 checkpoints must be copied there manually before\n"
        "# train_cross_attention_efficientnet_expanded.py's warm-start lookup can find them.\n"
        "expanded_ckpt_dst = \"/kaggle/working/logs/PAD_UFES20_Expanded/checkpoints\"\n"
        "os.makedirs(expanded_ckpt_dst, exist_ok=True)\n"
        "for s in (0, 1, 2):\n"
        "    name = f\"image_seed{s}_best.pt\"\n"
        "    shutil.copy(os.path.join(resolved_expanded_dir, name), os.path.join(expanded_ckpt_dst, name))\n"
        "print(\"copied 3 Step 3 EfficientNet-B0 checkpoints ->\", expanded_ckpt_dst)\n"
        "print(sorted(os.listdir(expanded_ckpt_dst)))\n\n"
        "print(\"\\nsetup OK, sys.path[0]:\", sys.path[0])\n"
        "```\n\n---\n\n"
    )

    cell_num = 3
    for source_name, kaggle_path in FILES:
        parts.append(writefile_cell(cell_num, source_name, kaggle_path))
        parts.append("\n---\n\n")
        cell_num += 1

    parts.append(
        f"## Cell {cell_num} — Sanity check (config load + both checkpoint sets + real image path resolution)\n\n"
        "```python\n"
        "import sys\n"
        "for mod in list(sys.modules):\n"
        "    if mod.startswith(\"src.\"):\n"
        "        del sys.modules[mod]\n\n"
        "import pandas as pd\n"
        "from src.models.config import get_dataset, resolve_image_path\n\n"
        "ds_config = get_dataset(\"PAD_UFES20\")\n"
        "expanded_ds_config = get_dataset(\"PAD_UFES20_Expanded\")\n"
        "print(\"num_classes:\", ds_config.num_classes)\n"
        "print(\"train_csv:\", ds_config.train_csv, \"exists:\", ds_config.train_csv.exists())\n"
        "print(\"val_csv:  \", ds_config.val_csv, \"exists:\", ds_config.val_csv.exists())\n"
        "print(\"test_csv: \", ds_config.test_csv, \"exists:\", ds_config.test_csv.exists())\n"
        "assert ds_config.train_csv.exists(), \"metadata_train.csv not found - check processed dataset nesting\"\n\n"
        "print(\"\\nstage1_checkpoints_dir:\", ds_config.stage1_checkpoints_dir)\n"
        "for seed in (0, 1, 2):\n"
        "    p = ds_config.stage1_checkpoints_dir / f\"metadata_seed{seed}_best.pt\"\n"
        "    assert p.exists(), f\"Missing Stage 1 metadata checkpoint: {p}\"\n"
        "print(\"all 3 Stage 1 metadata checkpoints resolved OK\")\n\n"
        "print(\"\\nexpanded checkpoints_dir:\", expanded_ds_config.checkpoints_dir)\n"
        "for seed in (0, 1, 2):\n"
        "    p = expanded_ds_config.checkpoints_dir / f\"image_seed{seed}_best.pt\"\n"
        "    assert p.exists(), f\"Missing Step 3 EfficientNet-B0 checkpoint: {p}\"\n"
        "print(\"all 3 Step 3 EfficientNet-B0 checkpoints resolved OK\")\n\n"
        "df = pd.read_csv(ds_config.train_csv)\n"
        "sample_image_path = df.iloc[0][\"image_path\"]\n"
        "resolved = resolve_image_path(sample_image_path)\n"
        "print(\"\\nsample image_path (from CSV):\", sample_image_path)\n"
        "print(\"resolved filesystem path:    \", resolved)\n"
        "print(\"resolved path exists:        \", resolved.exists())\n"
        "assert resolved.exists(), f\"Resolved image path does not exist: {resolved}\"\n\n"
        "import random\n"
        "random.seed(0)\n"
        "sample_rows = df.sample(n=min(20, len(df)), random_state=0)\n"
        "missing = []\n"
        "for _, row in sample_rows.iterrows():\n"
        "    p = resolve_image_path(row[\"image_path\"])\n"
        "    if not p.exists():\n"
        "        missing.append((row[\"image_path\"], str(p)))\n"
        "print(f\"\\nChecked {len(sample_rows)} random rows, missing: {len(missing)}\")\n"
        "if missing:\n"
        "    print(\"First few missing:\", missing[:5])\n"
        "assert not missing, \"Some resolved image paths do not exist - check folder-verification cell output\"\n\n"
        "print(\"\\nSANITY CHECK PASSED\")\n"
        "```\n\n---\n\n"
    )
    cell_num += 1

    parts.append(
        f"## Cell {cell_num} — Full model/GPU/dependency check (cross-attention model, expanded warm-start, one real batch)\n\n"
        "```python\n"
        "import torch, sys\n"
        "print(\"python:\", sys.version)\n"
        "print(\"torch:\", torch.__version__)\n"
        "print(\"cuda available:\", torch.cuda.is_available())\n"
        "if torch.cuda.is_available():\n"
        "    print(\"device:\", torch.cuda.get_device_name(0))\n\n"
        "from src.models.config import get_dataset\n"
        "from src.models.dataset import FusionDataset, MetadataPreprocessor\n"
        "from src.models.cross_attention_fusion_model import CrossAttentionFusionModel\n"
        "import pandas as pd\n\n"
        "ds_config = get_dataset(\"PAD_UFES20\")\n"
        "expanded_ds_config = get_dataset(\"PAD_UFES20_Expanded\")\n"
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n\n"
        "preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))\n"
        "model = CrossAttentionFusionModel(metadata_input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes)\n\n"
        "image_ckpt = expanded_ds_config.checkpoints_dir / \"image_seed0_best.pt\"\n"
        "metadata_ckpt = ds_config.stage1_checkpoints_dir / \"metadata_seed0_best.pt\"\n"
        "model.load_stage1_checkpoints(image_ckpt, metadata_ckpt, device)\n"
        "model.to(device)\n"
        "model.eval()  # single-sample batch below - BatchNorm1d needs eval mode, not train mode, to run on batch size 1\n"
        "print(\"Step 3 expanded EfficientNet-B0 + Stage 1 metadata checkpoints (seed 0) loaded OK\")\n\n"
        "train_ds = FusionDataset(ds_config.train_csv, ds_config, preprocessor, train=True)\n"
        "image, metadata, label = train_ds[0]\n"
        "assert image.shape == (3, 224, 224)\n"
        "assert metadata.shape == (preprocessor.output_dim,)\n\n"
        "with torch.no_grad():\n"
        "    out = model(image.unsqueeze(0).to(device), metadata.unsqueeze(0).to(device))\n"
        "assert out.shape == (1, ds_config.num_classes)\n"
        "print(\"cross-attention model forward pass OK, output shape:\", out.shape)\n\n"
        "print(\"\\nALL CHECKS PASSED - ready to train\")\n"
        "```\n\n---\n\n"
    )
    cell_num += 1

    for seed in SEEDS:
        parts.append(
            f"## Cell {cell_num} — Train: cross-attention (EfficientNet-B0), "
            f"dataset-expansion-only ablation, seed {seed}\n\n"
            "```python\n"
            f"!python -m src.models.train_cross_attention_efficientnet_expanded --seed {seed}\n"
            "```\n\n"
        )
        if seed != SEEDS[-1]:
            parts.append("---\n\n")
        cell_num += 1

    return "".join(parts)


def main():
    OUT_PATH.write_text(build_notebook(), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
