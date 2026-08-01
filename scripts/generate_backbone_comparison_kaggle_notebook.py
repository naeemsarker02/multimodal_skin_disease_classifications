"""Generates notebooks/pad_ufes20_expanded_backbone_comparison_kaggle_notebook.md
(Phase 8B Step 3 - 5-backbone comparison, image branch only, on
PAD-UFES-20-Expanded).

Reads the actual source files from src/models/ and embeds their real
content into %%writefile cells - same convention as every other
notebook generator in this project (never hand-typed, cannot silently
drift from the source-of-truth .py files).

Run: python scripts/generate_backbone_comparison_kaggle_notebook.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src" / "models"
OUT_PATH = PROJECT_ROOT / "notebooks" / "pad_ufes20_expanded_backbone_comparison_kaggle_notebook.md"

sys.path.insert(0, str(PROJECT_ROOT))
from src.models.config import KAGGLE_DATASET_SLUGS, KAGGLE_PROCESSED_SLUGS  # noqa: E402

DERM_OWNER, DERM_SLUG = KAGGLE_DATASET_SLUGS["DERM12345"]
MEDNODE_OWNER, MEDNODE_SLUG = KAGGLE_DATASET_SLUGS["MED-NODE"]
PROCESSED_OWNER, PROCESSED_SLUG = KAGGLE_PROCESSED_SLUGS["PAD_UFES20_Expanded"]

FILES = [
    ("config.py", "/kaggle/working/src/models/config.py"),
    ("dataset.py", "/kaggle/working/src/models/dataset.py"),
    ("backbones.py", "/kaggle/working/src/models/backbones.py"),
    ("metadata_model.py", "/kaggle/working/src/models/metadata_model.py"),
    ("train.py", "/kaggle/working/src/models/train.py"),
]

BACKBONES = ["efficientnet_b0", "mobilenet_v3_large", "densenet121", "resnet50", "convnext_tiny"]
SEEDS = (0, 1, 2)


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
        "# PAD-UFES-20-Expanded Kaggle Notebook — Phase 8B Step 3 "
        "(5-Backbone Comparison, Image Branch Only)\n\n"
        "> **WARNING — read before pasting any cell into Kaggle:** for every "
        "`%%writefile` cell below, `%%writefile <path>` MUST be the exact first "
        "line of the Kaggle cell, with absolutely nothing above it — no blank "
        "line, no comment, no stray character. **After pasting each "
        "`%%writefile` cell, re-open it and visually confirm `%%writefile` is "
        "line 1 before running it.**\n\n"
        "Structured for \"Save & Run All (Commit)\", same process as every "
        "prior notebook in this project. Requires **four** Kaggle \"Add Data\" "
        "sources attached before running:\n\n"
        "1. `mahdavi1202/skin-cancer` (raw PAD-UFES-20 image mirror) — "
        "**already used by every prior notebook, should already exist.**\n"
        f"2. `{DERM_OWNER}/{DERM_SLUG}` (raw DERM12345) — uploaded and "
        "published 2026-07-29.\n"
        f"3. `{MEDNODE_OWNER}/{MEDNODE_SLUG}` (raw MED-NODE) — uploaded and "
        "published 2026-07-29.\n"
        f"4. `{PROCESSED_OWNER}/{PROCESSED_SLUG}` (processed "
        "PAD_UFES20_Expanded) — uploaded and published 2026-07-29.\n\n"
        "This notebook's `%%writefile` cells and folder-verification cell "
        "(Cell 1) embed the real `(owner, slug)` values from "
        "`src/models/config.py`'s `KAGGLE_DATASET_SLUGS`/`KAGGLE_PROCESSED_SLUGS` "
        "as of generation time — re-run "
        "`scripts/generate_backbone_comparison_kaggle_notebook.py` any time "
        "those change to keep this notebook in sync.\n\n"
        "15 runs total: 5 backbones (EfficientNet-B0, MobileNetV3-Large, "
        "DenseNet121, ResNet50, ConvNeXt-Tiny) × 3 seeds, `--dataset "
        "PAD_UFES20_Expanded --branch image` (the `image_branch_only=True` "
        "guard added 2026-07-29 makes any other `--branch` value fail loudly "
        "for this dataset — see `Project_Tracking.md`, \"Pre-Step-3 Wiring "
        "Verification\").\n\n"
        "Paste each cell below into a separate Kaggle notebook cell, in order.\n\n"
        "---\n\n"
    )

    parts.append(
        "## Cell 1 — Folder verification (raw PAD-UFES-20 mirror + DERM12345 "
        "+ MED-NODE + processed PAD_UFES20_Expanded)\n\n"
        "```python\n"
        "import os\n\n"
        "def show(path, label):\n"
        "    print(f\"--- {label}: {path} ---\")\n"
        "    if not os.path.isdir(path):\n"
        "        print(\"  !! NOT FOUND\")\n"
        "        return\n"
        "    for entry in sorted(os.listdir(path))[:20]:\n"
        "        full = os.path.join(path, entry)\n"
        "        kind = \"dir\" if os.path.isdir(full) else \"file\"\n"
        "        print(f\"  [{kind}] {entry}\")\n\n"
        "raw_root = \"/kaggle/input/datasets/mahdavi1202/skin-cancer\"\n"
        "show(\"/kaggle/input/datasets\", \"datasets root\")\n"
        "show(raw_root, \"raw PAD-UFES-20 mirror\")\n\n"
        f"derm_root = \"/kaggle/input/datasets/{DERM_OWNER}/{DERM_SLUG}\"\n"
        f"mednode_root = \"/kaggle/input/datasets/{MEDNODE_OWNER}/{MEDNODE_SLUG}\"\n"
        f"processed_root = \"/kaggle/input/datasets/{PROCESSED_OWNER}/{PROCESSED_SLUG}\"\n"
        "show(derm_root, \"DERM12345 raw\")\n"
        "show(mednode_root, \"MED-NODE raw\")\n"
        "show(processed_root, \"PAD_UFES20_Expanded processed\")\n\n"
        "assert os.path.isdir(raw_root), \"raw PAD-UFES-20 mirror not found — check Add Data\"\n"
        "assert os.path.isdir(derm_root), \"DERM12345 raw dataset not found — check Add Data\"\n"
        "assert os.path.isdir(mednode_root), \"MED-NODE raw dataset not found — check Add Data\"\n"
        "assert os.path.isdir(processed_root), \"PAD_UFES20_Expanded processed dataset not found — check Add Data\"\n"
        "print(\"\\nOK — all 4 Add Data sources found\")\n\n"
        "# Sample-file resolution check (added 2026-07-30 after a MED-NODE\n"
        "# FileNotFoundError mid-training — resolve_image_path()'s candidate\n"
        "# order tried here directly, against one real CSV row per new\n"
        "# dataset, so any Kaggle-mount nesting mismatch is caught here\n"
        "# instead of after an epoch has already started).\n"
        "def check_candidates(root, dataset_dir, sample_rel, strip_prefix=None):\n"
        "    print(f\"--- resolving sample for {dataset_dir}: {sample_rel} ---\")\n"
        "    top, _, rest = sample_rel.partition(\"/\")\n"
        "    candidates = [\n"
        "        os.path.join(root, top, top, rest),\n"
        "        os.path.join(root, dataset_dir, sample_rel),\n"
        "        os.path.join(root, sample_rel),\n"
        "    ]\n"
        "    # Mirrors resolve_image_path()'s KAGGLE_REST_STRIP_PREFIX case: a\n"
        "    # leading segment our image_path values expect (e.g.\n"
        "    # \"complete_mednode_dataset\") that this specific mount omits\n"
        "    # entirely - opposite of the doubled/wrapped candidates above.\n"
        "    if strip_prefix and top == strip_prefix and rest:\n"
        "        s_top, _, s_rest = rest.partition(\"/\")\n"
        "        candidates.append(os.path.join(root, s_top, s_top, s_rest))\n"
        "        candidates.append(os.path.join(root, dataset_dir, rest))\n"
        "        candidates.append(os.path.join(root, rest))\n"
        "    found_any = False\n"
        "    for c in candidates:\n"
        "        exists = os.path.exists(c)\n"
        "        found_any = found_any or exists\n"
        "        print(f\"  [{'FOUND' if exists else 'missing'}] {c}\")\n"
        "    if not found_any:\n"
        "        print(\"  !! none of the expected candidate paths exist — \"\n"
        "              \"resolve_image_path() will fail for this dataset\")\n"
        "    return found_any\n\n"
        "mednode_ok = check_candidates(mednode_root, \"MED-NODE\", "
        "\"complete_mednode_dataset/melanoma/136733.jpg\", "
        "strip_prefix=\"complete_mednode_dataset\")\n"
        "derm_ok = check_candidates(derm_root, \"DERM12345\", \"images/alm/DERM_767235.jpg\")\n"
        "assert mednode_ok, \"No candidate path resolved a MED-NODE sample file — check Add Data / mount layout\"\n"
        "assert derm_ok, \"No candidate path resolved a DERM12345 sample file — check Add Data / mount layout\"\n"
        "```\n\n---\n\n"
    )

    parts.append(
        "## Cell 2 — Setup / os.makedirs\n\n"
        "```python\n"
        "import os, sys\n\n"
        "os.makedirs(\"/kaggle/working/src/models\", exist_ok=True)\n"
        "os.makedirs(\"/kaggle/working/src/evaluation\", exist_ok=True)\n"
        "open(\"/kaggle/working/src/__init__.py\", \"w\").close()\n"
        "open(\"/kaggle/working/src/models/__init__.py\", \"w\").close()\n"
        "open(\"/kaggle/working/src/evaluation/__init__.py\", \"w\").close()\n\n"
        "sys.path.insert(0, \"/kaggle/working\")\n"
        "print(\"setup OK, sys.path[0]:\", sys.path[0])\n"
        "```\n\n---\n\n"
    )

    cell_num = 3
    for source_name, kaggle_path in FILES:
        parts.append(writefile_cell(cell_num, source_name, kaggle_path))
        parts.append("\n---\n\n")
        cell_num += 1

    parts.append(
        f"## Cell {cell_num} — Sanity check (config load + real image path "
        "resolution, all 3 sources)\n\n"
        "```python\n"
        "import sys\n"
        "for mod in list(sys.modules):\n"
        "    if mod.startswith(\"src.\"):\n"
        "        del sys.modules[mod]\n\n"
        "import pandas as pd\n"
        "from src.models.config import get_dataset, resolve_image_path\n\n"
        "ds_config = get_dataset(\"PAD_UFES20_Expanded\")\n"
        "print(\"num_classes:\", ds_config.num_classes)\n"
        "print(\"image_branch_only:\", ds_config.image_branch_only)\n"
        "print(\"train_csv:\", ds_config.train_csv, \"exists:\", ds_config.train_csv.exists())\n"
        "print(\"val_csv:  \", ds_config.val_csv, \"exists:\", ds_config.val_csv.exists())\n"
        "print(\"test_csv: \", ds_config.test_csv, \"exists:\", ds_config.test_csv.exists())\n"
        "assert ds_config.train_csv.exists(), \"metadata_train_image_only.csv not found — check processed dataset nesting\"\n\n"
        "df = pd.read_csv(ds_config.train_csv)\n"
        "print(\"\\ntrain rows:\", len(df))\n"
        "print(df[\"dataset_source\"].value_counts())\n"
        "print(\"\\nclass counts:\")\n"
        "print(df[\"disease_label\"].value_counts())\n\n"
        "val_df = pd.read_csv(ds_config.val_csv)\n"
        "print(\"\\nval rows:\", len(val_df), \"| val dataset_source values:\", val_df[\"dataset_source\"].unique())\n"
        "assert set(val_df[\"dataset_source\"].unique()) == {\"PAD_UFES20\"}, (\n"
        "    \"val split contains a non-PAD_UFES20 row — the train-only-not-val rule is broken, STOP\"\n"
        ")\n\n"
        "import random\n"
        "random.seed(0)\n"
        "sample_rows = df.sample(n=min(30, len(df)), random_state=0)\n"
        "missing = []\n"
        "for _, row in sample_rows.iterrows():\n"
        "    p = resolve_image_path(row[\"image_path\"])\n"
        "    if not p.exists():\n"
        "        missing.append((row[\"dataset_source\"], row[\"image_path\"], str(p)))\n"
        "print(f\"\\nChecked {len(sample_rows)} random rows (mixed sources), missing: {len(missing)}\")\n"
        "if missing:\n"
        "    print(\"First few missing:\", missing[:5])\n"
        "assert not missing, \"Some resolved image paths do not exist — check folder-verification cell output\"\n\n"
        "print(\"\\nSANITY CHECK PASSED\")\n"
        "```\n\n---\n\n"
    )
    cell_num += 1

    parts.append(
        f"## Cell {cell_num} — Full model/GPU/dependency check (each backbone, "
        "one real batch, guard verification)\n\n"
        "```python\n"
        "import torch, sys\n"
        "print(\"python:\", sys.version)\n"
        "print(\"torch:\", torch.__version__)\n"
        "print(\"cuda available:\", torch.cuda.is_available())\n"
        "if torch.cuda.is_available():\n"
        "    print(\"device:\", torch.cuda.get_device_name(0))\n\n"
        "from src.models.config import get_dataset\n"
        "from src.models.dataset import ImageDataset\n"
        "from src.models.backbones import BACKBONE_NAMES, build_backbone\n\n"
        "ds_config = get_dataset(\"PAD_UFES20_Expanded\")\n"
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n\n"
        "train_ds = ImageDataset(ds_config.train_csv, ds_config, train=True)\n"
        "image, label = train_ds[0]\n"
        "assert image.shape == (3, 224, 224)\n\n"
        "print(\"BACKBONE_NAMES:\", BACKBONE_NAMES)\n"
        "for name in BACKBONE_NAMES:\n"
        "    model = build_backbone(name, num_classes=ds_config.num_classes).to(device)\n"
        "    model.eval()\n"
        "    with torch.no_grad():\n"
        "        out = model(image.unsqueeze(0).to(device))\n"
        "    assert out.shape == (1, ds_config.num_classes), f\"{name}: unexpected output shape {out.shape}\"\n"
        "    print(f\"{name}: forward pass OK, output shape {out.shape}\")\n"
        "    del model\n\n"
        "# Confirm the image_branch_only guard actually fires for this dataset\n"
        "from src.models.train import train_one_run\n"
        "try:\n"
        "    train_one_run(\"PAD_UFES20_Expanded\", \"metadata\", seed=0)\n"
        "    raise AssertionError(\"BUG: image_branch_only guard did not fire\")\n"
        "except ValueError as e:\n"
        "    print(\"\\nimage_branch_only guard fired correctly:\", str(e)[:100], \"...\")\n\n"
        "print(\"\\nALL CHECKS PASSED — ready to train\")\n"
        "```\n\n---\n\n"
    )
    cell_num += 1

    for backbone in BACKBONES:
        parts.append(f"### Backbone: `{backbone}`\n\n")
        for seed in SEEDS:
            parts.append(
                f"## Cell {cell_num} — Train: {backbone}, seed {seed}\n\n"
                "```python\n"
                f"!python -m src.models.train --dataset PAD_UFES20_Expanded --branch image "
                f"--backbone {backbone} --seed {seed}\n"
                "```\n\n---\n\n"
            )
            cell_num += 1

    return "".join(parts)


def main():
    OUT_PATH.write_text(build_notebook(), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
