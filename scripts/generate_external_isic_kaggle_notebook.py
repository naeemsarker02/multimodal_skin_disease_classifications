"""Generates notebooks/external_isic_evaluation_kaggle_notebook.md.

Reads the actual Phase 8 source files from src/models/ and
src/evaluation/evaluate_external_isic.py and embeds their real content
into %%writefile cells - same discipline as the other generate_*.py
scripts (e.g. generate_cross_attention_kaggle_notebook.py): the notebook
is never hand-typed, so it cannot silently drift from the source-of-truth
.py files.

This is an evaluation-only notebook (HAM10000-trained Stage 1 checkpoints
run zero-shot against the two ISIC archives, evaluate_external_isic.py's
Protocol A) - no training happens here, so there is no Stage-1-checkpoint
*output* to preserve; the 6 pre-trained HAM10000 Stage 1 checkpoints are
an *input* (Kaggle "Add Data" source), read-only.

Requires 6 Kaggle "Add Data" sources - all previously verified via
scripts/isic_full_verification_cell.py (2026-07-27) and
scripts/isic_mirror_verification_cell.py:
  1. nodoubttome/skin-cancer9-classesisic   (ISIC Archive 1 raw mirror, double-nested)
  2. andrewmvd/isic-2019                    (ISIC Archive 2 raw mirror, flat)
  3. naeemsarkertracer/isic-archive1-processed  (flat)
  4. naeemsarkertracer/isic-archive2-processed  (flat)
  5. naeemsarkertracer/ham10000-processed       (wrapped - needed for HAM10000's
     own train_csv, only read by the metadata-variant cells' preprocessor fit)
  6. naeemsarkertracer/ham10000-stage1-checkpoints  (flat - the actual model
     weights being evaluated)

Run: python scripts/generate_external_isic_kaggle_notebook.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_SRC = PROJECT_ROOT / "src" / "models"
EVAL_SRC = PROJECT_ROOT / "src" / "evaluation"
OUT_PATH = PROJECT_ROOT / "notebooks" / "external_isic_evaluation_kaggle_notebook.md"

# (source dir, source file, path to %%writefile on Kaggle)
FILES = [
    (MODELS_SRC, "config.py", "/kaggle/working/src/models/config.py"),
    (MODELS_SRC, "dataset.py", "/kaggle/working/src/models/dataset.py"),
    (MODELS_SRC, "image_model.py", "/kaggle/working/src/models/image_model.py"),
    (MODELS_SRC, "metadata_model.py", "/kaggle/working/src/models/metadata_model.py"),
    (EVAL_SRC, "evaluate_external_isic.py", "/kaggle/working/src/evaluation/evaluate_external_isic.py"),
]


def read_source(src_dir: Path, name: str) -> str:
    return (src_dir / name).read_text(encoding="utf-8").rstrip("\n")


def writefile_cell(cell_num: int, src_dir: Path, source_name: str, kaggle_path: str) -> str:
    code = read_source(src_dir, source_name)
    return (
        f"## Cell {cell_num} — `%%writefile {kaggle_path}`\n\n"
        f"```python\n%%writefile {kaggle_path}\n{code}\n```\n"
    )


def build_notebook() -> str:
    parts = []

    parts.append(
        "# External ISIC Evaluation Kaggle Notebook — Phase 8 "
        "(HAM10000 → ISIC Archive 1/2, zero-shot, Protocol A)\n\n"
        "> **WARNING - read before pasting any cell into Kaggle:** for every "
        "`%%writefile` cell below, `%%writefile <path>` MUST be the exact first "
        "line of the Kaggle cell, with absolutely nothing above it - no blank "
        "line, no comment, no stray character. Kaggle (like Jupyter) only "
        "treats a line as a cell magic if it is the first line of the cell; "
        "anything preceding it turns `%%writefile` into a plain, non-magic "
        "line and the file silently never gets written. **After pasting each "
        "`%%writefile` cell, re-open it and visually confirm `%%writefile` is "
        "line 1 before running it.**\n\n"
        "Evaluation-only notebook: no training happens here. HAM10000's "
        "already-finalized Stage 1 baseline checkpoints (image + metadata, "
        "seeds 0-2) are run zero-shot against ISIC Archive 1 and ISIC "
        "Archive 2, per `src/evaluation/evaluate_external_isic.py`'s "
        "Protocol A (native, unmodified 7-class argmax; scoring restricted "
        "to exact-string-match shared classes). Structured for \"Save & Run "
        "All (Commit)\" so all 9 evaluation runs and their "
        "`reports/HAM10000/external_isic/` outputs are preserved as the "
        "notebook's Kaggle Output.\n\n"
        "Requires **six** Kaggle \"Add Data\" sources attached before "
        "running, all previously verified via "
        "`scripts/isic_full_verification_cell.py` and "
        "`scripts/isic_mirror_verification_cell.py` (2026-07-27):\n\n"
        "- `nodoubttome/skin-cancer9-classesisic` (ISIC Archive 1 raw image "
        "mirror - double-nested under \"Skin cancer ISIC The International "
        "Skin Imaging Collaboration/\")\n"
        "- `andrewmvd/isic-2019` (ISIC Archive 2 raw image mirror - flat)\n"
        "- `naeemsarkertracer/isic-archive1-processed` (our processed "
        "metadata CSVs - flat layout)\n"
        "- `naeemsarkertracer/isic-archive2-processed` (our processed "
        "metadata CSVs - flat layout)\n"
        "- `naeemsarkertracer/ham10000-processed` (HAM10000's own processed "
        "metadata CSVs - needed to fit the metadata preprocessor for the "
        "Archive 2 metadata-variant runs; wrapped layout)\n"
        "- `naeemsarkertracer/ham10000-stage1-checkpoints` (published — "
        "https://www.kaggle.com/datasets/naeemsarkertracer/ham10000-stage1-checkpoints), "
        "the actual `image_seed{0,1,2}_best.pt` / `metadata_seed{0,1,2}_best.pt` "
        "weights being evaluated - flat layout)\n\n"
        "Paste each cell below into a separate Kaggle notebook cell, in order.\n\n"
        "---\n\n"
    )

    parts.append(
        "## Cell 1 — Folder verification (all 6 sources, confirmed layouts)\n\n"
        "```python\n"
        "import os, glob\n\n"
        "def show(path, label):\n"
        "    print(f\"--- {label}: {path} ---\")\n"
        "    if not os.path.isdir(path):\n"
        "        print(\"  !! NOT FOUND\")\n"
        "        return\n"
        "    for entry in sorted(os.listdir(path)):\n"
        "        full = os.path.join(path, entry)\n"
        "        kind = \"dir\" if os.path.isdir(full) else \"file\"\n"
        "        print(f\"  [{kind}] {entry}\")\n\n"
        "show(\"/kaggle/input/datasets\", \"datasets root\")\n\n"
        "# --- 1. ISIC Archive 1 raw mirror (double-nested, confirmed 2026-07-27) --\n"
        "archive1_root = \"/kaggle/input/datasets/nodoubttome/skin-cancer9-classesisic\"\n"
        "show(archive1_root, \"ISIC Archive 1 raw (root)\")\n"
        "archive1_nested = os.path.join(archive1_root, \"Skin cancer ISIC The International Skin Imaging Collaboration\")\n"
        "show(archive1_nested, \"ISIC Archive 1 raw (double-nested candidate)\")\n"
        "assert os.path.isdir(os.path.join(archive1_nested, \"Train\")) and os.path.isdir(os.path.join(archive1_nested, \"Test\")), (\n"
        "    \"Expected double-nested Train/Test not found - check archive1_nested path above\"\n"
        ")\n\n"
        "# --- 2. ISIC Archive 2 raw mirror (flat, confirmed 2026-07-27) -----------\n"
        "archive2_root = \"/kaggle/input/datasets/andrewmvd/isic-2019\"\n"
        "show(archive2_root, \"ISIC Archive 2 raw (root)\")\n"
        "assert os.path.isdir(os.path.join(archive2_root, \"ISIC_2019_Training_Input\")), (\n"
        "    \"ISIC_2019_Training_Input/ not found under Archive 2 raw root\"\n"
        ")\n\n"
        "# --- 3. ISIC Archive 1 processed (flat, confirmed 2026-07-27) -----------\n"
        "isic1_proc_root = \"/kaggle/input/datasets/naeemsarkertracer/isic-archive1-processed\"\n"
        "show(isic1_proc_root, \"ISIC Archive 1 processed (root)\")\n"
        "assert os.path.isfile(os.path.join(isic1_proc_root, \"metadata_train.csv\")), (\n"
        "    \"Expected flat metadata_train.csv not found at ISIC Archive 1 processed root\"\n"
        ")\n\n"
        "# --- 4. ISIC Archive 2 processed (flat, confirmed 2026-07-27) -----------\n"
        "isic2_proc_root = \"/kaggle/input/datasets/naeemsarkertracer/isic-archive2-processed\"\n"
        "show(isic2_proc_root, \"ISIC Archive 2 processed (root)\")\n"
        "assert os.path.isfile(os.path.join(isic2_proc_root, \"metadata_train.csv\")), (\n"
        "    \"Expected flat metadata_train.csv not found at ISIC Archive 2 processed root\"\n"
        ")\n\n"
        "# --- 5. HAM10000 processed (wrapped, published 2026-07-15) --------------\n"
        "ham_proc_root = \"/kaggle/input/datasets/naeemsarkertracer/ham10000-processed\"\n"
        "show(ham_proc_root, \"HAM10000 processed (root)\")\n"
        "ham_proc_wrapped = os.path.join(ham_proc_root, \"HAM10000\")\n"
        "show(ham_proc_wrapped, \"HAM10000 processed (wrapped candidate)\")\n"
        "assert (\n"
        "    os.path.isfile(os.path.join(ham_proc_wrapped, \"metadata_train.csv\"))\n"
        "    or os.path.isfile(os.path.join(ham_proc_root, \"metadata_train.csv\"))\n"
        "), \"metadata_train.csv not found at either HAM10000 processed candidate\"\n\n"
        "# --- 6. HAM10000 Stage 1 checkpoints (flat, confirmed 2026-07-27) -------\n"
        "ham_ckpt_root = \"/kaggle/input/datasets/naeemsarkertracer/ham10000-stage1-checkpoints\"\n"
        "show(ham_ckpt_root, \"HAM10000 Stage 1 checkpoints (root)\")\n"
        "expected_ckpts = [f\"{b}_seed{s}_best.pt\" for b in (\"image\", \"metadata\") for s in (0, 1, 2)]\n"
        "missing_ckpts = [f for f in expected_ckpts if not os.path.isfile(os.path.join(ham_ckpt_root, f))]\n"
        "assert not missing_ckpts, f\"Missing HAM10000 Stage 1 checkpoints at root: {missing_ckpts}\"\n\n"
        "print(\"\\nOK: all 6 Kaggle datasets found with their confirmed layouts.\")\n"
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
    for src_dir, source_name, kaggle_path in FILES:
        parts.append(writefile_cell(cell_num, src_dir, source_name, kaggle_path))
        parts.append("\n---\n\n")
        cell_num += 1

    parts.append(
        f"## Cell {cell_num} — Sanity check (config load + both archives' image "
        "paths + HAM10000 Stage 1 checkpoints + a real ExternalIsicEvalDataset row)\n\n"
        "```python\n"
        "import sys\n"
        "for mod in list(sys.modules):\n"
        "    if mod.startswith(\"src.\"):\n"
        "        del sys.modules[mod]\n\n"
        "import pandas as pd\n"
        "from src.models.config import get_dataset, resolve_image_path\n"
        "from src.evaluation.evaluate_external_isic import (\n"
        "    ARCHIVE_SHARED_CLASSES,\n"
        "    ExternalIsicEvalDataset,\n"
        "    build_ham_eval_preprocessor,\n"
        "    load_archive_metadata,\n"
        "    apply_exclusions,\n"
        ")\n\n"
        "ham_ds_config = get_dataset(\"HAM10000\")\n"
        "print(\"HAM10000 num_classes:\", ham_ds_config.num_classes)\n"
        "print(\"HAM10000 train_csv:\", ham_ds_config.train_csv, \"exists:\", ham_ds_config.train_csv.exists())\n"
        "assert ham_ds_config.train_csv.exists(), \"HAM10000 metadata_train.csv not found - check processed dataset nesting\"\n\n"
        "print(\"\\nHAM10000 stage1_checkpoints_dir:\", ham_ds_config.stage1_checkpoints_dir)\n"
        "for branch in (\"image\", \"metadata\"):\n"
        "    for seed in (0, 1, 2):\n"
        "        p = ham_ds_config.stage1_checkpoints_dir / f\"{branch}_seed{seed}_best.pt\"\n"
        "        assert p.exists(), f\"Missing HAM10000 Stage 1 checkpoint: {p}\"\n"
        "print(\"all 6 HAM10000 Stage 1 checkpoints resolved OK\")\n\n"
        "# --- both ISIC archives' processed metadata + image path resolution -----\n"
        "for archive in (\"ISIC_Archive_1\", \"ISIC_Archive_2\"):\n"
        "    df = load_archive_metadata(archive)\n"
        "    df = apply_exclusions(df, archive)\n"
        "    print(f\"\\n{archive}: {len(df)} rows after exclusions\")\n"
        "    sample_path = df.iloc[0][\"image_path\"]\n"
        "    resolved = resolve_image_path(sample_path)\n"
        "    print(f\"  sample image_path: {sample_path}\")\n"
        "    print(f\"  resolved:           {resolved}\")\n"
        "    print(f\"  exists:             {resolved.exists()}\")\n"
        "    assert resolved.exists(), f\"Resolved image path does not exist: {resolved}\"\n\n"
        "    import random\n"
        "    random.seed(0)\n"
        "    sample_rows = df.sample(n=min(20, len(df)), random_state=0)\n"
        "    missing = [row[\"image_path\"] for _, row in sample_rows.iterrows() if not resolve_image_path(row[\"image_path\"]).exists()]\n"
        "    print(f\"  checked {len(sample_rows)} random rows, missing: {len(missing)}\")\n"
        "    assert not missing, f\"Some resolved image paths do not exist for {archive}: {missing[:5]}\"\n\n"
        "# --- one real ExternalIsicEvalDataset row per shared-class configuration -\n"
        "eval_preprocessor = build_ham_eval_preprocessor(ham_ds_config)\n"
        "print(f\"\\nHAM10000 metadata preprocessor output_dim: {eval_preprocessor.output_dim}\")\n\n"
        "ds_a1_image = ExternalIsicEvalDataset(\"ISIC_Archive_1\", ham_ds_config.label_to_idx, need_metadata=False)\n"
        "print(f\"ISIC_Archive_1 image-eval dataset: {len(ds_a1_image)} rows, shared classes {ARCHIVE_SHARED_CLASSES['ISIC_Archive_1']}\")\n"
        "image, label = ds_a1_image[0]\n"
        "assert image.shape == (3, 224, 224)\n\n"
        "ds_a2_metadata = ExternalIsicEvalDataset(\"ISIC_Archive_2\", ham_ds_config.label_to_idx, need_metadata=True, eval_preprocessor=eval_preprocessor)\n"
        "print(f\"ISIC_Archive_2 metadata-eval dataset: {len(ds_a2_metadata)} rows, shared classes {ARCHIVE_SHARED_CLASSES['ISIC_Archive_2']}\")\n"
        "image, metadata, label = ds_a2_metadata[0]\n"
        "assert metadata.shape == (eval_preprocessor.output_dim,)\n\n"
        "print(\"\\nSANITY CHECK PASSED\")\n"
        "```\n\n---\n\n"
    )
    cell_num += 1

    parts.append(
        f"## Cell {cell_num} — Full model/GPU/dependency check (one real forward "
        "pass per branch type, both HAM10000-checkpoint-loaded models)\n\n"
        "```python\n"
        "import torch, sys\n"
        "print(\"python:\", sys.version)\n"
        "print(\"torch:\", torch.__version__)\n"
        "print(\"cuda available:\", torch.cuda.is_available())\n"
        "if torch.cuda.is_available():\n"
        "    print(\"device:\", torch.cuda.get_device_name(0))\n\n"
        "from src.models.config import get_dataset\n"
        "from src.evaluation.evaluate_external_isic import (\n"
        "    ExternalIsicEvalDataset,\n"
        "    build_ham_eval_preprocessor,\n"
        "    load_model_for_variant,\n"
        ")\n\n"
        "ham_ds_config = get_dataset(\"HAM10000\")\n"
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n\n"
        "# image branch, seed 0, against ISIC Archive 1\n"
        "image_model = load_model_for_variant(\"image\", 0, ham_ds_config, ham_ds_config.num_classes, None, device)\n"
        "ds_a1_image = ExternalIsicEvalDataset(\"ISIC_Archive_1\", ham_ds_config.label_to_idx, need_metadata=False)\n"
        "image, label = ds_a1_image[0]\n"
        "with torch.no_grad():\n"
        "    out = image_model(image.unsqueeze(0).to(device))\n"
        "assert out.shape == (1, ham_ds_config.num_classes)\n"
        "print(\"image-branch (seed 0) forward pass on ISIC Archive 1 OK, output shape:\", out.shape)\n\n"
        "# metadata branch, seed 0, against ISIC Archive 2\n"
        "eval_preprocessor = build_ham_eval_preprocessor(ham_ds_config)\n"
        "metadata_model = load_model_for_variant(\"metadata\", 0, ham_ds_config, ham_ds_config.num_classes, eval_preprocessor.output_dim, device)\n"
        "metadata_model.eval()  # single-sample batch below - BatchNorm1d needs eval mode on batch size 1\n"
        "ds_a2_metadata = ExternalIsicEvalDataset(\"ISIC_Archive_2\", ham_ds_config.label_to_idx, need_metadata=True, eval_preprocessor=eval_preprocessor)\n"
        "image, metadata, label = ds_a2_metadata[0]\n"
        "with torch.no_grad():\n"
        "    out = metadata_model(metadata.unsqueeze(0).to(device))\n"
        "assert out.shape == (1, ham_ds_config.num_classes)\n"
        "print(\"metadata-branch (seed 0) forward pass on ISIC Archive 2 OK, output shape:\", out.shape)\n\n"
        "print(\"\\nALL CHECKS PASSED - ready to evaluate\")\n"
        "```\n\n---\n\n"
    )
    cell_num += 1

    run_specs = (
        [("ISIC_Archive_1", "image", seed) for seed in (0, 1, 2)]
        + [("ISIC_Archive_2", "image", seed) for seed in (0, 1, 2)]
        + [("ISIC_Archive_2", "metadata", seed) for seed in (0, 1, 2)]
    )
    for i, (archive, variant, seed) in enumerate(run_specs):
        parts.append(
            f"## Cell {cell_num} — Evaluate: {archive}, {variant} branch, seed {seed}\n\n"
            "```python\n"
            f"!python -m src.evaluation.evaluate_external_isic --archive {archive} --variant {variant} --seed {seed}\n"
            "```\n\n"
        )
        if i != len(run_specs) - 1:
            parts.append("---\n\n")
        cell_num += 1

    return "".join(parts)


def main():
    OUT_PATH.write_text(build_notebook(), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
