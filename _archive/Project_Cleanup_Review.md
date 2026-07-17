# Project Cleanup & Simplification Review

Generated: 2026-07-08
Status: **Proposal only — nothing has been changed.** Every recommendation below requires your explicit approval before any file is merged, archived, or deleted.

Scope reviewed: full project tree (`docs/`, `data/`, `src/`, `reports/`, `logs/`, `results/`, `experiments/`, `notebooks/`, `papers/`, root files). `.venv/` (local virtual environment) was excluded — it is not project content.

---

## 1. Findings Table

| # | Item | Location | Why it is unnecessary / a problem | Recommendation |
|---|---|---|---|---|
| 1 | `README.md` | root | ~90% content-duplicate of `docs/Research_Plan.md` (research background, objectives, workflow, dataset roles, methodology, timeline, evaluation metrics all repeated near-verbatim). Additionally has a **leftover raw Bengali chat note pasted at the end** (lines 483–521: "আরেকটা ছোট recommendation...") — this is an accidental copy-paste artifact from a prior AI conversation, not real documentation. | **Merge & Clean**: keep `README.md` as the short public-facing overview only (title, 1-paragraph summary, link to `docs/`), delete the pasted chat fragment, and remove content that's fully covered by `Research_Plan.md`/`Project_Tracking.md`. |
| 2 | `docs/Research_Plan.md` | docs | Substantially duplicates `README.md` (same objectives, workflow diagram, dataset table, experiment plan, timeline). Neither file is clearly the "source of truth." | **Merge**: fold the *unique* content (research questions, contributions, evaluation metrics) into a single canonical `Project_Overview.md`; retire this file. |
| 3 | `docs/Dataset_Strategy.md` | docs | Thin, partially **stale** subset of `docs/AI_Assistant_Instructions.md` — e.g. still describes a single "ISIC Archive" (pre-dates the ISIC Archive 1 vs 2 split), and shows a final processed structure (`train.csv`/`val.csv`/`test.csv`) that doesn't match what was actually produced (`metadata_train.csv` etc., per `data/processed/*`). Everything correct in it is already covered, more accurately, in `AI_Assistant_Instructions.md` and `Project_Tracking.md`. | **Merge**: absorb any still-relevant strategy notes into `AI_Assistant_Instructions.md`, then retire this file to avoid two documents disagreeing about the pipeline. |
| 4 | `docs/AI_Assistant_Instructions.md` | docs | Not redundant — this is the operative working directive (role, phase-by-phase pipeline spec, hard rules). Some phase descriptions are now historical (phases are complete), but it's still the reference for "how this project works." | **Keep** (trim completed-phase narrative if desired, but no urgent action). |
| 5 | `docs/Project_Tracking.md` | docs | Up to date, actively maintained, is the real single source of truth for project status. | **Keep** — this should become the canonical "Project Tracking" doc per your target doc set. |
| 6 | `docs/Dataset_Preparation_Final_Report.md` | docs | Distinct, non-duplicated content — an independent critical-findings verification report (cross-dataset image leakage). Already correctly cross-referenced from `Project_Tracking.md`. | **Keep** as-is. |
| 7 | `docs/~$Multimodal Skin Lesion Classification Using Image and Clinical Metadata.xlsx` | docs | Microsoft Office **lock/temp file** (the `~$` prefix marks it as an auto-generated lock file created while the real `.xlsx` is/was open, not real content). | **Delete** — safe, it's not a data file. |
| 8 | `docs/Multimodal Skin Lesion Classification Using Image and Clinical Metadata.xlsx` | docs | Appears to be a literature/dataset tracking spreadsheet. Not obviously duplicated, but its purpose overlaps conceptually with `papers/Skin_Lesion_Related_Papers.docx`. | **Keep** — please confirm its exact purpose (paper tracker vs. dataset inventory) so it can be named clearly; not recommending deletion. |
| 9 | `src/data_audit/config.py` vs `src/data_cleaning/config.py` | src | ~30 lines of **duplicated constants** (all four datasets' raw-dir paths, filenames, image subdirs) copy-pasted between the two files. Risk: if a raw path ever changes, it must be edited in two places or the pipelines silently disagree. | **Merge (code, not yet executed)**: extract shared path constants into one `src/common/paths.py` imported by both `data_audit/config.py` and `data_cleaning/config.py`. Flagged for consolidation only — not refactored, per your instruction to preserve behavior. |
| 10 | `src/data_audit/common/` (`io_utils.py`, `logging_utils.py`) | src | Already a well-factored shared-utility module for the audit pipeline — no duplication found here. | **Keep** as-is; could serve as the model for consolidating `data_cleaning`'s equivalent needs. |
| 11 | `__pycache__/` directories (13 found under `src/`) | src (nested) | Regenerated automatically by Python on every run; contain no source content. | **Delete** — zero risk, will be recreated automatically. |
| 12 | `logs/ISIC_Archive_1/` — 2 audit logs, 2 cleaning logs; `logs/PAD_UFES20/` — 3 cleaning logs | logs | Multiple timestamped runs of the *same* script for the *same* dataset, consistent with iterative dev/debugging rather than distinct transformations. `AI_Assistant_Instructions.md` mandates "keep complete logs of every transformation," so these aren't unambiguously safe to delete. | **Archive**: move all but the latest successful run per dataset/script into `logs/archive/`, keeping the reproducibility trail without cluttering the active log directory. Requires your confirmation since it touches the "never delete logs" principle. |
| 13 | `notebooks/`, `experiments/`, `results/` | root | Currently empty. However, these map directly to upcoming required pipeline stages (EDA, model tracking, evaluation results) that `Project_Tracking.md` shows as "Pending," not abandoned. | **Keep** — these are legitimate forward scaffolding for phases not yet started, not orphaned placeholders. |
| 14 | `data/interim/*`, `data/processed/*` | data | Checked for multiple dataset versions — **only one version of each exists**, no stale/duplicate processed outputs found. | **Keep**, no action needed. |
| 15 | `reports/<dataset>/figures/` | reports | Each contains real generated figures (2 files each), not empty placeholders. | **Keep**. |
| 16 | `requirements.txt` | root | Minimal and accurate for the current (dataset-prep) phase; not bloated. | **Keep** — expect to grow once modeling starts (torch/sklearn etc.), not a cleanup issue now. |
| 17 | `papers/` (3 PDFs + `Skin_Lesion_Related_Papers.docx`) | root | Required literature-review source material and your reviewed-paper summary. | **Keep** — never remove per your reproducibility/research-integrity rule. |
| 18 | `data/raw/*` | data | Untouched raw datasets (2.8G–3.4G each), confirmed read-only by pipeline design. | **Keep** — never touch, per project's own hard rule. |

---

## 2. Proposed Simplified Documentation Set

Target (per your 5-document goal):

| Target file | Sourced from |
|---|---|
| `docs/Project_Overview.md` | Merge of `README.md` (cleaned) + `docs/Research_Plan.md` |
| `docs/AI_Assistant_Instructions.md` | Kept as-is |
| `docs/Dataset_Strategy.md` | Retired — content absorbed into `AI_Assistant_Instructions.md` where still accurate |
| `docs/Project_Tracking.md` | Kept as-is (canonical status doc) |
| `docs/Dataset_Preparation_Final_Report.md` | Kept as-is (stands alone — verification report, not a duplicate) |

Root `README.md` becomes a short pointer/summary (title + 1 paragraph + link to `docs/Project_Overview.md`), not a full duplicate document.

---

## 3. Proposed Simplified Folder Structure

No structural changes are proposed — the existing top-level layout already matches what's required:

```
Multimodal_Skin_Disease_Research/
├── data/            (raw / interim / processed — keep all three)
├── docs/            (5 canonical docs after merge, see §2)
├── logs/            (keep; archive stale re-run logs, see item 12)
├── notebooks/        (keep — reserved for EDA phase)
├── papers/           (keep)
├── reports/          (keep — per-dataset audit reports/figures)
├── experiments/       (keep — reserved for model-dev tracking)
├── results/           (keep — reserved for evaluation outputs)
├── src/
│   ├── common/        (NEW — proposed home for de-duplicated path config, see item 9)
│   ├── data_audit/
│   └── data_cleaning/
├── README.md          (shortened)
└── requirements.txt
```

No folder is recommended for removal — the placeholder folders (`notebooks/`, `experiments/`, `results/`) are legitimate future pipeline stages, not dead weight.

---

## 4. Summary Lists

### Keep (no action)
- `docs/AI_Assistant_Instructions.md`, `docs/Project_Tracking.md`, `docs/Dataset_Preparation_Final_Report.md`
- `data/raw/*`, `data/interim/*`, `data/processed/*`
- `reports/*`, `papers/*`, `requirements.txt`
- `notebooks/`, `experiments/`, `results/` (empty, reserved for upcoming phases)
- `src/data_audit/common/` (already well-factored)
- `docs/Multimodal Skin Lesion Classification Using Image and Clinical Metadata.xlsx` (pending purpose confirmation)

### Merge
- `README.md` + `docs/Research_Plan.md` → single `docs/Project_Overview.md` (+ shortened root `README.md`)
- `docs/Dataset_Strategy.md` → absorbed into `docs/AI_Assistant_Instructions.md`, then retired
- `src/data_audit/config.py` + `src/data_cleaning/config.py` → shared `src/common/paths.py` (code change, deferred until you approve — behavior-preserving only)

### Archive
- Superseded re-run logs in `logs/ISIC_Archive_1/` and `logs/PAD_UFES20/` (keep latest run per script, move earlier attempts to `logs/archive/`)

### Safe to Delete
- `docs/~$Multimodal Skin Lesion Classification Using Image and Clinical Metadata.xlsx` (Office lock file)
- All `__pycache__/` directories under `src/` (13 found; auto-regenerated)

---

## 5. Explicitly Not Touched

Per the research-integrity requirement, none of the following were flagged for any action:
- `data/raw/` (all four datasets)
- `papers/` (all PDFs and the reviewed-paper summary)
- `data/processed/`, `data/interim/` (single, current version each — nothing stale to archive)
- `reports/`, `logs/` content beyond the specific stale re-run logs called out in item 12

---

**Waiting for your approval before any merge, archive, or delete is carried out.** Please confirm or adjust the recommendations above (especially items 7–9, 12, and the doc-merge plan in §2) before I make any changes.
