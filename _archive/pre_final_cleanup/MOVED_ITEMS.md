# Moved Items — Pre-Final Cleanup (2026-08-03)

Fully reversible. Nothing was deleted — everything below was moved with
`git mv` (or plain `mv`), preserving relative structure under
`_archive/pre_final_cleanup/`. To undo, move a file back to the path shown
in "From".

| Item | From | To | Why |
|---|---|---|---|
| Master project document (PDF) | `docs/📘 MASTER PROJECT DOCUMENT.pdf` | `_archive/pre_final_cleanup/docs/MASTER PROJECT DOCUMENT.pdf` | Duplicate/superseded artifact of the same document already archived as `_archive/MASTER_PROJECT_DOCUMENT.md` (which carries an explicit `SUPERSEDED — see docs/Project_Tracking.md` banner, last updated 2026-07-09, predates Phases 6-8). This PDF was the un-flagged original still sitting live in `docs/` (dated 2026-07-11) and would have confused anyone opening `docs/` looking for the current status. |

## Reviewed and deliberately KEPT in place (not moved)

- `scripts/isic_mirror_verification_cell.py`, `scripts/isic_full_verification_cell.py`,
  `scripts/isic_archive2_id_comparison_cell.py` — look like one-off diagnostic
  Kaggle cells at first glance, but they are **actively cited by name** as
  provenance/evidence trail in currently-live files: `src/models/config.py:141`,
  `scripts/generate_external_isic_kaggle_notebook.py`, and 5 generated
  `notebooks/*.md` files (comments referencing "root cause confirmed via
  scripts/isic_archive2_id_comparison_cell.py"). Moving them would orphan
  those citations. Left in place per the "if uncertain, leave and flag" rule.
- `scripts/generate_*.py` (8 notebook generators) — each has a live
  corresponding `notebooks/*.md` output still part of the active
  documentation trail (methodology reproducibility: notebooks are generated
  from real `.py` source, not hand-typed). None are orphaned/superseded.
- `papers/1.pdf`, `papers/2.pdf`, `papers/3.pdf` — non-descriptive filenames,
  worth renaming to their actual titles at some point, but content/relevance
  not independently verified in this pass. Flagged only, not moved.
- `docs/Dataset_Strategy.md`, `docs/Dataset_Preparation_Final_Report.md`,
  `docs/PROJECT_OWNERSHIP.md`, `docs/PROJECT_PLAN.md` — overlapping scope
  with newer docs (`Project_AZ_Reference.md`, `Project_Tracking.md`) but each
  is a distinct, dated, non-contradictory report (not stale drafts — those
  were already archived per `PROJECT_PLAN.md`'s own cleanup log). Kept as
  historical record; `THESIS_OWNERSHIP_MASTER.md` is now the single
  up-to-date entry point so no one needs to reconcile them by hand.
- Everything under `data/`, `logs/`, active `src/models/*.py`,
  `notebooks/pad_ufes20_cross_attention_joint_kaggle_notebook.md`, and the
  two `generate_cross_attention_*joint*`/`train_cross_attention_joint_fusion.py`
  files — untouched, per explicit instruction not to touch anything related
  to the currently-running Phase 8E training.
