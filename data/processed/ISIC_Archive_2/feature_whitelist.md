# ISIC Archive 2 — Model-Input Feature Whitelist

Generated: 2026-07-08. Updated 2026-07-09 (sparse-field source-leak
exclusion — see below). This is the single reference for feature selection
in any future model. `metadata_train.csv` has 30 total columns; 7 are
allowed as model input, of which 4 are the active Phase 6 Stage 1 baseline.

Note: this whitelist excludes more columns than `docs/PROJECT_PLAN.md`
literally listed. Several extensions were made during this audit and are
recorded here and in `Project_Tracking.md` rather than applied silently —
see the "(extension...)" reason tags below.

## Allowed features (7)

`age_approx`, `anatom_site_1`, `anatom_site_2`, `anatom_site_general`,
`anatom_site_special`, `dermoscopic_type`, `sex`

**Active Phase 6 Stage 1 baseline (4 of the 7 above):** `age_approx`,
`sex`, `anatom_site_1`, `anatom_site_general` — chosen for reliable
population (1.5–10.2% missing) and no attribution-correlation signal
found. `anatom_site_2` (52.7% missing), `anatom_site_special`, and
`dermoscopic_type` (94.4% missing) remain allowed-in-principle but are
deferred out of the Stage 1 active feature set pending their own
missingness/attribution check, not yet run.

## Excluded columns and reasons

| Column | Reason |
|---|---|
| `image_id` | identifier |
| `lesion_id` | identifier |
| `patient_id` | identifier |
| `image_path` | path, not a feature |
| `dataset_source` | constant tag within this dataset's files, not discriminative |
| `disease_label` | the prediction target itself |
| `diagnosis_1` | label-source — coarsest level of the same diagnostic hierarchy `disease_label` was derived from (fully populated, 3 unique values: e.g. Benign/Malignant/Indeterminate) |
| `diagnosis_2` | label-source — mid level of the same hierarchy (fully populated, 8 unique values) |
| `diagnosis_3` | label-source — verified 1:1 with `disease_label` (every `diagnosis_3` value maps to exactly one `disease_label`) |
| `diagnosis_4` | label-source (extension beyond PROJECT_PLAN.md's literal list — 96.11% missing when absent, but a finer level of the same diagnostic hierarchy when present; same reasoning as `diagnosis_1`-`diagnosis_3`) |
| `diagnosis_5` | label-source (extension beyond PROJECT_PLAN.md's literal list — 98.35% missing when absent, finest level of the same hierarchy when present; same reasoning) |
| `diagnosis_confirm_type` | leakage feature — verified: malignant cases (Basal Cell Carcinoma/Melanoma/Squamous Cell Carcinoma) are **never** confirmed via "serial imaging showing no change" or "single image expert consensus" (0/8,473), and 90.7% (7,685/8,473) are confirmed via histopathology vs. 55.4% (9,205/16,603) for non-malignant. Phi=0.36, chi2=3171.74 (n=25,076) — real but weaker than PAD-UFES-20's `biopsed`. |
| `concomitant_biopsy` | leakage feature (extension — not named in PROJECT_PLAN.md). Produces the **identical** contingency table to `diagnosis_confirm_type == "histopathology"` (788/7,685 malignant, 7,398/9,205 non-malignant; same phi=0.36) — appears to be a duplicate encoding of the same confirmation-method signal, not independent information. Excluded for the same reason as `diagnosis_confirm_type`. |
| `melanocytic` | leakage feature (extension — not named in PROJECT_PLAN.md, confirmed with you before excluding). **Perfect deterministic split** of `disease_label`: Melanoma and Nevus are 100% `melanocytic=True` (17,395/17,395), all other 7 classes are 100% `melanocytic=False` (7,681/7,681), zero exceptions. A coarser restatement of the label itself, not a genuine pre-diagnosis clinical measurement. |
| `anatom_site_3` | source-leak risk (extension, confirmed via crosstab 2026-07-09 — see `Project_Tracking.md`). 87.56% missing overall; present for 0.00% of Hospital Clínic de Barcelona rows vs. 23.45%/27.75% for ViDIR/Anonymous — presence is a near-perfect proxy for source institution, same category as `attribution`. |
| `anatom_site_4` | source-leak risk (extension, confirmed via crosstab 2026-07-09). 99.05% missing overall; present for 0.00% of Hospital Clínic and ViDIR rows, only ever populated (8.20%) for Anonymous rows. |
| `anatom_site_5` | source-leak risk (extension, confirmed via crosstab 2026-07-09). 99.84% missing overall; present for 0.00% of Hospital Clínic and ViDIR rows, only ever populated (1.34%) for Anonymous rows. |
| `family_hx_mm` | source-leak risk (extension, confirmed via crosstab 2026-07-09). 97.83% missing overall; present for 0.00% of Hospital Clínic and ViDIR rows, only ever populated (18.75%) for Anonymous rows. |
| `personal_hx_mm` | source-leak risk (extension, confirmed via crosstab 2026-07-09). 97.79% missing overall; present for 0.00% of Hospital Clínic and ViDIR rows, only ever populated (19.10%) for Anonymous rows. |
| `clin_size_long_diam_mm` | source-leak risk (extension, confirmed via crosstab 2026-07-09). 97.79% missing overall; present for 0.00% of Hospital Clínic and ViDIR rows, only ever populated (19.13%) for Anonymous rows. |
| `attribution` | non-clinical, source-identifying (extension, judgment call — flagging for your review). Only 3 values; "ViDIR Group... Medical University of Vienna" = exactly 9,873 rows, matching this project's already-documented HAM10000↔ISIC-Archive-2 image overlap count exactly. This column identifies which sub-collection an image came from, not a clinical feature — and correlates with the cross-dataset overlap already handled via `external_validation_exclusions.csv`. |
| `copyright_license` | non-clinical, source-identifying (extension, judgment call). Only 2 values, directly redundant with `attribution` (`CC-0` co-occurs exactly with `attribution="Anonymous"`, 2,901/2,901). |
| `image_type` | zero-variance (extension, uncontroversial). Constant `"dermoscopic"` for all 25,076 rows — carries no information. |

Retained in the CSV for documentation/audit purposes only: `diagnosis_1`
through `diagnosis_5`, `diagnosis_confirm_type`, `concomitant_biopsy`,
`melanocytic`, `attribution`, `copyright_license`, `image_type`. Never pass
any of these into a model.

**Flag for your review:** the `attribution`/`copyright_license`/`image_type`
exclusions were my own judgment call (not identifier/path/label-source/
leakage in the strict PROJECT_PLAN.md sense — a 5th category: "non-clinical/
administrative, not a genuine feature"). If you'd rather keep
`copyright_license` or `image_type` in as harmless constants/near-constants,
that's a low-stakes reversal; `attribution` I'd push back on including, given
its exact correlation with the documented HAM10000 overlap.
