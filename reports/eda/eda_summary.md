# Phase 5 EDA — Summary

Generated: 2026-07-09. Covers all 4 datasets' `metadata_train.csv` (val/test
only used for split-balance checks) restricted to each dataset's
`feature_whitelist.md` columns + `disease_label` (+ `fitspatrick` for
PAD-UFES-20). Full detail, figures, and CSVs are in `reports/eda/<Dataset>/`;
narrative walkthroughs are in `notebooks/01-05_eda_*.ipynb`.

## Class distribution and imbalance

| Dataset | Train images | Classes | Majority class | Minority class | Imbalance ratio |
|---|---|---|---|---|---|
| PAD-UFES-20 | 1,606 | 6 | Basal Cell Carcinoma (586) | Melanoma (38) | ~15:1 |
| HAM10000 | 7,004 | 7 | Nevus (4,693) | Dermatofibroma (79) | ~59:1 |
| ISIC Archive 1 | 1,655 | 9 | Pigmented Benign Keratosis (393) | Seborrheic Keratosis (1) | ~393:1 |
| ISIC Archive 2 | 17,535 | 9 | Nevus (9,011) | Solar Lentigo (142) | ~63:1 |

Every dataset needs class-weighted loss / minority-aware sampling in Phase 6, per `PROJECT_PLAN.md`. ISIC Archive 1's Seborrheic Keratosis (1 image in train) is a known consequence of its 155-conflicting-pair exclusion during cleaning, not a bug.

## Demographics and metadata usability

- **PAD-UFES-20** has the richest metadata (21 whitelisted features) but ~34-35% of patients are missing `sex`, both diameters, `fitspatrick`, and most lifestyle/socioeconomic fields together (co-missing, likely one incomplete-intake subgroup) — symptom flags (`itch`/`hurt`/`bleed`/`elevation`) are >99.5% complete.
- **HAM10000** has only 3 usable features (`age`, `sex`, `anatomical_site`), all >97% complete.
- **ISIC Archive 1** has 0 usable metadata features — any model on this archive alone is necessarily image-only.
- **ISIC Archive 2** has 13 whitelisted features but most are sparse: `age_approx`/`sex`/`anatom_site_1` are usable (<10% missing), while `anatom_site_3`-`5`, `family_hx_mm`, `personal_hx_mm`, `clin_size_long_diam_mm`, `dermoscopic_type` are 88-100% missing — a Phase 6 decision (impute-and-flag vs. drop) is needed for these, not resolved here.

## Fitzpatrick

Only PAD-UFES-20 has a Fitzpatrick column (~34% missing, same subgroup as the other lifestyle fields). Cross-tab with `disease_label` saved for the Phase 8 fairness analysis. The other 3 datasets have no Fitzpatrick data at all.

## Image dimensions — the key Phase 6 input for resolution choice

| Dataset | Native resolution pattern | Aspect ratio | File size range |
|---|---|---|---|
| PAD-UFES-20 | Highly heterogeneous smartphone captures, 167px-3,120px | ~1.0 (square-cropped), max 1.16 | 44KB-12.4MB |
| HAM10000 | **Perfectly uniform**, 600x450 for all 250 sampled | 1.333, zero variance | 104KB-387KB |
| ISIC Archive 1 | Mixed — 600x450 dominant (163/250) but up to 3,872x2,592 | 0.8-1.5 | 25KB-1.5MB |
| ISIC Archive 2 | **Bimodal** — 600x450 cluster (103/250) + 1024x1024/1024x768 cluster (128/250), up to 6,641x4,401 | 1.0-1.5 | 15KB-1.9MB |

ISIC Archive 2's bimodality lines up exactly with its two `attribution` values (ViDIR Group Vienna ≈ HAM10000 resolution; Hospital Clínic de Barcelona ≈ higher resolution) — direct visual confirmation of the documented cross-dataset image-source overlap, independent of the `image_id`-based overlap check already done. **Implication for Phase 6:** a single fixed resize target is unavoidable across all 4 datasets; PAD-UFES-20 and ISIC Archive 2 will lose the most native resolution information regardless of the chosen target size.

## Cross-dataset shared-label comparison (descriptive only)

9 of 12 total union `disease_label` values appear in 2+ datasets. PAD-UFES-20 is the outlier in scale for its own frequent classes but contributes far fewer Melanoma/Nevus images than the other 3. Vascular Lesion appears only in HAM10000/ISIC Archive 1; Pigmented Benign Keratosis only in the two ISIC archives. **This chart is descriptive only** — HAM10000/ISIC Archive 1/ISIC Archive 2 share 40-99% of the same physical images (`Dataset_Preparation_Final_Report.md` §6), so it must never be read as 4 independent training pools without applying `external_validation_exclusions.csv`.

## Split balance

All 4 datasets' train/val/test class proportions stay close to their target split ratios (visual confirmation of what `split_quality_report.csv` already established numerically) — no new leakage or skew found.

## Open items surfaced by this EDA (not yet decided, flagging for Phase 6 planning)

1. ISIC Archive 2's sparse anatomical-site-hierarchy and clinical-history fields (`anatom_site_3-5`, `family_hx_mm`, `personal_hx_mm`, `clin_size_long_diam_mm`) need an explicit impute-and-flag-vs-drop decision before metadata-only/fusion modeling.
2. Input resolution choice for Phase 6 should account for the heterogeneity found above, not just default to a common CNN input size without discussion.
