# Dataset Preparation — Final Cross-Dataset Verification Report

Generated: 2026-07-08

This is an independent, read-only verification pass over the four processed
datasets (`data/processed/PAD_UFES20`, `HAM10000`, `ISIC_Archive_1`,
`ISIC_Archive_2`), performed after each dataset's own audit → cleaning
pipeline was already completed and documented in its own
`dataset_description.md`. No processed or raw file was modified to produce
this report — every check below re-derives its numbers directly from the
CSVs already on disk.

---

## 1. Summary of Every Dataset

| Dataset | Total images | Train / Val / Test | Classes | Imbalance ratio | Grouping key used for split | Patient ID coverage |
|---|---|---|---|---|---|---|
| PAD-UFES-20 | 2,298 | 1,606 / 338 / 354 | 6 | 16.2:1 | `patient_id` (true patient-wise) | 100% (of 2,298 processed) |
| HAM10000 | 10,015 | 7,004 / 1,501 / 1,510 | 7 | 58.3:1 | `lesion_id` (no patient ID exists) | 0% (none available) |
| ISIC Archive 1 | 2,047 (155 images excluded, see §6) | 1,655 / 292 / 100 | 9 | 239:1 | none (archive-provided Train/Test kept, val carved from Train) | 0% (none available) |
| ISIC Archive 2 | 25,076 (255 images excluded, see §6) | 17,535 / 3,769 / 3,772 | 9 | 61.6:1 | `lesion_id` where present, else `image_id` singleton | 1.6% (417/25,331 raw rows, *before* the 255-row exclusion — see note†) |

† **Denominator note (presentation fix, 2026-07-26):** every other column in
this row (Total images, Train/Val/Test) is reported **after** the 255-row
exclusion in §6, i.e. against a base of 25,076. The "Patient ID coverage"
figure is the one exception — it is computed against the 25,331 **raw**,
pre-exclusion rows, because patient-ID coverage was measured before the
exclusion step ran. The two numbers (25,076 vs. 25,331) are not
interchangeable; don't divide 417 by 25,076 or compare this percentage
directly against the other rows' coverage figures without accounting for
the different base.

Each dataset's own `dataset_description.md` and `split_quality_report.csv`
were re-derived independently below rather than taken on trust, and all
numbers matched.

---

## 2. Label Consistency Across Datasets

All four datasets standardize labels into a `disease_label` column. Diseases
shared across sources deliberately use **identical strings**
(`src/data_cleaning/common_label_mapping.py` is the single source of truth
for the two ISIC archives; PAD-UFES-20 and HAM10000's own label-standardization
steps were written to match those same strings for shared diseases).

Verified pairwise exact-string overlap:

| | PAD-UFES-20 | HAM10000 | ISIC Archive 1 | ISIC Archive 2 |
|---|---|---|---|---|
| **PAD-UFES-20** | — | Basal Cell Carcinoma, Melanoma, Nevus | Actinic Keratosis, Basal Cell Carcinoma, Melanoma, Nevus, Seborrheic Keratosis, Squamous Cell Carcinoma | Actinic Keratosis, Basal Cell Carcinoma, Melanoma, Nevus, Seborrheic Keratosis, Squamous Cell Carcinoma |
| **HAM10000** | | — | Basal Cell Carcinoma, Dermatofibroma, Melanoma, Nevus, Vascular Lesion | Basal Cell Carcinoma, Dermatofibroma, Melanoma, Nevus |
| **ISIC Archive 1** | | | — | Actinic Keratosis, Basal Cell Carcinoma, Dermatofibroma, Melanoma, Nevus, Pigmented Benign Keratosis, Seborrheic Keratosis, Squamous Cell Carcinoma |
| **ISIC Archive 2** | | | | — |

**Full union of standardized labels across all four datasets (12 total):**
Actinic Keratosis, Actinic Keratosis / Intraepithelial Carcinoma, Basal Cell
Carcinoma, Benign Keratosis-like Lesion, Dermatofibroma, Melanoma, Nevus,
Pigmented Benign Keratosis, Seborrheic Keratosis, Solar Lentigo, Squamous
Cell Carcinoma, Vascular Lesion.

**Intentional non-mergers (not a bug — confirmed against each dataset's own
`label_mapping.csv` reasoning):**
- HAM10000's `Actinic Keratosis / Intraepithelial Carcinoma` is **not** the
  same string as `Actinic Keratosis` used elsewhere. HAM10000's raw `akiec`
  code groups actinic keratosis together with Bowen's disease (SCC in situ),
  a clinically broader category than PAD-UFES-20/ISIC's narrower "Actinic
  Keratosis". Merging them would silently misrepresent the label.
- HAM10000's `Benign Keratosis-like Lesion` is **not** the same as
  `Seborrheic Keratosis` or `Pigmented Benign Keratosis`. HAM10000's raw
  `bkl` code groups solar lentigines, seborrheic keratoses, and
  lichen-planus-like keratoses together — broader than either ISIC category.

**Recommendation:** if a future experiment wants to merge PAD-UFES-20 and
HAM10000 into a single training pool, `Actinic Keratosis / Intraepithelial
Carcinoma` and `Benign Keratosis-like Lesion` must be treated as their own
classes (not silently collapsed onto the narrower ISIC/PAD names), or
explicitly re-mapped with a documented, reasoned decision at that time —
consistent with the project's "no invented data / no silent relabeling"
principle.

**Minor label disagreement found (not fixed, documented only):** on the
1,673 images shared between ISIC Archive 1 and ISIC Archive 2 (see §6), 3
images carry different `disease_label` values between the two archives
(2× `Seborrheic Keratosis` vs `Melanoma`, 1× `Nevus` vs `Actinic Keratosis`).
This is a genuine small residual label disagreement inherited from the two
source archives' own annotations, not introduced by this project's cleaning.
Volume is negligible (0.18% of the shared images) but should be kept in mind
if these images are ever combined into one training pool.

---

## 3. Metadata Schema Consistency

Column counts and content differ substantially by design — the four sources
provide fundamentally different metadata richness — but every processed
`metadata_{train,val,test}.csv` across all four datasets shares this common
column spine, confirmed present in all 12 files:

`image_id`, `image_path`, `dataset_source`, `disease_label`

| Dataset | Total columns | Clinical/demographic columns beyond the common spine |
|---|---|---|
| PAD-UFES-20 | 29 | 25 (age, sex, lesion/patient IDs, 6 symptom flags, 8 lifestyle/clinical flags, anatomical site, 2 diameters, Fitzpatrick, diagnostic_code) |
| HAM10000 | 10 | 6 (lesion_id, diagnostic_code, diagnosis_confirm_type, age, sex, anatomical_site) |
| ISIC Archive 1 | 6 | 2 (filename, class_label — this archive has no clinical metadata at all) |
| ISIC Archive 2 | 30 | 26 (full 27-column ISIC schema minus isic_id, renamed) |

**Naming consistency check:** `sex` and `anatomical_site` are named
identically in PAD-UFES-20 and HAM10000 (ISIC Archive 2 keeps `sex` but uses
`anatom_site_1..5` / `anatom_site_general` rather than a single
`anatomical_site` — this is intentional, since ISIC Archive 2's site
hierarchy is genuinely richer and collapsing it during cleaning would lose
information; documented in its own `dataset_description.md`, not an
oversight).

**Dtype spot-check:** boolean columns in PAD-UFES-20 round-trip through CSV
as literal `True`/`False`/blank (not native `bool` dtype on reload) — this
is already flagged with an explicit downstream-casting recommendation in
`data/processed/PAD_UFES20/dataset_description.md` and remains accurate.

No schema drift or inconsistent column naming was found beyond the
intentional, documented differences above.

---

## 4. Folder Structure Consistency

All four `data/processed/<Dataset>/` folders follow the same generated-file
convention:
- `metadata_train.csv`, `metadata_val.csv`, `metadata_test.csv`
- `label_mapping.csv`
- `split_quality_report.csv`
- `dataset_description.md`

Dataset-specific extras exist where the split method required them
(`patient_split_assignment.csv` for PAD-UFES-20, `lesion_split_assignment.csv`
for HAM10000, `group_split_assignment.csv` for ISIC Archive 2 — ISIC Archive
1 has none, since it reused the archive's own provided Train/Test split
rather than computing a new grouping). This is consistent, not a defect —
each extra file corresponds exactly to that dataset's documented split
method.

`data/raw/` was confirmed untouched (all cleaning/audit code routes through
`assert_not_raw_path`, and no raw file's mtime or content was altered by
this verification).

**Image path resolution check:** a random sample of 50 images per dataset
(200 total) had their `image_path` values checked for existence on disk —
0 missing files across all four datasets.

---

## 5. Split Integrity

Each dataset's `split_quality_report.csv` was **independently
re-derived** (not just re-read) directly from the four datasets'
`metadata_{train,val,test}.csv` files, using the grouping key appropriate to
each dataset:

| Dataset | Grouping key re-checked | Result |
|---|---|---|
| PAD-UFES-20 | `patient_id` | No overlap between train/val/test — confirmed |
| HAM10000 | `lesion_id` | No overlap between train/val/test — confirmed |
| ISIC Archive 1 | `filename` | No overlap between train/val/test — confirmed |
| ISIC Archive 2 | `lesion_id` (falling back to `image_id`) | No overlap between train/val/test — confirmed |

All four datasets are **internally leakage-free** on the grouping key their
own cleaning pipeline used. Split ratios are close to the intended
70/15/15 (PAD-UFES-20, HAM10000, ISIC Archive 2) or an intentional
archive-preserving variant (ISIC Archive 1: 80.9/14.3/4.9, because its own
provided Test set — kept untouched — is much smaller than 15%).

**This section only checks leakage *within* each dataset's own splits.**
Cross-dataset leakage is a separate, and materially more serious, finding —
see §6.

---

## 6. Cross-Dataset Statistics & Possible Data Leakage — CRITICAL FINDING

This is the most important result of this verification pass and was not
caught by any single dataset's own audit, because each dataset was audited
in isolation.

### 6.1 Massive image overlap between HAM10000, ISIC Archive 1, and ISIC Archive 2

Independently checking `image_id` overlap across all processed datasets
(after each dataset's own cleaning/exclusions):

| Pair | Shared `image_id`s | % of smaller dataset |
|---|---|---|
| HAM10000 ↔ ISIC Archive 2 | **9,873** | **98.6% of HAM10000** |
| ISIC Archive 1 ↔ ISIC Archive 2 | **1,673** | **81.7% of ISIC Archive 1** |
| HAM10000 ↔ ISIC Archive 1 | **1,362** | **66.5% of ISIC Archive 1** |
| PAD-UFES-20 ↔ any other dataset | **0** | — (genuinely independent source) |

**Interpretation:** HAM10000 is not an independent dataset from ISIC Archive
2 — it is almost entirely a *subset* of it (HAM10000 was itself contributed
to the ISIC Archive, which ISIC Archive 2 appears to bulk-export from).
ISIC Archive 1 is also heavily built from the same underlying image pool.
PAD-UFES-20 is confirmed to be the only source with zero image overlap
against the other three.

### 6.2 Consequence: cross-dataset splits are NOT leakage-free

Of the 9,873 images shared between HAM10000 and ISIC Archive 2, **4,588
(46%) are assigned to *different* splits in the two datasets** (e.g. an
image is in HAM10000's `train` but ISIC Archive 2's `test`, or vice versa),
because each dataset's split was computed independently without knowledge
of the other. The same problem exists for the other overlapping pairs:

| Pair | Shared images | Landing in different splits across the pair |
|---|---|---|
| HAM10000 ↔ ISIC Archive 2 | 9,873 | 4,588 (46%) |
| HAM10000 ↔ ISIC Archive 1 | 1,362 | 559 (41%) |
| ISIC Archive 1 ↔ ISIC Archive 2 | 1,673 | 667 (40%) |

**Why this matters:** the README (§6) and `Dataset_Strategy.md` designate
ISIC Archive 1 & 2 as the **external validation** dataset, used to test
generalization of a model trained on PAD-UFES-20/HAM10000. Given the overlap
above, any evaluation that trains on HAM10000 and validates/tests on ISIC
Archive 1 or 2 (or vice versa) **is not a valid measure of external
generalization** for the ~40–46% of images that are literally identical
files seen during training. This would silently inflate reported "external
validation" performance.

PAD-UFES-20 is unaffected — it has zero overlap with any other dataset and
remains a genuinely independent source.

### 6.3 What is NOT affected

- Each dataset's own internal train/val/test split (§5) is still valid in
  isolation — the leakage only appears when combining datasets.
- PAD-UFES-20-only experiments (Experiment 1–4 as currently scoped for the
  primary multimodal model) are unaffected.
- The label mismatches on overlapping images (§2) are a secondary, much
  smaller issue compared to the leakage risk itself.

---

## 7. Remaining Risks Before EDA and Model Development

1. **(Critical, new)** Cross-dataset image overlap (§6) must be resolved
   before HAM10000 and the ISIC archives are used together in any
   train/external-validation protocol. This is the single blocking issue
   found by this verification.
2. **(Already tracked, unresolved)** PAD-UFES-20's `biopsed` metadata field
   has not yet had the leakage/shortcut audit that `Project_Tracking.md`'s
   "Next Milestone" already calls for (per Watson et al., 2026) — still
   outstanding, not part of this verification's scope.
3. **(Documented, not a defect)** Severe class imbalance in every dataset
   (16:1 to 239:1) — already known, needs to inform loss weighting /
   sampling strategy during model development, not dataset preparation.
4. **(Documented, not a defect)** ISIC Archive 1's exclusion of 155
   conflicting-label images and ISIC Archive 2's exclusion of 255
   no-diagnosis rows (§1) reduce those datasets' usable size; both are
   already justified and documented in their own `dataset_description.md`.
5. **(Minor)** The 3 label disagreements between ISIC Archive 1 and ISIC
   Archive 2 on shared images (§2) — negligible volume, worth a one-line
   caveat if those images are used in a merged pool.

---

## 8. Overall Readiness Assessment

| Criterion | Status |
|---|---|
| Each dataset internally consistent, documented, reproducible (fixed seed) | ✅ Pass |
| Each dataset's own split leakage-free | ✅ Pass |
| Label taxonomy shared consistently where diseases overlap | ✅ Pass (with documented, intentional non-mergers) |
| Metadata schema consistent within each dataset's own scope | ✅ Pass |
| Folder/file structure consistent across all four datasets | ✅ Pass |
| Image paths resolve to real files | ✅ Pass (sampled) |
| **Cross-dataset (HAM10000 / ISIC Archive 1 / ISIC Archive 2) independence** | ❌ **Fail — see §6** |
| PAD-UFES-20 independence from all other sources | ✅ Pass |

---

## 9. Recommendations Before EDA

1. **Do not use HAM10000 and ISIC Archive 1/2 together in any
   train-vs-external-validation protocol until the overlap in §6 is
   resolved.** Two viable fixes, either is acceptable, both need a decision
   recorded in `Project_Tracking.md`:
   - **(a) Global de-duplication:** build a cross-dataset `image_id` dedup
     step that treats HAM10000/ISIC Archive 1/ISIC Archive 2 as one pool,
     assigns every physical image to exactly one split across all three
     datasets, then regenerates each dataset's train/val/test files from
     that single global assignment.
   - **(b) Restrict "external validation" claims:** only use the subset of
     ISIC Archive 1/2 images that do **not** overlap with whatever was used
     in training (i.e., filter external-validation sets to the
     non-overlapping images identified in §6 before evaluating), and report
     that filtering explicitly in the thesis methodology.
2. PAD-UFES-20 can proceed to EDA and Baseline Model Development
   immediately — it is unaffected by the cross-dataset overlap issue.
3. Complete the already-tracked `biopsed` leakage/shortcut audit for
   PAD-UFES-20 before building its metadata-only baseline (Next Milestone
   item, unchanged by this report).
4. When EDA begins, treat the 12-label union (§2) as the full taxonomy
   surface, but be explicit in any cross-dataset figure/table about which
   labels are shared strings vs. intentionally distinct (HAM10000's two
   broader categories).

---

## Verdict

**Dataset preparation cannot yet be declared fully complete for the project
as a whole.** Each individual dataset's preparation (audit → cleaning →
internal split) is complete, correct, and reproducible on its own terms.
However, this verification found a previously-undetected **critical
cross-dataset issue**: HAM10000, ISIC Archive 1, and ISIC Archive 2 share
between 40–99% of their images with each other, and none of the three
datasets' splits account for this when computed independently — undermining
the planned use of the ISIC archives as an *external* validation set against
HAM10000.

- **PAD-UFES-20** dataset preparation: **COMPLETE**, safe to proceed to EDA
  and Baseline Model Development now.
- **HAM10000 / ISIC Archive 1 / ISIC Archive 2**: dataset preparation is
  **complete only in isolation**; a follow-up decision (§9, item 1) is
  required before these three are used together in any cross-dataset
  training/validation protocol. Until that decision is made and implemented,
  the overall "Dataset Preparation" phase should remain **open, not closed**,
  specifically for the external-validation use case described in the README.
