# Phase 8 — HAM10000 → ISIC External Validation Results

**Date:** 2026-07-27
**Protocol:** A — full native 7-way argmax (never masked to the shared classes), scored only on the classes shared between HAM10000 and each ISIC archive's taxonomy (exact-string match only — see "Class taxonomy" below). Zero-shot transfer — both branches are trained only on HAM10000 and never see either ISIC archive during training; weights are frozen for this evaluation. Approved 2026-07-25, "Proposed ISIC External Validation Scope" in `Project_Tracking.md`.
**Script:** `src/evaluation/evaluate_external_isic.py`
**Models:** `image` and `metadata` Stage 1 branches only, 3 seeds each — the only branches trained on HAM10000 (`late_fusion`/`cross_attention` are PAD-UFES-20-only, Phase 7 scope, never trained on HAM10000).
**Archives, evaluated separately, never pooled:** ISIC Archive 1 (image branch only — 0 usable metadata columns) and ISIC Archive 2 (both branches). 9 total runs: 3 seeds × (1 branch × Archive 1 + 2 branches × Archive 2).
**Exclusions applied per archive (union of two lists):** `external_validation_exclusions.csv` (drops images already seen by the HAM10000-trained model) and `label_conflict_exclusions.csv` (drops the 3 images with disagreeing cross-archive ground truth).
**Raw per-run output:** `reports/HAM10000/external_isic/eval_{archive}_{variant}_seed{seed}.json`, per-row predictions `reports/HAM10000/external_isic/predictions_{archive}_{variant}_seed{seed}.csv`

---

## Class taxonomy — restricted to exact-string-matching shared classes

| | HAM10000 ∩ Archive 1 | HAM10000 ∩ Archive 2 |
|---|---|---|
| Shared classes (exact string match) | Basal Cell Carcinoma, Dermatofibroma, Melanoma, Nevus, Vascular Lesion (5) | Basal Cell Carcinoma, Dermatofibroma, Melanoma, Nevus (4 — Archive 2 has no Vascular Lesion class) |

Known, deliberately-unresolved naming/granularity mismatches (HAM10000's `"Actinic Keratosis / Intraepithelial Carcinoma"` vs. both archives' `"Actinic Keratosis"`; HAM10000's coarse `"Benign Keratosis-like Lesion"` vs. the archives' finer keratosis-family categories) are **not** merged into the shared-class set — consistent with this project's "don't guess, exclude or ask" precedent. Logged as a documented future-work item.

---

## Headline results — macro-F1 (shared classes)

| Archive | Variant | seed0 | seed1 | seed2 | mean | std (population) | spillover rate (mean) |
|---|---|---|---|---|---|---|---|
| Archive 1 (n=678) | image | 0.2563 | 0.2275 | 0.2426 | **0.2421** | 0.0118 | 19.1% |
| Archive 2 (n=12,508) | image | 0.5029 | 0.4908 | 0.4799 | **0.4912** | 0.0094 | 23.0% |
| Archive 2 (n=12,508) | metadata | 0.2009 | 0.2710 | 0.2510 | **0.2410** | 0.0295 | 27.3% |

For reference, HAM10000-internal (within-dataset) macro-F1 and PAD→HAM cross-dataset transfer numbers are reported in `docs/Phase8_CrossDataset_Generalization_Results.md`. As with the PAD→HAM direction, both ISIC transfer numbers sit well below HAM10000's own internal performance — the expected generalization-gap direction.

---

## Per-class F1 (shared classes)

### Archive 1 — image

| seed | Basal Cell Carcinoma | Dermatofibroma | Melanoma | Nevus | Vascular Lesion |
|---|---|---|---|---|---|
| 0 | 0.0000 | 0.1538 | 0.4314 | 0.6964 | 0.0000 |
| 1 | 0.0000 | 0.0000 | 0.4507 | 0.6869 | 0.0000 |
| 2 | 0.0000 | 0.1538 | 0.4085 | 0.6507 | 0.0000 |
| **mean** | **0.0000** | **0.1026** | **0.4302** | **0.6780** | **0.0000** |

### Archive 2 — image

| seed | Basal Cell Carcinoma | Dermatofibroma | Melanoma | Nevus |
|---|---|---|---|---|
| 0 | 0.6299 | 0.2264 | 0.4106 | 0.7449 |
| 1 | 0.5620 | 0.2007 | 0.4372 | 0.7634 |
| 2 | 0.5438 | 0.1439 | 0.4681 | 0.7636 |
| **mean** | **0.5786** | **0.1904** | **0.4386** | **0.7573** |

### Archive 2 — metadata

| seed | Basal Cell Carcinoma | Dermatofibroma | Melanoma | Nevus |
|---|---|---|---|---|
| 0 | 0.0000 | 0.0537 | 0.2196 | 0.5303 |
| 1 | 0.2402 | 0.0531 | 0.1505 | 0.6403 |
| 2 | 0.2847 | 0.0569 | 0.0635 | 0.5988 |
| **mean** | **0.1750** | **0.0546** | **0.1445** | **0.5898** |

**Archive 1's true-label support, verified directly from each seed's confusion matrix, is 0 for both Basal Cell Carcinoma and Vascular Lesion (0 of 678 eval rows), and only 7 for Dermatofibroma** — despite all 3 being in Archive 1's shared-class list (taxonomy overlap, not eval-set composition). Basal Cell Carcinoma's and Vascular Lesion's 0.0 F1 on Archive 1 is a direct consequence of zero true instances (undefined recall, scored 0 under `zero_division=0`), not a model failure — this is the same class-support caveat already established for HAM10000-internal and PAD→HAM class imbalance, just more extreme here (0 support rather than merely few). Dermatofibroma's 7-row support also explains its wide seed range (0.0–0.15) — a single flipped prediction moves its F1 by a large increment at that support size. Archive 2 (12,508 rows) has real support for Basal Cell Carcinoma (2,809), Dermatofibroma (124), Melanoma (3,407), and Nevus (6,168) — its Basal Cell Carcinoma and Melanoma F1 scores are genuine measurements, not support artifacts. Nevus is the most stable class across every archive/branch combination (0.53–0.76).

---

## Spillover — predictions falling outside the shared classes

"Spillover" = the model predicted a HAM10000 class not in the archive's shared-class list on a true-shared-class image. Under Protocol A this always counts as wrong; reported separately so the failure mode is visible rather than hidden inside the F1 numbers.

| Archive | Variant | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|---|
| Archive 1 | image | 20.4% (138/678) | 18.9% (128/678) | 18.0% (122/678) | 19.1% |
| Archive 2 | image | 26.2% (3,277/12,508) | 23.8% (2,977/12,508) | 19.0% (2,378/12,508) | 23.0% |
| Archive 2 | metadata | 32.3% (4,035/12,508) | 22.6% (2,824/12,508) | 27.2% (3,399/12,508) | 27.3% |

Metadata's spillover rate is both higher on average and more seed-variable than image's on Archive 2, mirroring the PAD→HAM finding that metadata-informed branches are the weaker generalizer, not a stronger, more conservative one.

---

## Bootstrap significance testing (2026-07-27) — Archive 2, image vs. metadata

**Scope:** ISIC Archive 2 only — the only archive where both branches were evaluated (Archive 1 is image-only, no comparison possible). Single comparison, so no Bonferroni correction is needed.

**Method:** paired bootstrap over the 12,508 Archive 2 eval rows (with replacement, 1,000 resamples, RNG seed 42, fixed and distinct from the 3 model-training seeds). The same resampled row-index set is applied to both variants within a given iteration (paired design — verified image/metadata predictions share identical row order before resampling). Per iteration, macro-F1 (4 shared classes) is computed separately for each of a variant's 3 seeds on the resampled rows, then averaged across the 3 seeds. Statistic: `diff = score(image) - score(metadata)`; 95% CI via the percentile method; two-sided p-value from the proportion of resamples whose sign disagrees with the observed sign, doubled. Script: `src/evaluation/bootstrap_significance_isic.py`. Per-row predictions consumed from `reports/HAM10000/external_isic/predictions_ISIC_Archive_2_{variant}_seed{seed}.csv`. Full output: `reports/HAM10000/external_isic/bootstrap_significance_isic_archive2.json`.

| Comparison | Observed diff | 95% CI | p-value (2-sided) | Sig. α=0.05 |
|---|---|---|---|---|
| image vs. metadata | **+0.2502** | **[+0.2368, +0.2641]** | **0.000** | **Yes** |

**Conclusion:** image's advantage over metadata on Archive 2 is real and highly significant — the CI excludes 0 by a wide margin (+0.24 to +0.26), consistent with the PAD→HAM direction's finding that the metadata-only branch is a materially weaker generalizer than image or any image-informed branch. This is the largest observed-diff / narrowest-relative-CI significance result of any bootstrap comparison run in this project to date, reflecting both the size of the true effect and Archive 2's large row count (12,508 vs. 1,253 for PAD→HAM) tightening the CI.

---

## Reading the results

- **Metadata generalizes far worse than image on Archive 2** (0.2410 vs. 0.4912 mean, a statistically significant −0.25 gap) — the same qualitative direction as the PAD→HAM result (metadata 0.2920 vs. image 0.4658), and here the effect is both larger in absolute terms and confirmed on 10× more eval rows.
- **Archive 1 (image-only, 0.2421 mean) scores far below Archive 2's image branch (0.4912 mean)** despite using the identical checkpoints — driven by Archive 1's eval-set composition, verified above: 2 of its 5 shared classes (Basal Cell Carcinoma, Vascular Lesion) have **zero** true instances in the 678-row filtered eval set, so their 0.0 F1 is baked into every seed's macro average regardless of model quality, and Dermatofibroma's 7-row support adds further noise. This is a between-archive eval-set-composition effect, not evidence that the same model performs differently on "the same task" — Archive 2's macro-F1 (0.4912) is the more representative measurement of this checkpoint's true transfer quality, since 3 of its 4 shared classes have hundreds-to-thousands of true rows.
- **Spillover confirms the same pattern seen in per-class F1**: metadata's higher, more variable spillover rate (27.3% mean, range 22.6–32.3%) versus image's lower, tighter rate (23.0% mean, range 19.0–26.2%) on Archive 2 is consistent with metadata carrying a weaker, noisier transfer signal — not a coincidence isolated to macro-F1.
- **Both ISIC transfer directions underperform HAM10000's own internal macro-F1** by a wide margin (see `docs/Phase8_CrossDataset_Generalization_Results.md` for internal numbers), confirming this is a genuine generalization gap rather than an evaluation artifact — the same reading already established for the PAD→HAM direction.
