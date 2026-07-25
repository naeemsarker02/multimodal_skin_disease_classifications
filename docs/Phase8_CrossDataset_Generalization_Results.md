# Phase 8, Experiment 1 — PAD-UFES-20 → HAM10000 Cross-Dataset Generalization Results

**Date:** 2026-07-25
**Protocol:** A — full native 6-way argmax (never masked to the 3 shared classes), scored only on the 3 classes shared between PAD-UFES-20 and HAM10000's taxonomies (Basal Cell Carcinoma, Melanoma, Nevus). 1,253 of HAM10000's 1,510 test rows fall in these 3 classes. Zero-shot transfer — all 4 models are trained only on PAD-UFES-20 and never see HAM10000 during training; weights are frozen for this evaluation.
**Script:** `src/evaluation/evaluate_cross_dataset.py`
**Checkpoints:** image uses the original `image_seed{0,1,2}_best.pt`; metadata/late_fusion/cross_attention use the reduced-feature (3-column: age, sex, anatomical_site) checkpoints `{metadata,fusion,cross_attention}_reduced_seed{0,1,2}_best.pt`, built specifically so PAD-UFES-20's metadata schema matches what HAM10000 provides.
**Raw per-run output:** `reports/PAD_UFES20/cross_dataset/eval_{variant}_seed{seed}_pad_to_ham.json`

---

## Bug fix during this run (does not affect any already-completed evaluation)

`evaluate_cross_dataset.py`'s eval loop unconditionally called `model(images, metadata)` whenever a variant needed metadata. `MetadataMLP.forward()` only accepts a single metadata tensor (no image), matching its usage everywhere else in the codebase (`src/evaluation/evaluate.py`) — the metadata-only branch crashed immediately with `TypeError: MetadataMLP.forward() takes 2 positional arguments but 3 were given`.

**No re-running of already-completed evaluations was required.** The crash happened before any output file was written, so no stale/incorrect JSON was ever produced for the metadata variant — it simply failed to run until fixed. The image variant doesn't pass metadata at all (unaffected). late_fusion and cross_attention's `forward()` signatures genuinely take `(image, metadata)`, so the unconditional call was already correct for those two — unaffected. The fix (in `evaluate_cross_dataset.py`) branches on `variant == "metadata"` to call `model(metadata)` only for that variant.

---

## Headline results — macro-F1 (3 shared classes)

| Variant | seed0 | seed1 | seed2 | mean | std (population) | spillover rate (mean) |
|---|---|---|---|---|---|---|
| image | 0.4577 | 0.4247 | 0.5149 | **0.4658** | 0.0373 | 17.7% |
| metadata | 0.3079 | 0.2787 | 0.2893 | **0.2920** | 0.0121 | 13.1% |
| late_fusion | 0.4521 | 0.4715 | 0.4557 | **0.4597** | 0.0084 | 12.2% |
| cross_attention | 0.4916 | 0.4442 | 0.4604 | **0.4654** | 0.0197 | 11.6% |

For reference, PAD-UFES-20-internal (within-dataset) macro-F1: image 0.5703±0.0130, metadata (original 21-feature) 0.5762±0.0072, late fusion 0.5731±0.0021, cross-attention 0.6209±0.0143, cross-attention (reduced-feature) 0.6449 mean. Every variant's cross-dataset transfer number is meaningfully lower than its own within-dataset number — the expected generalization-gap direction, not a bug.

---

## Per-class F1 (shared classes)

| Variant | seed | Basal Cell Carcinoma | Melanoma | Nevus |
|---|---|---|---|---|
| image | 0 | 0.1987 | 0.3642 | 0.8102 |
| image | 1 | 0.2092 | 0.2968 | 0.7681 |
| image | 2 | 0.3750 | 0.3395 | 0.8301 |
| image | **mean** | **0.2609** | **0.3335** | **0.8028** |
| metadata | 0 | 0.1304 | 0.1990 | 0.5942 |
| metadata | 1 | 0.1446 | 0.2255 | 0.4661 |
| metadata | 2 | 0.1429 | 0.1967 | 0.5284 |
| metadata | **mean** | **0.1393** | **0.2071** | **0.5296** |
| late_fusion | 0 | 0.2069 | 0.3505 | 0.7989 |
| late_fusion | 1 | 0.2581 | 0.3324 | 0.8239 |
| late_fusion | 2 | 0.2602 | 0.3252 | 0.7817 |
| late_fusion | **mean** | **0.2417** | **0.3360** | **0.8015** |
| cross_attention | 0 | 0.2676 | 0.3719 | 0.8352 |
| cross_attention | 1 | 0.1854 | 0.3487 | 0.7984 |
| cross_attention | 2 | 0.2667 | 0.3373 | 0.7773 |
| cross_attention | **mean** | **0.2399** | **0.3526** | **0.8037** |

Nevus dominates every variant's per-class F1 (0.53–0.83); Basal Cell Carcinoma and Melanoma lag well behind (0.14–0.38) across all 4 architectures. This mirrors PAD-UFES-20's own class imbalance rather than being an artifact of any one branch.

---

## Spillover — predictions falling outside the 3 shared classes

"Spillover" = the model predicted one of PAD-UFES-20's 3 non-transferable classes (Actinic Keratosis, Seborrheic Keratosis, Squamous Cell Carcinoma) on a true-shared-class HAM10000 image. Under Protocol A this always counts as wrong; it is reported separately so the failure mode is visible rather than hidden inside the F1 numbers.

| Variant | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| image | 16.5% (207/1253) | 26.7% (335/1253) | 10.0% (125/1253) | 17.7% |
| metadata | 13.7% (172/1253) | 12.8% (160/1253) | 12.8% (161/1253) | 13.1% |
| late_fusion | 9.9% (124/1253) | 16.8% (210/1253) | 9.8% (123/1253) | 12.2% |
| cross_attention | 9.2% (115/1253) | 14.8% (186/1253) | 10.8% (135/1253) | 11.6% |

Metadata-informed variants (late_fusion, cross_attention) show the lowest and most consistent spillover rates; image-only shows the highest and most variable (10.0%–26.7% across seeds).

---

## Metadata coverage on HAM10000 (metadata/late_fusion/cross_attention variants only)

Constant across all seeds (same preprocessor, fit once on PAD-UFES-20's reduced-feature train split):

| Field | Matched | Fell to `__MISSING__` | Total |
|---|---|---|---|
| sex | 1,244 | 9 | 1,253 |
| anatomical_site | 1,039 | 214 (17.1%) | 1,253 |

The anatomical_site gap is expected — HAM10000 has site categories (e.g. genital, trunk-adjacent) with no PAD-UFES-20-fitted counterpart per `docs/Phase8_Anatomical_Site_Mapping.csv` (2 of 14 raw sites unmapped: LIP, NOSE).

---

## Bootstrap significance testing (2026-07-25)

**Method:** paired bootstrap over the 1,253 HAM10000 eval rows (with replacement, 1,000 resamples, RNG seed 42, fixed and distinct from the 3 model-training seeds). The same resampled row-index set is applied to every variant within a given iteration (paired design). Per iteration, macro-F1 (3 shared classes) is computed separately for each of a variant's 3 seeds on the resampled rows, then averaged across the 3 seeds — folding in both row-level and seed-level variability. Statistic: `diff = score(cross_attention) - score(comparator)`; 95% CI via the percentile method; two-sided p-value from the proportion of resamples whose sign disagrees with the observed sign, doubled. Reported at both the uncorrected α=0.05 and the Bonferroni-adjusted α=0.05/3≈0.0167 (3 comparisons share the same cross-attention anchor). Script: `src/evaluation/bootstrap_significance.py`. Per-row predictions consumed from `reports/PAD_UFES20/cross_dataset/predictions_{variant}_seed{seed}_pad_to_ham.csv` (added this session — no model re-inference needed for the bootstrap itself, only for generating these CSVs once). Full output: `reports/PAD_UFES20/cross_dataset/bootstrap_significance.json`.

| Comparison | Observed diff | 95% CI | p-value (2-sided) | Sig. α=0.05 | Sig. Bonferroni α≈0.0167 |
|---|---|---|---|---|---|
| cross_attention vs. image | −0.0004 | [−0.0189, +0.0196] | 0.970 | No | No |
| cross_attention vs. metadata | **+0.1734** | **[+0.1341, +0.2097]** | **0.000** | **Yes** | **Yes** |
| cross_attention vs. late_fusion | +0.0057 | [−0.0139, +0.0247] | 0.590 | No | No |

**Conclusion:** cross-attention's advantage over metadata-alone is real and highly significant (CI excludes 0 by a wide margin under both thresholds) — expected, since metadata-alone is a fundamentally weaker signal on this transfer task. Cross-attention is **not** statistically distinguishable from image-alone or from late_fusion on this HAM10000 transfer set — both CIs comfortably straddle 0, consistent with the "cluster tightly" visual read below. This means the apparent architecture ranking among image/late_fusion/cross_attention (0.4597–0.4658 mean) should **not** be reported as one architecture "beating" the others on cross-dataset generalization specifically — only that all three clearly beat metadata-alone, and cross-attention/late_fusion/image are statistically tied here even though cross-attention remains the stronger architecture within-dataset (0.6209 vs. 0.5703/0.5731).

---

## Reading the results

- **Metadata alone generalizes worst** (0.2920 mean) — expected, given the reduced 3-feature schema and the 17.1% anatomical_site coverage gap on HAM10000; metadata distributions shift more across datasets than pixel statistics do. This gap is also the only one confirmed statistically significant by the bootstrap test above.
- **Image, late_fusion, and cross_attention cluster tightly** (0.4597–0.4658 mean, all overlapping in range across seeds) — fusing in the weak reduced-metadata signal neither helps nor hurts much under zero-shot transfer. Confirmed by the bootstrap test: neither pairwise difference against cross-attention is significant at α=0.05 or the Bonferroni-adjusted threshold.
- **cross_attention has the lowest spillover rate** (11.6% mean) and highest per-seed peak (0.4916, seed0) of the three multimodal-capable variants, consistent with — but not conclusive proof of — the metadata signal helping it stay within the shared-class boundary more often than image-only does. This is a descriptive observation, not something the bootstrap test above directly evaluates (that test targets macro-F1, not spillover rate).
- **All 4 variants underperform their own PAD-UFES-20-internal macro-F1** by a wide margin, confirming this is a genuine generalization gap rather than an evaluation artifact.
