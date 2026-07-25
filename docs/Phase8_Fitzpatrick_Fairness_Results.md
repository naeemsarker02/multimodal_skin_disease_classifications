# Phase 8 — Fitzpatrick Skin-Type Fairness Analysis Results

**Date:** 2026-07-25
**Scope:** within-dataset analysis on PAD-UFES-20's own test split (`metadata_test.csv`, n=354). Fitzpatrick (`fitspatrick`, scale 1–6, higher = darker skin / lower UV sensitivity) only exists in PAD-UFES-20 — HAM10000 has no such column — so this cannot be extended to the cross-dataset generalization experiment; it stands on its own.
**Models:** all 4 variants (image, metadata, fusion, cross_attention), original full-feature checkpoints (`{variant}_seed{0,1,2}_best.pt`), not the reduced-feature checkpoints built for HAM10000 schema matching.
**Script:** `src/evaluation/evaluate_fairness.py`
**Raw output:** `reports/PAD_UFES20/fairness/fairness_results.json`, per-row predictions `reports/PAD_UFES20/fairness/predictions_{variant}_seed{seed}_test.csv`

---

## Important process note — first use of the test split

This experiment's inference runs were **the first-ever evaluation of any PAD-UFES-20 model on the test split** (`metadata_test.csv`) — no `eval_*_test.json` existed anywhere in the repo before this, and the `--confirm-final` guard in `evaluate.py` had never actually been invoked (only ever described in code). Per user decision on 2026-07-25, these 12 runs (4 variants × 3 seeds) are being treated as **both** the fairness breakdown **and** PAD-UFES-20's official final Stage 1 test-set result — the single sanctioned use of the held-out test split, consistent with Project_Tracking.md's decision 4 (test touched only once, after all training/model-selection decisions are finalized, which they were as of Phase 7 Stage 2's completion).

---

## Fitzpatrick group sizes, PAD-UFES-20 test split (n=354)

| Fitzpatrick | 1 | 2 | 3 | 4 | 5 | 6 | Missing |
|---|---|---|---|---|---|---|---|
| n | 22 | 120 | 59 | 15 | 2 | 1 | 135 (38.1%) |
| Reportable? | Yes (small-sample caution) | Yes | Yes | Yes (small-sample caution) | **No** (n<15) | **No** (n<15) | Not a fairness group |

**38.1% of the test set has no Fitzpatrick value recorded at all** — a real data-quality/equity gap in its own right, reported for transparency but not scored as a group. **Groups 5 and 6 (the two darkest-skin categories) have only 2 and 1 rows respectively — too few to report any rate at all.** This means the fairness question that matters most (how does the model perform on the darkest skin tones it will encounter clinically) **cannot be answered from this dataset** — not a negative result, an absence-of-data result, and the most important finding of this analysis.

---

## Official PAD-UFES-20 Stage 1 test-set result (overall, not group-sliced)

| Variant | seed0 | seed1 | seed2 | mean | std (population) |
|---|---|---|---|---|---|
| image | 0.6019 | 0.6382 | 0.6123 | 0.6175 | 0.0153 |
| metadata | 0.5897 | 0.5975 | 0.6360 | 0.6077 | 0.0202 |
| fusion | 0.6261 | 0.6830 | 0.6606 | 0.6566 | 0.0234 |
| **cross_attention** | 0.6862 | 0.6721 | 0.7349 | **0.6977** | 0.0269 |

Architecture ranking on the test split matches the val-split ranking already established in Phase 7 Stage 2 (cross-attention > fusion > image ≈ metadata) — consistent, not a surprise reversal. All 4 variants score higher on test than they did on val (e.g. cross-attention 0.6977 test vs. 0.6209 val) — plausible test-split-composition effect, not evidence of anything wrong; this is the only test-set number that exists for any of these checkpoints, so there's no other test result to cross-check it against.

---

## Per-group macro-F1 (mean across 3 seeds, groups with n≥15 only)

| Variant | Group 1 (n=22) ⚠ | Group 2 (n=120) | Group 3 (n=59) | Group 4 (n=15) ⚠ |
|---|---|---|---|---|
| image | 0.2525 | 0.5446 | 0.4637 | 0.4051 |
| metadata | 0.3586 | 0.3690 | 0.5351 | 0.4053 |
| fusion | 0.2785 | 0.5638 | 0.5228 | 0.4190 |
| cross_attention | 0.3307 | 0.6212 | 0.5706 | 0.4365 |

⚠ = small-sample caution (n<30) — point estimates and their 95% CIs below should both be read with this in mind; per-seed and per-group CIs are wide.

Full per-seed values with 95% bootstrap CIs (1000 resamples, RNG seed 42, percentile method) are in `reports/PAD_UFES20/fairness/fairness_results.json`. Representative CI widths (cross_attention, seed2, the best-performing run): group 1 macro-F1 0.2294 [0.1042, 0.3021]; group 2 0.6897 [0.4929, 0.7852]; group 3 0.5866 [0.3522, 0.6877]; group 4 0.4074 [0.1159, 0.4848] — every group's CI spans 0.1–0.3 macro-F1, illustrating just how much sampling noise dominates at these group sizes even for the best-scoring group (2).

---

## Reading the results

- **Group 2 (n=120, the largest and majority group) scores highest for every architecture** (0.3690–0.6212) — consistent with it also being the largest slice of the training data; unsurprising and not itself evidence of a skin-tone-specific effect distinct from sample-size effect.
- **Group 1 (n=22, the lightest skin type) scores lowest or near-lowest for every architecture except metadata** (0.2525–0.3586) — counter to a naive "majority representation = only advantage" story, since group 1 isn't the smallest group. This could reflect a genuine per-group effect, or could be pure small-sample noise (CIs above are wide enough that this isn't distinguishable from noise with confidence at n=22).
- **cross_attention scores highest of the 4 architectures in every reportable group** (group 1: 0.3307, group 2: 0.6212, group 3: 0.5706, group 4: 0.4365) — consistent with its overall lead (0.6977 mean), and notably it doesn't achieve this lead by sacrificing the already-weakest groups; it's the best or tied-best performer everywhere reportable. This is a positive fairness signal for cross-attention specifically, though still bounded by the same small-sample caveats as every other variant.
- **No equalized-odds or other parity metric is computed** — deferred as scoped, since per-group instability at this sample size would make such a metric itself unreliable.
- **The headline fairness finding is an absence, not a rate:** with only 3 total rows across the two darkest Fitzpatrick categories in the entire 354-row test split, and 38% of rows missing Fitzpatrick entirely, **this dataset cannot support a claim about model fairness across the full skin-tone spectrum** — any fairness claim from this analysis is necessarily bounded to Fitzpatrick types I–IV, and even IV is small-sample. This should be stated explicitly wherever these results are cited (thesis, external validation discussion) rather than implied to cover the full clinical population.
