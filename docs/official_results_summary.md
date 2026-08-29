# Official Primary Results Summary — PAD-UFES-20

Extracted and independently recomputed from raw result files (not from
`Project_Tracking.md` prose). Test macro-F1 per seed was recomputed directly
with `sklearn.metrics.f1_score(average='macro')` on
`reports/PAD_UFES20/fairness/predictions_{model}_seed{seed}_test.csv`
(`true_label_idx` vs `pred_label_idx`); the resulting means match
`reports/PAD_UFES20/fairness/bootstrap_significance_primary.json` exactly.

**Split**: official test set, 354 rows, locked (single sanctioned use per
`TEST_SPLIT_CONSUMED.json`). **Metric**: macro-F1, 6-class, seeds [0,1,2].

## Validation macro-F1 (per seed, from `logs/PAD_UFES20/train_*_seed*_summary.json`)

| Model | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|---|---|---|---|---|
| Image-only | 0.5529 | 0.5741 | 0.5840 | 0.5703 ± 0.0130 |
| Metadata-only | 0.5861 | 0.5694 | 0.5732 | 0.5762 ± 0.0072 |
| Late fusion | 0.5723 | 0.5760 | 0.5711 | 0.5731 ± 0.0021 |
| **Cross-attention (headline)** | 0.6049 | 0.6182 | 0.6397 | **0.6209 ± 0.0143** |

## Test macro-F1 (per seed, recomputed from raw predictions CSVs)

| Model | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|---|---|---|---|---|
| Image-only | 0.6019 | 0.6382 | 0.6123 | 0.6175 ± 0.0152 |
| Metadata-only | 0.5897 | 0.5975 | 0.6360 | 0.6077 ± 0.0202 |
| Late fusion | 0.6261 | 0.6830 | 0.6606 | 0.6566 ± 0.0234 |
| **Cross-attention (headline)** | 0.6862 | 0.6721 | 0.7349 | **0.6977 ± 0.0269** |

## Significance (paired bootstrap, 1,000 resamples, seed 42, cross-attention as anchor)

| Comparison | Observed diff | 95% CI | p (two-sided) | Significant (Bonferroni α=0.0167) |
|---|---|---|---|---|
| Cross-attention vs. image | +0.0803 | [0.0509, 0.1168] | <0.001 | Yes |
| Cross-attention vs. metadata | +0.0900 | [0.0313, 0.1509] | 0.006 | Yes |
| Cross-attention vs. late fusion | +0.0412 | [0.0177, 0.0666] | 0.002 | Yes |

**Headline result**: Cross-attention fusion, test macro-F1 **0.6977 ± 0.0269**,
significantly better than every other primary variant (all comparisons
significant even under Bonferroni correction).

Source files: `docs/official_results_summary.json` (machine-readable version
of this table), `reports/PAD_UFES20/fairness/bootstrap_significance_primary.json`,
`logs/PAD_UFES20/train_*_seed*_summary.json`,
`reports/PAD_UFES20/fairness/predictions_*_seed*_test.csv`.
