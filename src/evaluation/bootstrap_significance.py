"""Phase 8 Experiment 1 - bootstrap significance testing for the
PAD-UFES-20 -> HAM10000 cross-dataset generalization results (approved
2026-07-25, Project_Tracking.md).

Usage:
    python -m src.evaluation.bootstrap_significance

Reads the 12 per-row prediction CSVs written by evaluate_cross_dataset.py
(reports/PAD_UFES20/cross_dataset/predictions_{variant}_seed{seed}_pad_to_ham.csv)
- no model inference happens here, this only resamples already-computed
predictions.

Method - paired bootstrap, row-level resampling, seed-averaged per
iteration:
- Unit resampled: the 1,253 HAM10000 eval rows, with replacement, 1,000
  iterations.
- Paired: the same resampled row-index set is applied to every variant
  within a given iteration, isolating the architecture effect from
  row-sampling noise rather than conflating the two.
- Each variant has 3 independently trained seeds. Per iteration, macro-F1
  (3 shared classes) is computed separately for each seed on that
  iteration's resampled rows, then averaged across the 3 seeds to give one
  score per variant per iteration - folding in both row-level and
  seed-level variability instead of picking one seed arbitrarily.
- Statistic: diff = score(cross_attention) - score(comparator) per
  iteration. Reported: observed mean diff, 95% CI via the percentile
  method (2.5th/97.5th percentile of the diff distribution), two-sided
  p-value (proportion of resamples where the sign of diff disagrees with
  the observed sign, doubled), significance at both the uncorrected
  alpha=0.05 and the Bonferroni-adjusted alpha=0.05/3 (~=0.0167, since 3
  comparisons share the same cross-attention anchor).
- RNG: fixed seed (42) for the resampling itself - distinct from and
  unrelated to the 3 model-training seeds.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.models.config import get_dataset

N_RESAMPLES = 1000
RNG_SEED = 42
ALPHA = 0.05
N_COMPARISONS = 3
BONFERRONI_ALPHA = ALPHA / N_COMPARISONS

VARIANTS = ["image", "metadata", "late_fusion", "cross_attention"]
SEEDS = [0, 1, 2]
ANCHOR = "cross_attention"
COMPARATORS = ["image", "metadata", "late_fusion"]

SHARED_CLASSES = ["Basal Cell Carcinoma", "Melanoma", "Nevus"]


def load_predictions(reports_dir):
    """variant -> list of 3 (true_label_idx, pred_label_idx) numpy array pairs,
    one pair per seed, all aligned to the same row order (verified below).
    """
    data = {}
    ref_paths = None
    for variant in VARIANTS:
        seed_arrays = []
        for seed in SEEDS:
            path = reports_dir / f"predictions_{variant}_seed{seed}_pad_to_ham.csv"
            df = pd.read_csv(path)
            if ref_paths is None:
                ref_paths = df["image_path"].to_numpy()
            elif not (df["image_path"].to_numpy() == ref_paths).all():
                raise ValueError(f"Row order mismatch in {path} - cannot pair-resample safely")
            seed_arrays.append((df["true_label_idx"].to_numpy(), df["pred_label_idx"].to_numpy()))
        data[variant] = seed_arrays
    return data, len(ref_paths)


def macro_f1_shared(true_idx, pred_idx, shared_indices):
    return f1_score(true_idx, pred_idx, labels=shared_indices, average="macro", zero_division=0)


def bootstrap_variant_scores(seed_arrays, shared_indices, resample_indices_per_iter):
    """Returns an (N_RESAMPLES,) array: for each iteration, the mean macro-F1
    across the variant's 3 seeds on that iteration's resampled rows.
    """
    n_iter = len(resample_indices_per_iter)
    per_iter_scores = np.zeros(n_iter)
    for seed_true, seed_pred in seed_arrays:
        seed_scores = np.array([
            macro_f1_shared(seed_true[idx], seed_pred[idx], shared_indices)
            for idx in resample_indices_per_iter
        ])
        per_iter_scores += seed_scores
    per_iter_scores /= len(seed_arrays)
    return per_iter_scores


def run_comparison(anchor_scores, comparator_scores, observed_anchor, observed_comparator):
    diffs = anchor_scores - comparator_scores
    observed_diff = observed_anchor - observed_comparator
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    # two-sided p-value: proportion of resamples whose sign disagrees with
    # the observed sign, doubled, capped at 1.0
    if observed_diff >= 0:
        p_one_sided = np.mean(diffs <= 0)
    else:
        p_one_sided = np.mean(diffs >= 0)
    p_value = min(1.0, 2 * p_one_sided)

    return {
        "observed_diff": float(observed_diff),
        "bootstrap_mean_diff": float(diffs.mean()),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "p_value_two_sided": float(p_value),
        "significant_alpha_0.05": bool(ci_low > 0 or ci_high < 0),
        "significant_bonferroni_alpha_{:.4f}".format(BONFERRONI_ALPHA): bool(p_value < BONFERRONI_ALPHA),
    }


def main():
    ds_config = get_dataset("PAD_UFES20")
    reports_dir = ds_config.baseline_reports_dir.parent / "cross_dataset"

    predictions, n_rows = load_predictions(reports_dir)
    shared_indices = [ds_config.label_to_idx[c] for c in SHARED_CLASSES]

    rng = np.random.default_rng(RNG_SEED)
    resample_indices_per_iter = [
        rng.integers(0, n_rows, size=n_rows) for _ in range(N_RESAMPLES)
    ]

    observed_scores = {}
    for variant in VARIANTS:
        seed_f1s = [
            macro_f1_shared(true_idx, pred_idx, shared_indices)
            for true_idx, pred_idx in predictions[variant]
        ]
        observed_scores[variant] = float(np.mean(seed_f1s))

    bootstrap_scores = {
        variant: bootstrap_variant_scores(predictions[variant], shared_indices, resample_indices_per_iter)
        for variant in VARIANTS
    }

    results = {
        "experiment": "phase8_bootstrap_significance_cross_attention_vs_others",
        "method": "paired bootstrap, row-level resampling (n=1253 HAM10000 eval rows), "
                  "seed-averaged (3 seeds) per iteration, percentile-method 95% CI",
        "n_resamples": N_RESAMPLES,
        "rng_seed": RNG_SEED,
        "alpha_uncorrected": ALPHA,
        "alpha_bonferroni": BONFERRONI_ALPHA,
        "n_comparisons": N_COMPARISONS,
        "anchor": ANCHOR,
        "observed_macro_f1_mean_across_seeds": observed_scores,
        "comparisons": {},
    }

    for comparator in COMPARATORS:
        results["comparisons"][f"{ANCHOR}_vs_{comparator}"] = run_comparison(
            bootstrap_scores[ANCHOR],
            bootstrap_scores[comparator],
            observed_scores[ANCHOR],
            observed_scores[comparator],
        )

    out_path = reports_dir / "bootstrap_significance.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Observed macro-F1 (mean across 3 seeds): {observed_scores}")
    print()
    for name, comp in results["comparisons"].items():
        print(f"{name}:")
        print(f"  observed diff: {comp['observed_diff']:+.4f}")
        print(f"  95% CI: [{comp['ci_95_low']:+.4f}, {comp['ci_95_high']:+.4f}]")
        print(f"  p-value (two-sided): {comp['p_value_two_sided']:.4f}")
        print(f"  significant at alpha=0.05: {comp['significant_alpha_0.05']}")
        print(f"  significant at Bonferroni alpha={BONFERRONI_ALPHA:.4f}: "
              f"{comp['significant_bonferroni_alpha_{:.4f}'.format(BONFERRONI_ALPHA)]}")
        print()
    print(f"written -> {out_path}")


if __name__ == "__main__":
    main()
