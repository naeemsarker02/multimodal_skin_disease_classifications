"""Bootstrap significance testing for the PAD-UFES-20 PRIMARY (native
test-set) headline comparison: cross_attention vs. image / metadata /
late_fusion, on PAD-UFES-20's own held-out test set (6-class macro-F1).

This is the one comparison that underpins the paper's headline claim
(0.6977 vs. 0.6566/0.6175/0.6077) but had never been bootstrap-tested,
unlike every other comparison in the project. Same methodology as
bootstrap_significance.py (PAD->HAM), adapted to:
- 6 classes (native PAD-UFES-20 taxonomy) instead of 3 shared classes.
- reports/PAD_UFES20/fairness/predictions_{variant}_seed{seed}_test.csv
  (variant names on disk: image, metadata, fusion, cross_attention;
  "fusion" == late fusion).

Usage:
    python -m src.evaluation.bootstrap_significance_primary
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

# variant -> filename stem on disk
VARIANT_FILE_STEM = {
    "image": "image",
    "metadata": "metadata",
    "late_fusion": "fusion",
    "cross_attention": "cross_attention",
}
SEEDS = [0, 1, 2]
ANCHOR = "cross_attention"
COMPARATORS = ["image", "metadata", "late_fusion"]


def load_predictions(reports_dir):
    data = {}
    ref_paths = None
    for variant, stem in VARIANT_FILE_STEM.items():
        seed_arrays = []
        for seed in SEEDS:
            path = reports_dir / f"predictions_{stem}_seed{seed}_test.csv"
            df = pd.read_csv(path)
            if ref_paths is None:
                ref_paths = df["image_path"].to_numpy()
            elif not (df["image_path"].to_numpy() == ref_paths).all():
                raise ValueError(f"Row order mismatch in {path} - cannot pair-resample safely")
            seed_arrays.append((df["true_label_idx"].to_numpy(), df["pred_label_idx"].to_numpy()))
        data[variant] = seed_arrays
    return data, len(ref_paths)


def macro_f1_all(true_idx, pred_idx, all_indices):
    return f1_score(true_idx, pred_idx, labels=all_indices, average="macro", zero_division=0)


def bootstrap_variant_scores(seed_arrays, all_indices, resample_indices_per_iter):
    n_iter = len(resample_indices_per_iter)
    per_iter_scores = np.zeros(n_iter)
    for seed_true, seed_pred in seed_arrays:
        seed_scores = np.array([
            macro_f1_all(seed_true[idx], seed_pred[idx], all_indices)
            for idx in resample_indices_per_iter
        ])
        per_iter_scores += seed_scores
    per_iter_scores /= len(seed_arrays)
    return per_iter_scores


def run_comparison(anchor_scores, comparator_scores, observed_anchor, observed_comparator):
    diffs = anchor_scores - comparator_scores
    observed_diff = observed_anchor - observed_comparator
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

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
    reports_dir = ds_config.baseline_reports_dir.parent / "fairness"

    predictions, n_rows = load_predictions(reports_dir)
    all_indices = list(ds_config.label_to_idx.values())

    rng = np.random.default_rng(RNG_SEED)
    resample_indices_per_iter = [
        rng.integers(0, n_rows, size=n_rows) for _ in range(N_RESAMPLES)
    ]

    observed_scores = {}
    for variant in VARIANT_FILE_STEM:
        seed_f1s = [
            macro_f1_all(true_idx, pred_idx, all_indices)
            for true_idx, pred_idx in predictions[variant]
        ]
        observed_scores[variant] = float(np.mean(seed_f1s))

    bootstrap_scores = {
        variant: bootstrap_variant_scores(predictions[variant], all_indices, resample_indices_per_iter)
        for variant in VARIANT_FILE_STEM
    }

    results = {
        "experiment": "primary_bootstrap_significance_cross_attention_vs_others_pad_ufes20_test",
        "method": "paired bootstrap, row-level resampling (native PAD-UFES-20 test rows), "
                  "seed-averaged (3 seeds) per iteration, percentile-method 95% CI, 6-class macro-F1",
        "n_resamples": N_RESAMPLES,
        "rng_seed": RNG_SEED,
        "n_rows": n_rows,
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

    out_path = reports_dir / "bootstrap_significance_primary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"n_rows: {n_rows}")
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
