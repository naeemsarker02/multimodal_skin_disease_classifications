"""Step 4 - paired bootstrap significance testing for the Option B
backbone-fusion models (ConvNeXt-Tiny/DenseNet121 cross-attention fusion,
plus their prediction-time ensemble) against the original locked
cross_attention (EfficientNet-B0) PAD-UFES-20 test result (macro-F1
0.6977, locked 2026-07-25). Approved in "Step 4 Scope Gap Found and
Resolved" (2026-07-31, Project_Tracking.md); the second test-split read
this required was logged and the guard reopened in "PAD-UFES-20
Test-Split Guard Reopened for Step 4" (2026-08-01).

Usage:
    python -m src.evaluation.bootstrap_significance_backbone_fusion

Reads only already-computed per-row prediction CSVs - no model inference
happens here:
- reports/PAD_UFES20/fairness/predictions_cross_attention_seed{seed}_test.csv
  (the original locked anchor, 2026-07-25)
- reports/PAD_UFES20/cross_attention_backbone/predictions_cross_attention_backbone_{backbone}_seed{seed}_test.csv
- reports/PAD_UFES20/cross_attention_backbone/predictions_cross_attention_backbone_ensemble_convnext_tiny_densenet121_seed{seed}_test.csv

Method - paired bootstrap, row-level resampling, seed-averaged per
iteration (same method as bootstrap_significance.py, adapted to a
single-dataset, all-6-classes comparison instead of cross-dataset
shared classes):
- Unit resampled: the 354 PAD-UFES-20 test rows, with replacement, 1,000
  iterations.
- Paired: the same resampled row-index set is applied to every variant
  within a given iteration.
- Each variant has 3 independently trained seeds. Per iteration,
  macro-F1 (all 6 classes) is computed separately for each seed on that
  iteration's resampled rows, then averaged across the 3 seeds.
- Statistic: diff = score(new_variant) - score(anchor) per iteration.
  Reported: observed mean diff, 95% CI via the percentile method, two-sided
  p-value, significance at both uncorrected alpha=0.05 and the
  Bonferroni-adjusted alpha=0.05/3 (3 comparisons share the same anchor).
- RNG: fixed seed (42), unrelated to the 3 model-training seeds.
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

SEEDS = [0, 1, 2]
ANCHOR = "cross_attention_original"
COMPARATORS = [
    "cross_attention_backbone_convnext_tiny",
    "cross_attention_backbone_densenet121",
    "cross_attention_backbone_ensemble",
]
VARIANTS = [ANCHOR] + COMPARATORS


def prediction_path(variant, seed, fairness_dir, backbone_dir):
    if variant == "cross_attention_original":
        return fairness_dir / f"predictions_cross_attention_seed{seed}_test.csv"
    if variant == "cross_attention_backbone_convnext_tiny":
        return backbone_dir / f"predictions_cross_attention_backbone_convnext_tiny_seed{seed}_test.csv"
    if variant == "cross_attention_backbone_densenet121":
        return backbone_dir / f"predictions_cross_attention_backbone_densenet121_seed{seed}_test.csv"
    if variant == "cross_attention_backbone_ensemble":
        return (
            backbone_dir
            / f"predictions_cross_attention_backbone_ensemble_convnext_tiny_densenet121_seed{seed}_test.csv"
        )
    raise ValueError(f"Unknown variant: {variant}")


def load_predictions(fairness_dir, backbone_dir):
    """variant -> list of 3 (true_label_idx, pred_label_idx) numpy array pairs,
    one pair per seed, all aligned to the same row order (verified below).
    """
    data = {}
    ref_paths = None
    for variant in VARIANTS:
        seed_arrays = []
        for seed in SEEDS:
            path = prediction_path(variant, seed, fairness_dir, backbone_dir)
            df = pd.read_csv(path)
            if ref_paths is None:
                ref_paths = df["image_path"].to_numpy()
            elif not (df["image_path"].to_numpy() == ref_paths).all():
                raise ValueError(f"Row order mismatch in {path} - cannot pair-resample safely")
            seed_arrays.append((df["true_label_idx"].to_numpy(), df["pred_label_idx"].to_numpy()))
        data[variant] = seed_arrays
    return data, len(ref_paths)


def macro_f1_all_classes(true_idx, pred_idx, num_classes):
    return f1_score(true_idx, pred_idx, labels=list(range(num_classes)), average="macro", zero_division=0)


def bootstrap_variant_scores(seed_arrays, num_classes, resample_indices_per_iter):
    n_iter = len(resample_indices_per_iter)
    per_iter_scores = np.zeros(n_iter)
    for seed_true, seed_pred in seed_arrays:
        seed_scores = np.array([
            macro_f1_all_classes(seed_true[idx], seed_pred[idx], num_classes)
            for idx in resample_indices_per_iter
        ])
        per_iter_scores += seed_scores
    per_iter_scores /= len(seed_arrays)
    return per_iter_scores


def run_comparison(new_scores, anchor_scores, observed_new, observed_anchor):
    diffs = new_scores - anchor_scores
    observed_diff = observed_new - observed_anchor
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
    fairness_dir = ds_config.baseline_reports_dir.parent / "fairness"
    backbone_dir = ds_config.fusion_reports_dir.parent / "cross_attention_backbone"
    num_classes = ds_config.num_classes

    predictions, n_rows = load_predictions(fairness_dir, backbone_dir)

    rng = np.random.default_rng(RNG_SEED)
    resample_indices_per_iter = [
        rng.integers(0, n_rows, size=n_rows) for _ in range(N_RESAMPLES)
    ]

    observed_scores = {}
    for variant in VARIANTS:
        seed_f1s = [
            macro_f1_all_classes(true_idx, pred_idx, num_classes)
            for true_idx, pred_idx in predictions[variant]
        ]
        observed_scores[variant] = float(np.mean(seed_f1s))

    bootstrap_scores = {
        variant: bootstrap_variant_scores(predictions[variant], num_classes, resample_indices_per_iter)
        for variant in VARIANTS
    }

    results = {
        "experiment": "step4_bootstrap_significance_backbone_fusion_vs_locked_original",
        "method": "paired bootstrap, row-level resampling (n=354 PAD-UFES-20 test rows), "
                  "seed-averaged (3 seeds) per iteration, percentile-method 95% CI, all 6 classes",
        "n_resamples": N_RESAMPLES,
        "rng_seed": RNG_SEED,
        "alpha_uncorrected": ALPHA,
        "alpha_bonferroni": BONFERRONI_ALPHA,
        "n_comparisons": N_COMPARISONS,
        "anchor": ANCHOR,
        "anchor_reference": "locked cross_attention (EfficientNet-B0) test result, 0.6977, "
                             "Project_Tracking.md 2026-07-25",
        "observed_macro_f1_mean_across_seeds": observed_scores,
        "comparisons": {},
    }

    for comparator in COMPARATORS:
        results["comparisons"][f"{comparator}_vs_{ANCHOR}"] = run_comparison(
            bootstrap_scores[comparator],
            bootstrap_scores[ANCHOR],
            observed_scores[comparator],
            observed_scores[ANCHOR],
        )

    out_path = backbone_dir / "bootstrap_significance_backbone_fusion.json"
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
