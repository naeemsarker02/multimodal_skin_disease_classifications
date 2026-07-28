"""Phase 8 - bootstrap significance testing for HAM10000 -> ISIC Archive 2
external validation, image vs. metadata (approved as part of the ISIC
external validation scope, Project_Tracking.md).

Usage:
    python -m src.evaluation.bootstrap_significance_isic

Reads the 6 per-row prediction CSVs written by evaluate_external_isic.py
(reports/HAM10000/external_isic/predictions_ISIC_Archive_2_{variant}_seed{seed}.csv)
- no model inference happens here, this only resamples already-computed
predictions.

Scope: ISIC Archive 2 only - the only archive where both image and
metadata branches were evaluated (Archive 1 has 0 usable metadata
columns, image-only). Single comparison (image vs. metadata), so no
Bonferroni correction is needed (n_comparisons=1).

Method - same paired-bootstrap, row-level resampling, seed-averaged
design as bootstrap_significance.py (PAD->HAM):
- Unit resampled: the 12,508 ISIC Archive 2 eval rows, with replacement,
  1,000 iterations.
- Paired: the same resampled row-index set is applied to both variants
  within a given iteration (verified below that image/metadata
  predictions share identical row order).
- Each variant has 3 independently trained seeds. Per iteration,
  macro-F1 (4 shared classes: Basal Cell Carcinoma, Dermatofibroma,
  Melanoma, Nevus) is computed separately for each seed on that
  iteration's resampled rows, then averaged across the 3 seeds.
- Statistic: diff = score(image) - score(metadata) per iteration.
  Reported: observed mean diff, 95% CI via the percentile method,
  two-sided p-value (proportion of resamples whose sign disagrees with
  the observed sign, doubled).
- RNG: fixed seed (42) for the resampling itself - same convention as
  bootstrap_significance.py, distinct from the 3 model-training seeds.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.models.config import get_dataset

N_RESAMPLES = 1000
RNG_SEED = 42
ALPHA = 0.05

VARIANTS = ["image", "metadata"]
SEEDS = [0, 1, 2]
ARCHIVE = "ISIC_Archive_2"

SHARED_CLASSES = ["Basal Cell Carcinoma", "Dermatofibroma", "Melanoma", "Nevus"]


def load_predictions(reports_dir):
    data = {}
    ref_ids = None
    for variant in VARIANTS:
        seed_arrays = []
        for seed in SEEDS:
            path = reports_dir / f"predictions_{ARCHIVE}_{variant}_seed{seed}.csv"
            df = pd.read_csv(path)
            if ref_ids is None:
                ref_ids = df["image_id"].to_numpy()
            elif not (df["image_id"].to_numpy() == ref_ids).all():
                raise ValueError(f"Row order mismatch in {path} - cannot pair-resample safely")
            seed_arrays.append((df["true_label_idx"].to_numpy(), df["pred_label_idx"].to_numpy()))
        data[variant] = seed_arrays
    return data, len(ref_ids)


def macro_f1_shared(true_idx, pred_idx, shared_indices):
    return f1_score(true_idx, pred_idx, labels=shared_indices, average="macro", zero_division=0)


def bootstrap_variant_scores(seed_arrays, shared_indices, resample_indices_per_iter):
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


def run_comparison(a_scores, b_scores, observed_a, observed_b):
    diffs = a_scores - b_scores
    observed_diff = observed_a - observed_b
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
    }


def main():
    ds_config = get_dataset("HAM10000")
    reports_dir = ds_config.baseline_reports_dir.parent / "external_isic"

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
        "experiment": "phase8_bootstrap_significance_isic_archive2_image_vs_metadata",
        "method": "paired bootstrap, row-level resampling (n=12508 ISIC Archive 2 eval rows), "
                  "seed-averaged (3 seeds) per iteration, percentile-method 95% CI",
        "n_resamples": N_RESAMPLES,
        "rng_seed": RNG_SEED,
        "alpha": ALPHA,
        "n_comparisons": 1,
        "observed_macro_f1_mean_across_seeds": observed_scores,
        "comparison": run_comparison(
            bootstrap_scores["image"],
            bootstrap_scores["metadata"],
            observed_scores["image"],
            observed_scores["metadata"],
        ),
    }

    out_path = reports_dir / "bootstrap_significance_isic_archive2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    comp = results["comparison"]
    print(f"Observed macro-F1 (mean across 3 seeds): {observed_scores}")
    print()
    print("image vs. metadata (ISIC Archive 2):")
    print(f"  observed diff: {comp['observed_diff']:+.4f}")
    print(f"  95% CI: [{comp['ci_95_low']:+.4f}, {comp['ci_95_high']:+.4f}]")
    print(f"  p-value (two-sided): {comp['p_value_two_sided']:.4f}")
    print(f"  significant at alpha=0.05: {comp['significant_alpha_0.05']}")
    print()
    print(f"written -> {out_path}")


if __name__ == "__main__":
    main()
