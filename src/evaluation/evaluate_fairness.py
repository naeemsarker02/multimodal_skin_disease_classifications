"""Phase 8 - Fitzpatrick skin-type fairness analysis on PAD-UFES-20's own
test split (approved 2026-07-25, Project_Tracking.md).

Usage:
    python -m src.evaluation.evaluate_fairness

Fitzpatrick (`fitspatrick`) only exists in PAD-UFES-20 - HAM10000 has no
such column - so this is a within-dataset analysis on PAD-UFES-20's test
split, not an extension of the cross-dataset generalization experiment.
Runs all 4 variants (image, metadata, fusion, cross_attention) x 3 seeds
using the ORIGINAL full-feature checkpoints (not the reduced-feature
ones - those exist only for HAM10000 schema matching and have no reason
to be used here). No per-row prediction script existed for PAD-UFES-20's
own test split before this, so inference is run once here and the
per-row predictions are saved for reuse/audit.

This re-runs inference on the test split (already used once for Phase
6/7's final Stage 1 evaluation, per Project_Tracking.md decision 4) but
does not touch model selection or training in any way - purely a
post-hoc diagnostic slice of an already-finalized model's predictions,
not a new tuning decision, so the "touch test once" discipline is not
violated.

Small-sample handling (approved 2026-07-25):
- n < 15: excluded from the macro-F1 table entirely (count reported only,
  "insufficient sample" - not reported as a rate).
- 15 <= n < 30: included but flagged as small-sample / wide-variance-
  expected.
- n >= 30: included, no caution flag.
- Missing/NaN fitspatrick: reported as its own count/percentage row, not
  treated as a fairness group (data-availability gap, not a skin-tone
  category).
- No val+test pooling - keeps the test-only split discipline used
  throughout the project, even though it costs power on small groups.

Per-group 95% CIs reuse the same paired-bootstrap machinery introduced
for src/evaluation/bootstrap_significance.py (percentile method, 1000
resamples, fixed RNG seed 42) - here resampling within each Fitzpatrick
group's rows (unpaired across groups, since groups are disjoint subsets,
not the same rows scored by different models).
"""

import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.evaluation.evaluate import build_eval_dataset, load_model
from src.evaluation.test_split_guard import check_test_split_available, mark_test_split_consumed
from src.models.config import BATCH_SIZE, get_dataset
from src.models.dataset import MetadataPreprocessor

VARIANTS = ["image", "metadata", "fusion", "cross_attention"]
SEEDS = [0, 1, 2]

N_RESAMPLES = 1000
RNG_SEED = 42
MIN_N_REPORTABLE = 15
MIN_N_NO_CAUTION = 30


def run_inference(variant: str, seed: int, ds_config, device):
    preprocessor = None
    if variant in ("metadata", "fusion", "cross_attention"):
        preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))

    checkpoint_path = ds_config.checkpoints_dir / f"{variant}_seed{seed}_best.pt"
    model = load_model(variant, checkpoint_path, ds_config, device, preprocessor)

    dataset = build_eval_dataset(variant, ds_config.test_csv, ds_config, preprocessor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if variant in ("fusion", "cross_attention"):
                images, metadata, labels = batch
                outputs = model(images.to(device), metadata.to(device))
            else:
                inputs, labels = batch
                outputs = model(inputs.to(device))
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    predictions_df = dataset.df.copy()
    predictions_df["true_label_idx"] = all_labels
    predictions_df["pred_label_idx"] = all_preds
    return predictions_df


def macro_f1(true_idx, pred_idx, num_classes):
    return f1_score(true_idx, pred_idx, labels=list(range(num_classes)), average="macro", zero_division=0)


def bootstrap_ci(true_idx, pred_idx, num_classes, rng):
    n = len(true_idx)
    scores = np.empty(N_RESAMPLES)
    for i in range(N_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        scores[i] = macro_f1(true_idx[idx], pred_idx[idx], num_classes)
    ci_low, ci_high = np.percentile(scores, [2.5, 97.5])
    return float(ci_low), float(ci_high)


def group_label(fitz_value) -> str:
    if pd.isna(fitz_value):
        return "missing"
    return str(int(fitz_value))


def analyze_variant_seed(predictions_df, num_classes, rng):
    predictions_df = predictions_df.copy()
    predictions_df["fitz_group"] = predictions_df["fitspatrick"].apply(group_label)

    groups = {}
    total_n = len(predictions_df)
    for group, sub in predictions_df.groupby("fitz_group"):
        n = len(sub)
        true_idx = sub["true_label_idx"].to_numpy()
        pred_idx = sub["pred_label_idx"].to_numpy()

        if group == "missing":
            groups[group] = {"n": n, "pct_of_total": n / total_n, "reported": False,
                              "reason": "not a fairness group - data-availability gap"}
            continue

        if n < MIN_N_REPORTABLE:
            groups[group] = {"n": n, "pct_of_total": n / total_n, "reported": False,
                              "reason": f"insufficient sample (n<{MIN_N_REPORTABLE})"}
            continue

        f1 = macro_f1(true_idx, pred_idx, num_classes)
        ci_low, ci_high = bootstrap_ci(true_idx, pred_idx, num_classes, rng)
        groups[group] = {
            "n": n,
            "pct_of_total": n / total_n,
            "reported": True,
            "macro_f1": float(f1),
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "small_sample_caution": n < MIN_N_NO_CAUTION,
        }
    return groups


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset("PAD_UFES20")
    check_test_split_available(ds_config, "evaluate_fairness.py")
    num_classes = ds_config.num_classes

    fairness_dir = ds_config.baseline_reports_dir.parent / "fairness"
    fairness_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)

    all_results = {
        "experiment": "phase8_fitzpatrick_fairness_pad_ufes20_test",
        "n_resamples": N_RESAMPLES,
        "rng_seed": RNG_SEED,
        "min_n_reportable": MIN_N_REPORTABLE,
        "min_n_no_caution": MIN_N_NO_CAUTION,
        "variants": {},
    }

    for variant in VARIANTS:
        all_results["variants"][variant] = {"seeds": {}}
        for seed in SEEDS:
            print(f"=== {variant} seed{seed} ===")
            predictions_df = run_inference(variant, seed, ds_config, device)

            pred_out_path = fairness_dir / f"predictions_{variant}_seed{seed}_test.csv"
            predictions_df.to_csv(pred_out_path, index=False)

            overall_f1 = macro_f1(
                predictions_df["true_label_idx"].to_numpy(),
                predictions_df["pred_label_idx"].to_numpy(),
                num_classes,
            )
            groups = analyze_variant_seed(predictions_df, num_classes, rng)

            all_results["variants"][variant]["seeds"][str(seed)] = {
                "overall_macro_f1": float(overall_f1),
                "groups": groups,
            }

            print(f"  overall macro-F1: {overall_f1:.4f}")
            for g, info in sorted(groups.items()):
                if info.get("reported"):
                    caution = " [SMALL SAMPLE]" if info["small_sample_caution"] else ""
                    print(f"  fitzpatrick={g}: n={info['n']} macro-F1={info['macro_f1']:.4f} "
                          f"95% CI=[{info['ci_95_low']:.4f}, {info['ci_95_high']:.4f}]{caution}")
                else:
                    print(f"  fitzpatrick={g}: n={info['n']} - not reported ({info['reason']})")
            print(f"  written -> {pred_out_path}")

    out_path = fairness_dir / "fairness_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwritten -> {out_path}")

    marker = mark_test_split_consumed(
        ds_config,
        caller="evaluate_fairness.py",
        consumed_on="2026-07-25",
        reference="docs/Project_Tracking.md#phase-8-fitzpatrick-fairness-complete, docs/Phase8_Fitzpatrick_Fairness_Results.md",
        detail={"variants": VARIANTS, "seeds": SEEDS, "n_runs": len(VARIANTS) * len(SEEDS)},
    )
    print(f"test split now marked consumed -> {marker}")


if __name__ == "__main__":
    main()
