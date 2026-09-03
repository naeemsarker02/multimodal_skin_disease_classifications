"""Post-hoc logit adjustment (Menon et al., ICLR 2021, "Long-tail learning
via logit adjustment") applied to the already-trained, already-locked
cross_attention (EfficientNet-B0) PAD_UFES20 checkpoints - INFERENCE-TIME
ONLY. No retraining, no changes to the model or training pipeline.

    adjusted_logit[y] = original_logit[y] - tau * log(class_prior[y])
    class_prior[y] = count(y in TRAINING split) / total TRAINING samples

This is the 3rd sanctioned read of PAD_UFES20's test split - see
data/processed/PAD_UFES20/TEST_SPLIT_CONSUMED.json for the pre-registered
decision rule (tau selected on validation only, applied to test exactly
once). Per-sample logits for val and test (all 3 seeds) are cached to
reports/logit_cache/ so this script (or any related post-hoc analysis)
never needs a 4th test read.

Usage:
    python -m src.evaluation.logit_adjustment
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.evaluation.evaluate import build_eval_dataset, load_model
from src.models.config import BATCH_SIZE, get_dataset

SEEDS = [0, 1, 2]
TAU_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
BRANCH = "cross_attention"
ORIGINAL_TEST_MACRO_F1_PER_SEED = {0: 0.6862, 1: 0.6721, 2: 0.7349}
ORIGINAL_TEST_HEADLINE_MEAN = 0.6977
ORIGINAL_TEST_HEADLINE_STD = 0.0269

CACHE_DIR = Path("reports/logit_cache/PAD_UFES20/cross_attention")
OUT_DIR = Path("reports/PAD_UFES20/logit_adjustment")


def cache_path(seed: int, split: str) -> Path:
    return CACHE_DIR / f"cross_attention_seed{seed}_{split}_logits.npz"


def compute_and_cache_logits(ds_config, seed: int, split: str, device, preprocessor):
    out_path = cache_path(seed, split)
    if out_path.exists():
        data = np.load(out_path)
        return data["logits"], data["labels"]

    checkpoint_path = ds_config.checkpoints_dir / f"{BRANCH}_seed{seed}_best.pt"
    model = load_model(BRANCH, checkpoint_path, ds_config, device, preprocessor)

    split_csv = {"val": ds_config.val_csv, "test": ds_config.test_csv}[split]
    dataset = build_eval_dataset(BRANCH, split_csv, ds_config, preprocessor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, metadata, labels in loader:
            outputs = model(images.to(device), metadata.to(device))
            all_logits.append(outputs.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.array(all_labels)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, logits=logits, labels=labels)
    print(f"cached logits -> {out_path}  (shape={logits.shape})")
    return logits, labels


def compute_class_prior(ds_config) -> np.ndarray:
    train_df = pd.read_csv(ds_config.train_csv)
    counts = np.zeros(ds_config.num_classes, dtype=np.float64)
    for label, idx in ds_config.label_to_idx.items():
        counts[idx] = (train_df["disease_label"] == label).sum()
    prior = counts / counts.sum()
    return prior


def macro_f1_for_tau(logits: np.ndarray, labels: np.ndarray, log_prior: np.ndarray, tau: float, num_classes: int):
    adjusted = logits - tau * log_prior[None, :]
    preds = adjusted.argmax(axis=1)
    return f1_score(labels, preds, average="macro", labels=list(range(num_classes)), zero_division=0), preds


def per_class_f1_dict(labels, preds, class_names):
    scores = f1_score(labels, preds, average=None, labels=list(range(len(class_names))), zero_division=0)
    return dict(zip(class_names, scores.tolist()))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset("PAD_UFES20")
    class_names = ds_config.class_names
    num_classes = ds_config.num_classes

    from src.models.dataset import MetadataPreprocessor
    preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))

    # --- Step 1: forward passes on val AND test, all 3 seeds, cached to disk.
    val_logits, val_labels = {}, {}
    test_logits, test_labels = {}, {}
    for seed in SEEDS:
        val_logits[seed], val_labels[seed] = compute_and_cache_logits(
            ds_config, seed, "val", device, preprocessor
        )
        test_logits[seed], test_labels[seed] = compute_and_cache_logits(
            ds_config, seed, "test", device, preprocessor
        )

    # --- Step 2: class prior from TRAINING split only.
    prior = compute_class_prior(ds_config)
    log_prior = np.log(prior)
    print("\nclass_prior (from training split only):")
    for name, p in zip(class_names, prior):
        print(f"  {name}: {p:.4f}")

    # --- Step 3: tau sweep on VAL only.
    val_f1_grid = {tau: [] for tau in TAU_GRID}
    per_seed_val_table = {}
    for seed in SEEDS:
        per_seed_val_table[seed] = {}
        for tau in TAU_GRID:
            f1, _ = macro_f1_for_tau(val_logits[seed], val_labels[seed], log_prior, tau, num_classes)
            val_f1_grid[tau].append(f1)
            per_seed_val_table[seed][tau] = f1

    mean_val_f1_per_tau = {tau: float(np.mean(scores)) for tau, scores in val_f1_grid.items()}
    std_val_f1_per_tau = {tau: float(np.std(scores)) for tau, scores in val_f1_grid.items()}
    best_tau = max(mean_val_f1_per_tau, key=mean_val_f1_per_tau.get)

    print("\nVAL macro-F1 per tau (mean +/- std across 3 seeds):")
    for tau in TAU_GRID:
        marker = "  <-- selected" if tau == best_tau else ""
        print(f"  tau={tau:.2f}: {mean_val_f1_per_tau[tau]:.4f} +/- {std_val_f1_per_tau[tau]:.4f}{marker}")
    print(f"\nSelected tau (maximizes mean VAL macro-F1 across seeds): {best_tau}")

    original_val_f1 = {seed: per_seed_val_table[seed][0.0] for seed in SEEDS}

    # --- Step 4: apply the single selected tau to TEST logits exactly once.
    test_results = {}
    for seed in SEEDS:
        f1_adj, preds_adj = macro_f1_for_tau(
            test_logits[seed], test_labels[seed], log_prior, best_tau, num_classes
        )
        f1_orig, preds_orig = macro_f1_for_tau(
            test_logits[seed], test_labels[seed], log_prior, 0.0, num_classes
        )
        cm_adj = confusion_matrix(test_labels[seed], preds_adj, labels=list(range(num_classes)))
        test_results[seed] = {
            "original_test_macro_f1_recomputed": f1_orig,
            "original_test_macro_f1_recorded": ORIGINAL_TEST_MACRO_F1_PER_SEED[seed],
            "adjusted_test_macro_f1": f1_adj,
            "adjusted_per_class_f1": per_class_f1_dict(test_labels[seed], preds_adj, class_names),
            "original_per_class_f1": per_class_f1_dict(test_labels[seed], preds_orig, class_names),
            "adjusted_confusion_matrix": cm_adj.tolist(),
        }

    adjusted_test_f1s = [test_results[s]["adjusted_test_macro_f1"] for s in SEEDS]
    adjusted_mean = float(np.mean(adjusted_test_f1s))
    adjusted_std = float(np.std(adjusted_test_f1s))

    # --- Step 5: report + save.
    report = {
        "tau_grid": TAU_GRID,
        "selected_tau": best_tau,
        "class_prior": dict(zip(class_names, prior.tolist())),
        "val": {
            "per_seed_original_macro_f1": original_val_f1,
            "per_seed_per_tau_macro_f1": per_seed_val_table,
            "mean_macro_f1_per_tau": mean_val_f1_per_tau,
            "std_macro_f1_per_tau": std_val_f1_per_tau,
        },
        "test": {
            "original_headline_mean": ORIGINAL_TEST_HEADLINE_MEAN,
            "original_headline_std": ORIGINAL_TEST_HEADLINE_STD,
            "adjusted_mean": adjusted_mean,
            "adjusted_std": adjusted_std,
            "per_seed": test_results,
        },
        "test_split_read_note": (
            "This is the 3rd sanctioned read of PAD_UFES20's test split (see "
            "data/processed/PAD_UFES20/TEST_SPLIT_CONSUMED.json), performed once "
            "per the pre-registered rule: tau selected on validation only, "
            "applied to test exactly once, no further tau tuning on test."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "logit_adjustment_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nfull report written -> {out_path}")

    # --- Console summary table.
    print("\n" + "=" * 72)
    print("BEFORE / AFTER  -  Logit Adjustment (tau = {:.2f})".format(best_tau))
    print("=" * 72)
    print(f"{'seed':<6}{'val orig':<12}{'val adj (tau*)':<16}{'test orig':<12}{'test adj':<12}")
    for seed in SEEDS:
        val_adj_f1, _ = macro_f1_for_tau(val_logits[seed], val_labels[seed], log_prior, best_tau, num_classes)
        print(
            f"{seed:<6}{original_val_f1[seed]:<12.4f}{val_adj_f1:<16.4f}"
            f"{test_results[seed]['original_test_macro_f1_recorded']:<12.4f}"
            f"{test_results[seed]['adjusted_test_macro_f1']:<12.4f}"
        )
    print("-" * 72)
    print(
        f"{'mean':<6}{'':<12}{'':<16}"
        f"{ORIGINAL_TEST_HEADLINE_MEAN:<12.4f}{adjusted_mean:<12.4f}"
    )
    print(
        f"{'std':<6}{'':<12}{'':<16}"
        f"{ORIGINAL_TEST_HEADLINE_STD:<12.4f}{adjusted_std:<12.4f}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
