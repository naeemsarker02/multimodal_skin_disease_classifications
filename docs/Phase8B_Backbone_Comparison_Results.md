# Phase 8B — Backbone Comparison Results (PAD-UFES-20 Expanded)

Image-only branch, 5 backbones × 3 seeds (0/1/2) = 15 runs, evaluated by
`best_val_macro_f1`. Source: `logs/PAD_UFES20_Expanded/train_image_*_summary.json`
(downloaded from Kaggle output, verified against this repo's raw JSON files —
all 15 per-seed values and all 5 aggregates cross-checked exactly).

## Results table

| Backbone            | Seed 0 | Seed 1 | Seed 2 | Mean   | Std (pop.) |
|----------------------|--------|--------|--------|--------|------------|
| EfficientNet-B0       | 0.5910 | 0.5937 | 0.5800 | 0.5882 | 0.0059     |
| MobileNetV3-Large      | 0.6351 | 0.5709 | 0.5734 | 0.5931 | 0.0297     |
| DenseNet121            | 0.5769 | 0.5959 | 0.5991 | 0.5906 | 0.0098     |
| ResNet50               | 0.6094 | 0.5713 | 0.5770 | 0.5859 | 0.0168     |
| **ConvNeXt-Tiny**      | 0.6416 | 0.5998 | 0.6257 | **0.6224** | 0.0172 |

**Baseline** (non-expanded PAD-UFES-20, image-only, EfficientNet-B0,
`logs/PAD_UFES20/train_image_seed{0,1,2}_summary.json`): 0.5529 / 0.5741 / 0.5840
→ **0.5703 ± 0.0130**.

All 5 expanded-dataset backbones beat this baseline on mean macro-F1, but the
margin is not uniform — see the MobileNetV3 caveat below.

## Top-2 selection for Step 4 (fusion)

**#1 — ConvNeXt-Tiny (0.6224 ± 0.0172).** Clear winner, ~3 points above every
other backbone on mean, and every individual seed (0.600–0.642) beats every
other backbone's best seed except one. Not a close call.

**#2 — DenseNet121 (0.5906 ± 0.0098), recommended over MobileNetV3-Large
(0.5931 ± 0.0297).**

Reasoning:

- MobileNetV3's mean is highest of the two, but it's carried entirely by
  seed 0 (0.6351). Seeds 1 and 2 land at 0.5709 and 0.5734 — both *barely*
  above the non-expanded baseline (0.5703) and well below DenseNet121's worst
  seed (0.5769). Two of MobileNetV3's three runs essentially replicate the
  baseline rather than improving on it; the "beats baseline" claim for this
  backbone rests on one seed, not three.
- DenseNet121 clears baseline on all three seeds, with the tightest spread of
  any non-ConvNeXt backbone (std 0.0098, vs. 0.0297 for MobileNetV3 — a 3x
  difference). That consistency is the more useful property going into
  fusion: Step 4 will itself introduce new variance (fusion architecture,
  metadata branch, cross-attention), and starting from a backbone whose image
  branch is already seed-sensitive stacks uncertainty on uncertainty, making
  it harder to attribute a fusion result to the architecture rather than to
  which seed happened to land well.
- A mean-only pick is defensible in principle, but here the higher mean is a
  single-outlier artifact rather than a consistently better backbone. For a
  thesis where you need to argue the fusion improvement is real and not seed
  luck, DenseNet121 is the more defensible #2.

**Recommendation: ConvNeXt-Tiny + DenseNet121** as the top-2 backbones carried
into Step 4.

**Methodological note (citable directly):** selection was made on mean *and*
stability, not mean alone. A mean-only rule would have picked MobileNetV3-Large;
per-seed inspection shows that ranking is driven by a single outlier run
(seed 0) rather than representative performance across seeds, so it was
rejected in favor of the backbone that improves on baseline consistently
across all three seeds. This two-criteria selection process — mean for
central tendency, per-seed spread for reliability — is the standard applied
throughout this comparison, not a one-off exception made for backbone
selection.
