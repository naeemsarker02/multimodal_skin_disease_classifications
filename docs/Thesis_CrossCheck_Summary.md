# THESIS FULL-CONTENT CROSS-CHECK SUMMARY
> এই document-এ পুরো thesis-এ লেখা প্রতিটা factual claim, সংখ্যা, citation সংক্ষিপ্তভাবে listed। Claude Code-কে দিয়ে actual project file/data-এর সাথে cross-check করাও।

> **✅ VERIFIED & CORRECTED — 2026-09-22.** A full cross-check pass against project source files (Literature_Review_Extraction.md, Dataset_Preparation_Final_Report.md, official_results_summary.md, Phase8*_Results.md, Project_Tracking.md, reports/*.json, data/processed/*/feature_whitelist.md, external_validation_exclusions.csv) was completed. Most claims PASSED exactly. Five items below were corrected — each is marked `⚠ CORRECTED` at its location with the original claim struck through and the verified replacement given.

---

# A. Chapter 1 — Introduction (Claims)

```
- Six diagnostic classes: AK, BCC, MEL, NEV, SEK, SCC
- Primary dataset: PAD-UFES-20
- Secondary: HAM10000 (cross-dataset), ISIC Archive 1 & 2 (external validation)
- 6 stated Research Objectives (architecture design, leakage audit, baseline 
  comparison, cross-dataset/external validation, fairness, extended ablations)
- 5 "core strength" claims (leakage-free audit, patient-wise protocol, 
  cross-dataset validation, pre-registered ablation, clinical-safety philosophy)
```
**Cross-check করো:** এই ৬টা objective ও ৫টা strength-claim কি thesis defense/
project scope-এর সাথে মেলে? কোনো objective বাদ পড়েছে কিনা?

---

# B. Chapter 2 — Literature Review (১৯টা Paper + Citation)

## B.1 প্রতিটা Paper-এর নির্দিষ্ট Claim
```
1. Yap, Yolland & Tschandl (2018) — tri-modal dataset n=2,917; melanoma AUC 
   0.866 vs 0.784 (image-only); multiclass mAP 0.729 vs 0.598
2. Pacheco & Krohling (2019/2020) — 1,612-image precursor dataset; 6 CNN 
   backbones; balanced accuracy 0.650→0.718; SCC/BCC confusion noted
3. Pacheco & Krohling, MetaBlock (2021) — 74.8% PAD-UFES-20, 80.7% ISIC 2019; 
   channel-wise gating, NOT Transformer attention
4. Mridha & Islam (2026) — 1,568 lesions, binary; AUC 0.9818, F1 0.9769
5. Suresh et al., TG-CAVNet (2026) — 6,194-sample; 90.75% accuracy, macro 
   Jaccard 0.82
6. Tran-Van & Le (2025) — cross-attention + Hadamard ~98.8% accuracy 
   (paywalled, unverified primary text)
7. Frontiers in AI comparative study (2025) — cross-attention 98.86% accuracy 
   on HAM10000 (accuracy/weighted-F1, NOT macro-F1)
8. Zuo, Wang & Wang, CosCatNet (2025) — no quantitative results available
9. Zeng et al., MM-Skin (2025) — ~10,000 image-text pairs, 90-94% accuracy
10. Yang & Yang (2024) — 37,000-image custom dataset, 91% accuracy
11. Shrestha & Palit (2026) — ISIC 2024 subset, binary, 96% accuracy, 0.987 AUROC
12. Garib, Mery & Navarrete-Dechent (2025) — 17 models, 3 fusion methods; 
    balanced accuracy +10.43pp PAD-UFES-20; age = most informative feature
13. Watson et al. (2026) — 5,481 images/4,538 patients; free-text "leading 
    language" leakage (NOT structured metadata)
14. Badhon et al. (2025) — Grad-CAM, 91-93% accuracy (unverified dataset)
15. Frederich, Himawan & Rizkinia (2024) — EfficientNet-B0/B1, 90-94% accuracy 
    (unverified dataset)
16. Islam et al. (2025/2026) — 79,246 images/19,295 patients (Check4Cancer UK); 
    91.11% accuracy, 94.06% AUC; only 10% biopsied; Fitzpatrick not collected
17. Daneshjou et al. (2022) — DDI dataset, 656 cases; 27-36pp ROC-AUC drop on 
    dark skin
18. Alipour, Burke & Courtney (2024) — systematic review, systemic 
    underrepresentation finding
19. Xu et al. (2024) — 3-axis framework; reduced sensitivity at pretrained 
    embedding level for dark Fitzpatrick tones
```
**Cross-check করো:** 
1. এই ১৯টা paper-এর প্রতিটা citation (author, year, venue, DOI) কি সঠিক ও 
   verified? (আগের session-এ কিছু reference [5] সংশোধন হয়েছিল — বাকিগুলো 
   double-check করা দরকার)
2. এই ১৯টা paper ছাড়া project-এ আর কোনো paper review করা হয়েছিল যেটা এখানে 
   বাদ পড়েছে?
3. প্রতিটা paper-এর সংখ্যা (accuracy, AUC ইত্যাদি) কি মূল audit file-এর সাথে 
   মেলে?

## B.2 Architecture Comparison Table Claims
```
This thesis: EfficientNet-B0, Channel gate + 8-head cross-attention, 
Leakage audit = Yes (22 columns)
Others: leakage audit = "Not reported" for all 5 compared papers
```

---

# C. Chapter 3 — Methodology (সবচেয়ে বেশি Verify করা দরকার)

## C.1 Dataset Numbers
```
PAD-UFES-20: 2,298 images, 6 classes, 21 usable features, Primary
HAM10000: 10,015 images, 7 classes, Sparse metadata
ISIC Archive 1: 2,357 images, 9 classes (native, pre-exclusion) ⚠ CORRECTED (see note below)
ISIC Archive 2: 25,076 images, 9 classes (native)
DERM12345: 666 images (expansion only)
MED-NODE: 70 images (expansion only)
Patients: 1,373 unique, 1,641 unique lesions
Class imbalance ratio: 16.2:1
```
> ⚠ **CORRECTED — ISIC Archive 1 count has TWO valid numbers, used for different purposes:**
> - **2,357** = native/raw image count before external-validation exclusion. **This is the number `docs/main.tex` Table I (dataset overview) actually uses** — confirmed at line 130 (`ISIC Archive 1 & 2,357 & 9 & External validation`). Correct as-is for a "dataset overview" table describing raw dataset size.
> - **2,047** = post-exclusion count after removing 155 images that overlap with HAM10000/other splits (per `docs/Dataset_Preparation_Final_Report.md` §6). This is the count that reflects the actual *evaluation* set size after de-duplication — **not currently stated anywhere in the thesis text**.
> - **No correction needed in the thesis's Table 3.1** — 2,357 is the correct label for a native dataset-overview table. But if anywhere in Chapter 3/4 the thesis implies 2,357 images were *evaluated* (rather than just "available"), that would be wrong — the actual external-validation evaluation set for Archive 1 is smaller (2,047, or fewer once the 1,362-ID exclusion list below is also applied). Recommend adding one clarifying sentence near Table 3.1 or in the external-validation subsection: "2,357 native images; N used in external validation after exclusion of overlapping/duplicate cases."

## C.2 Leakage Audit — সম্পূর্ণ ২২-Column Breakdown (সবচেয়ে জরুরি verify)
```
মোট raw excluded: 45 columns (23 structural + 22 genuine leakage)
⚠ CORRECTED — see note immediately below the category breakdown.

Category 1 — Statistically tested (4 columns):
  - biopsed (PAD-UFES-20): φ=0.80, χ²=1474.5
  - diagnosis_confirm_type (HAM10000): φ=0.41, χ²=1700.67
  - diagnosis_confirm_type (ISIC 2): φ=0.36, χ²=3171.74
  - concomitant_biopsy (ISIC 2): duplicate encoding

Category 2 — Deterministic-split (1 column):
  - melanocytic (ISIC 2): 100%/0% split

Category 3 — Label-source (8 columns):
  - diagnostic_code (PAD, HAM) ×2
  - class_label (ISIC 1)
  - diagnosis_1 through diagnosis_5 (ISIC 2) ×5

Category 4 — Institution-proxy (7 columns):
  - anatom_site_3/4/5, family_hx_mm, personal_hx_mm, 
    clin_size_long_diam_mm, dermoscopic_type (all ISIC 2)

Category 5 — Non-clinical (2 columns):
  - attribution, copyright_license (ISIC 2)

Total: 4+1+8+7+2 = 22
```
> ⚠ **CORRECTED — "23 structural" bucket relabeled.** Verified against `data/processed/*/feature_whitelist.md`: the true "structural" exclusion bucket is **22 columns** (PAD 6 + HAM 5 + ISIC1 5 + ISIC2 6 — identifiers, file paths, `dataset_source`, `disease_label`), not 23. The 45-total is still arithmetically correct, but the missing 1 is a **separate, uncategorized zero-variance column**: `image_type` (ISIC Archive 2 — constant value "dermoscopic" for every row, so it carries zero information and was excluded, but it fits none of the 5 leakage categories or the structural bucket).
> **Correct reconciliation: 45 = 22 structural + 22 statistically-audited leakage + 1 zero-variance (`image_type`, ISIC Archive 2).**
> Update the thesis text wherever it says "23 structural columns" to say "22 structural columns" and add one sentence noting the separate zero-variance exclusion.
**Cross-check করো:** এই categorization ও প্রতিটা column-নাম কি source audit 
file/script-এর সাথে হুবহু মেলে? phi/chi-square সংখ্যাগুলো exact কিনা?

## C.3 Data Split
```
seed=42, split 70/15/15
Train: 1,606 rows | Val: 338 rows | Test: 354 rows
89-dimensional metadata vector (21 features, one-hot encoded)
Image preprocessing: 224×224, aspect-preserving resize
```

## C.4 Class-Imbalance Correction
```
Melanoma: 38 → 508 images (+1,237%)
SCC: 135 → 401 images (+197%)
Source: DERM12345 + MED-NODE (real, biopsy-confirmed, zero patient overlap)
```

## C.5 Architecture Specifics
```
Image branch: EfficientNet-B0, 49 tokens × 1280-d
Metadata branch: MLP, 89-d → 64-d
Channel Gate: Sigmoid(Linear(metadata)) × Image Tokens
Cross-Attention: 8 heads, d_model=256, Q=Metadata, K/V=Image
Classifier: FC(320→128) → BatchNorm → ReLU → Dropout → FC(128→6)
```

## C.6 Training Config
```
Optimizer: Adam
LR: image-only 1e-4, metadata-only 1e-3, fusion/cross-attention 1e-5
Batch size: 32, Max epochs: 30, Patience: 7
Seeds: 0, 1, 2
Weight decay: 1e-4
Hardware: NVIDIA Tesla T4 (Kaggle)
Software: Python 3.12.13, PyTorch 2.10.0 (CUDA 12.8)
```
> ⚠ **CORRECTED — UNVERIFIED for the headline run.** No logged diagnostic output (`python -V` / `torch.__version__` print) was found for the headline cross-attention training run, or for the SupCon run, anywhere in the project (`Project_Tracking.md`, `PROJECT_OWNERSHIP.md`, `THESIS_OWNERSHIP_MASTER.md`, notebook `.md` files). Both Kaggle notebooks contain the diagnostic print *code*, but no captured *output* was saved for either run. This spec string exists only inside this summary file — it is not sourced from anywhere else in the project.
> **Recommendation: do not state exact GPU/software versions as confirmed fact in the thesis.** Hedge instead, e.g.: *"Training was performed on Kaggle's cloud GPU environment (NVIDIA T4-class GPU); exact software versions were not logged for this run."* Do not re-run training solely to capture a version-print cell — not worth the compute cost for a reproducibility footnote.

---

# D. Chapter 4 — Results (সব সংখ্যা Verify করা দরকার)

## D.1 Primary Results
```
Image-only: Val 0.5703±0.0130, Test 0.6175±0.0153
Metadata-only: Val 0.5762±0.0072, Test 0.6077±0.0202
Late Fusion: Val 0.5731±0.0021, Test 0.6566±0.0234
Cross-Attention: Val 0.6209±0.0143, Test 0.6977±0.0269
Accuracy (same predictions): 0.763
```

## D.2 Bootstrap Significance (Primary, n=354, 1000 resamples)
```
cross_attention vs image: diff=0.0803, CI=[0.051, 0.117], p<0.001
cross_attention vs metadata: diff=0.0900, CI=[0.031, 0.151], p=0.006
cross_attention vs late_fusion: diff=0.0412, CI=[0.018, 0.067], p=0.002
```

## D.3 Confusion Matrix (summed 3 seeds, n=1,062)
```
     AK   BCC  MEL  NEV  SEK  SCC
AK   268   22   0    0    17   20
BCC   30  328   0    4    3    34
MEL    0    0  16    8    0    0
NEV    1    8   1   87    8    0
SEK    3    4   2   12   89    1
SCC   15   51   1    0    7   22

68.3% of errors (172/252) in AK/BCC/SCC cluster
```

## D.4 Per-Class Metrics
```
AK: n=327, P=0.8454, R=0.8196, F1=0.8323
BCC: n=399, P=0.7942, R=0.8221, F1=0.8079
MEL: n=24, P=0.8000, R=0.6667, F1=0.7273
NEV: n=105, P=0.7838, R=0.8286, F1=0.8056
SEK: n=111, P=0.7177, R=0.8018, F1=0.7574
SCC: n=96, P=0.2857, R=0.2292, F1=0.2543
Macro avg: P=0.7045, R=0.6947, F1=0.6975
```
**Cross-check করো:** Table III (0.6977) vs Table V macro-avg (0.6975) — 
এই ছোট gap আসলেই "per-seed mean vs pooled-confusion-matrix" পার্থক্যের 
কারণে কিনা, সঠিকভাবে documented আছে কিনা।

## D.5 Cross-Dataset (PAD→HAM10000)
```
Image-only: 0.4658±0.0373
Metadata-only: 0.2920±0.0121
Late Fusion: 0.4597±0.0084
Cross-Attention: 0.4654±0.0197

Bootstrap: CA vs image p=0.970 CI[-0.019,0.020]; CA vs late-fusion 
p=0.590 CI[-0.014,0.025]; CA vs metadata p<0.001 CI[0.134,0.210]
```

## D.6 External Validation
```
Archive 1 overlap: 66.5% | Archive 2 overlap: 98.6%
Exclusion list: 1,362 image IDs ⚠ CORRECTED — see note below
Archive 2: image-only 0.4912±0.0094, metadata-only 0.2410±0.0295, 
diff p<0.001
Archive 1: image-only 0.2421±0.0118
```
> ⚠ **CORRECTED — "1,362" is only the Archive 1 exclusion count, not a combined figure.** Verified directly against the actual exclusion files:
> - `data/processed/ISIC_Archive_1/external_validation_exclusions.csv` = **1,362 rows** (matches the summary/paper's "1,362 IDs" exactly — this is the HAM10000↔Archive 1 overlap exclusion list).
> - `data/processed/ISIC_Archive_2/external_validation_exclusions.csv` = **9,873 rows** (HAM10000↔Archive 2 overlap exclusion list — a separate, much larger file, consistent with Archive 2's 98.6% overlap vs. Archive 1's 66.5%).
> - **Combined total across both archives: 1,362 + 9,873 = 11,235 excluded IDs**, not 1,362.
> `docs/main.tex` (line 267) currently states the exclusion "via a dedicated exclusion list (1,362 IDs)" in a sentence covering *both* archives together — this reads as if 1,362 is the combined figure, which is misleading. **Recommend clarifying in the thesis: "Archive 1: 1,362 excluded overlapping images; Archive 2: 9,873 excluded overlapping images (11,235 total)" instead of a single unqualified 1,362.**

## D.7 Fairness
```
Fitzpatrick groups 1-4 tested; Types V/VI only 2+1 test images
38% missing Fitzpatrick value
Cross-attention best/tied-best in every sufficiently-sized group
```

## D.8 Extended Ablations (৮টা experiment)
```
1. 5-Backbone comparison: 0.5859-0.6224 val, ConvNeXt-Tiny best
2. Dataset-expansion-only: 0.6186 val
3. Dual-backbone ensemble: 0.6856 val → 0.7321 test, p=0.062, 
   CI=[-0.002, 0.077]
4. Joint 3-way fusion: 0.6721 val (bar=0.6710, missed by 0.0011)
5. Ensemble+TTA: 0.6213→0.5886 aggregate; Melanoma F1 0.3636→0.20 
   (REJECTED)
6. Post-hoc logit adjustment: τ=0 selected, no change (0.6977 unchanged)
7. Two-stage decoupled retraining (cRT): did NOT clear the pre-registered
   validation bar — raw val macro-F1 per seed, stage1→stage2:
   seed0 0.6049→0.5920, seed1 0.6182→0.5963, seed2 0.6397→0.6904
   (stage1 mean 0.6209, stage2 mean 0.6262). Per-class SCC/Melanoma F1
   decreased in seeds 0-1 (e.g. seed0 Melanoma 0.286→0.250, SCC
   0.314→0.286). No bootstrap significance test was ever formally run
   for this comparison. ⚠ CORRECTED — see note below (item was
   previously mis-stated as "p=0.074").
8. Supervised contrastive loss: Val 0.6420±0.0261 vs 0.6209±0.0143 
   baseline, p=0.074, CI=[-0.0108, +0.0433]
```
> ⚠ **CORRECTED — MOST IMPORTANT FIX.** The previous entry #7 stated "p=0.074" for cRT — this is now confirmed to be an **erroneous copy from ablation #8's SupCon p-value** (identical number, no independent source). Checked directly: `reports/PAD_UFES20/cross_attention_crt/crt_val_results.json` contains only raw per-seed stage1/stage2 macro-F1 and per-class F1 — **no bootstrap test, no p-value, no CI was ever computed or saved for cRT.** `Project_Tracking.md` only says cRT was "closed... tested negative or inconclusive" with no number given.
> **Correct statement for the thesis: cRT's stage-2 result (mean val macro-F1 ≈0.6262) did not clearly and meaningfully exceed the pre-registered decision bar relative to stage-1/baseline, and — unlike the SupCon ablation — was never formally bootstrap-tested. Report it as "inconclusive/not formally tested" rather than citing a specific p-value.** If the thesis currently prints "p=0.074" for cRT anywhere, that must be removed or replaced with this framing.
**Cross-check করো:** এই ৮টা ablation-ই কি project-এ করা সব extended experiment 
কভার করে, নাকি আরও কোনো experiment বাদ পড়েছে?

---

# E. Chapter 5 — Conclusion (Summary Claims)
```
5টা key finding, 6টা contribution, 7টা limitation, 5টা future-work direction
Final quote: "the goal was not to chase the highest score, but to build 
an evaluation that can be trusted"
```

---

# F. References — সব ২০টা (Chapter 2 citation + PAD-UFES-20 dataset paper)

```
[1]-[19]: উপরে B.1-এ listed
[20] Pacheco et al. (2020) — PAD-UFES-20 dataset paper, Data in Brief, 
     32, 106221
```

---

# G. 🎯 Claude Code-কে দেওয়ার Prompt (সরাসরি Copy করো)

```
I'm doing a final cross-check pass before printing my thesis. Please
verify the following against the actual project files (audit scripts,
JSON reports, Project_Tracking.md, Literature_Review_Extraction.md,
etc.) and report back ANY mismatch, missing detail, or gap:

1. All 19 literature-review paper citations (author/year/venue/DOI) —
   confirm each is still accurate per our verified extraction file.
2. The complete 22-column leakage audit breakdown (5 categories, with
   exact column names, phi/chi-square values) — confirm this matches
   the actual audit output exactly, with no columns miscounted.
3. All primary/cross-dataset/external-validation/ablation numbers
   listed in the attached summary — confirm every single number
   against the source JSON/CSV files.
4. Confirm whether the hardware/software spec (Tesla T4, Python
   3.12.13, PyTorch 2.10.0) was verified specifically for the
   HEADLINE cross-attention training run, or only for a different
   experiment (e.g., the SupCon run) — flag if this needs re-verification.
5. Confirm the 8 extended ablation experiments listed cover
   EVERYTHING in the project's ablation history — flag any experiment
   that exists in the project but is missing from this list.
6. Flag any thesis claim below that you cannot find supporting
   evidence for in the project files.

[PASTE THE FULL SUMMARY ABOVE HERE]

Report back a clear pass/fail per section, and for anything wrong or
missing, give me the corrected/complete information so I can update
the thesis text.
```

---
**STATUS: এই summary Claude Code-কে দাও। Output পেলে আমাকে ফিরিয়ে দাও — আমি 
সেই অনুযায়ী thesis-এ correction/addition করে দেব।**
