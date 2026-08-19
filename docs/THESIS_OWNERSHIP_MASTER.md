# THESIS OWNERSHIP MASTER DOCUMENT

**Purpose:** the single, complete, self-contained reference for viva defense and paper writing. Built 2026-08-03 by directly re-reading `docs/Project_AZ_Reference.md` (compiled 2026-07-28, itself file-verified), `docs/Project_Tracking.md` (3,776 lines, dated decision log through 2026-08-02), `docs/Phase8B_Backbone_Comparison_Results.md`, and the actual Phase 8B–8E source files (`src/models/cross_attention_backbone_fusion_model.py`, `cross_attention_joint_fusion_model.py`, `spatial_backbone_embedder.py`) and their training-run entries in `Project_Tracking.md`. Every number below is either (a) traced to a summary JSON / results doc, or (b) explicitly flagged **UNVERIFIED** if it only exists as prose in `Project_Tracking.md` with no locally-downloaded JSON to check it against — this happens for two numbers only, both called out in §3 and §6.

**How to use this document:** read top to bottom once for orientation, then use it as a lookup table — every section is self-contained enough to answer a viva question or write a paper subsection without re-opening the codebase, **except** the two flagged-unverified numbers (§2.3, §2.6 license/citation gaps). Phase 8E's result (§3.8, §8.5) is now locally verified — see below.

---

## 1. Project Folder/File Map (post-cleanup, 2026-08-03)

| Path | What it is |
|---|---|
| `data/raw/{PAD_UFES20,HAM10000,ISIC_Archive_1,ISIC_Archive_2}/` | Untouched original downloads. No images ever copied elsewhere — `image_path` in processed CSVs always resolves back here. |
| `data/raw/DERM12345/`, `data/raw/MED-NODE/` | Raw images/metadata for the two Phase 8C dataset-expansion sources (§2.5–2.6). Not independently audited/cleaned as standalone datasets — folded directly into `PAD_UFES20_Expanded`'s training CSV. |
| `data/interim/<dataset>/` | Intermediate cleaning-pipeline outputs (`metadata_standardized.csv`, `value_validation_report.csv`) — currently showing as locally-deleted in git status (uncommitted, pre-existing at session start, not part of this cleanup pass). |
| `data/processed/{PAD_UFES20,HAM10000,ISIC_Archive_1,ISIC_Archive_2}/` | Final train/val/test CSVs, `label_mapping.csv`, `feature_whitelist.md`, `dataset_description.md`, split-quality reports, `TEST_SPLIT_CONSUMED.json` guards. Source of truth for every dataset fact in §2. |
| `data/processed/PAD_UFES20_Expanded/` | Image-only expanded training CSV (`metadata_train_image_only.csv`) built from PAD-UFES-20 train + DERM12345 + MED-NODE; val/test byte-identical to original PAD-UFES-20. |
| `src/data_audit/<dataset>/m0N_*.py` | Per-dataset audit pipeline modules (image inventory, corruption/orphan checks, class distribution, missing-value analysis). Rolled up into `reports/<dataset>/*_dataset_audit_summary.md`. |
| `src/data_cleaning/<dataset>/c0N_*.py` | Per-dataset cleaning pipeline: column standardization → value validation → label standardization → split → split-quality report → description doc. |
| `src/data_cleaning/pad_ufes20_expanded/c01_build_expanded_dataset.py` | Builds `PAD_UFES20_Expanded` from PAD-UFES-20 + DERM12345 + MED-NODE, train-only, val/test frozen. |
| `src/data_cleaning/cross_dataset_leakage_filter.py`, `label_conflict_filter.py` | Generate `external_validation_exclusions.csv` and `label_conflict_exclusions.csv` for the ISIC archives; same ID-matching method reused to clear DERM12345 against ISIC (§2.5). |
| `src/eda/` | Phase 5 EDA code; outputs in `reports/eda/` and `notebooks/01–05_eda_*.ipynb`. |
| `src/models/` | All model architectures (`image_model.py`, `metadata_model.py`, `fusion_model.py`, `cross_attention_fusion_model.py`, `cross_attention_backbone_fusion_model.py`, `cross_attention_joint_fusion_model.py`, `spatial_backbone_embedder.py`, `backbones.py`), all `train*.py` entry points, `dataset.py` (shared `FusionDataset`/`MetadataPreprocessor`), `config.py` (dataset registry, hyperparameters, `resolve_image_path()`). |
| `src/evaluation/` | `evaluate.py` (main eval entrypoint, all variants), `evaluate_cross_dataset.py`, `evaluate_external_isic.py`, `evaluate_fairness.py`, `bootstrap_significance*.py` (3 variants), `test_split_guard.py` (the `TEST_SPLIT_CONSUMED.json` enforcement added 2026-07-25). |
| `scripts/generate_*_kaggle_notebook.py` | Generators that embed the real `.py` source into Kaggle-notebook `%%writefile` cells — the committed `notebooks/*.md` files cannot silently drift from source. 8 generators, one per training notebook. |
| `scripts/isic_*_cell.py` (3 files) | One-off diagnostic Kaggle cells (mirror ID verification), kept in place — actively cited by filename in `src/models/config.py` and 5 live notebooks as the evidence trail for a resolved data-integrity question. |
| `notebooks/*.md` | Generated (8) and hand-maintained (2: `ham10000_kaggle_notebook.md`, `pad_ufes20_fusion_kaggle_notebook.md`) Kaggle notebooks, one per training run type. |
| `notebooks/0{1-5}_eda_*.ipynb` | Phase 5 EDA notebooks, one per dataset + cross-dataset comparison. |
| `reports/<dataset>/` | Audit CSVs/summaries (numbered `0N_*.csv`), plus per-model evaluation output (`fairness/`, `cross_dataset/`, `cross_attention_backbone/`, `score_experiments/`, `baseline/`, `fusion/`) — confusion matrices, per-class F1, prediction CSVs. |
| `reports/eda/` | Phase 5 EDA figures/summaries per dataset + cross-dataset. |
| `logs/<dataset>/` | Training logs, checkpoints, per-run summary JSONs (gitignored — not pushed to remote, exists locally/on Kaggle output only). |
| `docs/Project_AZ_Reference.md` | File-verified reference for the 4 original datasets + 4 Phase 6/7 model variants (compiled 2026-07-28) — this document's primary source for §2.1–2.4 and §3.1–3.4. |
| `docs/Project_Tracking.md` | The living, dated decision log — every judgment call in §5 is sourced here with a date. |
| `docs/PROJECT_PLAN.md` | Canonical confirmed design decisions (do not re-litigate list). |
| `docs/PROJECT_OWNERSHIP.md` | Earlier-scoped ownership doc, Phase 1–5 only (superseded in scope by this document, not deleted — historical record). |
| `docs/Phase8_*.md`, `docs/Phase8B_*.md` | Standalone results write-ups per Phase 8 sub-experiment (cross-dataset generalization, fairness, ISIC external validation, backbone comparison). |
| `docs/Dataset_Strategy.md`, `docs/Dataset_Preparation_Final_Report.md` | Earlier dataset-scoped reports, kept as historical record. |
| `docs/Literature_Review.md` (+ `.xlsx`) | 20-paper literature review log (row #20, Shrestha & Palit, added 2026-08-14 — `.xlsx` not yet updated to match, see §9.3). |
| `_archive/` | `MASTER_PROJECT_DOCUMENT.md` (superseded, flagged in-file), `Project_Cleanup_Review.md`, `Research_Plan.md`, plus new `pre_final_cleanup/` (this session's cleanup — see `MOVED_ITEMS.md`). |
| `papers/1.pdf,2.pdf,3.pdf` | Reference papers, non-descriptive filenames (flagged, not renamed this pass). |

---

## 2. Datasets

### 2.1 PAD-UFES-20 (primary, multimodal)
- **Citation**: Pacheco et al., *"PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones,"* *Data in Brief*, 32, 106221 (2020). https://doi.org/10.1016/j.dib.2020.106221. Also arXiv:2007.00478, Mendeley Data DOI 10.17632/zr7vgbcyr2.1.
- **License**: not independently confirmed — no license text found in-repo or via web search for the Mendeley Data page itself. **Open the Mendeley page directly before submission.**
- **Size**: 2,298 images (`.png`), 1,373 unique patients, 1,641 unique lesions. 26 raw metadata columns.
- **Class distribution**: BCC 845 (36.77%), ACK 730 (31.77%), NEV 244 (10.62%), SEK 235 (10.23%), SCC 192 (8.36%), MEL 52 (2.26%). Imbalance ratio 16.2:1.
- **Leakage audit**: `biopsed` excluded — phi=0.80, chi²=1474.5, n=2,298 (100% of malignant cases have `biopsed=True`, zero counter-examples across all 3 splits). `diagnostic_code` excluded as raw label source (1:1 with the label).
- **Feature whitelist**: 21 of 29 columns allowed (`smoke, drink, background_father, background_mother, age, pesticide, sex, skin_cancer_history, cancer_history, has_piped_water, has_sewage_system, fitspatrick, anatomical_site, diameter_1, diameter_2, itch, grew, hurt, changed, bleed, elevation`).
- **Split**: patient-wise, dominant-diagnosis-stratified, seed 42, ratios 0.70/0.15/0.15. Verified: train 1,606 / val 338 / test 354 rows; 948/205/220 patients. Zero cross-split overlap. Test consumed twice (sanctioned): Stage 1 final result (2026-07-25) and Step 4 Option B second read (2026-08-01).
- **Limitations**: 16.2:1 imbalance; ~35% missingness on lifestyle/socioeconomic fields (real, not imputed); 179 patients have images spanning >1 diagnosis (dominant-diagnosis stratification accepted as a documented limitation); Fitzpatrick under-representation. **Positive**: the only one of the 4 original datasets with zero image overlap with any other source.

### 2.2 HAM10000 (benchmark / cross-dataset generalization target)
- **Citation**: Tschandl, Rosendahl & Kittler, *"The HAM10000 dataset..."*, *Scientific Data*, 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161.
- **License**: CC BY-NC 4.0 (Harvard Dataverse deposit; mirrored on ISIC Archive).
- **Size**: 10,015 images (600×450 uniform), 7,470 unique lesions (no patient ID).
- **Class distribution**: nv 6,705 (66.95%), mel 1,113 (11.11%), bkl 1,099 (10.97%), bcc 514 (5.13%), akiec 327 (3.27%), vasc 142 (1.42%), df 115 (1.15%). Imbalance ratio 58.3:1.
- **Leakage audit**: `diagnosis_confirm_type` excluded — phi=0.41, chi²=1700.67, n=10,015 (100% of malignant images confirmed via `histo`, deterministic in the malignant→histo direction).
- **Feature whitelist**: 3 allowed (`age, sex, anatomical_site`).
- **Split**: lesion-wise, stratified, seed 42, ratios 0.70/0.15/0.15. Train 7,004 (5,247 lesions) / val 1,501 (1,114) / test 1,510 (1,109). Test consumed once, 2026-07-25 (PAD→HAM generalization run).
- **Critical caveat**: **not independent** of the ISIC archives — 98.6% of images (9,873/10,015) overlap with ISIC Archive 2, 66.5% with ISIC Archive 1; 46% of the Archive-2 overlap is cross-split. Resolved via exclusion lists, never pooled in training.

### 2.3 ISIC Archive 1 (external validation)
- **Citation**: **not found, not safely inferable** — no metadata.csv, readme, or license file in-repo. Circumstantial evidence (66.5%/81.7% image overlap with HAM10000/Archive 2) suggests a repackaging of the ISIC image pool, but the specific release/mirror is unconfirmed. **Recommendation**: cite generically as "a repackaging of images from the ISIC Archive (isic-archive.com), exact release unconfirmed" and disclose this as a known provenance gap, or trace the original download source if recoverable. Do not guess a specific paper.
- **Size**: 2,357 raw images (folder-name-as-label, no metadata.csv), reduced to 2,047 after excluding 155 images (310 folder-entries) filed under conflicting class labels.
- **Class distribution (post-cleaning)**: 9 classes; test split stark — BCC/Dermatofibroma/Melanoma/Nevus/Pigmented Benign Keratosis/SCC each 16, Actinic Keratosis 1, Vascular Lesion 3, Seborrheic Keratosis 0.
- **Leakage audit**: 0 model-input columns exist at all — image-only dataset. `external_validation_exclusions.csv` (1,362 IDs, HAM10000 overlap) applies only when scored as HAM10000 external validation.
- **Split**: archive's own Test kept untouched; val carved from Train only, 15%, stratified, seed 42. Train 1,655 / val 292 / test 100.
- **Limitations**: small/imbalanced test set; no patient ID; not independent of HAM10000/Archive 2. In the HAM10000→Archive1 transfer eval (n=678), mean macro-F1 only 0.2421 — attributed to near-zero support for 2 of 5 shared classes, not worse true generalization (Archive 2's 0.4912 explicitly documented as the more representative number).

### 2.4 ISIC Archive 2 (external validation)
- **Provenance**: Kaggle mirror `andrewmvd/isic-2019` = ISIC 2019 Challenge training pool, 3 contributing sources (Hospital Clínic de Barcelona 12,302 rows / ViDIR Group Vienna 9,873 / Anonymous 2,901).
- **Citations** (medium-high confidence, externally researched — not present in-repo docs before 2026-07-28): ViDIR/Vienna portion = same Tschandl et al. 2018 HAM10000 citation. Hospital Clínic portion = Combalia et al., *"BCN20000: Dermoscopic Lesions in the Wild,"* arXiv:1908.02288 (2019). "Anonymous" portion (2,901 rows, likely MSK) — **not independently confirmed**, flagged for direct verification before citing a specific paper.
- **License**: CC0 1.0 Universal (the 2,901 "Anonymous"/CC-0 rows, confirmed 1:1) + CC BY-NC 4.0 (the rest). Per-row in `metadata.csv`'s `copyright_license` column.
- **Size**: 25,331 images, 25,076 with a populated `diagnosis_3` label (used as class label).
- **Class distribution**: Nevus 50.81%, Melanoma NOS 16.34%, BCC 13.12%, ... down to Epidermal Nevus/Atypical Melanocytic Neoplasm (1 each). Imbalance ratio 61.6:1.
- **Leakage audit**: `diagnosis_confirm_type` (phi=0.36, chi²=3171.74), `concomitant_biopsy` (duplicate signal), `melanocytic` (perfect deterministic split of Melanoma+Nevus vs. rest), plus 6 institution-proxy fields (`anatom_site_3/4/5, family_hx_mm, personal_hx_mm, clin_size_long_diam_mm, dermoscopic_type`) all excluded.
- **Feature whitelist**: 6 allowed in principle (`age_approx, anatom_site_1, anatom_site_2, anatom_site_general, anatom_site_special, sex`); active Stage 1 baseline uses only 4 (`age_approx, sex, anatom_site_1, anatom_site_general`).
- **Split**: group-wise (`lesion_id` else singleton `image_id`), stratified, seed 42, ratios 0.70/0.15/0.15. Train 17,535 / val 3,769 / test 3,772 (of 25,076 labeled rows).
- **Limitations**: severe imbalance; `patient_id` populated for <2%; massive HAM10000 overlap (essentially a superset).

### 2.5 DERM12345 (dataset-expansion source, folded into `PAD_UFES20_Expanded`)
- **Citation**: Harvard Dataverse, DOI 10.7910/DVN/DAXZ7P; secondary coverage PMC11604664. Turkish origin (3 institutions), 2008–2021, 1,627 patients, 12,345 dermoscopic images, 40 subclasses, official train/test split.
- **License**: CC BY 4.0.
- **Labels**: malignant biopsy-confirmed; benign labels by two dermatologists (20+ yrs) where no histology/follow-up existed.
- **Used in this project**: Melanoma-family (400 images, all 5 sub-classes) + SCC (266 images) = 666 images (not "400+266=666" independently re-verified against `c01_build_expanded_dataset.py` row counts in this pass — cross-check against the actual expanded CSV before citing exact per-class DERM12345-only counts in the thesis; the *combined* new-row totals below (§2.7) are the numbers directly confirmed from `data/processed/PAD_UFES20_Expanded/`).
- **Leakage/overlap audit — hard blocker, resolved**: exact `image_id` string-matching (same method/script as the HAM10000↔ISIC discovery) against the full combined ISIC Archive 1+2 ID set (27,123 IDs) found **zero overlap** (`DERM_*` vs `ISIC_*` disjoint namespaces, confirmed not just assumed). The DERM12345 paper's claim of future ISIC-Archive availability is future-tense (2024) and postdates this project's pre-2024 ISIC pulls — corroborating, not the basis for the decision.
- **Not independently audited/cleaned** the way the 4 original datasets were (no `src/data_audit/derm12345/` module tree) — integrated directly as a training-data supplement via `c01_build_expanded_dataset.py`, not as a standalone dataset object with its own train/val/test splits.

### 2.6 MED-NODE (dataset-expansion source, folded into `PAD_UFES20_Expanded`)
- **Citation**: Giotis et al., *MED-NODE: A computer-assisted melanoma diagnosis system using non-dermoscopic images*, Univ. Medical Center Groningen (2015) — full text paywalled, confirmed only via abstract/secondary sources.
- **License**: CC BY 4.0 (confirmed via a secondary aggregator listing, not stated on the dataset's own page).
- **Size**: 170 macroscopic clinical images total — 70 melanoma, 100 nevus, no SCC. Image-only, no tabular metadata.
- **Used in this project**: 70 melanoma images added to `PAD_UFES20_Expanded`'s training set.
- **Modality note**: notably modality-matched to PAD-UFES-20 (both macroscopic/clinical photography, unlike ISIC/HAM10000's dermoscopic images) — a genuine compatibility point.
- **Gap, disclosed not hidden**: the exact labeling methodology (biopsy vs. clinical diagnosis) could not be confirmed from any freely accessible source — reported as a clinically-plausible inference (hospital dermatology dept.), not a verified fact.
- **Overlap**: no evidence found connecting MED-NODE to ISIC/HAM10000; independent single-institution origin, non-dermoscopic modality.

### 2.7 PAD_UFES20_Expanded (derived training set, not an independent dataset)
- Built 2026-07-29 (`src/data_cleaning/pad_ufes20_expanded/c01_build_expanded_dataset.py`). Adds DERM12345 (Melanoma-family + SCC) + MED-NODE (Melanoma) = **736 new rows**, **train-only, image-branch-only** (`metadata_train_image_only.csv` = 2,342 rows = 1,606 original + 736 new). New rows have no compatible clinical metadata schema, so this variant is never used for metadata or fusion training — image branch only, enforced by a hard code guard in `config.py` that raises if `--branch metadata/fusion/*` is requested against it.
- **Class effect**: Melanoma 38→508 train images, SCC 135→401; other 4 classes unchanged (BCC 586, ACK 513, Nevus 167, SEK 167).
- **Val/test**: byte-identical to original PAD-UFES-20 (`DataFrame.equals()`-verified) — never touched by expansion. This "Step 2 binding rule" exists specifically to keep a valid *paired* bootstrap comparison possible against the already-locked 0.6977 headline result.
- **Leakage note specific to this variant**: `dataset_source` itself is a modality-confound risk (DERM12345 is dermoscopic vs. PAD-UFES-20's macroscopic — only the Melanoma/SCC classes are mixed-modality within this expanded set) — flagged, not resolved by exclusion (accepted as an inherent property of the expansion, disclosed in the thesis if this variant's results are used).
- **Bug caught and fixed pre-training**: new rows' `image_path` was initially written as absolute Windows paths, incompatible with `resolve_image_path()`'s relative-path assumption. Fixed and re-verified (736/736 resolve) before any Kaggle run.

---

## 3. Models

All variants share `src/models/dataset.py` (shared `FusionDataset`/`MetadataPreprocessor`) and `src/models/config.py` (hyperparameters, dataset registry). No LR scheduler anywhere except the abandoned "improved" cross-attention variant. Class-weighted `CrossEntropyLoss` (inverse train-split frequency), batch size 32, 30 max epochs, early-stopping patience 7 on val macro-F1, 3 seeds `[0,1,2]` throughout, Adam (plain, not AdamW), weight_decay 1e-4 — this recipe is constant across every variant below unless stated otherwise.

### 3.1 Image-Only (Stage 1 baseline)
- **Architecture**: EfficientNet-B0, ImageNet-pretrained, fully fine-tuned (no frozen layers), final `Linear` head replaced (1280→num_classes).
- **Input**: `ResizePad(224)` (aspect-preserving) + train-only flip/rotation(20°)/color-jitter → ImageNet-normalized `[B,3,224,224]`.
- **LR**: 1e-4. Warm-start: none (ImageNet only).
- **Results**: PAD-UFES-20 val 0.5703±0.0130, **test 0.6175±0.0153** (official). HAM10000 val 0.6940±0.0041. PAD→HAM cross-dataset 0.4658±0.0373. HAM→ISIC-Archive1 0.2421±0.0118. HAM→ISIC-Archive2 0.4912±0.0094.

### 3.2 Metadata-Only (Stage 1 baseline)
- **Architecture**: MLP, `Linear(input_dim,128)→BN→ReLU→Dropout(0.3)→Linear(128,64)→BN→ReLU→Dropout(0.3)→Linear(64,num_classes)`. PAD-UFES-20 fitted `input_dim=89` (confirmed via checkpoint metadata, not hardcoded — re-verify against checkpoint if reused for a new schema).
- **LR**: 1e-3. Warm-start: none (random init).
- **Results**: PAD-UFES-20 val 0.5762±0.0072, **test 0.6077±0.0202**. HAM10000 val (3-feature) 0.2521±0.0104. PAD→HAM cross-dataset 0.2920±0.0121. HAM→ISIC-Archive2 0.2410±0.0295. Bootstrap: metadata significantly weaker than image on both cross-dataset transfers (p<0.001 both).

### 3.3 Late Fusion (Phase 7 Stage 1)
- **Architecture**: `ImageEmbedder` (EfficientNet-B0 minus head, →1280-d) ⊕ `MetadataEmbedder` (MLP minus head, →64-d), plain `torch.cat` → 1344-d → `Linear(1344,128)→BN→ReLU→Dropout(0.3)→Linear(128,num_classes)`.
- **LR**: 1e-5. Warm-start: seed-matched Stage 1 `image_seed{N}_best.pt` + `metadata_seed{N}_best.pt`, `strict=True`, then fully fine-tuned end-to-end.
- **Diagnosed limitation** (motivates §3.4): 1280:64 dimension imbalance lets the image branch numerically dominate even through a 2-layer head.
- **Results**: PAD-UFES-20 val 0.5731±0.0021, **test 0.6566±0.0234**. `fusion_reduced` (3-feature, cross-dataset variant) val ≈0.5838. No ISIC/HAM10000 results exist by scope (Phase 7 was PAD-UFES-20-only), not a failed attempt.

### 3.4 Cross-Attention Fusion (Phase 7 Stage 2) — **thesis headline model**
- **Architecture**: `SpatialImageEmbedder` (EfficientNet-B0 conv features, `[B,1280,7,7]`→flatten→`[B,49,1280]` spatial tokens) + `MetadataEmbedder` (→64-d) + optional `MetadataChannelGate` (metadata-conditioned sigmoid gate over the 1280 channels, on by default) → `query_proj: Linear(64,256)` (metadata=Query), `kv_proj: Linear(1280,256)` (image tokens=Key/Value) → one `nn.MultiheadAttention(d_model=256, num_heads=8, dropout=0.1, batch_first=True)` → attended `[B,256]` ⊕ raw metadata `[B,64]` = 320-d joint → `Linear(320,128)→BN→ReLU→Dropout(0.3)→Linear(128,num_classes)`.
- **Explicitly not a MetaBlock reproduction**: MetaBlock computes one uniform gate across all spatial positions; this model computes genuine per-spatial-location attention weights.
- **LR**: 1e-5. Warm-start: same as late fusion, then fine-tunes everything including the new query/kv/attention/gate/head params.
- **Reduced-feature variant**: separate checkpoints (`metadata_input_dim=16`), image embedder still warm-starts from the unchanged original checkpoint, used only for the PAD→HAM cross-dataset run. Never mixed with the full-feature checkpoints.
- **Abandoned "improved" variant**: FocalLoss+WeightedRandomSampler+stronger augmentation+CosineAnnealingLR scored mean 0.509 (well below 0.6209) — **these specific numbers are user-reported and have no surviving checkpoint/summary JSON in the repo**; cite with that caveat. Decision to not adopt stands regardless of exact numbers.
- **Results**: PAD-UFES-20 val 0.6209±0.0143, **test 0.6977±0.0269 — locked, official, single sanctioned Stage-1 test-set use** (also the source of the Fitzpatrick fairness result, same run). PAD→HAM cross-dataset (`cross_attention_reduced`) 0.4654±0.0197. Cross-attention's minimum val seed (0.6049) exceeds every other Stage-1/7 variant's maximum val seed — zero overlap across the 4-way comparison.
- **Fitzpatrick fairness** (test split, mean of 3 seeds): group 1 (n=22) 0.3307, group 2 (n=120) 0.6212, group 3 (n=59) 0.5706, group 4 (n=15) 0.4365 — best or tied-best of all 4 Stage-1/7 architectures in every reportable group; groups 5/6 unreportable (n=2, n=1).
- **Bootstrap significance (PAD→HAM)**: vs. image −0.0004 (n.s.); vs. metadata +0.1734 (p<0.001, significant); vs. late-fusion +0.0057 (n.s.) — cross-attention's within-dataset lead does not transfer to a *statistically significant* cross-dataset edge over image-only or late-fusion.
- **Ensemble/TTA explored and permanently rejected** (2026-07-19): Melanoma F1 collapses 0.364→0.20 under ensemble+TTA — disqualifying, not just "no improvement."

### 3.5 5-Backbone Comparison (Phase 8B Step 3, image-only, `PAD_UFES20_Expanded`)
Source: `docs/Phase8B_Backbone_Comparison_Results.md`, `logs/PAD_UFES20_Expanded/train_image_*_summary.json`, 3 seeds each.

| Backbone | Seed0 | Seed1 | Seed2 | Mean | Std |
|---|---|---|---|---|---|
| EfficientNet-B0 | 0.5910 | 0.5937 | 0.5800 | 0.5882 | 0.0059 |
| MobileNetV3-Large | 0.6351 | 0.5709 | 0.5734 | 0.5931 | 0.0297 |
| DenseNet121 | 0.5769 | 0.5959 | 0.5991 | 0.5906 | 0.0098 |
| ResNet50 | 0.6094 | 0.5713 | 0.5770 | 0.5859 | 0.0168 |
| **ConvNeXt-Tiny** | 0.6416 | 0.5998 | 0.6257 | **0.6224** | 0.0172 |

Baseline (non-expanded, EfficientNet-B0): 0.5703±0.0130 — all 5 expanded-dataset backbones beat it on mean. **Top-2 selected for Step 4: ConvNeXt-Tiny + DenseNet121**, chosen on mean **and** stability (not mean alone) — MobileNetV3's higher mean is a single-seed (seed0=0.6351) outlier artifact; DenseNet121 clears baseline on all 3 seeds with 3× tighter spread (std 0.0098 vs. 0.0297). This two-criteria rule is documented project methodology, citable directly.

### 3.6 Step 4 (Option B) — Cross-Attention Backbone Fusion (ConvNeXt-Tiny / DenseNet121, single-backbone) + Ensemble
- **Architecture**: `CrossAttentionBackboneFusionModel` — structurally identical to §3.4's cross-attention model (metadata Query, image spatial-token K/V, `d_model=256`, `num_heads=8`, single `nn.MultiheadAttention`, optional channel gate), but the image embedder is `SpatialBackboneEmbedder` parameterized per backbone: ConvNeXt-Tiny → 49×768 tokens (`.features(x)`); DenseNet121 → 49×1024 tokens (explicit `F.relu(.features(x))`, since DenseNet applies that ReLU outside `.features`). Two **independent** full models (no shared parameters) — combined only at *inference* via unweighted softmax-probability averaging (`cross_attention_backbone_ensemble`), not jointly trained (contrast with §3.8).
- **Warm-start**: image embedder ← Step 3's `PAD_UFES20_Expanded` backbone checkpoint (`image_convnext_tiny_seed{N}_best.pt` / `image_densenet121_seed{N}_best.pt`); metadata embedder ← PAD-UFES-20's own Phase 7 Stage 1 `metadata_seed{N}_best.pt`. Fine-tuned on PAD-UFES-20's **original, unexpanded** train/val (expanded rows have no metadata). LR 1e-5 (same as §3.4).
- **Val**: ConvNeXt-Tiny 0.6542/0.6731/0.6856; DenseNet121 0.6363/0.6714/0.6601; ensemble 0.6661/0.6952/0.6956 (mean 0.6856).
- **Test** (second sanctioned PAD-UFES-20 test-split read, 2026-08-01):

| Variant | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| ConvNeXt-Tiny | 0.6897 | 0.6994 | 0.7425 | **0.7105** |
| DenseNet121 | 0.6630 | 0.6946 | 0.6980 | **0.6852** |
| **Ensemble** | 0.7081 | 0.7338 | 0.7542 | **0.7321** |
| Original (§3.4, locked, reused) | — | — | — | 0.6977 |

- **Paired bootstrap** (1,000 resamples, row-level, seed-averaged): ConvNeXt-Tiny vs. original +0.0128 (p=0.470, n.s.); DenseNet121 vs. original −0.0126 (p=0.656, n.s.); **ensemble vs. original +0.0343, 95% CI [−0.0020, +0.0771], p=0.062 — not significant** (CI crosses zero, barely). Bonferroni-corrected α=0.0167: also not significant.
- **Conclusion**: ensemble (0.7321) is the numerically best result to date and closest to significance, but per the pre-committed decision rule, **the original locked cross-attention (§3.4, test 0.6977) remains the thesis headline**; Option B is documented as a positive, not-yet-significant exploratory finding.

### 3.7 Dataset-Expansion-Only Ablation (EfficientNet-B0, original architecture)
- **Purpose**: isolate the dataset-expansion effect from the backbone-architecture-change effect observed in Step 4 — trains the **original** `CrossAttentionFusionModel` (§3.4 architecture, EfficientNet-B0), warm-starting the image embedder from Step 3's **expanded**-dataset EfficientNet-B0 checkpoint instead of the original Stage 1 checkpoint. Fine-tunes on PAD-UFES-20's original, unexpanded train/val. Val-only.
- **Pre-registered prediction** (2026-08-01, logged before training): val macro-F1 range 0.61–0.66.
- **✅ STATUS — VERIFIED 2026-08-03**: confirmed against local summary JSONs `logs/PAD_UFES20/train_cross_attention_efficientnet_expanded_seed{0,1,2}_summary.json` — `best_val_macro_f1` = 0.620970 (seed0) / 0.602331 (seed1) / 0.632408 (seed2), mean **0.6186**. Matches `Project_Tracking.md`'s Phase 8E inline citation exactly.
- **Working hypothesis being tested**: architecture (backbone choice) is the dominant driver of Step 4's gain, not dataset expansion alone (0.6186 confirms this — within the pre-registered 0.61–0.66 range and well below the 0.6710 architecture-change result).

### 3.8 Phase 8E (Option A) — Genuine Joint Three-Way Fusion — **COMPLETE (val-only), did not clear its own bar**
- **Scientific question**: does *jointly* fusing two backbones (one shared attention mechanism, trained together) capture complementary signal beyond Step 4's ensemble (0.7321 test), which only late-averages two independently-trained models?
- **Architecture** (`src/models/cross_attention_joint_fusion_model.py`, read in full): `CrossAttentionJointFusionModel(metadata_input_dim, num_classes, d_model=256, num_heads=8, dropout=0.3)`. ConvNeXt-Tiny tokens `[B,49,768]` and DenseNet121 tokens `[B,49,1024]` are each projected by their **own** `kv_proj_a`/`kv_proj_b: Linear(·,256)` into the shared 256-d space, then **concatenated along the token axis** → one `[B,98,256]` K/V sequence. Metadata supplies a single Query token (`query_proj: Linear(64,256)`) into one `nn.MultiheadAttention` (same config as §3.4/§3.6). Optional per-backbone `MetadataChannelGate` applied before projection. Joint = `cat(attended[256], metadata_embedding[64])` = 320-d → same 2-layer head as every other variant. The *only* new mechanism vs. Step 4 is the per-backbone projection + token-concatenation step — no new attention variant.
- **Warm-start**: `image_embedder_a`/`image_embedder_b` ← Step 3's `PAD_UFES20_Expanded` checkpoints (convnext_tiny/densenet121 seed-matched); metadata ← PAD-UFES-20's own Stage 1 `metadata_seed{N}_best.pt` — identical pattern to Step 4. Trains on PAD-UFES-20's original train/val (unexpanded rows have no metadata). LR 1e-5.
- **Naming**: `cross_attention_joint_convnext_densenet_seed{0,1,2}` — verified non-colliding with every prior checkpoint prefix before training started (repeated project safety drill).
- **Test-split discipline**: validation only this round. The test split has already been consumed twice (§2.1). A third read is **not automatic** — requires validation to "clearly and meaningfully exceed" Step 4's best single-backbone val result (0.6710, ConvNeXt-Tiny), not a marginal beat.
- **Pre-registered prediction** (2026-08-02, logged before training): val macro-F1 range **0.66–0.71**, explicitly framed as genuinely uncertain whether it even clears the 0.6710 single-backbone bar, let alone Step 4's ensemble — reasoning: joint training compresses both backbones through one shared bottleneck and typically captures *less* complementary diversity than late-averaging two independently-optimized models.
- **Result (all 3 seeds, Kaggle run, reported 2026-08-03)**: best val macro-F1 seed0 0.6575 / seed1 0.6927 / seed2 0.6660, **mean 0.6721** — inside the pre-registered 0.66–0.71 range. **Locally verified 2026-08-03**: `logs/PAD_UFES20/train_cross_attention_joint_convnext_densenet_seed{0,1,2}_summary.json` confirmed present on disk with `best_val_macro_f1` = 0.657466 / 0.692674 / 0.665969 (mean 0.672036, rounds to 0.6721) — matches the reported numbers exactly, same verification standard used for §3.7's 0.6186.
- **Decision-rule outcome**: mean (0.6721) exceeds the 0.6710 bar by only **+0.0011** — not the "clear and meaningful" margin the pre-registered rule (§3.8 above, set 2026-08-02) required to trigger a third test-split read. Per that rule, **no test-split evaluation was performed**. Phase 8E is reported as a val-only result that narrowly missed its own bar, not as a headline candidate. See §6 for the honestly-documented limitation this creates (observed val→test gaps elsewhere in the project suggest a test score could plausibly have been higher — deliberately not chased, to preserve the pre-registration's integrity).

---

## 4. Methodology Flow (full pipeline, start to finish)

1. **Raw data acquisition** → `data/raw/`. No acquisition/download script exists (manual acquisition, Kaggle "Add Data" slugs referenced only in notebook headers).
2. **Audit** → `src/data_audit/<dataset>/m0N_*.py` (PAD-UFES-20: 12 modules; HAM10000: 11; ISIC Archive 1: 4, folder-derived; ISIC Archive 2: 8). Each dataset's final module produces `reports/<dataset>/*_dataset_audit_summary.md`.
3. **Cleaning** → `src/data_cleaning/<dataset>/c0N_*.py`: column standardization → value validation → label standardization → split → split-quality report → dataset description doc.
4. **Splitting** → seed 42 throughout (`src/data_cleaning/config.py`), `np.random.default_rng(42)`. Method varies by available grouping key: PAD-UFES-20 patient-wise; HAM10000 & ISIC Archive 2 lesion-/group-wise; ISIC Archive 1 archive-preserving (own Test kept, val carved from Train).
5. **Feature whitelist generation** → hand-authored per dataset (`feature_whitelist.md`), not script-generated; phi/chi-square leakage evidence computed ad hoc during a documented review-and-approve step.
6. **Cross-dataset leakage fix** → `cross_dataset_leakage_filter.py` (HAM10000-overlap exclusion lists for both ISIC archives) + `label_conflict_filter.py` (3 cross-archive label disagreements). Fix (b) — restrict external-validation claims — chosen over fix (a) — global re-split — to avoid discarding already-verified individual splits to solve a problem specific to one use case.
7. **Dataset expansion** (Phase 8C, PAD-UFES-20 only) → DERM12345 + MED-NODE folded into `PAD_UFES20_Expanded`'s train-only, image-only CSV, ID-matched against the full ISIC corpus first as a hard blocker, not a soft check.
8. **Training** → `src/models/train*.py`, one script per variant (§3), all image-only/metadata-only/fusion training uses `src/models/dataset.py`'s shared `FusionDataset`/`MetadataPreprocessor`.
9. **Evaluation** → `src/evaluation/evaluate.py`, single entrypoint for all variants/splits. `--confirm-final` guard required for any `--split test` call; `test_split_guard.py`'s `TEST_SPLIT_CONSUMED.json` marker enforces single/sanctioned reuse across `evaluate.py`, `evaluate_fairness.py`, `evaluate_cross_dataset.py` (added 2026-07-25 after `evaluate_fairness.py` was found to bypass the CLI flag entirely).
10. **Cross-dataset validation** → `evaluate_cross_dataset.py` (PAD-UFES-20→HAM10000, 3 shared classes) and `evaluate_external_isic.py` (HAM10000→both ISIC archives, both exclusion lists applied).
11. **Fairness testing** → `evaluate_fairness.py`, all 4 Stage-1/7 variants × 3 seeds on PAD-UFES-20's test split (Fitzpatrick doesn't exist in HAM10000). 1,000 resamples, seed 42, percentile 95% CI, n<15 excluded / 15≤n<30 flagged / n≥30 clean.
12. **Significance testing** → `bootstrap_significance.py` (PAD→HAM), `bootstrap_significance_isic.py` (HAM→ISIC-Archive2, image vs. metadata), `bootstrap_significance_backbone_fusion.py` (Step 4 variants vs. locked headline). All: 1,000 resamples, seed 42, paired row-level resampling, per-iteration macro-F1 averaged across 3 seeds, two-sided p, significance = 95% CI excludes 0.

---

## 5. Key Decisions and Why

**Leakage exclusions treated as hard, evidence-gated blockers, not soft warnings** (2026-07-08 onward). Every excluded feature across all 4 original datasets carries a quantified phi/chi-square value computed before exclusion, not a judgment call alone (`biopsed` phi=0.80; HAM10000 `diagnosis_confirm_type` phi=0.41; ISIC Archive 2's `melanocytic` a perfect deterministic split). **Why**: motivated explicitly by Watson et al.'s leakage-methodology caution in the literature review — a feature that trivially predicts the label (even indirectly, via institution-of-origin proxies) invalidates any downstream performance claim, so the bar was set at "prove it's clean," not "assume it's clean."

**Cross-dataset overlap fixed via exclusion lists (fix b), not a global re-split (fix a)** (2026-07-08). Discovering 40–46% of HAM10000/ISIC-overlap images land in different splits across the independently-split datasets could have triggered a full re-split of all 3 datasets together. Rejected because it would discard already-verified, already-approved individual splits to solve a problem specific only to the external-validation use case — the exclusion-list approach scopes the fix to exactly where it's needed.

**Patient-/lesion-/group-wise splitting only, never image-wise** (`PROJECT_PLAN.md` confirmed decision, applied throughout). Prevents the same patient/lesion appearing in both train and test, which would let a model memorize lesion-specific texture rather than learn generalizable features — enforced per-dataset using whatever grouping key actually exists (patient ID, lesion ID, or archive-preserving fallback when neither exists).

**Cross-attention chosen over late fusion / MetaBlock** (2026-07-18). Late fusion's diagnosed limitation is a raw 1280:64 image:metadata dimension imbalance that lets the image branch numerically dominate even through a deeper joint head. Cross-attention projects both modalities into a shared `d_model=256` space *before* interaction, removing that mechanical bias. Explicitly framed as "contrasted with MetaBlock's channel-gating," never "MetaBlock-inspired" — MetaBlock computes one uniform gate across all spatial positions; this model computes genuine per-spatial-location attention weights, a structurally different mechanism.

**Ensemble/TTA rejected outright despite no macro-F1 regression** (2026-07-19). Aggregate macro-F1 looked fine, but per-class inspection showed Melanoma F1 collapsing 0.364→0.20 under ensembling+TTA — the minority, clinically highest-stakes class. Treated as disqualifying on clinical-relevance grounds, not merely "no improvement," even though the project's headline metric alone wouldn't have caught this.

**Option B (independent-model ensemble) run before Option A (genuine joint fusion)** (2026-07-31). The literal supervisor instruction was "one final joint fusion model" (Option A). Option B was built first anyway as a risk-managed sequencing choice: it reuses the already-validated Stage 2 cross-attention mechanism unchanged (low technical risk, guaranteed usable result), while Option A requires solving a genuine architectural problem (fusing two backbones with mismatched token dimensions, 768 vs. 1024). Whichever result beats the locked 0.6977 headline *and* is statistically significant becomes the new thesis headline; if neither does, 0.6977 stands and both become documented ablations — this decision rule was set before either result was known.

**Test-split single-use safeguard added after a real gap was found** (2026-07-25). `evaluate_fairness.py` was found to bypass `evaluate.py`'s `--confirm-final` CLI guard entirely by calling its helper functions directly. Rather than patch that one script, a dataset-scoped `TEST_SPLIT_CONSUMED.json` marker file was added and enforced across all three evaluation entrypoints — closes the class of bug, not just the one instance found.

**DERM12345's ISIC-overlap risk treated as a hard blocker, reject-if-found** (2026-07-29, explicit user direction). Rather than adding a second exclusion-list mechanism to maintain alongside the existing HAM10000/ISIC one, the rule was: if DERM12345 shares any image ID with the existing ISIC pulls, reject it outright. The check (exact ID-matching, same method as the original HAM10000 discovery) came back clean (0/12,345 overlap) before any integration work began.

**Lesion segmentation/cropping deliberately deferred** (2026-07-29). Reasoned through and explicitly rejected for this phase specifically to avoid stacking three simultaneous variables (dataset expansion + backbone change + segmentation) in one experiment — the project had already lived through one uninterpretable negative result (the abandoned "improved cross-attention" variant, which changed 4 things at once and couldn't attribute its failure to any one of them) and did not want to repeat that failure mode.

---

## 6. Current Final Status

- **LOCKED, final, thesis headline as of 2026-08-03**: Cross-Attention Fusion (§3.4), PAD-UFES-20 test macro-F1 **0.6977 ± 0.0269** (mean of seeds 0.6862/0.6721/0.7349). This is final and will only change if Phase 8E's result clears the pre-committed bar below. **Note: the same 3-seed test predictions also give mean accuracy 0.763 ± ~0.024 (seeds 0.7712/0.7345/0.7825) — accuracy and macro-F1 are both legitimate, both computed on the exact same predictions, but are not interchangeable (macro-F1 is unweighted across the 6 imbalanced classes, accuracy is dominated by the majority classes BCC/AK); macro-F1 (0.6977) is the reported headline metric per this project's metrics discipline, accuracy is reference-only.**
- **Complete, documented, exploratory (not headline)**: Step 4 Option B backbone-fusion ensemble, test 0.7321, +0.0343 over headline, **not statistically significant** (p=0.062, 95% CI crosses zero by 0.002). Stands as a positive finding to present alongside the headline, not a replacement for it.
- **COMPLETE, val-only, did not clear its own bar**: Phase 8E (Option A, genuine joint three-way fusion, §3.8) — val macro-F1 mean 0.6721 (seeds 0.6575/0.6927/0.6660), exceeds the pre-registered 0.6710 bar by only +0.0011, not the "clear and meaningful" margin required. **No third test-split read performed.** Reported as a val-only exploratory result, not a headline candidate.
- **VERIFIED 2026-08-03**: the dataset-expansion-only ablation's 0.6186 figure (§3.7) — confirmed against local summary JSONs, mean of seeds 0.6210/0.6023/0.6324.
- **Decision rule for the thesis headline** (set before any Phase 8E result was known, 2026-07-31/08-02): whichever of {0.6977 original, Option B ensemble, Option A joint} both beats 0.6977 numerically *and* clears statistical significance becomes the headline. If none does, 0.6977 remains the headline and the others are reported as documented, informative negative/inconclusive results — this is a pre-registered rule, not a post-hoc rationalization, and should be stated as such in the thesis. **Outcome (2026-08-03): none of the three cleared both bars — Option B's ensemble is numerically best but not significant (p=0.062), and Option A didn't clear its own pre-condition to even attempt a test read. 0.6977 (§3.4) is the final thesis headline.**

**Limitation, honestly documented — val-to-test gap pattern (2026-08-03):**
every variant evaluated on both val and the locked test split scored
higher on test than on val: headline 0.6209→0.6977 (+0.0768), Step 4
ensemble 0.6856→0.7321 (+0.0465). Phase 8E's val mean (0.6721) missed
its 0.6710 decision-rule bar by only +0.0011 — under the observed
pattern, a hypothetical test evaluation could plausibly have scored
higher, possibly even above 0.6977. **The test split was deliberately
not reopened for this reason** — doing so specifically because of a
post-hoc "it might do better" observation would defeat the purpose of
pre-registering the 0.6710 bar in the first place (the bar exists
precisely to prevent rationalized re-testing of a near-miss). This is
disclosed here as a disciplined choice to preserve test-set integrity,
at the possible cost of a marginal improvement not chased this thesis
cycle — and as a candidate future-work item (a *fresh*, independently
held-out evaluation set, not a third read of this project's
already-twice-consumed PAD-UFES-20 test split). The pattern itself
rests on only 2 prior val/test pairs — an honest observation, not a
statistically established property of the pipeline.
- **Not yet started**: Phase 9 (Thesis Writing Support), Phase 10 (arXiv preprint/submission).
- **Outstanding, low-priority**: `src/common/paths.py` config-duplication refactor (flagged 2026-07-08, never actioned); PAD-UFES-20's exact Mendeley license text (needs the user to open the page directly); ISIC Archive 1's citation (genuinely unrecoverable from repo evidence alone).

---

## 7. Paper-Writing Quick-Reference

### Headline numbers (PAD-UFES-20, official test split, macro-F1 mean ± std)
| Model | Test macro-F1 |
|---|---|
| Image-only (EfficientNet-B0) | 0.6175 ± 0.0153 |
| Metadata-only (MLP) | 0.6077 ± 0.0202 |
| Late fusion | 0.6566 ± 0.0234 |
| **Cross-attention fusion (headline)** | **0.6977 ± 0.0269** |
| Step 4 ConvNeXt-Tiny cross-attention | 0.7105 |
| Step 4 DenseNet121 cross-attention | 0.6852 |
| Step 4 dual-backbone ensemble (best, n.s.) | 0.7321 |
| Phase 8E joint fusion | val-only 0.6721 — **did not clear its bar for a test read, no test number exists** |

### Literature comparison caveat — TRACE / DANET (arXiv:2411.08701), PAD-UFES-20

**Correction, verified directly from the TRACE paper (2026-08-03):** TRACE
reports DANET at macro-F1 0.625 and TRACE itself at 0.783 on
PAD-UFES-20 — but **these numbers are not directly comparable to this
thesis's 0.6977 test result**, for two independent reasons:

1. **Split mismatch.** TRACE evaluates on a **90/10 train-validation
   split with no held-out test set** — their 0.625/0.783 are
   validation-style numbers, evaluated on data the model selection
   process could still be influenced by, not numbers from a locked,
   single-use test split the way this thesis's 0.6977 is (§2.1, `TEST_SPLIT_CONSUMED.json`
   discipline). Comparing their val-style number directly against our
   test number overstates the gap in either direction.
2. **Different task.** TRACE is **metadata-only (tabular)** — it does
   not fuse image and metadata. This thesis's headline (§3.4) is an
   image+metadata cross-attention fusion model. A metadata-only tabular
   model and a multimodal fusion model are not architecturally
   comparable as "competitors" on the same task.

**Fairer comparison:** this thesis's own **validation** macro-F1
(0.6209, §3.4) is the correct like-for-like comparison point against
TRACE/DANET's validation-style numbers (both evaluated without a
locked test-split read). On that basis, DANET (0.625) and this
thesis's cross-attention fusion (0.6209 val) are close; TRACE's 0.783
is higher, but on a different task (tabular-only) — cite it as **a
separate, informative data point about how far metadata-only signal
alone can go on PAD-UFES-20's clinical fields**, not as a direct
competitor result to be beaten by this thesis's multimodal model. This
thesis's 0.6977 remains the number to report as the headline (it is a
genuine held-out test score for a fusion model), but must not be
placed in a same-row comparison table against TRACE/DANET's
val-only, metadata-only numbers without this caveat stated explicitly.

**Citation note:** TRACE paper identified here only by arXiv ID
(2411.08701) as given — full title/author citation not independently
looked up in this pass; confirm exact citation details before final
submission.

**Class-count claim, verified (2026-08-13):** TRACE's evaluation on
PAD-UFES-20 uses the dataset's native 6-class taxonomy (same 6 classes
this thesis uses), confirmed by direct inspection of Table II in the
TRACE paper (arXiv:2411.08701). Previously flagged as an unverified
"6 classes" claim not documented in tracked notes — now sourced and no
longer flagged.

### Cross-dataset transfer (val/test as noted)
- PAD→HAM (3 shared classes): image 0.4658, metadata 0.2920, late-fusion 0.4597, cross-attention 0.4654 — statistically indistinguishable except metadata (significantly worse, p<0.001).
- HAM→ISIC Archive 2 (n=12,508, image-only checkpoints): image 0.4912, metadata 0.2410 (diff +0.2502, 95% CI [0.2368,0.2641], p<0.001 — largest, most significant effect in the whole project).
- HAM→ISIC Archive 1 (n=678, image-only): 0.2421 — low due to eval-set class-support composition, not worse true generalization (Archive 2's number is the representative one).

### Exact architecture dimensions for the Methods section
- Image branch (all variants): EfficientNet-B0 (or ConvNeXt-Tiny/DenseNet121 for Phase 8B+), ImageNet-pretrained, input `224×224×3`.
- Metadata branch: MLP `input_dim → 128 → 64 → num_classes`, BatchNorm+ReLU+Dropout(0.3) after each hidden layer. PAD-UFES-20 fitted `input_dim=89`.
- Late fusion joint dim: 1280 (image) + 64 (metadata) = 1344 → hidden 128 → num_classes.
- Cross-attention: `d_model=256`, `num_heads=8` (head_dim=32), single `nn.MultiheadAttention`, dropout 0.1 in attention / 0.3 in head. Image spatial tokens: 49 (7×7 conv grid) × 1280-d (EfficientNet-B0) / 768-d (ConvNeXt-Tiny) / 1024-d (DenseNet121), each projected to 256-d via its own `kv_proj`. Metadata → Query via `Linear(64,256)`. Joint = attended (256) ⊕ raw metadata (64) = 320-d → hidden 128 → num_classes. Joint (Phase 8E) variant concatenates two backbones' 49-token sequences into 98 tokens before attention.
- num_classes = 6 for PAD-UFES-20, 7 for HAM10000 (independent per-dataset heads, no shared taxonomy).

### Training hyperparameters (constant unless noted)
Adam, weight_decay=1e-4, class-weighted CE (inverse train-split frequency), batch 32, max 30 epochs, early-stopping patience 7 (val macro-F1), 3 seeds [0,1,2], no LR scheduler. LR: image-only 1e-4, metadata-only 1e-3, all fusion/cross-attention variants 1e-5 (deliberately conservative given warm-starting).

### Exact citations to use
- PAD-UFES-20: Pacheco et al. 2020, *Data in Brief*, 32, 106221. https://doi.org/10.1016/j.dib.2020.106221
- HAM10000: Tschandl, Rosendahl & Kittler 2018, *Scientific Data*, 5, 180161. https://doi.org/10.1038/sdata.2018.161
- ISIC Archive 2 (Vienna portion): same as HAM10000 above. (Barcelona portion): Combalia et al. 2019, arXiv:1908.02288 (BCN20000).
- DERM12345: Harvard Dataverse DOI 10.7910/DVN/DAXZ7P; secondary coverage PMC11604664.
- MED-NODE: Giotis et al. 2015 (Univ. Medical Center Groningen) — full citation details paywalled/incomplete, verify before submission.
- ISIC Archive 1: **do not cite a specific paper** — provenance unconfirmed, see §2.3.

### Leakage/methodology framing to cite verbatim in the paper's leakage section
"Feature exclusion was gated on quantified evidence (phi coefficient, chi-square statistic) computed on the full labeled set before any exclusion decision, not on judgment alone" — applies to `biopsed` (PAD-UFES-20, phi=0.80), `diagnosis_confirm_type` (HAM10000, phi=0.41; ISIC Archive 2, phi=0.36), `melanocytic` (ISIC Archive 2, perfect deterministic split), and the institution-proxy fields (ISIC Archive 2).

### Statistical methodology to cite verbatim
"All significance testing uses paired bootstrap resampling (1,000 resamples, seed 42), with per-iteration macro-F1 averaged across 3 training seeds per resample, two-sided p-values, and significance declared when the 95% percentile-method confidence interval excludes zero."

---

## 8. Viva Quick-Reference: Comparison, Honest Assessment, Novelty, Future Work

### 8.1 Cross-Model Comparison Table (one row per variant, viva lookup)

| Model variant | Trained on | Tested on | Why this combination | Final metric (macro-F1) |
|---|---|---|---|---|
| Image-only (EfficientNet-B0) | PAD-UFES-20 train/val | PAD-UFES-20 test | Single-modality baseline | **0.6175 ± 0.0153** |
| Metadata-only (MLP) | PAD-UFES-20 train/val | PAD-UFES-20 test | Single-modality baseline | **0.6077 ± 0.0202** |
| Late fusion | PAD-UFES-20 train/val | PAD-UFES-20 test | Naive-concatenation fusion baseline (motivates cross-attention) | **0.6566 ± 0.0234** |
| **Cross-attention fusion** | PAD-UFES-20 train/val | PAD-UFES-20 test | **Thesis headline** — fixes late fusion's 1280:64 dimension imbalance | **0.6977 ± 0.0269** |
| Image-only | PAD-UFES-20 (trained) | HAM10000 (3 shared classes) | Cross-dataset generalization check | 0.4658 ± 0.0373 |
| Metadata-only | PAD-UFES-20 (trained) | HAM10000 | Cross-dataset generalization check | 0.2920 ± 0.0121 |
| Cross-attention (reduced-feature) | PAD-UFES-20 (trained) | HAM10000 | Cross-dataset generalization check | 0.4654 ± 0.0197 |
| Image-only | HAM10000 (trained) | ISIC Archive 1 | External validation | 0.2421 ± 0.0118 (low support, not worse generalization — see §2.3) |
| Image-only | HAM10000 (trained) | ISIC Archive 2 | External validation, more representative n | 0.4912 ± 0.0094 |
| Metadata-only | HAM10000 (trained) | ISIC Archive 2 | External validation | 0.2410 ± 0.0295 |
| 5-backbone comparison (ConvNeXt-Tiny best) | PAD_UFES20_Expanded train | PAD_UFES20_Expanded val | Backbone selection for Step 4 | val mean 0.6224 |
| Step 4 ConvNeXt-Tiny cross-attention | PAD-UFES-20 (warm-start from expanded backbone) | PAD-UFES-20 test | Isolate backbone-swap effect | **0.7105** |
| Step 4 DenseNet121 cross-attention | same | PAD-UFES-20 test | Isolate backbone-swap effect | 0.6852 |
| Step 4 dual-backbone ensemble | same (2 independent models, softmax-averaged) | PAD-UFES-20 test | Test whether decorrelated errors help | **0.7321** (numerically best, **not significant**, p=0.062) |
| Dataset-expansion-only ablation (EfficientNet-B0, original arch) | PAD-UFES-20 (warm-start from expanded EfficientNet-B0) | PAD-UFES-20 **val only** | Isolate dataset-expansion effect from backbone-change effect | **0.6186** (mean of 3 seeds, verified §3.7) |
| Phase 8E joint three-way fusion | PAD-UFES-20 (warm-start from expanded ConvNeXt+DenseNet) | PAD-UFES-20 **val only** | Test genuine joint (not ensembled) two-backbone fusion | **0.6721** (mean of 3 seeds; missed 0.6710 bar by +0.0011, no test read — see §8.5) |

### 8.2 What We Improved / Attempted — Honest Results

- **Leakage audit** (quantified phi/chi-square exclusion of `biopsed`, `diagnosis_confirm_type`, `melanocytic`, institution-proxy fields): **success** — every exclusion is evidence-gated, not assumed; prevents inflated/invalid downstream metrics (§5, §2.1–2.4).
- **Cross-attention vs. late fusion** (shared 256-d projection space instead of raw 1280:64 concatenation): **success** — +0.0411 test macro-F1 over late fusion (0.6977 vs. 0.6566), became the thesis headline (§3.4).
- **Dataset expansion** (DERM12345 + MED-NODE, 736 new rows, Melanoma/SCC-focused): **inconclusive** — isolated effect (0.6186 val, §3.7) is smaller than the backbone-swap effect (0.6710 val) observed in the same expanded data; expansion alone did not clearly drive Step 4's gain.
- **Backbone comparison** (5 backbones on expanded data, ConvNeXt-Tiny/DenseNet121 selected by mean+stability): **success** — ConvNeXt-Tiny reached test 0.7105, the best single-backbone result to date (§3.5–3.6).
- **Ensemble** (independent ConvNeXt+DenseNet cross-attention models, softmax-averaged): **numerically positive, not statistically confirmed** — 0.7321 test, +0.0343 over headline, 95% CI crosses zero by 0.002 (p=0.062) — reported as a promising, not proven, finding (§3.6).
- **Joint fusion** (single shared-attention model over both backbones' concatenated tokens): **negative/inconclusive** — val mean 0.6721 across 3 seeds, missed the pre-registered 0.6710 decision-rule bar by only +0.0011 (not a "clear and meaningful" exceed), so no test-split read was performed. Consistent with the working hypothesis that joint training captures less complementary signal than late-averaging two independently-trained models (§3.6's ensemble). See §6 for the honestly-disclosed val→test gap limitation this creates.

### 8.3 Future Scope (honest, concrete)

1. **Lesion segmentation/cropping** — deliberately deferred (§5) to avoid stacking three simultaneous variables (dataset expansion + backbone change + segmentation) in one experiment; a clean next step once the current ablation chain concludes.
2. **Larger, Melanoma/SCC-specific dataset** — the dataset-expansion ablation's inconclusive result (§8.2) suggests 736 rows across two classes wasn't enough signal to isolate from backbone effects; a substantially larger, disease-targeted corpus is a more promising lever than incremental expansion.
3. **Equalized-odds or calibration-based fairness metric** — current fairness analysis (§3.4) reports only per-Fitzpatrick-group macro-F1; adding equalized odds or expected calibration error (per Literature Review row 19's framework) would strengthen the fairness claim beyond a single metric.
4. **Multi-institution external validation** — current cross-dataset checks (HAM10000, ISIC Archives 1/2) are all ISIC-family/dermoscopic sources; validating against an independent, non-ISIC-derived clinical cohort (e.g. a Check4Cancer-style teledermatology set, per Literature Review row 15) would test generalization beyond one data lineage.
5. **Larger-sample significance testing** — several key comparisons (Step 4 ensemble vs. headline, p=0.062) are borderline-non-significant on the current test-set size; a pre-registered follow-up with a larger held-out set (not a re-read of the already-consumed test split) could resolve these rather than leaving them as "not yet significant."

### 8.4 Novelty/Strength vs. the 19 Reviewed Papers (`docs/Literature_Review.md`)

- **Leakage-audit rigor**: this thesis quantifies phi/chi-square evidence before excluding any feature (`biopsed` phi=0.80, `diagnosis_confirm_type` phi=0.41/0.36, `melanocytic` perfect split). Of the 19 reviewed papers, only **row 3 (Watson et al.)** even raises leakage/shortcut-feature caution as a methodological point — none of the other 18 report a quantified feature-leakage audit at all.
- **Cross-dataset generalization testing**: this thesis explicitly evaluates PAD→HAM and HAM→ISIC-Archive-1/2 transfer. Rows 1, 4, 6–12, 14–16 train and evaluate on a single dataset only; **row 13 explicitly lists external validation as a recommended future step rather than performing it** — this thesis already does what that paper flags as unfinished.
- **Statistical significance testing**: this thesis runs paired bootstrap resampling (1,000 resamples) for every major model comparison, with pre-registered decision rules. None of rows 1, 9, 10, 12, 13, 15, 16 report confidence intervals or significance tests on their fusion-vs-baseline comparisons — only point-estimate accuracy/AUC/F1.
- **Fairness analysis embedded in the model comparison itself**: this thesis's Fitzpatrick per-skin-tone breakdown is applied directly to its own cross-attention fusion model, not as a separate benchmark paper. Rows 1–16 (the 16 core comparative/architecture papers) perform **zero** skin-tone or fairness analysis — **row 15 self-flags this exact gap** ("Fitzpatrick skin type wasn't even collected"). Only rows 17–19 (added specifically to fill this gap, §Literature Gap section) address fairness at all, and they are benchmark/review papers, not fusion-architecture papers.
- **Metric discipline (macro-F1 over accuracy)**: rows 4, 6–9, 15 report plain accuracy only; **row 13 uses accuracy on a 58:1-imbalanced 7-class split** despite that split's severity — a documented, explicit trap this thesis's `PROJECT_PLAN.md` rejects by mandating macro-F1 throughout (see the standing citation caveat already in `Literature_Review.md`).

### 8.5 Phase 8E Status — Resolved, Val-Only, Did Not Clear Its Bar

- **All 3 seeds complete** (reported 2026-08-03): val macro-F1 seed0 0.6575 / seed1 0.6927 / seed2 0.6660, mean **0.6721**. Inside the pre-registered 0.66–0.71 prediction range (§3.8, logged 2026-08-02 before training).
- **Locally verified 2026-08-03**: `logs/PAD_UFES20/train_cross_attention_joint_convnext_densenet_seed{0,1,2}_summary.json` confirmed present with `best_val_macro_f1` = 0.657466/0.692674/0.665969, mean 0.672036 — matches exactly, same verification standard as §3.7's 0.6186.
- **Decision-rule outcome**: mean (0.6721) exceeds the 0.6710 bar by only +0.0011 — not the "clear and meaningful" margin the pre-registered rule required. **No third test-split read was performed.** Phase 8E is reported as a completed, val-only exploratory result, not a headline candidate; §3.4's 0.6977 remains the thesis headline.
- **Honest limitation this creates**: every prior variant with both a val and test number scored higher on test than val (headline +0.0768, Step 4 ensemble +0.0465) — under that pattern, Phase 8E's test score, had it been read, could plausibly have been higher, possibly even above 0.6977. This was deliberately not chased: reopening the test split because of a post-hoc "it might do better" observation would defeat the purpose of pre-registering the 0.6710 bar in the first place. Disclosed in §6 as a disciplined choice made at the possible cost of a marginal improvement, and as a candidate future-work item using a fresh, independently held-out set — not a third read of this project's already-twice-consumed test split.

---

## 9. Paper-Writing Session Addendum (2026-08-13)

Cross-check performed 2026-08-14 against `docs/Phase8_ConfusionMatrix.csv` and `docs/Literature_Review.md` directly (not transcribed from the paper-writing session summary without verification). One number from that session did not reproduce and is corrected below rather than recorded as-is.

### 9.1 Confusion matrix (headline cross_attention_fusion, PAD-UFES-20 test, `SUMMED_3_SEEDS_n1062` row)

Full 6×6 matrix, rows = true label, columns = predicted (AK/BCC/MEL/NEV/SEK/SCC):

| True | Pred | AK | BCC | MEL | NEV | SEK | SCC | Row total |
|---|---|---|---|---|---|---|---|
| AK | 268 | 22 | 0 | 0 | 17 | 20 | 327 |
| BCC | 30 | 328 | 0 | 4 | 3 | 34 | 399 |
| MEL | 0 | 0 | 16 | 8 | 0 | 0 | 24 |
| NEV | 1 | 8 | 1 | 87 | 8 | 0 | 105 |
| SEK | 3 | 4 | 2 | 12 | 89 | 1 | 111 |
| SCC | 15 | 51 | 1 | 0 | 7 | 22 | 96 |

Total misclassifications (all off-diagonal cells, summed over 3 seeds, n=1,062 test-set reads): **252**.

**AK/BCC/SCC error-cluster finding — verified, with one correction.** Confusions strictly among {AK, BCC, SCC} (off-diagonal cells where both true and predicted label are in this set): AK→BCC 22, AK→SCC 20, BCC→AK 30, BCC→SCC 34, SCC→AK 15, SCC→BCC 51 = **172 of 252 misclassifications (68.3%)**, not the 187/252 (74%) figure from the paper-writing session summary — that number did not reproduce from the CSV under any grouping I tried (cluster-only, true-in-cluster-only, or union-with-predicted-in-cluster all gave different totals, none landing on 187). Use **172/252 = 68.3%** going forward; if 187/74% is the number actually printed in the submitted paper, that discrepancy needs resolving before further citation.
**SCC→BCC = 51 is confirmed** as the single largest off-diagonal confusion in the entire matrix (next-largest: BCC→SCC 34, BCC→AK 30).

### 9.2 Test-set per-class distribution (derived: row totals ÷ 3 seeds)

| Class | Row total (3 seeds) | ÷3 |
|---|---|---|
| AK | 327 | 109 |
| BCC | 399 | 133 |
| MEL | 24 | 8 |
| NEV | 105 | 35 |
| SEK | 111 | 37 |
| SCC | 96 | 32 |
| **Sum** | **1,062** | **354** |

Matches §2.1's known PAD-UFES-20 test split count of 354 rows exactly. All six per-class figures confirmed against the user's summary (AK 109, BCC 133, MEL 8, NEV 35, SEK 37, SCC 32).

### 9.3 Shrestha & Palit (2026) citation — RESOLVED 2026-08-14

Added as row #20 in `docs/Literature_Review.md`'s reconciled table (title/authors/journal/DOI verified via direct publisher lookup: *Biomedical Physics & Engineering Express*, vol. 12, no. 2, art. 025043, DOI 10.1088/2057-1976/ae4eeb). Table header updated from "19 papers" to "20 papers". **`docs/Literature_Review_Master.xlsx` was NOT updated** — no tool access to edit the spreadsheet in this session; the `.md` and `.xlsx` are now out of sync until someone manually mirrors row #20 into the Excel file. Flag this the next time the `.xlsx` is opened.

### 9.4 Final IEEE paper — structural choices (for consistency in future edits)

Recorded from the paper-writing session; **not independently verifiable from this repo** since no paper source file (`.tex` or otherwise) was found locally (§9.5). Treat as the user's own report of the submitted structure until a source file can be checked against it:
- **Related Work** split into four subsections: ML-Based, DL-Based, Fusion-Mechanisms, Fairness.
- **Discussion** contains an explicit **Limitations** subsection.
- A standalone **Reproducibility Statement** section.
- **Acknowledgment** deliberately scoped to dataset credit only — no AI-tool-usage disclosure included (a conscious choice, not an omission).

### 9.5 Final paper file location — NOT FOUND, unresolved

Searched the full repo (`**/*.tex`, `**/main_FINAL*.tex`) and a broader disk search for `main_FINAL*`/`*final*` — **no paper source file exists locally**. Asked the user directly (2026-08-14): confirmed the final paper file is **not saved on this machine** (written/held elsewhere — e.g. Overleaf — or not yet saved locally). §1's file map is **not** updated with a row for it; this stays an open item until the user provides an actual local path, at which point add it to §1 as `papers/` or wherever it lands.

---

*This document supersedes nothing — `Project_AZ_Reference.md` and `Project_Tracking.md` remain the underlying source of truth and should be consulted directly for anything not covered here or flagged as unverified above.*


