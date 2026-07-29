# Literature Review

**Source of record:** `docs/Literature_Review_Master.xlsx` ("Master Literature
Review" sheet, 19 rows, and "Gap Analysis & Next Steps" sheet). This file is
a readable summary generated from that spreadsheet — if the two ever
disagree, the xlsx is authoritative; update the xlsx first, then regenerate
this summary.

**Update (2026-07-28):** 3 papers added specifically for the previously-flagged
Fitzpatrick/skin-tone fairness gap (rows 17-19 below), bringing the total from
16 to **19 unique papers**. See "Literature Gap: Fitzpatrick/Skin-Tone
Fairness" below — that section is no longer describing an unfilled gap.

**Reconciliation (2026-07-17, arithmetic corrected 2026-07-28):** 16 unique
papers, combining the 3 papers already tracked in `Project_Tracking.md`
(Mridha & Islam 2026; Suresh et al. 2026 TG-CAVNet; Watson et al. 2026),
6 net-new papers from the user's working Excel sheet (rows 4-9 below — one
Excel duplicate was found and resolved, see "Duplicate Resolution" below),
and 7 new candidates found via web search (rows 10-16 below). **Correction:**
the original 2026-07-17 write-up stated this as "8 new candidates found via
web search," which doesn't reconcile (3+6+8=17, not 16); the table itself
has always had exactly 7 web-search-sourced rows (10-16) — 3 + 6 + 7 = 16,
consistent with the actual row count. This was a wording slip in the prose
summary only, not a missing/extra paper; caught while resolving the row #5
duplicate check below.

**Status (2026-07-28):** all 16 rows now have at least abstract-level
coverage; 6 have been read in full (rows 1, 3 already full-text at
reconciliation; rows 11, 13, 15 read in full this session). Rows 10, 12,
14, 16 remain abstract-level only — genuinely paywalled with no free full
text found anywhere (checked arXiv, Unpaywall, Semantic Scholar's OA index,
ResearchGate, publisher/author sites; see each row's entry and the
"Priority Full-Text Reads" section for the specific dead ends checked).
Row 2 (TG-CAVNet) and row 6-9 still need their own full-text pass. See
`Project_Tracking.md`'s Phase 2 status entry. **Rows 17-19 (added 2026-07-28,
see "Update" note above) are all full-text-read already** — 2 free full texts
(rows 17, 18) plus 1 free preprint (row 19); none are abstract-only.

**Row #10 (MetaBlock) update (2026-07-18, re-confirmed 2026-07-28):** its
core mechanism is confirmed (see the reconciled table and "Priority
Full-Text Reads" below) via the paper's official code repository plus an
independently-agreeing secondary source — not via reading the primary IEEE
PDF, which is paywalled with no free copy anywhere (re-checked 2026-07-28
via Unpaywall and Semantic Scholar's OA index, both return no open-access
location). Treat row #10 as mechanism-confirmed but not full-text-read.

---

## Duplicate Resolution (logged so it is not re-investigated)

**Finding:** the user's Excel rows 1 and 2 are the **same paper**, listed
twice under slightly different titles — both report identical metrics
(AUC=0.9818, AUPRC=0.9924, F1=0.9769), which is how the duplication was
caught. This paper is the medRxiv preprint already tracked in
`Project_Tracking.md` as **"Mridha & Islam 2026"** (Cross-Attention Enables
Context-Aware Multimodal Skin Lesion Diagnosis) — confirmed via GitHub
author match. It is row #1 in the reconciled table below.

**Net effect:** the Excel's 8 rows contribute only 6 new unique papers, not
8 — do not recount this as a fresh duplicate-check task later.

**Row #5 identity check — RESOLVED 2026-07-28.** "Multimodal Skin Lesion
Classification Using Deep Learning" (Excel row 4, previously described as
"ISIC Archive," ~93-96% accuracy) is **confirmed to be Yap, Yolland &
Tschandl (2018, *Experimental Dermatology* 27(11):1261-1267, DOI
10.1111/exd.13777, PubMed 30187575)** — exact title match, same year, same
combine-modalities-for-diagnosis topic; a second unrelated paper sharing
this exact title in the same year is not a plausible coincidence.
**This does not reduce the paper count from 16 to 15** — unlike the
Mridha & Islam case above, there was never a second table row duplicating
this paper; row #5 simply needed its factual description corrected in
place (dataset was a custom 2,917-case tri-modal set, not "ISIC Archive";
reported metrics were AUC 0.866/mAP 0.729, not "~93-96% accuracy" — see the
reconciled table's row #5 for full detail). **Final count: 16 unique
papers, confirmed, not 15.**

---

## Reconciled Table (19 papers)

| # | Year | Title | Dataset(s) | Methodology Summary | Reported Metric | Limitation | Relevance to This Thesis |
|---|---|---|---|---|---|---|---|
| 1 | 2026 | Cross-Attention Enables Context-Aware Multimodal Skin Lesion Diagnosis (medRxiv) — Mridha & Islam | PAD-UFES-20 (1,568 lesions) | ViT image encoder + metadata tokens; cross-attention lets metadata tokens query image tokens before classification. Compared vs. metadata-only logistic regression, image-only ResNet18, and late fusion. | AUC=0.9818, AUPRC=0.9924, F1=0.9769, ECE=0.0379 (best of all 4 variants) | Small dataset, binary malignant/benign framing (not multi-class), basic metadata only, no external-dataset validation | Direct architectural precedent for Phase 7 Stage 2 cross-attention design; this thesis extends to multi-class + cross-dataset generalization, which this paper does not attempt |
| 2 | 2026 | TG-CAVNet — Suresh et al. | Not fully captured — verify on next read | TG-CAVNet architecture (text-guided cross-attention variant) | Not fully captured — verify on next read | Not fully captured — verify on next read | Primary architectural reference for Phase 7 Stage 2 cross-attention fusion design |
| 3 | 2026 | (Leakage/shortcut-feature warning paper) — Watson et al. | Not fully captured — verify on next read | Methodological caution: post-diagnosis fields cause data leakage | N/A (methodological) | N/A | Direct inspiration for this thesis's entire leakage-audit methodology (22 excluded columns) — cite prominently in Methodology chapter |
| 4 | 2025 | MM-Skin: A Vision-Language Model for Dermatology (SkinVL) | Large VL dermatology dataset (custom) | Large vision-language dermatology dataset; SkinVL model combines image+text via VQA, supervised fine-tuning, zero-shot learning | ~90–94% accuracy | High compute need, dataset not fully clinical, limited real-world validation | Different modality (free text vs. structured metadata) — useful contrast in Related Work, not directly comparable architecture |
| 5 | 2018 | **Duplicate — confirmed 2026-07-28.** Multimodal Skin Lesion Classification Using Deep Learning = Yap, Yolland & Tschandl, *Experimental Dermatology* 27(11):1261-1267 (DOI 10.1111/exd.13777, PubMed 30187575). Exact title match, same year, same combine-modalities-for-diagnosis topic — a coincidental second unrelated paper with this exact title in the same year is not plausible. Authors affiliated with MetaOptima Technology Inc. (Vancouver) and Medical University of Vienna (Tschandl — also a HAM10000 co-creator). | **Correction:** a custom 2,917-case tri-modal dataset (dermatoscopic image + macroscopic photo + patient metadata per case) — **not** "ISIC Archive" as the Excel review recorded; that dataset label was inaccurate, caught the same way row 14's "JI-ADF" mislabel was. | Two-tower CNN feature extraction (one tower per image modality: dermatoscopic, macroscopic) + patient metadata vector, all concatenated (late fusion) and passed through an embedding network for final classification. | **Correction:** binary melanoma AUC 0.866 (multimodal) vs. 0.784 (macroscopic-image-only baseline); multiclass mean average precision 0.729 (multimodal) vs. 0.598 (image-only). **Not** "~93-96% accuracy" as the Excel review recorded — that figure doesn't match any metric this paper actually reports; likely an imprecise paraphrase in the original review, not a citable number. | Early (2018), foundational three-modality (not just image+metadata) fusion paper — useful for Related Work historical framing, but any numbers cited from it must use the corrected AUC/mAP figures above, not the erroneous accuracy percentage. |
| 6 | 2024 | Explainable AI for Skin Disease Classification | HAM10000 | Multiple CNN transfer-learning models + Grad-CAM visualization | ~91–93% accuracy | Image-only, no metadata/symptom integration, limited to visual explanation | Relevant for Discussion/Future Work (explainability is a natural extension of this thesis) |
| 7 | 2024 | A Multimodal Approach to Skin Disease Detection Using Patient Symptoms | DermNet Dataset | Image model gives initial prediction; LLM-based reasoning (Chain of Options) refines using symptom text | ~90–92% accuracy | Symptom data partially synthetic/non-standardized, weak clinical reliability | Different fusion paradigm (LLM reasoning vs. learned cross-attention) — contrast point in Related Work |
| 8 | 2024 | Skin Lesion Classification Using EfficientNet | HAM10000 | EfficientNet-B0/B1 transfer learning, dermoscopic images only | ~90–94% accuracy | No metadata/symptoms, purely image-based | Directly validates this thesis's own architecture choice (EfficientNet-B0) as a reasonable image-branch baseline |
| 9 | 2025 | Multimodal Learning with Clinical Metadata for Skin Cancer Diagnosis | PAD-UFES-20 | Dual-branch: CNN for image + TabNet for clinical metadata, fused representations | ~94–97% accuracy | Skin-cancer-focused only, limited disease variety, structured metadata only | Closest prior-art comparison for this thesis's PAD-UFES-20 image+metadata baseline numbers — cite directly in Results discussion |
| 10 | 2021 | MetaBlock: An Attention-Based Mechanism to Combine Images and Metadata in Deep Learning Models Applied to Skin Cancer Classification — Pacheco & Krohling | PAD-UFES-20 (creators' own dataset) | **Mechanism confirmed 2026-07-18 (see below) — NOT Transformer-style Q/K/V attention.** MetaBlock is a channel-wise gated feature-modulation block: metadata vector U passes through two independent Linear+BatchNorm branches producing t1, t2 (same channel-dim as CNN feature vector V); output V' = sigmoid(tanh(V·t1) + t2) — a multiplicative gate plus additive bias, squashed by sigmoid, broadcast uniformly across spatial positions within each channel (no per-location weighting). Their own simpler baseline MetaNet is closer to squeeze-and-excitation (metadata → conv+ReLU+sigmoid → per-channel scale map → elementwise multiply). | Per abstract: improves classification across all tested models on both datasets (ISIC 2019, PAD-UFES-20); beats MetaNet + concatenation in 6/10 scenarios. Secondary-source figures, unconfirmed against primary text: ~80.7%±0.008 accuracy (ISIC 2019), ~74.8%±0.018 (PAD-UFES-20). | Channel-wise-only gating, no explicit spatial/token-level attention (unlike our proposed cross-attention). Primary IEEE PDF paywalled and not directly read — mechanism verified via the official code repo (`github.com/paaatcha/MetaBlock`) + a secondary paper's description, not the primary text itself. | **CRITICAL** — written by PAD-UFES-20's own creators; near-mandatory citation. Our Phase 7 Stage 2 cross-attention design (metadata=Q, image spatial tokens=K/V) is confirmed **not** a MetaBlock reproduction — a related-but-distinct mechanism pursuing the same goal via genuine spatial attention rather than MetaBlock's per-channel gating. Correct framing going forward: "cross-attention, contrasted with MetaBlock's channel-gating approach," not "MetaBlock-inspired." |
| 11 | 2019/2020 | The impact of patient clinical information on automated skin cancer detection — Pacheco & Krohling | PAD dataset precursor (1,612 clinical images, 6 classes — smaller/earlier version of the public PAD-UFES-20 this thesis uses) | **Full text read 2026-07-28** (arXiv:1909.12912, freely available). Simple concatenation-based fusion, not MetaBlock's gating and not cross-attention: CNN feature extractor kept frozen-then-fine-tuned; flattened image features pass through a "feature reducer" NN block whose output size is controlled by a tunable "combination factor" `cf` (0.5-0.9) to balance against the fixed 28 clinical features (8 raw fields, one-hot encoded except age); reduced image features concatenated with clinical features, fed to classifier. `T = ceil(cf·Nimg + (1-cf)·Ncli)`. Best `cf`=0.7-0.8 (no significant difference between them), found via sensitivity analysis on ResNet-50. Tested 6 backbones (ResNet-50/101, GoogleNet, MobileNet, VGGNet-13/19), 5-fold CV, class-weighted loss (upsampling tried and rejected — biased toward Melanoma). | Image-only avg across 6 models: BACC 0.650±0.031. Image+clinical avg: BACC 0.718±0.022 (~7% absolute improvement), AUC 0.929→0.948. Best single model: ResNet-50 image+clinical — ACC 0.788±0.025, BACC 0.750±0.033, F1 0.790±0.027, AUC 0.958±0.007. Friedman+Wilcoxon confirm all 6 models significantly improved by adding clinical data. | **Key finding directly relevant to this thesis's own results:** clinical features improve differentiation of ACK/MEL/NEV/SEK but do **not** help separate SCC vs. BCC — the two share near-identical clinical profiles (both bleed, hurt, itch, same age range, same anatomical-site preference), so the model keeps confusing them even with metadata added. Authors' own dataset (1,612 images, precursor to PAD-UFES-20) is smaller/less rich than the 2,298-image, 21-whitelisted-feature PAD-UFES-20 this thesis uses. Authors acknowledge smartphone image quality (vs. dermoscopy), patient self-report subjectivity, and unresolved SCC/BCC confusion as limitations. | **Foundational justification for this thesis's entire multimodal premise — cite in Introduction.** The SCC/BCC confusion-persists-with-metadata finding is a useful forward citation for this thesis's own per-class results (worth checking whether the same pair is confused in our PAD-UFES-20 confusion matrices). This 2020 paper's fusion method (tunable concatenation ratio) is a direct historical precursor to this thesis's own late-fusion Stage 1 baseline (Phase 7 Stage 1) — both are "naive concatenation" approaches that motivate a more sophisticated fusion mechanism (MetaBlock's gating here; cross-attention in this thesis). **Important side-finding: this paper's dataset is the direct precursor to the public PAD-UFES-20 dataset** (later formally released as "PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones," Pacheco et al., *Data in Brief* — see note below); this Data in Brief paper, not this 2019/2020 preprint, is the correct citation for the dataset itself and is not yet in this 16-paper table. |
| 12 | 2025 | A multimodal skin lesion classification through cross-attention fusion and collaborative edge computing (Computerized Medical Imaging and Graphics, vol. 124, p.102588) | Not fully captured — abstract-level only, paywalled (ScienceDirect HTTP 403, no arXiv preprint found) | **Abstract-level only (2026-07-28).** Three-module architecture: modality-wise feature extraction, cross-attention-based feature fusion (dermoscopic images + patient metadata), multimodal classifier. Paired with a "collaborative inference scheme" distributing compute across IoT/edge devices — local processing of sensitive data for privacy/latency, not just an architecture paper. | Per abstract/secondary summary: cross-attention fusion outperforms unimodal and simpler-fusion baselines; cross-attention and Hadamard-product fusion both near state-of-the-art (~98.86%/98.85% accuracy — dataset/task not yet confirmed, treat with caution until full text read). | Not yet captured — full text needed for dataset, exact metrics definition (accuracy alone would be a red flag for imbalance-sensitivity vs. this thesis's macro-F1 discipline), and edge-computing evaluation. | Recent (2025) cross-attention precedent structurally similar to this thesis's own Phase 7 Stage 2 (metadata=Q, image=K/V is not yet confirmed here — needs full text) — but the edge-computing/privacy angle is a different motivation than this thesis's (generalization + fairness), useful contrast in Related Work. The very high reported accuracy (~98.8%) with no macro-F1 given yet is worth scrutinizing once full text is available — this thesis's own cross-attention macro-F1 (0.62-0.70 range) is far lower, likely reflecting a much harder multi-class/imbalanced setup than whatever this paper evaluated on. |
| 13 | 2025 | Comparative analysis of multimodal architectures for effective skin lesion detection using clinical and image data (Frontiers in AI) | **HAM10000 — the same benchmark dataset this thesis uses** (10,015 images, 7 classes; clinical fields: diagnosis-confirmation method, age, sex, anatomical location) | **Full text read 2026-07-28** (open access via PMC). Two feature extractors: Clinical MLP (128→256-dim on tabular metadata) and DermiResNet (a modified ResNet with a learnable weighted skip connection, `y = F(x) + α·x`, 512-dim output). **8 fusion methods compared, head-to-head:** simple concatenation, weighted concatenation, Hadamard product (element-wise, both modalities projected to 256D), tensor fusion (outer product, 512×256), bilinear fusion (learnable pairwise weight tensor), gated fusion (sigmoid gate blending both modalities), self-attention (intra-modality only), and **cross-attention** (image=Query, clinical metadata=Key/Value — same Q/K-V assignment direction as this thesis's own Phase 7 Stage 2 design). | **Reports plain accuracy/precision/recall/F1/specificity, not macro-F1** — see Limitation flag. Cross-attention 98.86% accuracy (best), Hadamard product 98.85% (near-tied), bilinear 98.76%; weighted concatenation 97.15%; simple concat/tensor fusion ~96.5%; gated fusion 93.0% and self-attention 92.70% (both underperform despite architectural complexity). Unimodal ablation: image-only (DermiResNet) 92.0% accuracy, metadata-only (Clinical MLP) 77.0%. Cross-attention per-class AUC: Melanoma 0.99, Nevus 0.98, all others 1.0. | **Critical limitation for citing this paper: it reports only accuracy/weighted-F1 on HAM10000's 58:1-imbalanced 7-class split (Nevus 6,705 vs. Dermatofibroma 115) — a metric choice this thesis's own `PROJECT_PLAN.md` explicitly rejects for exactly this reason** (macro-F1 mandated precisely because plain accuracy on imbalanced data can look deceptively high by matching the majority class). The paper's own stated limitations (self-flagged): heavy compute cost of cross-attention/tensor fusion; HAM10000 alone doesn't capture real-world population diversity (external validation on ISIC/PH2 recommended — this thesis already does this, see Phase 8.2); limited clinical-feature richness in HAM10000; Grad-CAM-only interpretability; persistent melanoma↔nevus/benign-keratosis misclassification despite fusion. | **Directly supports this thesis's own baseline-vs-fusion comparison narrative and is a strong, same-dataset external benchmark — but the accuracy-vs-macro-F1 mismatch must be stated explicitly wherever this paper is cited**, since a naive comparison of their reported "98.86%" against this thesis's own HAM10000 cross-attention macro-F1 (0.6209, PAD-UFES-20; different transfer numbers for HAM10000-trained models, see Phase 8) would be apples-to-oranges and could look like this thesis underperforms when the difference is actually the metric, not the model. The finding that "cross-attention and Hadamard product both win, while self-attention/gated-fusion underperform despite complexity" is a useful citation for this thesis's own Phase 7 Stage 1→2 narrative (naive fusion underperforming, then cross-attention improving it) — same qualitative story, different specific mechanism ranking. The Q=image/KV=metadata assignment matches this thesis's own cross-attention design exactly, worth citing as external validation of that architectural choice. |
| 14 | 2025 | **Title correction (2026-07-28): "A multi-stage multi-modal learning algorithm with adaptive multimodal fusion for improving multi-label skin lesion classification"** — Lihan Zuo, Zizhou Wang, Yan Wang (Artificial Intelligence in Medicine). *The "JI-ADF" name previously recorded here was wrong — that title belongs to a completely different, unrelated paper (Phan Nguyen et al., arXiv:2604.27343, evaluated on the MILK10k benchmark) that happens to address a similar problem. Caught while searching for this row's full text; do not conflate the two going forward.* | Not fully captured — abstract-level only, paywalled (ScienceDirect HTTP 403, no arXiv preprint; PubMed/GitHub repo README give abstract-depth detail only) | **Abstract-level only (2026-07-28).** "CosCatNet" — a two-stage hybrid fusion: (1) image-fusion stage combining clinical photos + dermoscopy images via an intermediate fusion using cosine similarity (captures correlated cross-image information) plus concatenation (captures complementary information); (2) multimodal fusion stage combining the fused image representation with metadata via an uncertainty-based adaptive late fusion (dynamically weights modality contributions per-sample based on estimated confidence). Code at `github.com/Zuo-Lihan/CosCatNet-Adaptive_Fusion_Algorithm`. | Abstract states "experiments demonstrate the effectiveness of our proposed method" on "a popular publicly available skin disease diagnosis dataset" — likely a trimodal (clinical + dermoscopy + metadata) dataset such as Derm7pt given the three-modality setup, but not confirmed; no specific quantitative metrics available without the paywalled full text. | Not yet captured — full text needed for dataset confirmation, quantitative results, and comparison methodology. | Relevant for multi-label/multi-modal framing and adaptive (uncertainty-weighted) fusion design ideas — a third distinct fusion paradigm in this literature set (cosine-similarity/concatenation hybrid + uncertainty-weighted late fusion), contrasted with MetaBlock's channel-gating (row 10) and this thesis's own cross-attention. Uses clinical + dermoscopy image pairs (a modality this thesis's datasets don't have — PAD-UFES-20/HAM10000/ISIC are single-image), so architecturally it's not directly reproducible here, but the uncertainty-based adaptive weighting idea is a citable alternative to this thesis's own fixed cross-attention mechanism for Future Work. |
| 15 | 2026 | Advancing skin cancer detection through deep learning and fusion of patient metadata and skin lesion images — Islam, Wishart, Walls, Hall, Seco de Herrera, Gan & Raza (Scientific Reports) | Check4Cancer (UK private teledermatology network), 2015–2022: 79,246 images (39,623 dermoscopic + 39,623 DSLR pairs), 39,623 unique lesions, 19,295 patients; 22 metadata features (7 core "C4C risk factors": lesion pinkness/size/colour/inflammation/shape/age, natural hair colour; plus age, sex, body location, itch/bleed/pain/growth/pattern-change/elevation, BMI, ethnicity, family history) | **Full text read 2026-07-28** (freely available via PMC — Scientific Reports is fully open access). EfficientNet-B2 backbone, hair removed via a variational-autoencoder method, images resized to 1024×1024, 16-technique Albumentations augmentation pipeline. Six model variants tested (DER-only, DSLR-only, DER+meta, DSLR+meta, DER+DSLR, DER+DSLR+meta) — fusion is simple feature concatenation (image vector ⊕ metadata vector) + Swish activation + 0.5-ratio dropout, not attention-based. Final system uses **decision-level majority-vote fusion** across two of the six trained variants, not a single end-to-end model. Patient-wise 80/20 split, 5-fold CV. | Best single model (DER+DSLR+meta, tested on DER+metadata): 88.77% accuracy, 92.98% AUC, sens 99.83%/spec 77.71%. Best majority-vote fusion: **91.11% accuracy, 94.06% AUC, sens 99.50%/spec 82.72%** — beats every single-modality/single-model variant. Image-only baseline (DER alone): 81.28% accuracy — so metadata fusion adds ~10 points of accuracy here, mostly by lifting *specificity* (63%→78-83%) while sensitivity was already near-ceiling image-only. Benchmarked against real teledermatology clinical performance (Cochrane systematic review, 96% sensitivity in-person; a competing commercial system "Skin Analytics," 95-98%) and reports beating both at matched specificity. | **Ground-truth caveat, self-acknowledged:** primary training/eval ground truth is *expert visual triage rating*, not biopsy-confirmed diagnosis — only 10% of lesions were ever biopsied, and using only those would cut the dataset by 90%. This is the **opposite risk profile from this thesis's own leakage concern** (we worry a *biopsy-confirmed* field like `biopsed` leaks the malignancy answer; this paper instead worries its *labels themselves* are unconfirmed expert opinion, not ground truth) — worth citing as a contrasting methodological trade-off. **Also self-acknowledged: Fitzpatrick skin type wasn't even collected**, and the patient population is predominantly types I-IV — directly parallel to this thesis's own Phase 8 finding (PAD-UFES-20's test split has only 2-3 patients in Fitzpatrick V/VI, too few to report) — this is independent evidence from a much larger (79K-image) UK cohort that the same skin-tone-representation gap recurs across datasets, strengthening this thesis's fairness-analysis contribution claim. | Large-scale precedent (79,246 images vs. this thesis's leakage-audited, deliberately smaller 2,298-image PAD-UFES-20) with a genuinely strong result, but built on a simpler concatenation fusion + decision-level majority voting rather than a jointly-trained attention mechanism — a useful contrast point for framing this thesis's cross-attention as architecturally more sophisticated even at smaller scale. **Two citable parallels:** (1) its unconfirmed-ground-truth limitation is a mirror image of this thesis's `biopsed`-leakage concern, both examples of how clinical confirmation status shapes what a label actually means; (2) its independent finding of Fitzpatrick V/VI under-representation (in a completely different country/cohort/dataset) is strong corroborating evidence for this thesis's own Phase 8 fairness finding and argument that this is a systemic gap in the field, not an artifact of PAD-UFES-20 specifically — cite both in the Related Work/fairness framing. |
| 16 | 2025 | Evaluation of the importance of metadata in skin lesion classification — Garib, Mery & Navarrete-Dechent (Signal, Image and Video Processing) | PAD-UFES-20 (clinical images) + ISIC 2019 (dermoscopic images) | **Abstract-level only (2026-07-28) — paywalled, no free full text found** (Springer redirects to institutional login; ResearchGate blocked 403; author's personal site unreachable; no OA copy indexed by any aggregator checked). 17 deep learning models tested across 3 fusion methods on both datasets; separately trained models on different metadata subsets to isolate each feature's individual contribution to performance. | Image+metadata beat image-only baseline by **+10.43% balanced accuracy on PAD-UFES-20**, **+2.22% on ISIC 2019** (PAD-UFES-20's much larger gain is consistent with this thesis's own experience — its rich clinical metadata carries more signal than a dermoscopy-only archive's sparser tags). Per-feature importance ranking: **age most useful**, then body/anatomical location, then sex. | Not yet captured — full text needed to assess methodology details, exact model list, and statistical rigor of the per-feature importance ranking. | **Directly parallels this thesis's own `feature_whitelist.md` exercise** — the age > location > sex ranking is a useful comparison point for our own feature set (PAD-UFES-20's 21 whitelisted features vs. HAM10000's 3). Same PAD-UFES-20 dataset as this thesis, so its reported image+metadata gain (+10.43% BACC) is a direct external benchmark for our own Phase 6→7 fusion improvement, worth citing even at abstract-only depth — but any specific number should be flagged as "per abstract, not full-text-verified" until/unless full text becomes available. |
| 17 | 2022 | Disparities in Dermatology AI Performance on a Diverse, Curated Clinical Image Set — Daneshjou et al. (*Science Advances*) | DDI (Diverse Dermatology Images) — 656 images, pathologically confirmed, curated for diverse Fitzpatrick skin-tone representation | **Full text freely available** (arXiv:2203.08807). Introduces the DDI dataset and benchmarks existing SOTA dermatology classifiers (trained on standard datasets) against it; separately compares dermatologist diagnostic performance across the same skin-tone/disease splits; also tests fine-tuning models on DDI itself. **Date exception, logged:** published 2022, outside this project's preferred 2023-2025 fairness-search window — included anyway because it is the field's foundational fairness-benchmark paper and the closest methodological analogue to this thesis's own Phase 8.3 analysis. | **27–36 percentage-point ROC-AUC drop** for SOTA models vs. their originally reported test performance, concentrated on dark skin tones and uncommon diseases; dermatologists also perform worse on dark-skin/uncommon-disease images; fine-tuning on DDI narrows but does not eliminate the light/dark performance gap. | DDI itself is relatively small (656 images); disparity demonstrated at the model-benchmark level, not root-caused to a single mechanism. | **Near-mandatory citation for Phase 8.3.** Direct methodological analogue — an external, pathologically-confirmed test set built specifically to expose skin-tone performance gaps, exactly the kind of evaluation this thesis's own Fitzpatrick fairness analysis performs (albeit within-dataset on PAD-UFES-20 rather than via a dedicated external benchmark). Cite when framing/motivating Phase 8.3 and when discussing why the fairness analysis matters as a contribution. |
| 18 | 2024 | Skin Type Diversity in Skin Lesion Datasets: A Review — Alipour, Burke & Courtney (*Current Dermatology Reports*, DOI 10.1007/s13671-024-00440-0) | Systematic review across multiple public skin lesion datasets (not a single dataset) | **Full text freely available** (PMC11343783). Reviews publicly available skin lesion datasets and their metadata; evaluates both whether Fitzpatrick skin type is reported at all and, separately, how diverse/representative that reporting actually is — explicitly notes prior work does one or the other but not both together. | Review paper, no single quantitative headline metric; core finding is that under-representation of darker skin types is a **systemic, field-wide dataset problem**, not an isolated instance in any one dataset. | Review-level (not a new dataset or model); scope limited to skin lesion datasets with publicly available metadata. | **Strongest citation for the "Literature Gap: Fitzpatrick/Skin-Tone Fairness" section below** — independently confirms under-representation is systemic across the field, directly corroborating this thesis's own Phase 8.3 finding (PAD-UFES-20 test split: only 3 rows total across Fitzpatrick V/VI, 38% missing entirely). Cite this paper wherever this thesis states that the fairness gap it addresses is field-wide, not an artifact of PAD-UFES-20 specifically. |
| 19 | 2024 | A Framework for Evaluating the Efficacy of Foundation Embedding Models in Healthcare — Xu, Gui, Rotemberg, Wang, Chen & Daneshjou (medRxiv 2024.04.17.24305983) | Google Health's Derm Foundation Model pretrained embeddings, evaluated on dermatology image classification tasks (not a new dataset) | **Free preprint**, posted 2024-04-19. Proposes and pilots a 3-axis framework (general performance / bias-fairness / confounders) for evaluating medical foundation models, applied to dermatology; evaluates Derm Foundation Model embeddings plus general-purpose CLIP embeddings across all 3 axes; measures per-Fitzpatrick-group sensitivity directly on the pretrained embedding space. | Foundation-model embeddings exceed SOTA classification accuracy; general-purpose CLIP embeddings are also informative for medical tasks; **lower sensitivity for darker Fitzpatrick tones (4–6)**, present even at the pretrained-embedding level before any task-specific fine-tuning; image quality also significantly affects performance. | Single foundation-model family evaluated in depth (Derm Foundation Model); the 3-axis framework is proposed/piloted, not yet an established field standard. | Confirms skin-tone-linked bias exists even inside pretrained foundation-model embeddings, not just end-task classifiers — relevant caution if future work on this thesis ever considers a foundation-model backbone. **The 3-axis evaluation framework (general performance / bias-fairness / confounders) is a citable structure for organizing this thesis's own Phase 8.3 write-up** in the Discussion/Results chapter — worth adopting explicitly rather than presenting the fairness results as a standalone table. |

---

## Standing Citation Caveat — Row #13 (must carry forward to thesis Discussion chapter)

**Whenever row #13 ("Comparative analysis of multimodal architectures...",
Frontiers in AI, HAM10000) is cited anywhere in this thesis, both of the
following must be stated together, not left implicit:**

1. **Independent architectural validation:** their cross-attention fusion
   uses the same Query/Key-Value assignment direction as this thesis's own
   Phase 7 Stage 2 design (image = Query, metadata = Key/Value) — a genuine,
   independent confirmation that this is a sound design choice, found by a
   different group on the same dataset (HAM10000).
2. **Metric-choice caveat, not a performance gap:** their headline 98.86%
   is **plain accuracy** on HAM10000's 58:1-imbalanced 7-class split, not
   macro-F1. This thesis's own numbers (e.g. cross-attention macro-F1
   0.6209 on PAD-UFES-20; HAM10000-trained transfer numbers in Phase 8) are
   **not directly comparable** to that 98.86% figure. If both numbers ever
   appear near each other in the same paragraph, explicitly state that the
   difference is a metric-choice artifact (accuracy vs. macro-F1 on
   imbalanced data), not evidence that this thesis's model underperforms
   theirs — per `PROJECT_PLAN.md`'s own metrics rule, this is exactly the
   accuracy-on-imbalanced-data trap the project's macro-F1 discipline
   exists to avoid.

---

## Literature Gap: Fitzpatrick/Skin-Tone Fairness — RESOLVED 2026-07-28

**Originally, none of the first 16 papers focused on Fitzpatrick/skin-tone
fairness in dermatology AI.** This has been addressed: 3 targeted papers
were added (rows 17–19 above), specifically to fill this gap ahead of
Phase 9 (Thesis Writing Support):

- **Row 17 (Daneshjou et al. 2022, DDI dataset)** — quantifies dermatology
  AI performance disparity across skin tones directly, the closest
  methodological analogue to this thesis's own Phase 8.3 analysis. Included
  despite being outside the 2023-2025 window preferred for this search — see
  its row for the logged date-exception justification.
- **Row 18 (Alipour, Burke & Courtney 2024)** — a systematic review
  confirming that Fitzpatrick under-representation in public skin lesion
  datasets is a **systemic, field-wide problem**, not specific to
  PAD-UFES-20. This is now the primary citation for the claim below.
- **Row 19 (Xu, Gui, Rotemberg, Wang, Chen & Daneshjou 2024)** — finds
  skin-tone-linked bias even inside pretrained foundation-model embeddings,
  and contributes a citable 3-axis evaluation framework (general
  performance / bias-fairness / confounders) worth adopting when writing up
  Phase 8.3.

This remains directly relevant because `PROJECT_PLAN.md`'s Phase 8
(Experiments & Evaluation) includes a dedicated Fitzpatrick fairness
analysis (see `Phase8_Fitzpatrick_Fairness_Results.md`). With rows 17–19 now
in hand, this thesis's own fairness analysis should be framed as **both** a
routine evaluation step **and** a contribution that extends a small but
real prior-work base (rather than the earlier framing of "no prior work
exists at all," which understated the field while the gap was genuinely
unfilled) — cite row 18 specifically when stating that skin-tone
under-representation is a systemic issue this thesis's dataset also
exhibits, not an artifact of PAD-UFES-20.

---

## Priority Full-Text Reads

Two papers are near-mandatory citations and should be read in full before
relying on this table further, since they are written by **PAD-UFES-20's own
dataset creators**:

1. **Pacheco & Krohling (2021)** — *MetaBlock: An Attention-Based Mechanism
   to Combine Images and Metadata in Deep Learning Models Applied to Skin
   Cancer Classification* (row #10). **Mechanism now confirmed (2026-07-18)**
   — see the reconciled table above. **Caveat: the primary IEEE JBHI PDF
   itself is paywalled and was not directly read.** The mechanism
   (channel-wise gated modulation, `V' = sigmoid(tanh(V·t1) + t2)`, not
   Transformer-style attention) was verified from the paper's official code
   repository (`github.com/paaatcha/MetaBlock`) plus a secondary paper's
   description of the same method, which independently agreed — high
   confidence, but not equivalent to reading the primary text. **Re-checked
   2026-07-28: no free full text exists anywhere** (arXiv has no preprint of
   this paper; ResearchGate's copy returned HTTP 403; the author's own
   publications page (`pachecoandre.com.br/research`) links only to the
   paywalled IEEE Xplore page, not a self-hosted PDF) — still blocked on
   institutional access. If the thesis needs to quote specific numeric
   results (accuracy/balanced-accuracy tables) or cite methodology details
   beyond the mechanism itself, the primary PDF should still be obtained
   (e.g. via institutional access) and read before those specific claims are
   finalized.
2. **Pacheco & Krohling (2020)** — *The impact of patient clinical
   information on automated skin cancer detection* (row #11). **Full text
   read 2026-07-28** via the free arXiv preprint (arXiv:1909.12912) — see
   the reconciled table above for full methodology/results/limitations.
   Foundational justification for the multimodal premise this thesis is
   built on — a thesis using PAD-UFES-20 that doesn't engage this pair of
   papers is an easily-flagged gap in a viva.

**New finding, 2026-07-28 (not yet in the 16-paper table, not literature-review
material — a dataset citation):** while reading row #11, its dataset was
identified as the direct precursor to the public PAD-UFES-20 dataset this
thesis uses (1,612 images / 6 classes / 8 raw clinical fields here, vs. the
public release's 2,298 images / 6 classes / richer metadata). The public
dataset's own citation paper — **Pacheco et al., "PAD-UFES-20: A skin lesion
dataset composed of patient data and clinical images collected from
smartphones," *Data in Brief*** — is the correct reference for the dataset
itself (distinct from either of the two priority papers above, which are
about *methods*, not the dataset release). **This paper is not yet cited
anywhere in this project's literature review and should be added as the
mandatory dataset citation**, separate from the 16-paper comparative table
(dataset citations aren't "related work" to compare against, but omitting it
would be a citation gap in the Methodology chapter's dataset section).

Recommended order after these two: "Comparative analysis of multimodal
architectures" (row #13, broad fusion-strategy context) and "Evaluation of
the importance of metadata" (row #16, parallels `feature_whitelist.md`
directly), then the remaining abstract-only rows as time allows.
