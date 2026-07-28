# Literature Review

**Source of record:** `docs/Literature_Review_Master.xlsx` ("Master Literature
Review" sheet, 16 rows, and "Gap Analysis & Next Steps" sheet). This file is
a readable summary generated from that spreadsheet — if the two ever
disagree, the xlsx is authoritative; update the xlsx first, then regenerate
this summary.

**Reconciliation (2026-07-17):** 16 unique papers, combining the 3 papers
already tracked in `Project_Tracking.md` (Mridha & Islam 2026; Suresh et al.
2026 TG-CAVNet; Watson et al. 2026), 8 papers the user reviewed in a working
Excel sheet, and 8 new candidates found via web search. This is **not**
3 + 8 + 8 = 19 — one duplicate was found and resolved (see below), so the
Excel's 8 rows contribute 6 net-new papers, giving 3 + 6 + 8 = 16 unique
papers total, not 19.

**Status:** 8 of the 16 papers (rows 10–16, minus row 12 which duplicates
into that count — effectively rows 10, 11, 12, 13, 14, 15, 16, i.e. all the
web-search-found candidates) are abstract-only and still need a full-text
read before the Literature Review chapter can be written. See
`Project_Tracking.md`'s Phase 2 status entry.

**Row #10 (MetaBlock) update (2026-07-18):** its core mechanism is now
confirmed (see the reconciled table and "Priority Full-Text Reads" below),
via the paper's official code repository plus an independently-agreeing
secondary source — not via reading the primary IEEE PDF, which is paywalled
and still unread. Treat row #10 as mechanism-confirmed but not fully
full-text-read; the remaining 7 rows (11, 12, 13, 14, 15, 16, and the
still-open row #11 read) are unchanged, still abstract-only.

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

**Still open, not yet resolved (flagged, not a duplicate finding):** row #5
("Multimodal Skin Lesion Classification Using Deep Learning", ISIC Archive,
2018) is a *possible* match to Yap, Yolland & Tschandl (2018, Experimental
Dermatology), found independently via the web search. This has **not** been
confirmed — needs a full-text author check. If confirmed identical, the
paper count drops to 15; if different, it stays at 16 and a 17th distinct
paper may exist.

---

## Reconciled Table (16 papers)

| # | Year | Title | Dataset(s) | Methodology Summary | Reported Metric | Limitation | Relevance to This Thesis |
|---|---|---|---|---|---|---|---|
| 1 | 2026 | Cross-Attention Enables Context-Aware Multimodal Skin Lesion Diagnosis (medRxiv) — Mridha & Islam | PAD-UFES-20 (1,568 lesions) | ViT image encoder + metadata tokens; cross-attention lets metadata tokens query image tokens before classification. Compared vs. metadata-only logistic regression, image-only ResNet18, and late fusion. | AUC=0.9818, AUPRC=0.9924, F1=0.9769, ECE=0.0379 (best of all 4 variants) | Small dataset, binary malignant/benign framing (not multi-class), basic metadata only, no external-dataset validation | Direct architectural precedent for Phase 7 Stage 2 cross-attention design; this thesis extends to multi-class + cross-dataset generalization, which this paper does not attempt |
| 2 | 2026 | TG-CAVNet — Suresh et al. | Not fully captured — verify on next read | TG-CAVNet architecture (text-guided cross-attention variant) | Not fully captured — verify on next read | Not fully captured — verify on next read | Primary architectural reference for Phase 7 Stage 2 cross-attention fusion design |
| 3 | 2026 | (Leakage/shortcut-feature warning paper) — Watson et al. | Not fully captured — verify on next read | Methodological caution: post-diagnosis fields cause data leakage | N/A (methodological) | N/A | Direct inspiration for this thesis's entire leakage-audit methodology (22 excluded columns) — cite prominently in Methodology chapter |
| 4 | 2025 | MM-Skin: A Vision-Language Model for Dermatology (SkinVL) | Large VL dermatology dataset (custom) | Large vision-language dermatology dataset; SkinVL model combines image+text via VQA, supervised fine-tuning, zero-shot learning | ~90–94% accuracy | High compute need, dataset not fully clinical, limited real-world validation | Different modality (free text vs. structured metadata) — useful contrast in Related Work, not directly comparable architecture |
| 5 | 2018 | Multimodal Skin Lesion Classification Using Deep Learning (possible match: Yap, Yolland & Tschandl 2018 — unconfirmed) | ISIC Archive | Combines dermoscopic images, clinical images, and patient metadata; per-modality features merged for classification | ~93–96% accuracy | Small/less diverse dataset, no symptom-level info, limited real-world variability | Early foundational multimodal paper — good for Related Work historical framing |
| 6 | 2024 | Explainable AI for Skin Disease Classification | HAM10000 | Multiple CNN transfer-learning models + Grad-CAM visualization | ~91–93% accuracy | Image-only, no metadata/symptom integration, limited to visual explanation | Relevant for Discussion/Future Work (explainability is a natural extension of this thesis) |
| 7 | 2024 | A Multimodal Approach to Skin Disease Detection Using Patient Symptoms | DermNet Dataset | Image model gives initial prediction; LLM-based reasoning (Chain of Options) refines using symptom text | ~90–92% accuracy | Symptom data partially synthetic/non-standardized, weak clinical reliability | Different fusion paradigm (LLM reasoning vs. learned cross-attention) — contrast point in Related Work |
| 8 | 2024 | Skin Lesion Classification Using EfficientNet | HAM10000 | EfficientNet-B0/B1 transfer learning, dermoscopic images only | ~90–94% accuracy | No metadata/symptoms, purely image-based | Directly validates this thesis's own architecture choice (EfficientNet-B0) as a reasonable image-branch baseline |
| 9 | 2025 | Multimodal Learning with Clinical Metadata for Skin Cancer Diagnosis | PAD-UFES-20 | Dual-branch: CNN for image + TabNet for clinical metadata, fused representations | ~94–97% accuracy | Skin-cancer-focused only, limited disease variety, structured metadata only | Closest prior-art comparison for this thesis's PAD-UFES-20 image+metadata baseline numbers — cite directly in Results discussion |
| 10 | 2021 | MetaBlock: An Attention-Based Mechanism to Combine Images and Metadata in Deep Learning Models Applied to Skin Cancer Classification — Pacheco & Krohling | PAD-UFES-20 (creators' own dataset) | **Mechanism confirmed 2026-07-18 (see below) — NOT Transformer-style Q/K/V attention.** MetaBlock is a channel-wise gated feature-modulation block: metadata vector U passes through two independent Linear+BatchNorm branches producing t1, t2 (same channel-dim as CNN feature vector V); output V' = sigmoid(tanh(V·t1) + t2) — a multiplicative gate plus additive bias, squashed by sigmoid, broadcast uniformly across spatial positions within each channel (no per-location weighting). Their own simpler baseline MetaNet is closer to squeeze-and-excitation (metadata → conv+ReLU+sigmoid → per-channel scale map → elementwise multiply). | Per abstract: improves classification across all tested models on both datasets (ISIC 2019, PAD-UFES-20); beats MetaNet + concatenation in 6/10 scenarios. Secondary-source figures, unconfirmed against primary text: ~80.7%±0.008 accuracy (ISIC 2019), ~74.8%±0.018 (PAD-UFES-20). | Channel-wise-only gating, no explicit spatial/token-level attention (unlike our proposed cross-attention). Primary IEEE PDF paywalled and not directly read — mechanism verified via the official code repo (`github.com/paaatcha/MetaBlock`) + a secondary paper's description, not the primary text itself. | **CRITICAL** — written by PAD-UFES-20's own creators; near-mandatory citation. Our Phase 7 Stage 2 cross-attention design (metadata=Q, image spatial tokens=K/V) is confirmed **not** a MetaBlock reproduction — a related-but-distinct mechanism pursuing the same goal via genuine spatial attention rather than MetaBlock's per-channel gating. Correct framing going forward: "cross-attention, contrasted with MetaBlock's channel-gating approach," not "MetaBlock-inspired." |
| 11 | 2019/2020 | The impact of patient clinical information on automated skin cancer detection — Pacheco & Krohling | PAD dataset precursor (1,612 clinical images, 6 classes — smaller/earlier version of the public PAD-UFES-20 this thesis uses) | **Full text read 2026-07-28** (arXiv:1909.12912, freely available). Simple concatenation-based fusion, not MetaBlock's gating and not cross-attention: CNN feature extractor kept frozen-then-fine-tuned; flattened image features pass through a "feature reducer" NN block whose output size is controlled by a tunable "combination factor" `cf` (0.5-0.9) to balance against the fixed 28 clinical features (8 raw fields, one-hot encoded except age); reduced image features concatenated with clinical features, fed to classifier. `T = ceil(cf·Nimg + (1-cf)·Ncli)`. Best `cf`=0.7-0.8 (no significant difference between them), found via sensitivity analysis on ResNet-50. Tested 6 backbones (ResNet-50/101, GoogleNet, MobileNet, VGGNet-13/19), 5-fold CV, class-weighted loss (upsampling tried and rejected — biased toward Melanoma). | Image-only avg across 6 models: BACC 0.650±0.031. Image+clinical avg: BACC 0.718±0.022 (~7% absolute improvement), AUC 0.929→0.948. Best single model: ResNet-50 image+clinical — ACC 0.788±0.025, BACC 0.750±0.033, F1 0.790±0.027, AUC 0.958±0.007. Friedman+Wilcoxon confirm all 6 models significantly improved by adding clinical data. | **Key finding directly relevant to this thesis's own results:** clinical features improve differentiation of ACK/MEL/NEV/SEK but do **not** help separate SCC vs. BCC — the two share near-identical clinical profiles (both bleed, hurt, itch, same age range, same anatomical-site preference), so the model keeps confusing them even with metadata added. Authors' own dataset (1,612 images, precursor to PAD-UFES-20) is smaller/less rich than the 2,298-image, 21-whitelisted-feature PAD-UFES-20 this thesis uses. Authors acknowledge smartphone image quality (vs. dermoscopy), patient self-report subjectivity, and unresolved SCC/BCC confusion as limitations. | **Foundational justification for this thesis's entire multimodal premise — cite in Introduction.** The SCC/BCC confusion-persists-with-metadata finding is a useful forward citation for this thesis's own per-class results (worth checking whether the same pair is confused in our PAD-UFES-20 confusion matrices). This 2020 paper's fusion method (tunable concatenation ratio) is a direct historical precursor to this thesis's own late-fusion Stage 1 baseline (Phase 7 Stage 1) — both are "naive concatenation" approaches that motivate a more sophisticated fusion mechanism (MetaBlock's gating here; cross-attention in this thesis). **Important side-finding: this paper's dataset is the direct precursor to the public PAD-UFES-20 dataset** (later formally released as "PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones," Pacheco et al., *Data in Brief* — see note below); this Data in Brief paper, not this 2019/2020 preprint, is the correct citation for the dataset itself and is not yet in this 16-paper table. |
| 12 | 2025 | A multimodal skin lesion classification through cross-attention fusion and collaborative edge computing (Computerized Medical Imaging and Graphics, vol. 124, p.102588) | Not fully captured — abstract-level only, paywalled (ScienceDirect HTTP 403, no arXiv preprint found) | **Abstract-level only (2026-07-28).** Three-module architecture: modality-wise feature extraction, cross-attention-based feature fusion (dermoscopic images + patient metadata), multimodal classifier. Paired with a "collaborative inference scheme" distributing compute across IoT/edge devices — local processing of sensitive data for privacy/latency, not just an architecture paper. | Per abstract/secondary summary: cross-attention fusion outperforms unimodal and simpler-fusion baselines; cross-attention and Hadamard-product fusion both near state-of-the-art (~98.86%/98.85% accuracy — dataset/task not yet confirmed, treat with caution until full text read). | Not yet captured — full text needed for dataset, exact metrics definition (accuracy alone would be a red flag for imbalance-sensitivity vs. this thesis's macro-F1 discipline), and edge-computing evaluation. | Recent (2025) cross-attention precedent structurally similar to this thesis's own Phase 7 Stage 2 (metadata=Q, image=K/V is not yet confirmed here — needs full text) — but the edge-computing/privacy angle is a different motivation than this thesis's (generalization + fairness), useful contrast in Related Work. The very high reported accuracy (~98.8%) with no macro-F1 given yet is worth scrutinizing once full text is available — this thesis's own cross-attention macro-F1 (0.62-0.70 range) is far lower, likely reflecting a much harder multi-class/imbalanced setup than whatever this paper evaluated on. |
| 13 | 2025 | Comparative analysis of multimodal architectures for effective skin lesion detection using clinical and image data (Frontiers in AI) | **HAM10000 — the same benchmark dataset this thesis uses** (10,015 images, 7 classes; clinical fields: diagnosis-confirmation method, age, sex, anatomical location) | **Full text read 2026-07-28** (open access via PMC). Two feature extractors: Clinical MLP (128→256-dim on tabular metadata) and DermiResNet (a modified ResNet with a learnable weighted skip connection, `y = F(x) + α·x`, 512-dim output). **8 fusion methods compared, head-to-head:** simple concatenation, weighted concatenation, Hadamard product (element-wise, both modalities projected to 256D), tensor fusion (outer product, 512×256), bilinear fusion (learnable pairwise weight tensor), gated fusion (sigmoid gate blending both modalities), self-attention (intra-modality only), and **cross-attention** (image=Query, clinical metadata=Key/Value — same Q/K-V assignment direction as this thesis's own Phase 7 Stage 2 design). | **Reports plain accuracy/precision/recall/F1/specificity, not macro-F1** — see Limitation flag. Cross-attention 98.86% accuracy (best), Hadamard product 98.85% (near-tied), bilinear 98.76%; weighted concatenation 97.15%; simple concat/tensor fusion ~96.5%; gated fusion 93.0% and self-attention 92.70% (both underperform despite architectural complexity). Unimodal ablation: image-only (DermiResNet) 92.0% accuracy, metadata-only (Clinical MLP) 77.0%. Cross-attention per-class AUC: Melanoma 0.99, Nevus 0.98, all others 1.0. | **Critical limitation for citing this paper: it reports only accuracy/weighted-F1 on HAM10000's 58:1-imbalanced 7-class split (Nevus 6,705 vs. Dermatofibroma 115) — a metric choice this thesis's own `PROJECT_PLAN.md` explicitly rejects for exactly this reason** (macro-F1 mandated precisely because plain accuracy on imbalanced data can look deceptively high by matching the majority class). The paper's own stated limitations (self-flagged): heavy compute cost of cross-attention/tensor fusion; HAM10000 alone doesn't capture real-world population diversity (external validation on ISIC/PH2 recommended — this thesis already does this, see Phase 8.2); limited clinical-feature richness in HAM10000; Grad-CAM-only interpretability; persistent melanoma↔nevus/benign-keratosis misclassification despite fusion. | **Directly supports this thesis's own baseline-vs-fusion comparison narrative and is a strong, same-dataset external benchmark — but the accuracy-vs-macro-F1 mismatch must be stated explicitly wherever this paper is cited**, since a naive comparison of their reported "98.86%" against this thesis's own HAM10000 cross-attention macro-F1 (0.6209, PAD-UFES-20; different transfer numbers for HAM10000-trained models, see Phase 8) would be apples-to-oranges and could look like this thesis underperforms when the difference is actually the metric, not the model. The finding that "cross-attention and Hadamard product both win, while self-attention/gated-fusion underperform despite complexity" is a useful citation for this thesis's own Phase 7 Stage 1→2 narrative (naive fusion underperforming, then cross-attention improving it) — same qualitative story, different specific mechanism ranking. The Q=image/KV=metadata assignment matches this thesis's own cross-attention design exactly, worth citing as external validation of that architectural choice. |
| 14 | 2025 | JI-ADF: A multi-stage multi-modal learning algorithm with adaptive multimodal fusion for improving multi-label skin lesion classification — Zuo, Wang & Wang (Artificial Intelligence in Medicine) | Not yet captured — read full text | Combines clinical images, dermoscopy images, and metadata via uncertainty-based hybrid/adaptive fusion | Not yet captured — read full text | Not yet captured — read full text | Relevant for multi-class framing and adaptive-fusion design ideas |
| 15 | 2025–26 | Advancing skin cancer detection through deep learning and fusion of patient metadata and skin lesion images (Scientific Reports) | 79,246 images (large custom collection) | AI framework fusing patient metadata with image data at scale | Not yet captured — read full text | Not yet captured — read full text | Large-scale precedent; useful contrast given this thesis intentionally uses smaller, leakage-audited PAD-UFES-20 |
| 16 | 2025 | Evaluation of the importance of metadata in skin lesion classification — Garib, Mery & Navarrete-Dechent (Signal, Image and Video Processing) | PAD-UFES-20 (clinical images) + ISIC 2019 (dermoscopic images) | **Abstract-level only (2026-07-28) — paywalled, no free full text found** (Springer redirects to institutional login; ResearchGate blocked 403; author's personal site unreachable; no OA copy indexed by any aggregator checked). 17 deep learning models tested across 3 fusion methods on both datasets; separately trained models on different metadata subsets to isolate each feature's individual contribution to performance. | Image+metadata beat image-only baseline by **+10.43% balanced accuracy on PAD-UFES-20**, **+2.22% on ISIC 2019** (PAD-UFES-20's much larger gain is consistent with this thesis's own experience — its rich clinical metadata carries more signal than a dermoscopy-only archive's sparser tags). Per-feature importance ranking: **age most useful**, then body/anatomical location, then sex. | Not yet captured — full text needed to assess methodology details, exact model list, and statistical rigor of the per-feature importance ranking. | **Directly parallels this thesis's own `feature_whitelist.md` exercise** — the age > location > sex ranking is a useful comparison point for our own feature set (PAD-UFES-20's 21 whitelisted features vs. HAM10000's 3). Same PAD-UFES-20 dataset as this thesis, so its reported image+metadata gain (+10.43% BACC) is a direct external benchmark for our own Phase 6→7 fusion improvement, worth citing even at abstract-only depth — but any specific number should be flagged as "per abstract, not full-text-verified" until/unless full text becomes available. |

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

## Literature Gap: Fitzpatrick/Skin-Tone Fairness

**None of the 16 papers above focus on Fitzpatrick/skin-tone fairness in
dermatology AI.** This is a genuine gap in the current literature-review
table, not just an unread-paper issue — it held even after the 8 new
web-search candidates were added.

This is directly relevant because `PROJECT_PLAN.md`'s Phase 8 (Experiments &
Evaluation) already includes a dedicated Fitzpatrick fairness analysis. The
absence of prior work specifically on skin-tone bias in dermatology AI means
this thesis's own fairness analysis is a citable contribution, not just a
routine evaluation step — worth stating explicitly in the thesis (e.g. in
the Related Work or Contributions section) rather than leaving implicit.

**Recommended action (not yet done):** search for 2–3 papers specifically on
skin-tone/Fitzpatrick bias in dermatology AI before Phase 9 (Thesis Writing
Support) begins, both to properly situate this contribution and to see
whether any existing fairness-evaluation methodology should inform Phase 8's
design.

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
