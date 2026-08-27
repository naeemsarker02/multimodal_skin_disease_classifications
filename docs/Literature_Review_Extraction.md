# Literature Review — Full Extraction (for Thesis Chapter Drafting)

Source: `docs/Literature_Review.md` (reconciled table, 20 rows) + `docs/Literature_Review_Master.xlsx`
("Master Literature Review" sheet). Where the two disagree, the .md is newer/authoritative (it
carries later corrections and full-text-read updates the xlsx was never updated with).

**Read-depth key** (carried over from the source — do not upgrade a paper's status without
actually reading it):
- 🟢 **Full text read** — claims below are as reliable as the source project's own verification
- 🟡 **Abstract-level only** — paywalled, no free full text found; treat all quantitative claims
  as "per abstract," not verified against primary text
- 🔵 **Mechanism-confirmed via secondary source** (code repo + independent paper), not primary PDF

**Citation completeness warning:** several rows are missing pieces of a full citation (author list,
DOI, venue detail) in the source project's own notes — these are marked `[INCOMPLETE CITATION]`
below. Do not fabricate the missing piece when writing the thesis; either track it down from the
publisher/DOI or cite it exactly as the source project has it, flagged as needing verification.

---

## 1. Cross-Attention Enables Context-Aware Multimodal Skin Lesion Diagnosis 🟢
**Citation:** Mridha & Islam (2026). *Cross-Attention Enables Context-Aware Multimodal Skin Lesion
Diagnosis.* medRxiv preprint. `[INCOMPLETE CITATION — no DOI/URL recorded in source; full author
first names not recorded]`

**Method:** ViT image encoder + metadata tokens; cross-attention lets metadata tokens query image
tokens before classification. Compared against metadata-only logistic regression, image-only
ResNet18, and late fusion (4 variants total).

**Strength:** Best of all 4 tested variants — AUC 0.9818, AUPRC 0.9924, F1 0.9769, ECE 0.0379 (low
calibration error). Dataset: PAD-UFES-20 (1,568 lesions).

**Limitation (as noted):** Small dataset; binary malignant/benign framing only (not multi-class);
basic metadata only; no external-dataset validation.

**Relation to our gap:** Direct architectural precedent for the thesis's cross-attention design
(Phase 7 Stage 2). This thesis extends the idea to multi-class classification and cross-dataset
generalization, neither of which this paper attempts.

---

## 2. TG-CAVNet 🟡 `[INCOMPLETE CITATION — high priority to fix]`
**Citation:** Suresh et al. (2026). *TG-CAVNet* [full title, venue, DOI all unrecorded in source —
**this paper has never had its bibliographic details captured** despite being called a "primary
architectural reference"].

**Method:** Text-guided cross-attention variant ("TG-CAVNet" architecture) — mechanism details not
yet captured.

**Strength / Limitation:** Not fully captured — the source project explicitly flags this row as
needing a re-read before it can be cited with specifics.

**Relation to our gap:** Currently used as "primary architectural reference for cross-attention
fusion design," but **this claim cannot be substantiated without re-finding and reading the actual
paper** — do not cite specific numbers or mechanism claims from this row in the thesis until that
happens.

⚠️ **Action needed before thesis writing:** this is flagged in your own source docs as
under-verified. If TG-CAVNet is going to anchor part of your Related Work section, it needs a
literature search pass to even confirm it exists under that name/venue.

---

## 3. (Leakage/shortcut-feature warning paper) — Watson et al. 🟡 `[INCOMPLETE CITATION — same problem as #2]`
**Citation:** Watson et al. (year unrecorded). Title unrecorded — referred to only as "the
leakage/shortcut-feature warning paper."

**Method:** Methodological — identifies that post-diagnosis fields in clinical metadata cause data
leakage in model evaluation.

**Strength / Limitation:** N/A (methodological caution paper, not an empirical benchmark).

**Relation to our gap:** Cited as "direct inspiration for this thesis's entire leakage-audit
methodology (22 excluded columns)" — but again, **the actual title/venue/DOI is missing**. This is
a load-bearing citation for your Methodology chapter's leakage-audit section and currently cannot
be placed in a bibliography. This needs to be re-found before the thesis is finalized.

---

## 4. MM-Skin: A Vision-Language Model for Dermatology (SkinVL) 🟡
**Citation:** (2025). *MM-Skin: A Vision-Language Model for Dermatology (SkinVL).*
`[INCOMPLETE CITATION — authors, venue, DOI all unrecorded]`

**Method:** Large custom vision-language dermatology dataset; SkinVL model combines image + free
text via VQA, supervised fine-tuning, and zero-shot learning.

**Strength:** ~90–94% accuracy (per abstract-level source note).

**Limitation:** High compute requirement; dataset not fully clinical; limited real-world
validation.

**Relation to our gap:** Different modality (free text vs. this thesis's structured metadata) —
useful as a contrast point in Related Work, not a directly comparable architecture.

---

## 5. Multimodal Skin Lesion Classification Using Deep Learning (= Yap, Yolland & Tschandl 2018) 🟢
**Citation:** Yap, J., Yolland, W., & Tschandl, P. (2018). *Multimodal Skin Lesion Classification
Using Deep Learning.* **Experimental Dermatology**, 27(11), 1261–1267.
**DOI: 10.1111/exd.13777** (PubMed 30187575).

**Method:** Two-tower CNN feature extraction (one tower per image modality: dermatoscopic image,
macroscopic photo) + patient metadata vector, all concatenated (late fusion), passed through an
embedding network for final classification. Dataset: custom 2,917-case tri-modal set (**not** ISIC
Archive — this was a correction made in the source project after an earlier miscategorization).

**Strength:** Binary melanoma AUC 0.866 (multimodal) vs. 0.784 (macroscopic-image-only baseline);
multiclass mean average precision 0.729 (multimodal) vs. 0.598 (image-only).

**Limitation (as noted):** Small/less diverse dataset relative to later work; no symptom-level
information; limited real-world variability.

**Relation to our gap:** Early (2018) three-modality fusion paper — useful for Related Work
historical framing as an early proof that combining modalities beats image-only. **Caution:** an
earlier draft of the source review misreported this paper's accuracy as "~93–96%" — that figure is
not real; use only the corrected AUC 0.866 / mAP 0.729 figures above.

---

## 6. Explainable AI for Skin Disease Classification 🟡
**Citation:** (2024). *Explainable AI for Skin Disease Classification.*
`[INCOMPLETE CITATION — authors, venue, DOI all unrecorded]`

**Method:** Multiple CNN transfer-learning models + Grad-CAM visualization. Dataset: HAM10000.

**Strength:** ~91–93% accuracy (per abstract-level source note).

**Limitation:** Image-only, no metadata/symptom integration; explanation limited to visual (Grad-CAM).

**Relation to our gap:** Relevant for Discussion/Future Work — explainability is a natural
extension beyond this thesis's scope.

---

## 7. A Multimodal Approach to Skin Disease Detection Using Patient Symptoms 🟡
**Citation:** (2024). `[INCOMPLETE CITATION — authors, venue, DOI all unrecorded]`

**Method:** Image model produces an initial prediction; an LLM-based reasoning step ("Chain of
Options") refines the prediction using symptom text. Dataset: DermNet Dataset.

**Strength:** ~90–92% accuracy (per abstract-level source note).

**Limitation:** Symptom data partially synthetic/non-standardized; weak clinical reliability.

**Relation to our gap:** Different fusion paradigm (LLM reasoning vs. this thesis's learned
cross-attention) — a contrast point, not a direct precedent.

---

## 8. Skin Lesion Classification Using EfficientNet 🟡
**Citation:** (2024). `[INCOMPLETE CITATION — authors, venue, DOI all unrecorded]`

**Method:** EfficientNet-B0/B1 transfer learning on dermoscopic images only. Dataset: HAM10000.

**Strength:** ~90–94% accuracy (per abstract-level source note).

**Limitation:** No metadata/symptoms; purely image-based.

**Relation to our gap:** Validates this thesis's own choice of EfficientNet-B0 as a reasonable
image-branch backbone/baseline.

---

## 9. Multimodal Learning with Clinical Metadata for Skin Cancer Diagnosis 🟡
**Citation:** (2025). `[INCOMPLETE CITATION — authors, venue, DOI all unrecorded]`

**Method:** Dual-branch architecture — CNN for image, TabNet for clinical metadata — fused
representations. Dataset: PAD-UFES-20.

**Strength:** ~94–97% accuracy (per abstract-level source note).

**Limitation:** Skin-cancer-focused only; limited disease variety; structured metadata only.

**Relation to our gap:** Closest prior-art comparison for this thesis's own PAD-UFES-20
image+metadata baseline — but note the accuracy figure is abstract-level/unverified, so treat any
head-to-head comparison in Results with the same caution the source project applies elsewhere
(accuracy vs. macro-F1 mismatch risk — see Paper #13's standing caveat).

---

## 10. MetaBlock: An Attention-Based Mechanism to Combine Images and Metadata in Deep Learning Models Applied to Skin Cancer Classification 🔵
**Citation:** Pacheco, A. G. C., & Krohling, R. A. (2021). *MetaBlock: An Attention-Based Mechanism
to Combine Images and Metadata in Deep Learning Models Applied to Skin Cancer Classification.*
IEEE Journal of Biomedical and Health Informatics (JBHI). **Primary PDF is paywalled** — no free
full text found (checked arXiv, ResearchGate, author's site — all dead ends).

**Method (mechanism confirmed via official GitHub repo `github.com/paaatcha/MetaBlock` +
independently agreeing secondary source, NOT read from primary text):** Despite its own
"attention-based" framing, MetaBlock is **not** Transformer-style Q/K/V attention. It is a
channel-wise gated feature-modulation block: metadata vector *U* passes through two independent
Linear+BatchNorm branches producing *t1, t2* (same channel-dim as CNN feature vector *V*); output
*V′ = sigmoid(tanh(V·t1) + t2)* — a multiplicative gate plus additive bias, squashed by sigmoid,
broadcast uniformly across spatial positions within each channel (no per-location weighting).
Dataset: PAD-UFES-20 (this is the dataset's own creators) + ISIC 2019.

**Strength:** Per abstract — improves classification across all tested models on both datasets;
beats MetaNet + concatenation baselines in 6/10 tested scenarios. Secondary-source (unconfirmed
against primary) figures: ~80.7%±0.8% accuracy (ISIC 2019), ~74.8%±1.8% (PAD-UFES-20).

**Limitation (as noted):** Channel-wise-only gating — no explicit spatial/token-level attention
weighting, unlike genuine cross-attention.

**Relation to our gap:** **Near-mandatory citation** — written by PAD-UFES-20's own creators. This
thesis's cross-attention design (metadata=Q, image spatial tokens=K/V) is confirmed to be a
*related-but-distinct* mechanism from MetaBlock, not a reproduction of it — correct framing is
"cross-attention, contrasted with MetaBlock's channel-gating approach."

---

## 11. The Impact of Patient Clinical Information on Automated Skin Cancer Detection 🟢
**Citation:** Pacheco, A. G. C., & Krohling, R. A. (2019/2020). *The Impact of Patient Clinical
Information on Automated Skin Cancer Detection.* arXiv preprint **arXiv:1909.12912** (freely
available).

**Method:** Simple concatenation-based fusion (not gating, not cross-attention). CNN feature
extractor (kept frozen, then fine-tuned) → flattened image features → "feature reducer" NN block
whose output size is set by a tunable "combination factor" `cf` (0.5–0.9) that balances image
feature dimensionality against the fixed 28 clinical features (8 raw fields, one-hot encoded except
age); reduced image features concatenated with clinical features → classifier.
`T = ceil(cf·N_img + (1-cf)·N_cli)`. Best `cf` = 0.7–0.8. Tested 6 backbones (ResNet-50/101,
GoogleNet, MobileNet, VGGNet-13/19), 5-fold CV, class-weighted loss. Dataset: precursor to
PAD-UFES-20 — 1,612 clinical images, 6 classes, 8 raw clinical fields (smaller/earlier than the
public 2,298-image PAD-UFES-20 release).

**Strength:** Image-only avg across 6 models: BACC 0.650±0.031. Image+clinical avg: BACC
0.718±0.022 (~7-point absolute improvement), AUC 0.929→0.948. Best single model (ResNet-50
image+clinical): ACC 0.788±0.025, BACC 0.750±0.033, F1 0.790±0.027, AUC 0.958±0.007.
Friedman+Wilcoxon tests confirm all 6 models were significantly improved by adding clinical data.

**Limitation (as noted, and self-acknowledged by the authors):** Clinical features improve
differentiation of ACK/MEL/NEV/SEK but do **not** help separate SCC vs. BCC — the two classes share
near-identical clinical profiles (both bleed, hurt, itch, same age range, same anatomical-site
preference), so metadata doesn't resolve this specific confusion. Smartphone image quality (vs.
dermoscopy) and patient self-report subjectivity also flagged as limitations.

**Relation to our gap:** **Foundational justification for this thesis's entire multimodal
premise** — cite in the Introduction. Worth checking whether SCC/BCC confusion persists in your own
PAD-UFES-20 confusion matrices as a direct forward-citation. This paper's tunable-concatenation
fusion is a direct historical precursor to this thesis's own late-fusion Stage 1 baseline. **Also
note:** this paper's dataset is the direct precursor to the public PAD-UFES-20 release — the
correct citation for the *dataset itself* is a separate paper (Pacheco et al., "PAD-UFES-20: A skin
lesion dataset composed of patient data and clinical images collected from smartphones," *Data in
Brief*), which is **not yet in this 20-paper table** and should be added as the mandatory dataset
citation in your Methodology chapter.

---

## 12. A Multimodal Skin Lesion Classification Through Cross-Attention Fusion and Collaborative Edge Computing 🟡
**Citation:** (2025). *A Multimodal Skin Lesion Classification Through Cross-Attention Fusion and
Collaborative Edge Computing.* **Computerized Medical Imaging and Graphics**, vol. 124, article
102588. `[INCOMPLETE CITATION — author names unrecorded; DOI not captured, only volume/page]`
Paywalled (ScienceDirect HTTP 403), no arXiv preprint found.

**Method:** Three-module architecture — modality-wise feature extraction, cross-attention-based
feature fusion (dermoscopic images + patient metadata), multimodal classifier — paired with a
"collaborative inference scheme" distributing compute across IoT/edge devices for
privacy/latency reasons.

**Strength:** Per abstract — cross-attention fusion outperforms unimodal and simpler-fusion
baselines; cross-attention and Hadamard-product fusion both near state-of-the-art (~98.86%/98.85%
accuracy — **dataset/task not yet confirmed**, treat with caution).

**Limitation:** Full text not read — dataset, exact metric definitions, and edge-computing
evaluation details all unconfirmed. The very high accuracy figure with no macro-F1 reported is
flagged in the source as worth scrutinizing.

**Relation to our gap:** Structurally similar cross-attention precedent, but the edge-computing/
privacy motivation differs from this thesis's generalization/fairness motivation — useful contrast
in Related Work, but do not cite the ~98.8% figure as directly comparable to this thesis's own
(much lower) macro-F1 numbers without the metric-mismatch caveat used elsewhere in this table.

---

## 13. Comparative Analysis of Multimodal Architectures for Effective Skin Lesion Detection Using Clinical and Image Data 🟢
**Citation:** (2025). *Comparative Analysis of Multimodal Architectures for Effective Skin Lesion
Detection Using Clinical and Image Data.* **Frontiers in Artificial Intelligence** (open access via
PMC). `[INCOMPLETE CITATION — author names and DOI not recorded in source, despite full text having
been read]`

**Method:** Two feature extractors — Clinical MLP (128→256-dim on tabular metadata) and
DermiResNet (a modified ResNet with a learnable weighted skip connection, `y = F(x) + α·x`,
512-dim output). **8 fusion methods compared head-to-head:** simple concatenation, weighted
concatenation, Hadamard product, tensor fusion (outer product), bilinear fusion, gated fusion,
self-attention (intra-modality), and cross-attention (image=Query, clinical metadata=Key/Value —
same Q/KV direction as this thesis's own design). Dataset: **HAM10000** — the same benchmark this
thesis uses (10,015 images, 7 classes; clinical fields: diagnosis-confirmation method, age, sex,
anatomical location).

**Strength:** Cross-attention: 98.86% accuracy (best); Hadamard product: 98.85% (near-tied);
bilinear: 98.76%. Unimodal ablation: image-only 92.0%, metadata-only 77.0%. Cross-attention
per-class AUC: Melanoma 0.99, Nevus 0.98, all others 1.0.

**Limitation (as noted — critical, self-flagged by authors too):** Reports only accuracy/weighted-F1
on HAM10000's 58:1-imbalanced 7-class split (Nevus 6,705 vs. Dermatofibroma 115), **not macro-F1**
— exactly the metric choice this thesis's own project rules reject for imbalanced data. Also
self-flagged: heavy compute cost for cross-attention/tensor fusion; HAM10000 alone doesn't capture
real-world population diversity; limited clinical-feature richness; Grad-CAM-only
interpretability; persistent melanoma↔nevus/benign-keratosis misclassification despite fusion.

**Relation to our gap:** Strong same-dataset external benchmark. **Standing caveat that must
accompany every citation of this paper in the thesis:** (1) independent architectural validation —
same Q=image/KV=metadata direction as this thesis's cross-attention design, found by a different
group on the same dataset; (2) their 98.86% is plain accuracy on an imbalanced split, not
comparable to this thesis's macro-F1 numbers — any side-by-side mention must state explicitly that
the gap is a metric artifact, not a performance gap.

---

## 14. A Multi-Stage Multi-Modal Learning Algorithm with Adaptive Multimodal Fusion for Improving Multi-Label Skin Lesion Classification ("CosCatNet") 🟡
**Citation:** Zuo, L., Wang, Z., & Wang, Y. (2025). *A Multi-Stage Multi-Modal Learning Algorithm
with Adaptive Multimodal Fusion for Improving Multi-Label Skin Lesion Classification.* Artificial
Intelligence in Medicine. `[INCOMPLETE CITATION — DOI/volume/page not captured]` Paywalled
(ScienceDirect HTTP 403), no arXiv preprint; PubMed/GitHub README give abstract-depth detail only.
Code: `github.com/Zuo-Lihan/CosCatNet-Adaptive_Fusion_Algorithm`.

⚠️ **Title correction already made in the source:** an earlier note incorrectly called this paper
"JI-ADF" — that name belongs to a *different, unrelated* paper (Phan Nguyen et al., arXiv:2604.27343,
evaluated on MILK10k). Do not conflate the two.

**Method ("CosCatNet"):** Two-stage hybrid fusion — (1) image-fusion stage combining clinical
photos + dermoscopy images via cosine-similarity-based intermediate fusion (captures correlated
cross-image information) plus concatenation (captures complementary information); (2) multimodal
fusion stage combining the fused image representation with metadata via uncertainty-based adaptive
late fusion (dynamically weights modality contributions per-sample based on estimated confidence).

**Strength:** Abstract states effectiveness demonstrated on "a popular publicly available skin
disease diagnosis dataset" — likely trimodal (clinical + dermoscopy + metadata, possibly Derm7pt),
not confirmed. No specific quantitative metrics available without full text.

**Limitation:** Full text needed for dataset confirmation, quantitative results, and comparison
methodology.

**Relation to our gap:** A third distinct fusion paradigm in this literature set (cosine-similarity/
concatenation hybrid + uncertainty-weighted late fusion), contrasted with MetaBlock's channel-gating
(#10) and this thesis's cross-attention. Uses clinical+dermoscopy image pairs — a modality this
thesis's datasets (PAD-UFES-20/HAM10000/ISIC, single-image) don't have, so not architecturally
reproducible here, but the uncertainty-weighted fusion idea is citable for Future Work.

---

## 15. Advancing Skin Cancer Detection Through Deep Learning and Fusion of Patient Metadata and Skin Lesion Images 🟢
**Citation:** Islam, Wishart, Walls, Hall, Seco de Herrera, Gan, & Raza (2025/2026). *Advancing Skin
Cancer Detection Through Deep Learning and Fusion of Patient Metadata and Skin Lesion Images.*
**Scientific Reports** (fully open access via PMC). `[INCOMPLETE CITATION — DOI not recorded]`

**Method:** EfficientNet-B2 backbone; hair removed via a variational-autoencoder method; images
resized to 1024×1024; 16-technique Albumentations augmentation pipeline. Six model variants tested
(DER-only, DSLR-only, DER+meta, DSLR+meta, DER+DSLR, DER+DSLR+meta) — fusion is simple feature
concatenation (⊕) + Swish activation + 0.5-ratio dropout, **not attention-based**. Final system uses
decision-level majority-vote fusion across two of the six trained variants, not a single end-to-end
model. Patient-wise 80/20 split, 5-fold CV. Dataset: Check4Cancer (UK private teledermatology
network), 2015–2022: 79,246 images (39,623 dermoscopic + 39,623 DSLR pairs), 19,295 patients; 22
metadata features.

**Strength:** Best single model (DER+DSLR+meta): 88.77% accuracy, 92.98% AUC, sens 99.83%/spec
77.71%. Best majority-vote fusion: **91.11% accuracy, 94.06% AUC, sens 99.50%/spec 82.72%** — beats
every single-modality/single-model variant. Image-only baseline: 81.28% accuracy, so metadata
fusion adds ~10 accuracy points, mostly via specificity gains. Benchmarked against real
teledermatology clinical performance and a competing commercial system ("Skin Analytics").

**Limitation (self-acknowledged):** Ground truth is *expert visual triage rating*, not
biopsy-confirmed diagnosis — only 10% of lesions were ever biopsied. Fitzpatrick skin type was
**not even collected**; population predominantly types I–IV.

**Relation to our gap:** Large-scale precedent using simpler concatenation + decision-level voting
rather than a jointly-trained attention mechanism — useful contrast for framing this thesis's
cross-attention as architecturally more sophisticated even at smaller scale. **Two citable
parallels:** (1) its unconfirmed-ground-truth limitation mirrors this thesis's own `biopsed`-field
leakage concern, from the opposite direction; (2) its independent finding of Fitzpatrick V/VI
under-representation (different country/cohort/dataset) corroborates this thesis's own Phase 8
fairness finding that the gap is systemic, not PAD-UFES-20-specific.

---

## 16. Evaluation of the Importance of Metadata in Skin Lesion Classification 🟡
**Citation:** Garib, Mery, & Navarrete-Dechent (2025). *Evaluation of the Importance of Metadata in
Skin Lesion Classification.* **Signal, Image and Video Processing** (Springer).
`[INCOMPLETE CITATION — DOI not recorded]` Paywalled (Springer institutional login required;
ResearchGate blocked; author site unreachable; no OA copy found).

**Method:** 17 deep learning models tested across 3 fusion methods on two datasets; separately
trained models on different metadata subsets to isolate each feature's individual contribution.
Datasets: PAD-UFES-20 (clinical images) + ISIC 2019 (dermoscopic images).

**Strength:** Image+metadata beat image-only baseline by **+10.43% balanced accuracy on
PAD-UFES-20**, +2.22% on ISIC 2019 — consistent with the intuition that PAD-UFES-20's richer
clinical metadata carries more signal than a dermoscopy-only archive's sparser tags. Per-feature
importance ranking: **age most useful**, then body/anatomical location, then sex.

**Limitation:** Full text needed to assess exact model list and statistical rigor of the
per-feature ranking.

**Relation to our gap:** Directly parallels this thesis's own `feature_whitelist.md` exercise — the
age > location > sex ranking is a comparison point for your own feature set. Same PAD-UFES-20
dataset as this thesis, so the +10.43% BACC gain is a direct external benchmark for your own
Phase 6→7 fusion improvement — but flag any number cited from this row as "per abstract, not
full-text-verified."

---

## 17. Disparities in Dermatology AI Performance on a Diverse, Curated Clinical Image Set 🟢
**Citation:** Daneshjou, R., Yekrang-Sis, K., Cai, Z. R., et al. (2022). *Disparities in Dermatology
AI Performance on a Diverse, Curated Clinical Image Set.* **Science Advances**. Free full text:
**arXiv:2203.08807**. `[INCOMPLETE CITATION — full author list beyond first three and DOI not
recorded]`

**Method:** Introduces the **DDI (Diverse Dermatology Images)** dataset — 656 images, pathologically
confirmed diagnoses, curated for diverse Fitzpatrick skin-tone representation. Benchmarks existing
SOTA dermatology classifiers (trained on standard datasets) against DDI; compares dermatologist
diagnostic performance across the same skin-tone/disease splits; also tests fine-tuning models on
DDI itself. (Published 2022 — outside the source project's preferred 2023–2025 window, kept as a
logged exception because it's the field's foundational fairness-benchmark paper.)

**Strength:** N/A framing — this paper's main "finding" is itself the deficiency it exposes (see
Limitation), which is why it's being cited for the gap rather than as a positive-result precedent.

**Limitation/gap found by this paper (this IS the paper's headline finding, not a shortcoming of
it):** 27–36 percentage-point ROC-AUC drop for SOTA models vs. their originally reported test
performance, concentrated on dark skin tones and uncommon diseases; dermatologists also perform
worse on dark-skin/uncommon-disease images; fine-tuning on DDI narrows but does not eliminate the
light/dark performance gap. Its own limitation: DDI itself is relatively small (656 images);
disparity is demonstrated at the model-benchmark level, not root-caused to a single mechanism.

**Relation to our gap:** **Near-mandatory citation for your Phase 8.3 fairness section.** Direct
methodological analogue — an external, pathologically-confirmed test set built specifically to
expose skin-tone performance gaps, exactly the kind of evaluation your own Fitzpatrick fairness
analysis performs (within-dataset on PAD-UFES-20 rather than via a dedicated external benchmark).

---

## 18. Skin Type Diversity in Skin Lesion Datasets: A Review 🟢
**Citation:** Alipour, N., Burke, T., & Courtney, J. (2024). *Skin Type Diversity in Skin Lesion
Datasets: A Review.* **Current Dermatology Reports**. **DOI: 10.1007/s13671-024-00440-0**
(PMC11343783, freely available).

**Method:** Systematic review across multiple public skin lesion datasets. Evaluates both whether
Fitzpatrick skin type is reported at all, and separately, how diverse/representative that reporting
actually is — explicitly notes prior work does one or the other but not both together.

**Strength:** N/A (review paper) — its contribution is synthesis/confirmation across the field
rather than a single quantitative finding.

**Limitation/gap (this IS the finding):** Under-representation of darker skin types is a
**systemic, field-wide dataset problem**, not an isolated instance in any one dataset. Its own
limitation: review-level, not a new dataset or model; scope limited to datasets with publicly
available metadata.

**Relation to our gap:** **Strongest citation for your "Fitzpatrick/Skin-Tone Fairness" gap
framing** — independently confirms under-representation is systemic, directly corroborating this
thesis's own finding (PAD-UFES-20 test split: only 3 rows total across Fitzpatrick V/VI, 38%
missing entirely). Cite wherever you state the fairness gap is field-wide, not a PAD-UFES-20
artifact.

---

## 19. A Framework for Evaluating the Efficacy of Foundation Embedding Models in Healthcare 🟢
**Citation:** Xu, Y., Gui, H., Rotemberg, V., Wang, K., Chen, Q., & Daneshjou, R. (2024). *A
Framework for Evaluating the Efficacy of Foundation Embedding Models in Healthcare.* medRxiv
preprint **2024.04.17.24305983** (posted 2024-04-19, free).

**Method:** Proposes and pilots a 3-axis framework (general performance / bias-fairness /
confounders) for evaluating medical foundation models, applied to dermatology. Evaluates Google
Health's Derm Foundation Model embeddings plus general-purpose CLIP embeddings across all 3 axes;
measures per-Fitzpatrick-group sensitivity directly on the pretrained embedding space (not just
after task-specific fine-tuning).

**Strength:** Foundation-model embeddings exceed SOTA classification accuracy; general-purpose CLIP
embeddings are also informative for medical tasks. The 3-axis framework itself is a reusable
methodological contribution.

**Limitation (as noted):** **Lower sensitivity for darker Fitzpatrick tones (4–6)**, present even at
the pretrained-embedding level before any fine-tuning — this is the paper's key negative finding.
Its own limitation: single foundation-model family evaluated in depth; the 3-axis framework is
proposed/piloted, not yet an established field standard.

**Relation to our gap:** Confirms skin-tone-linked bias exists even inside pretrained
foundation-model embeddings, not just end-task classifiers — a relevant caution if future work ever
considers a foundation-model backbone. The 3-axis framework (general performance / bias-fairness /
confounders) is a citable structure for organizing your own Phase 8.3 write-up.

---

## 20. Multimodal Skin Disease Classification Using Vision Transformers, Medical Captioning, and Metadata Fusion: An Analysis on the ISIC 2024 Dataset 🟡
**Citation:** Shrestha & Palit (2026). *Multimodal Skin Disease Classification Using Vision
Transformers, Medical Captioning, and Metadata Fusion: An Analysis on the ISIC 2024 Dataset.*
**Biomedical Physics & Engineering Express**, vol. 12, no. 2, article 025043.
**DOI: 10.1088/2057-1976/ae4eeb**. `[INCOMPLETE CITATION — full first names not recorded]`
Bibliographic details verified via publisher lookup; full methodology not yet read in depth.

**Method:** Combines MedCLIP-derived image-text embeddings with patient metadata through early and
attention-based fusion; a vision-language captioning stage precedes the fusion step. Dataset:
curated ISIC 2024 subset, image+text+metadata.

**Strength:** 96% accuracy, AUROC 0.987 — but on a **binary** classification task.

**Limitation:** Binary framing and accuracy-as-primary-metric make this not directly comparable to
this thesis's 6-class, macro-F1-based evaluation. Full text not yet read for methodology depth
(fusion architecture details, dataset split discipline, leakage-audit presence/absence unconfirmed).

**Relation to our gap:** Already cited in `docs/main.tex` (Related Work — Deep Learning-Based
Approaches, and the litcompare table) as the vision-language-fusion comparator; illustrates the
value of combining vision-language representations with metadata, contrasted with this thesis's
structured-metadata-only cross-attention design.

---

## Summary: what still needs work before this table is fully thesis-ready

1. **Papers #2, #3, #6, #7, #8, #9, #12, #16 have incomplete citations** (missing authors, venue,
   and/or DOI) — several of these (#2 TG-CAVNet, #3 the leakage paper) are load-bearing citations
   for your Methodology chapter and currently cannot be placed in a bibliography as-is.
2. **Abstract-only papers (🟡: #2, #3, #4, #6, #7, #8, #9, #12, #14, #16, #20)** — any quantitative
   number pulled from these must be flagged "per abstract" in the thesis, not stated as verified
   fact, per the source project's own discipline.
3. **The dataset citation for PAD-UFES-20 itself** (Pacheco et al., *Data in Brief*, "PAD-UFES-20: A
   skin lesion dataset composed of patient data and clinical images collected from smartphones") is
   **not in this 20-paper table at all** and should be added as a separate mandatory dataset
   citation in your Methodology chapter — it's distinct from papers #10/#11, which are about
   *methods*, not the dataset release.
4. **Paper #13's standing caveat** (accuracy vs. macro-F1 — see its entry above) must be repeated
   verbatim wherever it's cited in the Discussion chapter, per the source project's own instruction.
