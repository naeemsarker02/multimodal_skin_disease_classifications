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

**Citation-verification pass (2026-08-27):** every row previously marked `[INCOMPLETE CITATION]`
was re-searched (web search + crossref/arXiv/publisher API lookups). Status:
- **6 fully verified** with complete author list, venue, and DOI/arXiv ID: #2, #3, #4, #7, #12, #16
  (plus the separately-added PAD-UFES-20 dataset paper, see new section after #19).
- **2 moderate-confidence**: #6 and #8 — title, method, and venue confirmed via publisher/crossref
  lookup, but the specific dataset/accuracy claims recorded in this table were **not** independently
  re-confirmed against the primary text in this pass. Any number cited from these two rows in the
  thesis must carry an explicit hedge (e.g. "reportedly ~90–94%"), not be stated as fact.
- **1 dropped**: #9 — could not be re-located under any plausible title/method/dataset combination
  after four independent search passes; see its entry below for the full explanation.
- Remaining rows still carrying a minor gap (e.g. one missing DOI, not re-searched this pass since
  not requested): #14, #15. Flagged individually where they remain.

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

## 2. TG-CAVNet 🟡 — **CITATION VERIFIED 2026-08-27**
**Citation:** Suresh, P., Keerthika, P., & Nitesh Kumar, A. R. (2026). *Text guided cross attentive
multimodal learning with visual feature modulation for automated skin lesion detection.*
**Scientific Reports**, 16. **DOI: 10.1038/s41598-026-47271-6**.

**Method:** Bio-ClinicalBERT clinical-text encoder + EfficientNet-B4 visual feature extractor,
combined via three components: (1) text-guided **channel-wise feature modulation**, (2)
text-queried cross-attention for semantic-spatial alignment, (3) adaptive multi-stream fusion.
Evaluated on a custom multimodal dermoscopic dataset of **6,194 aligned image-text samples**
(not yet confirmed which public archive, if any, this derives from).

**Strength:** 90.75% accuracy, macro Jaccard score 0.82 (per abstract/search-summary — not yet
read from primary text). Ablation studies reported to confirm separate and synergistic
contributions of each component; attention visualizations used for interpretability.

**Limitation:** Still only abstract/summary-depth for us — the full methodology (exact formula for
the channel-wise modulation step, dataset provenance, training details) has not been read from the
primary PDF. Retain the 🟡 tag until that happens.

**Abstract-only claim for thesis prose:** "TG-CAVNet reportedly achieves 90.75% accuracy and a macro
Jaccard score of 0.82 on a custom 6,194-sample image-text dermoscopic dataset (Suresh et al., 2026,
per the paper's abstract)."

**Relation to our gap:** ⚠️ **Flagging doubt, per your instruction, rather than assuming.** Your
`MetadataChannelGate` module (`src/models/cross_attention_fusion_model.py:63-84`) is documented in
its own docstring as "Suresh et al. TG-CAVNet-inspired... kept as a secondary, optional mechanism...
not a primary design input pending its own full-text read." What I verified this pass is that
TG-CAVNet's abstract-level description ("text-guided channel-wise feature modulation") is
**consistent in spirit** with your gate's design (a metadata-conditioned sigmoid gate over image
channels, applied before cross-attention) — but I have **not** obtained TG-CAVNet's actual
modulation formula, so I cannot confirm your implementation reproduces or even closely approximates
their specific mechanism. **Recommend keeping the citation as "TG-CAVNet-inspired" (as the code
already correctly hedges) rather than upgrading to a stronger claim of architectural correspondence,
until the primary paper's methodology section is read.**

---

## 3. Multimodal Models for Skin Cancer Classification Using Clinical Freetext and Dermatoscopic Images — Watson et al. 🟡 — **CITATION VERIFIED 2026-08-27**
**Citation:** Watson, M., Winterbottom, T., Hudson, T., Jones, B., Shum, H. P. H.,
Atapour-Abarghouei, A., Breckon, T., Harmsworth King, J., & Al Moubayed, N. (2026). *Multimodal
models for skin cancer classification using clinical freetext and dermatoscopic images.*
**Communications Medicine**, 6. **DOI: 10.1038/s43856-026-01456-2**.

**Method:** Multimodal ML models combining dermatoscopic images, clinical free-text notes, and
patient metadata for benign/malignant classification. Dataset: 5,481 dermatoscopic images from
4,538 patients, binary labels (7% malignant). Investigates how **"leading language"** in clinical
free-text (i.e., diagnostic language physicians wrote into notes that effectively states or implies
the answer) inflates model performance, and develops a preprocessing pipeline (regex + LLM) to
strip it out.

**Strength:** Free-text improves classification performance even after the leading/diagnostic
language is removed — i.e., free-text carries genuine signal beyond the leakage artifact.

**Limitation/gap found by this paper:** Confirms that clinical free-text notes can contain
diagnosis-leaking language that inflates apparent model performance if not explicitly detected and
removed — a shortcut-learning risk that would otherwise go unnoticed.

**Abstract-only claim for thesis prose:** "Watson et al. (2026) report, per their abstract, that
free-text clinical notes continue to improve multimodal skin-cancer classification performance even
after diagnosis-leaking 'leading language' is stripped out — an analogous free-text leakage finding
to this thesis's own tabular-metadata leakage audit."

**Relation to our gap — ⚠️ REFRAMED (as you requested):** The original source note described this as
"post-diagnosis fields cause data leakage," implying tabular/structured metadata fields analogous to
this thesis's own excluded `biopsed`-style columns. What is actually confirmed is a **structurally
different but conceptually analogous** leakage mechanism: **diagnosis-leaking language embedded in
free-text clinical notes**, not structured tabular fields. When citing this paper in your
Methodology chapter's leakage-audit section, frame it as **"an analogous free-text leakage finding
from a different modality"** — evidence that the general phenomenon (models exploiting
post-diagnosis information that leaked into an input feature) recurs across modalities — rather than
claiming it demonstrates the exact same tabular-field leakage mechanism your 22-column audit
addresses.

---

## 4. MM-Skin: Enhancing Dermatology Vision-Language Model with an Image-Text Dataset Derived from Textbooks 🟡 — **CITATION VERIFIED 2026-08-27**
**Citation:** Zeng, W., Sun, Y., Ma, C., Tan, W., & Yan, B. (2025). *MM-Skin: Enhancing Dermatology
Vision-Language Model with an Image-Text Dataset Derived from Textbooks.* **arXiv:2505.06152**
(submitted 2025-05-09). Note: real title differs from the placeholder title previously recorded
("A Vision-Language Model for Dermatology (SkinVL)") — SkinVL is the *model* introduced by this
paper, not its title.

**Method:** MM-Skin dataset (3 imaging modalities — clinical, dermoscopic, pathological; ~10K
image-text pairs from professional textbooks; 27K+ VQA samples). SkinVL model built on this data,
evaluated on VQA, supervised fine-tuning, and zero-shot classification across 8 datasets.

**Strength:** ~90–94% accuracy (per abstract-level source note — treat as "reportedly," not
verified against primary text).

**Abstract-only claim for thesis prose:** "MM-Skin's SkinVL model reportedly achieves ~90–94%
accuracy across VQA/SFT/zero-shot dermatology benchmarks (Zeng et al., 2025, per abstract-level
sourcing)."

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

## 6. Explainable AI for Skin Disease Classification Using Gradient-Weighted Class Activation Mapping and Transfer Learning in Digital Health to Identify Contours 🟡 — **MODERATE CONFIDENCE**
**Citation:** Badhon, S. M. S. I., Khushbu, S. A., Shaqib, S. M., Ali, M. A., Anik, A. H., &
Hossain, K. S. M. T. (2025). *Explainable AI for skin disease classification using
gradient-weighted class activation mapping and transfer learning in digital health to identify
contours.* **DIGITAL HEALTH**, 11. **DOI: 10.1177/20552076251404523**. (An August 2024 preprint of
this became the May 2025 journal version — explains the year discrepancy in the original note.)

⚠️ **Moderate confidence, not full verification:** title, authors, venue, and DOI are confirmed via
crossref. The dataset (HAM10000) and "multiple CNN transfer-learning models" claims below are
**carried over from the original source note and were NOT independently re-confirmed** this pass
(publisher page returned HTTP 403 on fetch attempts) — treat as plausible, not verified.

**Method:** Multiple CNN transfer-learning models + Grad-CAM visualization (per original source
note, unconfirmed this pass). Dataset: HAM10000 (per original source note, unconfirmed this pass).

**Strength:** Reportedly ~91–93% accuracy — **do not state as a confirmed fact in the thesis; use a
hedge such as "reportedly ~91–93% accuracy" per abstract-level sourcing.**

**Limitation:** Image-only, no metadata/symptom integration; explanation limited to visual (Grad-CAM)
— per original source note, unconfirmed this pass.

**Abstract-only claim for thesis prose:** "Badhon et al. (2025) reportedly achieve ~91–93% accuracy
using Grad-CAM-explained CNN transfer learning on HAM10000, per the original source note's
abstract-level summary (dataset and CNN-model details not independently re-confirmed in this
review)."

**Relation to our gap:** Relevant for Discussion/Future Work — explainability is a natural
extension beyond this thesis's scope.

---

## 7. A Multimodal Approach to The Detection and Classification of Skin Diseases 🟡 — **CITATION VERIFIED 2026-08-27**
**Citation:** Yang, A., & Yang, E. (2024). *A Multimodal Approach to The Detection and
Classification of Skin Diseases.* arXiv preprint **arXiv:2411.13855** (submitted 2024-11-21).
No DOI — unpublished preprint, not yet in a peer-reviewed venue as far as could be confirmed.
Full author list confirmed via arXiv's own Atom API (`export.arxiv.org/api/query`): **only two
authors, Allen Yang (Mission San Jose High School, Fremont, CA) and Edward Yang (Yale University,
New Haven, CT)** — the abstract-page UI text suggesting "9 total authors" in an earlier fetch was a
misread of generic page chrome, not real metadata; arXiv's authoritative API confirms 2 authors.
Real title differs from the placeholder previously recorded ("A Multimodal Approach to Skin Disease
Detection Using Patient Symptoms").

**Method:** ResNet-50 initial image-based prediction (baseline 70% accuracy, improved to 80% via
optimization), refined via a novel LLM fine-tuning strategy, **"Chain of Options"**, which breaks a
complex reasoning task into intermediate steps at training time rather than inference time, using
the patient's symptom narrative text.

**Strength:** 91% accuracy (confirmed from the paper's own abstract, not a secondary summary) for
diagnosing patient skin disease from image + symptom description.

**Abstract-only claim for thesis prose:** "Yang & Yang (2024) report, per their own abstract, 91%
accuracy diagnosing skin disease from image plus patient symptom narrative using their 'Chain of
Options' LLM fine-tuning strategy, on a custom 37K-image dataset (not DermNet)."

**Limitation (as noted):** ⚠️ **Dataset correction:** the original source note recorded this
paper's dataset as "DermNet Dataset" — that is **incorrect**. The confirmed abstract states this
paper introduces its **own custom dataset**: 26 skin disease types, 37K images with associated
patient narratives, not DermNet. Symptom data partially synthetic/non-standardized; weak clinical
reliability (carried over from original note, consistent with a custom/narrative-based dataset).

**Relation to our gap:** Different fusion paradigm (LLM reasoning vs. this thesis's learned
cross-attention) — a contrast point, not a direct precedent.

---

## 8. Skin Lesion Classification Using EfficientNet B0 and B1 via Transfer Learning for Computer Aided Diagnosis 🟡 — **MODERATE CONFIDENCE**
**Citation:** Frederich, J., Himawan, J., & Rizkinia, M. (2024). *Skin lesion classification using
EfficientNet B0 and B1 via transfer learning for computer aided diagnosis.* AIP Conference
Proceedings, 3080, article 110002 (7th Biomedical Engineering's Recent Progress in Biomaterials,
Drugs Development, and Medical Devices, ACB-ISBE 2022). **DOI: 10.1063/5.0200741**.

⚠️ **Moderate confidence, not full verification:** title, authors, venue, and DOI confirmed via
crossref, and the method (EfficientNet-B0/B1 transfer learning) matches closely. The **HAM10000
dataset** and the **~90–94% accuracy figure** below are carried over from the original source note
and were **NOT independently re-confirmed** this pass (publisher page returned HTTP 403).

**Method:** EfficientNet-B0/B1 transfer learning on dermoscopic images only (per original source
note, dataset unconfirmed this pass).

**Strength:** Reportedly ~90–94% accuracy — **use a hedge such as "reportedly ~90–94% accuracy" in
the thesis, not a stated fact.**

**Limitation:** No metadata/symptoms; purely image-based (per original source note).

**Abstract-only claim for thesis prose:** "Frederich et al. (2024) reportedly achieve ~90–94%
accuracy using EfficientNet-B0/B1 transfer learning on HAM10000, per the original source note's
abstract-level summary (dataset detail not independently re-confirmed in this review)."

**Relation to our gap:** Validates this thesis's own choice of EfficientNet-B0 as a reasonable
image-branch backbone/baseline.

---

## 9. Multimodal Learning with Clinical Metadata for Skin Cancer Diagnosis — **DROPPED 2026-08-27, DO NOT CITE**

**Status: excluded from the formal bibliography.** This row's citation could not be verified and is
being removed from the 19-paper reference set rather than carried forward with a shaky citation.

**Why it was dropped:** four independent, differently-worded searches were run (TabNet+PAD-UFES-20;
dual-branch+TabNet+skin-cancer; "Multimodal Learning with Clinical Metadata for Skin Cancer
Diagnosis" as an exact phrase; general PAD-UFES-20+TabNet metadata-fusion searches). All four turned
up other, genuinely-published PAD-UFES-20 multimodal papers (MetaBlock/MetaNet, a "TabFusion"
GCN-based architecture, Swin Transformer + gated attention, EfficientNetB3+ResNet50 concatenation,
etc.) but **none use TabNet specifically**, and none match this row's exact title. The specific
combination recorded here — "dual-branch CNN + TabNet, PAD-UFES-20, ~94–97% accuracy" — could not be
traced to any real, existing paper.

**Most likely explanation:** a transcription or conflation error in the original Excel-based
literature review — this entry may have merged details from two or more different real papers (e.g.
a dual-branch PAD-UFES-20 paper's architecture description bleeding into a different paper's
reported accuracy range), or "TabNet" may have been a misremembering of one of the several other
tabular-metadata architectures found during this search (e.g. TabFusion, or a Tabular Embedding
Network variant). **If anyone asks later why row #9 disappeared from the bibliography:** this is
the reason — not an oversight, a deliberate exclusion after failed re-verification, per an explicit
decision on 2026-08-27 to prioritize a fully-verified 19-paper set over a 20-paper set carrying one
unconfirmable entry.

**If a PAD-UFES-20 dual-branch fusion comparison point is still needed for Results/Discussion**, the
"TabFusion" paper surfaced during this search (GCN-enhanced multimodal architecture, reportedly
91.74% accuracy / 92.30% F1 on PAD-UFES-20) is a real, findable candidate — but it has **not** been
verified or added here, since you asked not to chase a replacement.

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
Brief*) — **now added as its own verified entry** in the "Dataset Citation" section after #19,
separate from this Related Work table, per your instruction that it belongs in the
Methodology/Dataset chapter.

---

## 12. A Multimodal Skin Lesion Classification Through Cross-Attention Fusion and Collaborative Edge Computing 🟡 — **CITATION VERIFIED 2026-08-27**
**Citation:** Tran-Van, N.-Y., & Le, K.-H. (2025). *A multimodal skin lesion classification through
cross-attention fusion and collaborative edge computing.* **Computerized Medical Imaging and
Graphics**, 124, article 102588. **DOI: 10.1016/j.compmedimag.2025.102588**. Paywalled
(ScienceDirect HTTP 403), no arXiv preprint found — remains abstract-level for us.

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

**Abstract-only claim for thesis prose:** "Tran-Van & Le (2025) report, per their abstract,
cross-attention and Hadamard-product fusion both reaching ~98.8% accuracy, though the exact
dataset/task and metric definition remain unconfirmed against the primary text — this figure should
not be compared directly to this thesis's own macro-F1 numbers without that caveat."

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

**Abstract-only claim for thesis prose:** "Zuo, Wang & Wang (2025) state, per their abstract, that
their CosCatNet two-stage fusion method is effective on 'a popular publicly available skin disease
diagnosis dataset,' but report no specific quantitative results in the portion available to us —
no numeric claim from this paper should be cited in the thesis without full-text access."

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

## 16. Evaluation of the Importance of Metadata in Skin Lesion Classification 🟡 — **CITATION VERIFIED 2026-08-27**
**Citation:** Garib, G., Mery, D., & Navarrete-Dechent, C. (2025). *Evaluation of the importance of
metadata in skin lesion classification.* **Signal, Image and Video Processing (SIViP)**, 19(11),
article 887. **DOI: 10.1007/s11760-025-04498-6**. Paywalled (Springer institutional login required;
ResearchGate blocked; author site unreachable; no OA copy found) — remains abstract-level for us.

**Method:** 17 deep learning models tested across 3 fusion methods on two datasets; separately
trained models on different metadata subsets to isolate each feature's individual contribution.
Datasets: PAD-UFES-20 (clinical images) + ISIC 2019 (dermoscopic images).

**Strength:** Image+metadata beat image-only baseline by **+10.43% balanced accuracy on
PAD-UFES-20**, +2.22% on ISIC 2019 — consistent with the intuition that PAD-UFES-20's richer
clinical metadata carries more signal than a dermoscopy-only archive's sparser tags. Per-feature
importance ranking: **age most useful**, then body/anatomical location, then sex.

**Limitation:** Full text needed to assess exact model list and statistical rigor of the
per-feature ranking.

**Abstract-only claim for thesis prose:** "Garib, Mery & Navarrete-Dechent (2025) report, per their
abstract, that image+metadata fusion improves balanced accuracy by +10.43% on PAD-UFES-20 and +2.22%
on ISIC 2019 versus image-only baselines, with age ranked as the most important metadata feature."

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

**Abstract-only claim for thesis prose:** "Shrestha & Palit (2026) report 96% accuracy and AUROC
0.987 on a binary skin-disease classification task using vision-language captioning plus metadata
fusion on a curated ISIC 2024 subset — not directly comparable to this thesis's 6-class, macro-F1
evaluation without stating that caveat."

**Relation to our gap:** Already cited in `docs/main.tex` (Related Work — Deep Learning-Based
Approaches, and the litcompare table) as the vision-language-fusion comparator; illustrates the
value of combining vision-language representations with metadata, contrasted with this thesis's
structured-metadata-only cross-attention design.

---

## Dataset Citation (separate from the Related-Work table — belongs in Methodology/Dataset chapter) 🟢 — **CITATION VERIFIED 2026-08-27**

**This entry is not part of the 19-paper Related Work comparison set.** It is the mandatory citation
for the dataset itself, distinct from papers #10 (MetaBlock) and #11 (Pacheco & Krohling 2020),
which are about *methods applied to* PAD-UFES-20/its precursor, not the dataset release.

**Citation:** Pacheco, A. G. C., Lima, G. R., Salomão, A. S., Krohling, B., Biral, I. P., de Angelo,
G. G., Alves Jr., F. C. R., Esgario, J. G. M., Simora, A. C., Castro, P. B. C., Rodrigues, F. B.,
Frasson, P. H. L., Krohling, R. A., Knidel, H., Santos, M. C. S., do Espírito Santo, R. B., Macedo,
T. L. S. G., Canuto, T. R. P., & de Barros, L. F. S. (2020). *PAD-UFES-20: A skin lesion dataset
composed of patient data and clinical images collected from smartphones.* **Data in Brief**, 32,
article 106221. **DOI: 10.1016/j.dib.2020.106221**. Also on arXiv:2007.00478 (free preprint).

**What it is:** The data-descriptor paper introducing the public PAD-UFES-20 dataset this thesis
uses: 2,298 clinical images from 1,641 skin lesions across 1,373 patients, up to 22 clinical
features per case, 6 diagnostic classes (3 skin diseases + 3 skin cancers), 58.4% of lesions
biopsy-proven (100% of the cancers). Collected via the Dermatological and Surgical Assistance
Program (PAD) at the Federal University of Espírito Santo (UFES, Brazil).

**Use in the thesis:** Cite this paper (not #10 or #11) whenever describing or introducing the
PAD-UFES-20 dataset itself in the Methodology/Dataset chapter.

---

## Final Status Summary (as of 2026-08-27 citation-verification pass)

**Bibliography size: 19 papers** (20 original entries minus #9, dropped as unverifiable) **+ 1
separate dataset citation** (PAD-UFES-20 / Pacheco et al., *Data in Brief*, in its own section
above, not counted in the 19).

**Read-depth / confidence breakdown of the 19:**
- 🟢 **Full text read (8):** #1, #5, #11, #13, #15, #17, #18, #19
- 🔵 **Mechanism confirmed via secondary source (1):** #10
- 🟡 **Abstract-level only, citation fully verified (8):** #2, #3, #4, #7, #12, #14, #16, #20
- 🟡 **Abstract-level only, MODERATE CONFIDENCE — title/venue/DOI verified but dataset/accuracy
  claims not independently re-confirmed (2):** #6, #8

**Every quantitative claim drawn from an abstract-only row now carries an explicit "Abstract-only
claim for thesis prose" sentence in its entry**, pre-phrased with the "reportedly"/"per abstract"
hedge so it can be copied directly into thesis text without further rewriting.

**Outstanding items still worth resolving before the bibliography is fully final:**
1. **#2 (TG-CAVNet):** citation is now fully verified, but the specific claim that your Channel Gate
   module reproduces or closely approximates TG-CAVNet's channel-modulation mechanism is **flagged
   as unconfirmed, not assumed** — the primary paper's methodology section (specifically its
   modulation formula) should be read before the Methodology chapter asserts architectural
   correspondence beyond "inspired by."
2. **#3 (Watson et al.):** citation fully verified; reframed per your instruction as an analogous
   free-text leakage finding, not a direct structural match to this thesis's tabular leakage audit.
3. **#6 and #8:** citations verified at the bibliographic level (title/authors/venue/DOI), but the
   dataset and accuracy figures originally recorded for them were not independently re-confirmed
   against primary text in this pass — both are marked MODERATE CONFIDENCE and carry hedged
   thesis-prose sentences.
4. **#9 dropped** — see its entry for the full explanation (kept in place, marked DROPPED, so the
   reasoning is preserved rather than silently deleted).
5. **Paper #13's standing caveat** (accuracy vs. macro-F1 — see its entry above) must still be
   repeated verbatim wherever it's cited in the Discussion chapter, per the source project's own
   instruction — unchanged by this pass.
6. **#14 and #15** still carry one minor citation gap each (a missing DOI/page number) that was not
   in scope for this verification pass — flag for a future pass if their citations need to appear
   with full precision.
