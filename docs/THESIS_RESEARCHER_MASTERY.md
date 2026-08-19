# COMPLETE RESEARCHER-LEVEL THESIS MASTERY DOCUMENT
> Metadata-Guided Cross-Attention Fusion for Multimodal Skin Lesion Classification
> Source of truth: your thesis, paper, and this entire project journey. Where information wasn't documented, it is explicitly flagged as [NOT DOCUMENTED] / [UNCERTAIN] / [NEEDS VERIFICATION] rather than invented.

---

## PART 1 — THESIS AT A GLANCE

**1. What is this research about?** Building an AI model that classifies skin lesions into 6 diagnostic classes by combining dermoscopic/clinical images with structured patient metadata, using a cross-attention mechanism.

**2. Real-world problem:** Skin cancer detection benefits from early, accurate diagnosis; most automated systems use images only, ignoring clinical context a real dermatologist would use.

**3. Why important:** Early detection changes outcomes; automated screening has practical value especially where specialist access is limited.

**4. Why difficult:** High class imbalance (some classes have <30 test images), visually similar classes (AK/BCC/SCC), limited diverse metadata, dataset-to-dataset variation in imaging/metadata schema.

**5. Role of AI/ML:** Supportive screening tool — NOT a diagnostic replacement. [Explicitly not claimed as clinically deployable.]

**6. What we built:** A two-branch model (EfficientNet-B0 image branch + MLP metadata branch) fused via 8-head cross-attention (metadata=Query, image=Key/Value), evaluated under a leakage-audited, patient-wise-split, cross-dataset-tested protocol.

**7. Central research question:** Can combining image and metadata through cross-attention improve skin lesion classification over single-modality or naive-fusion baselines, under rigorous, leakage-free, generalization-tested evaluation?

**8. Main hypothesis:** Metadata-guided cross-attention avoids the dimensional imbalance of naive concatenation (1,280-d image vs. 64-d metadata) and yields a statistically real improvement.

**9. Main contribution:** Not primarily the architecture (a prior comparative study already found similar cross-attention designs superior for this problem — see Part 4/28) but the **rigorous, transparent evaluation methodology** wrapped around it: leakage audit, patient-wise split, cross-dataset test, external validation with overlap exclusion, fairness check, pre-registered ablation protocol, and honest reporting of non-significant/rejected results.

**10. Final outcome:** 0.6977 test macro-F1 (Cross-Attention), statistically significant vs. all baselines (p ≤ 0.006 — image p<0.001, metadata p=0.006, late fusion p=0.002, Bonferroni-corrected); this is the strongest **evaluated** result, not claimed as SOTA.

### Explain my thesis in...

**30 seconds:** "We built a model that classifies skin lesions using both the image and patient information, fused through cross-attention. We rigorously audited the data for leakage, tested it on new datasets, checked fairness, and reported everything honestly — including experiments that didn't work. Our result, 0.6977 macro-F1, is statistically proven and methodologically trustworthy."

**1 minute:** Add: "Most published work in this space either skips leakage auditing, tests on only one dataset, or uses an easier binary/accuracy setup that inflates scores. We deliberately chose the harder six-class, macro-F1 setup. We also found and rejected an approach that looked good on aggregate but collapsed on the rarest class — showing our priority was clinical validity over a bigger number."


**3 minutes:** Add the full pipeline (data → leakage audit → architecture → primary result → confusion analysis → cross-dataset drop → fairness limits → ensemble honesty → conclusion that methodology matters as much as accuracy). [See Part 2 below for full detail to expand into 3 minutes.]


**To a non-technical person:** "Imagine a doctor looking at a skin spot AND asking about your age and symptoms before deciding. We taught a computer to do both at once, instead of just looking at the photo. We were also very careful to make sure the computer wasn't cheating by using clues it shouldn't have access to, and we tested it honestly on new patients it had never seen."


**To an AI/ML researcher:** "Metadata-guided cross-attention fusion (with a metadata-conditioned channel gate on image tokens preceding the attention step; metadata=Q, image=K/V) on PAD-UFES-20, with a statistically-audited leakage removal (phi/chi-square; commonly cited as '22 columns' though this figure does not cleanly reconcile against the per-dataset feature whitelists — see Part 8 note), patient-wise splitting, cross-dataset transfer to HAM10000, external validation with image-overlap exclusion, Fitzpatrick fairness analysis, and a pre-registered ablation protocol covering backbone comparison, dataset expansion, ensembling, and joint fusion. Headline: 0.6977 macro-F1, p≤0.006 vs. baselines (Bonferroni-corrected)."

---

## PART 2 — COMPLETE RESEARCH JOURNEY (Pipeline, What/Why/How)

| Stage | What | Why | How |
|---|---|---|---|
| Problem ID | Multimodal skin lesion classification | Image-only is the norm; metadata is under-used | Reviewed literature gap |
| Literature Review | TRACE, MetaBlock, Shrestha & Palit, others | Establish what's missing | See Part 4/5 |
| Research Gap | Leakage audit + fairness + cross-dataset rarely combined | Motivates our protocol | Table comparison |
| Dataset Selection | PAD-UFES-20 primary; HAM10000, ISIC 1&2 secondary | Only PAD-UFES-20 has rich metadata | See Part 6 |
| Dataset Audit | Corrupted image check, class distribution, missing values | Standard data-quality practice | Manual + scripted checks |
| Leakage Detection | Phi coefficient + chi-square per metadata column | Prevent shortcut learning | Commonly cited as "22 columns removed" (incl. `main.tex`); this total does not cleanly reconcile against the 4 datasets' `feature_whitelist.md` files — strict phi/chi-square-flagged leakage columns total ~5–12 depending on definition, all-exclusions-summed total is 45. [NEEDS VERIFICATION before final publication — see Part 8.] |
| Patient-Level Split | 70/15/15, seed=42 | Prevent patient overlap | Split by patient ID |
| Preprocessing | Image resize/normalize; metadata one-hot encode | Model input requirements | See Part 10 |
| Baseline Models | Image-only, Metadata-only, Late Fusion | Establish comparison points | Same architecture skeleton |
| Architecture Design | Cross-attention fusion | Avoid late-fusion dimensional imbalance | See Part 12-13 |
| Training | Adam, class-weighted CE, 3 seeds, early stop (patience 7 within a 30-epoch cap) | Standard robust training | LR: 1e-4 (image-only), 1e-3 (metadata-only), 1e-5 (late fusion & cross-attention, warm-started); batch size 32; see Part 15 |
| Validation | Macro-F1 on val split | Model selection criterion | Per-seed tracking |
| Test Evaluation | Single-use guarded test read | Prevent repeated-access leakage | Code-enforced guard |
| Cross-Dataset Eval | PAD→HAM10000 | Test generalization | 3 shared classes only |
| External Validation | HAM→ISIC 1 & 2 | Further generalization check | Overlap-excluded (1,362 IDs) |
| Ablations | Backbone, dataset expansion, ensemble, joint fusion | Test if legitimate changes improve result | Pre-registered rules |
| Ensemble | Dual-backbone | Test if combining helps | One-time test read, rule-gated |
| Error Analysis | Confusion matrix, per-class F1 | Understand failure modes | Table IV/V of paper |
| Statistical Analysis | Bootstrap (1,000 resamples), Bonferroni | Confirm findings aren't chance | scipy-based |
| Interpretation | Discussion section | Explain results honestly | See Part 17-21 |
| Limitations | Fairness sample size, ensemble unconfirmed, geographic scope | Scientific honesty | See Part 26 |
| Final Contribution | Methodology + honestly-reported result | Real thesis contribution | See Part 28 |
| Paper Preparation | IEEE format, 6 pages | Dissemination | See Part 30 |
| Compute/Timeline | No local GPU available; training run on Kaggle (environment-detection/path-resolution logic in `src/models/config.py` handles this) | Examiner may ask about hardware/compute budget | Phase 4 (Dataset Prep) closed 2026-07-08 · Phase 5 (EDA) done 2026-07-09 · Phase 7 Stage 1 done 2026-07-18 · Headline result locked 2026-08-03 |

**What would happen if we skipped key steps?**
- Skip leakage audit → inflated, untrustworthy scores (e.g., "biopsed" alone would nearly solve malignancy detection without real learning)
- Skip patient-wise split → model could partly memorize patients, inflating test score
- Skip cross-dataset test → no evidence of generalization beyond training distribution
- Skip pre-registration on ablations → risk of p-hacking / cherry-picking the best-looking result

---

## PART 3 — INTRODUCTION LOGIC

The introduction's argument flows: **(1)** skin cancer is common and early detection matters → **(2)** most AI is image-only, but doctors use more → **(3)** evaluation rigor is often weak (binary framing, accuracy metric, no leakage check, single dataset) → **(4)** class imbalance is usually handled with synthetic data, not real data → **(5)** fairness is rarely measured → **(6)** therefore, this work proposes a fusion architecture AND a rigorous protocol addressing all of the above. Each paragraph builds toward: **the contribution is as much about HOW we evaluate as WHAT we build.**

---

## PART 4 — RESEARCH GAP (Honest Analysis)

| Prior Work | Modality | Held-out Test | Leakage Audit | Cross-Dataset | Fairness |
|---|---|---|---|---|---|
| TRACE | Metadata only | No (val-only, 90/10) | Not reported | No | No |
| MetaBlock | Image+Meta | [UNCERTAIN — primary source paywalled, N/C] | N/C | N/C | N/C |
| Shrestha & Palit | Image+Meta (VLM) | Yes | Not reported | No | No |
| **Ours** | Image+Meta | **Yes** | **Yes (phi/chi-square; commonly cited as ~22 cols., figure NEEDS VERIFICATION — see Part 8)** | **Yes** | **Yes** |

**Strongly supported gap:** Comparable work rarely combines leakage auditing + cross-dataset testing + fairness checking in one pipeline.
**Reasonable interpretation:** Our multi-dimensional rigor is a genuine methodological contribution.
**Weak/uncertain claim:** That our architecture itself is novel — a prior comparative study already found similar cross-attention (image=K/V, metadata=Q) designs superior on HAM10000. **[Do not claim architectural novelty without qualification.]**
**Claims that should NOT be made:** "State of the art," "proven clinically safe," "proven fair" — none of these are supported by current evidence.

---

## PART 5 — RELATED WORK COMPARISON (Detail)

**TRACE:** Transformer on structured metadata only, no image input. PAD-UFES-20, 90/10 train-val split, NO held-out test set. Reported: DANET 0.625, TRACE 0.783 macro-F1 — both **validation-style estimates**, not comparable to a true test score. Strength: shows metadata alone carries real signal. Weakness: no test-set discipline, no image modality. **Documentation-hygiene note:** TRACE is cited in `main.tex`'s bibliography/comparison table and discussed extensively in `Project_Tracking.md`/`THESIS_OWNERSHIP_MASTER.md`, but is absent from `docs/Literature_Review.md`'s formal 20-paper review table — worth adding there for consistency, though this is not a factual error in this document.

**MetaBlock:** Channel-wise metadata-conditioned gating on image features (uniform across spatial locations) — architecturally different from spatial cross-attention (per-location). From PAD-UFES-20's creators. Most evaluation details **[N/C — primary paper paywalled, only secondary-source accuracy figures found: ~80.7% ISIC2019, ~74.8% PAD-UFES-20, unverified against primary text]**.

**Shrestha & Palit (2026):** MedCLIP-derived embeddings + metadata, early/attention fusion, ISIC 2024 subset. **Binary** classification, 96% accuracy, AUROC 0.987. Not directly comparable — binary task + accuracy metric vs. our 6-class macro-F1.

**Is our score actually better?** Not a clean "yes/no" — different tasks, metrics, and splits make direct comparison invalid. Under comparably rigorous multi-class, macro-F1, true-test-set evaluation, our result compares favorably.

**Is our work SOTA?** **No, and this should not be claimed.** SOTA requires benchmarking against the current best on an identical, standardized task/split — we have not done this.

**What makes it valuable even without SOTA:** methodological rigor, transparency, and reproducibility — arguably more valuable for clinical AI than a marginally higher number on an easier task.

---

## PART 6 — DATASETS (Deep Dive)

| Dataset | Images | Classes | Metadata | Role | Key Property |
|---|---|---|---|---|---|
| PAD-UFES-20 | 2,298 | 6 | Yes (21 usable features post-audit) | Primary train/val/test | Only dataset enabling genuine multimodal fusion |
| HAM10000 | 10,015 | 7 | Sparse | Cross-dataset generalization target | Never trained on |
| ISIC Archive 1 | 2,357 | 9 (native) | N/A for our use | External validation | Never trained on |
| ISIC Archive 2 | 25,076 | 9 (native) | N/A for our use | External validation | 98.6% image overlap with HAM10000 — excluded via 1,362-ID list |
| DERM12345 | 666 (added) | Melanoma expansion | — | Dataset expansion only | Real, biopsy-confirmed |
| MED-NODE | 70 (added) | Melanoma expansion | — | Dataset expansion only | Real, biopsy-confirmed |

**Expansion dataset provenance (from `THESIS_OWNERSHIP_MASTER.md` §2.5–2.6):** DERM12345 — Harvard Dataverse (DOI-registered), CC BY 4.0 license, Turkish-origin, dermatologist-labeled. MED-NODE — full primary citation is paywalled; benign cases dermatologist-labeled. Both verified to have zero patient-ID overlap with PAD-UFES-20 before inclusion.

**Class imbalance (PAD-UFES-20 test set support):** AK 109/327, BCC 133/399, MEL 8/24, NEV 35/105, SEK 37/111, SCC 32/96 (denominator ×3 seeds). Melanoma and SCC are the weakest-represented — directly explains volatility in their per-seed scores.

**Biases/limitations:** Geographic concentration (limited regions), Fitzpatrick skin-tone underrepresentation (esp. Types V/VI, 1-2 test images), potential imaging-equipment differences across datasets.

---

## PART 7 — DATASET PREPARATION AND CLEANING

**Pipeline:** Raw sourcing → audit (corrupted/missing images, class distribution, missing-value check) → column standardization → value validation → label standardization → leakage audit → patient-wise split → split-quality report.

**Why each step:**
- **Audit** prevents training on corrupted/mislabeled data silently degrading results.
- **Standardization** ensures consistent taxonomy across sources.
- **Leakage audit** (see Part 8) is the single most consequential cleaning decision.
- **Dataset expansion (real, biopsy-confirmed)** addresses class imbalance without the "no new signal" problem of synthetic augmentation.

**What makes this more responsible than "download and train":** an explicit, statistically-justified leakage audit; patient-wise (not image-wise) splitting; a code-enforced test-access guard (not literally single-use — see Part 9); verified zero patient overlap for expansion images.

---

## PART 8 — DATA LEAKAGE AUDIT (Deep)

**What is data leakage?** Information that would not genuinely be available at prediction time accidentally helps the model, inflating apparent performance without real learning.

**Why dangerous in medical AI:** A leaky model can appear highly accurate in development but fail or mislead in real deployment, since the "shortcut" signal won't exist for a new, undiagnosed patient.

**Why metadata is especially risky:** Clinical metadata fields are often recorded *because of* or *as part of* the diagnostic workup (e.g., biopsy status), so some fields inherently encode the answer.

**The "biopsed" issue:** True in 100% of malignant cases (phi=0.80) — because clinicians biopsy lesions they already suspect are malignant. Using this field would let the model "know" the answer via a proxy for clinical suspicion, not learn diagnostic patterns from image/pre-diagnosis metadata.

**Phi coefficient:** Measures strength of association between a binary (or categorical) metadata field and the label, 0 (no association) to 1 (perfect association).
**Chi-square test:** Tests whether an observed association is statistically significant (unlikely due to chance).

**Why "22 columns" excluded — [NEEDS VERIFICATION]:** The figure "22" is the one repeated across `main.tex`, `Project_Tracking.md`, and earlier drafts, but on direct reconciliation against the 4 datasets' `feature_whitelist.md` files it does not cleanly resolve to any single population: the columns with a *statistically-tested, near-deterministic* phi/chi-square leakage association (e.g. `biopsed` phi=0.80; HAM10000's and ISIC Archive 2's `diagnosis_confirm_type`; ISIC Archive 2's `concomitant_biopsy`, `melanocytic`) total roughly 5–12 depending on how strictly "leakage" is defined versus administrative/identifier/path exclusions; the all-exclusions-summed total across all 4 datasets is 45 (PAD-UFES-20: 8, HAM10000: 7, ISIC Archive 1: 6, ISIC Archive 2: 24). **Before final publication, re-derive the exact population "22" is meant to describe (likely a specific subset/date snapshot in `Project_Tracking.md`) and correct both this document and `main.tex` line 46, which repeats the same figure.**

**Why not just "feature selection"?** Feature selection typically removes low-value or redundant features; leakage removal removes features that would make the task trivially/artificially easy by encoding post-diagnosis information — a categorically different concern (validity, not just performance).

**How leakage creates artificially high performance:** The model learns to read the shortcut field rather than the actual visual/clinical diagnostic pattern, producing high in-sample metrics that won't transfer to real-world use.

**Likely examiner questions:** "How do you know [the exact count] is the right number, not too many/few?" → answer: the number is not chosen a priori; it's the count of columns that failed the statistical test at the audit's threshold — it's a *result*, not a design choice (though see the reconciliation note above — the exact headline figure needs re-derivation before publication). "Could legitimate features have been accidentally removed?" → **CORRECTED:** there was no independent *clinical/dermatologist* review, but there WAS a documented manual researcher/supervisor review-and-approve step for each flagged column — e.g. `docs/Project_Tracking.md` records the `feature_whitelist.md` as "reviewed and approved by you," and ISIC Archive 2's `melanocytic` column as "flagged to you directly before exclusion; you confirmed exclude." The honest framing is: **statistically audited and researcher-approved, but not independently validated by a clinical/dermatology expert** — that remains the real limitation, not "no review at all."

---

## PART 9 — DATA SPLITTING

**Patient-level split:** All of a patient's images stay in exactly one of train/val/test.
**Why not image-level:** If the same patient appears in both train and test, the model can partially "recognize" that patient's skin/lesion texture rather than learning generalizable diagnostic patterns — inflating test performance without real generalization.
**70/15/15, seed=42:** Standard train/val/test proportions; fixed seed for reproducibility.
**Held-out test set:** Evaluated under a **code-enforced access guard** (raises an error on unsanctioned repeated access) — this operationalizes "independent held-out test" as a hard constraint, not just a stated intention. **CORRECTED:** this is not literally "single-use" — the PAD-UFES-20 test split was legitimately consumed twice under pre-registered, sanctioned rules: the Stage 1 final result (2026-07-25) and Step 4 Option B's (ensemble) second read (2026-08-01). The guard's real function is to prevent *unsanctioned* re-access, not to cap access at exactly one read — see Part 21 for how the second read was authorized.
**Why repeated access is dangerous:** Every additional *unsanctioned* look at the test set (e.g. re-tuning based on test performance) risks the test set functioning like a second validation set — undermining its role as an unbiased final estimate. This is why every additional read in this project required a pre-registered rule set *before* the read, not after.

**Random image split vs. patient-level split:** Random image split is easier to implement but clinically meaningless — a real deployed model will never see the *same patient* twice from training; patient-level split simulates the true deployment scenario (new, unseen patients).

---

## PART 10 — DATA PREPROCESSING

**Image — RESOLVED:** Resized to **224×224** (`IMAGE_INPUT_SIZE = 224`, matching EfficientNet-B0's native input), using a custom aspect-ratio-preserving `ResizePad` transform: the longer side is scaled to 224, then the shorter side is zero-padded to complete the square — deliberately avoiding the distortion a naive stretch-resize would introduce, given the documented heterogeneity in source image aspect ratios. Normalized with standard ImageNet statistics — `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` — verified 2026-07-29 by directly querying each of the 5 tested backbones' own pretrained-weight `.transforms()` mean/std (confirmed identical across all 5, not merely assumed). Source: `src/models/dataset.py:26-51`.

**Metadata:** 21 usable clinical features (post-audit) → one-hot encoded categorical + numerical fields → 89-dimensional input vector → fed to MLP.

**How metadata becomes model input:** Raw clinical fields (age, lesion duration, itching, etc., post-leakage-audit) → one-hot/numeric encoding → 89-d vector → MLP → 64-d embedding → Query projection → 256-d.

---

## PART 11 — MODELS AND PRETRAINED BACKBONES

**EfficientNet-B0 (primary/headline backbone):** ImageNet-pretrained CNN, chosen for strong accuracy-to-compute efficiency; fully fine-tuned (not frozen) end-to-end with the metadata branch.

**Backbones tested in the 5-backbone comparison ablation (Section IV-E context):** MobileNetV3, DenseNet121, ResNet50, ConvNeXt-Tiny, alongside EfficientNet-B0. All exceeded the non-expanded baseline (range 0.5859–0.6224 val macro-F1) when paired with the expanded dataset. **[Individual per-backbone architectural trade-off discussion — e.g., MobileNetV3's mobile-efficiency focus vs. ConvNeXt's modernized-CNN design — is general ML knowledge, not something the thesis itself elaborates on in depth; be ready to discuss this from general knowledge if asked, clearly distinguishing it as such.]**

**Transfer learning:** All backbones initialized from ImageNet weights, then fully fine-tuned (not linear-probe/frozen-backbone) on the target task.

---

## PART 12 — PROPOSED ARCHITECTURE (Stage by Stage)

```
Input Image (RGB)
  -> EfficientNet-B0 (fine-tuned)
  -> Pre-pool feature map [B, 1280, 7, 7]
  -> Reshaped to 49 spatial tokens (1280-d each)
  -> Metadata-Conditioned Channel Gate (sigmoid, TG-CAVNet-inspired; see note below)
  -> Key/Value Projection: Linear(1280 -> 256)

Input Metadata (89-d, one-hot encoded)
  -> MLP
  -> 64-d embedding
  -> Query Projection: Linear(64 -> 256)

Cross-Attention (8 heads, d_model=256)
  Query = projected metadata
  Key/Value = projected, channel-gated image tokens
  -> 256-d attended vector

Concatenation: [attended (256-d) ; raw metadata embedding (64-d)] -> 320-d joint vector

Classification Head: FC(320->128) -> BatchNorm -> ReLU -> Dropout -> FC(128->6)

-> Logits -> Softmax -> Predicted class (1 of 6)
```

**CORRECTED — architecture omission fixed:** The actual headline `CrossAttentionFusionModel` includes a component earlier versions of this document omitted entirely: a `MetadataChannelGate` (`src/models/cross_attention_fusion_model.py:63-84`), **enabled by default** (`use_channel_gate=True`, never disabled in the headline runs). Before cross-attention, it applies a metadata-conditioned sigmoid gate across the image tokens' channels — a TG-CAVNet-inspired mechanism (documented in `Project_Tracking.md:1159-1193`) that lets metadata modulate *which image channels* are emphasized, in addition to the cross-attention step that determines *which spatial locations* are attended to. This is a real, always-active part of the deployed model, not a minor implementation detail — it should be described alongside cross-attention whenever the architecture is explained.

**Why each choice:**
- **EfficientNet-B0:** efficient, strong ImageNet-pretrained features, standard practice-appropriate for a bachelor's-thesis-scale compute budget.
- **49 spatial tokens:** natural consequence of EfficientNet-B0's 7×7 pre-pool feature map (7×7=49) — not an independently tuned hyperparameter, but a structural property of the backbone.
- **1,280-d image features:** EfficientNet-B0's native channel depth at that layer.
- **89-d metadata input:** post-leakage-audit feature count after one-hot encoding.
- **64-d metadata embedding, 256-d shared space, 8 heads:** architectural design choices consistent with standard transformer-style fusion practice. **CONFIRMED — no ablation exists:** `d_model=256` and `num_heads=8` are hardcoded constructor defaults in `src/models/cross_attention_fusion_model.py:98-106` with no call site anywhere in the repo passing alternate values (e.g. 4 heads, 128-d); no sweep script or tracking-doc entry documents such a test. These are reasoned defaults, not independently ablated hyperparameters — acknowledge this honestly if asked "did you test other head counts?"
- **Metadata = Query, Image = Key/Value:** lets the (smaller, clinically-grounded) metadata representation actively select which image regions to attend to, rather than the reverse — matches the clinical intuition of "given what I know about this patient, where should I look in the image?" This directionality is also the design independently corroborated as superior in prior comparative fusion studies (see Part 5).
- **320-d joint vector, FC->BN->ReLU->Dropout->FC:** standard, well-established classification head design for regularized fine-tuning.
- **6 output classes:** matches PAD-UFES-20's native diagnostic taxonomy.

---

## PART 13 — CROSS-ATTENTION DEEPLY

**Intuition:** The metadata "asks a question" (Query) about the image; the image's spatial regions (Key) are scored for relevance to that question, and the most relevant regions' content (Value) is aggregated into the answer.

**Math:** `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`
- **Q (Query):** projected metadata embedding, what we're "looking for."
- **K (Key):** projected image tokens, what's being matched against.
- **V (Value):** projected image tokens (same source as K here), the actual content aggregated.
- **Softmax** over QK^T turns similarity scores into attention weights summing to 1.
- **Weighted sum** of V using those weights produces the attended output.
- **Multi-head (8 heads):** runs this process in 8 parallel, independently-learned subspaces, then concatenates — allows attending to different types of relevant patterns simultaneously.

**Why Metadata=Query, Image=Key/Value (not the reverse):** The clinically meaningful direction is "given the patient's context, which visual evidence matters" — not "given the image, which patient facts matter" (metadata is much lower-dimensional and less naturally decomposable into a K/V sequence).

**Fusion strategy comparison:**
| Strategy | Mechanism | Weakness for this problem |
|---|---|---|
| Early fusion | Combine raw inputs before any processing | Modality-specific structure lost immediately |
| Late fusion (ours, baseline) | Concatenate final feature vectors | 1,280-d image dominates 64-d metadata numerically |
| Self-attention | Attend within one modality only | Doesn't model cross-modal interaction directly |
| Metadata-conditioned gating (MetaBlock) | Channel-wise scaling of image features by metadata | Uniform across space — no per-location attention |
| **Cross-attention (ours)** | Metadata=Q attends over image K/V (image tokens first pass through our own metadata-conditioned channel gate, then cross-attention adds per-location selectivity — see Part 12) | Fair, shared-space interaction; spatially selective, with channel-level gating as a complementary mechanism rather than a substitute |

---

## PART 14 — HOW THE MODEL MAKES A PREDICTION

Image representation (49×256-d) + Metadata representation (64-d) → Cross-modal interaction (attention) → Joint representation (320-d) → Classification logits (6-d) → Softmax probabilities → Argmax = predicted class.

**What "probability" means here:** The model's learned confidence distribution over 6 classes, calibrated only insofar as training encouraged it — **NOT** a directly validated clinical probability of disease.

**Why a prediction is NOT a clinical diagnosis:** No prospective clinical validation, no calibration study, no sensitivity/specificity analysis against a clinical gold-standard workflow, no regulatory review. This model is a research-stage decision-support signal at most, evaluated retrospectively on public benchmark data.

**Concepts needed before clinical use:** calibration (does 80% predicted confidence mean 80% real-world accuracy?), external multi-site validation, sensitivity/specificity trade-off tuning for clinical risk tolerance, human-in-the-loop oversight, regulatory approval pathway, ongoing monitoring for distribution drift.

---

## PART 15 — TRAINING CONFIGURATION

| Setting | Value | Source |
|---|---|---|
| Optimizer | Adam | Documented |
| Loss | Class-weighted cross-entropy | Documented |
| Seeds | 3 random seeds per configuration | Documented |
| Early stopping | Patience = 7 epochs, on validation macro-F1, within a 30-epoch cap | Documented |
| Learning rate | **RESOLVED:** Image-only 1e-4 · Metadata-only 1e-3 · Late Fusion 1e-5 · Cross-Attention (headline) 1e-5 (warm-started); same recipe reused for backbone-comparison, ensemble, and joint-fusion variants | `src/models/config.py:419-435` |
| Batch size | **RESOLVED:** 32, constant across all variants | `src/models/config.py:419` |
| Total epoch budget | **RESOLVED:** `NUM_EPOCHS = 30` max; early-stop patience 7 operates within this cap | `src/models/config.py:420-421` |
| Fine-tuning scope | Full end-to-end (both branches), warm-started from independently pretrained single-modality checkpoints | Documented |

**Why class-weighted loss:** Counteracts class imbalance so rare classes (e.g., Melanoma) aren't ignored by the loss function in favor of majority classes.
**Why early stopping:** Prevents overfitting to the training set beyond the point of genuine validation improvement.
**Why 3 seeds:** Captures run-to-run variance, especially important given small per-class test counts (e.g., only 8 Melanoma images).
**Why warm-start:** Leverages already-learned single-modality representations as a better initialization than training the fused model from scratch.

---

## PART 16 — COMPLETE EXPERIMENT INVENTORY

**Total distinct experiments/variants: 13+** — Image-only, Metadata-only, Late Fusion, Cross-Attention (headline), 5-backbone comparison (5 sub-runs, incl. individually-bootstrapped ConvNeXt-Tiny test 0.7105 and DenseNet121 test 0.6852, both n.s. vs. headline: p=0.470 and p=0.656 respectively), Dataset-expansion-only ablation, Dual-backbone ensemble, Joint 3-way fusion, Ensemble+TTA (rejected), Abandoned "improved" cross-attention variant (FocalLoss + WeightedRandomSampler + stronger augmentation + CosineAnnealingLR — mean macro-F1 0.509, well below the 0.6209 baseline; no surviving checkpoint/summary JSON, numbers user-reported only — see `THESIS_OWNERSHIP_MASTER.md` §3.4), PAD→HAM10000 cross-dataset (using reduced-feature checkpoints matched to HAM10000's sparser metadata schema — `train_cross_attention_fusion_reduced.py`/`train_metadata_reduced.py`, not the full-feature checkpoints), HAM→ISIC1/ISIC2 external validation, Fitzpatrick fairness analysis.

| Experiment | Purpose | Dataset | Result | Status |
|---|---|---|---|---|
| Image-only | Baseline | PAD-UFES-20 | Test 0.6175 | Primary baseline |
| Metadata-only | Baseline | PAD-UFES-20 | Test 0.6077 | Primary baseline |
| Late Fusion | Baseline | PAD-UFES-20 | Test 0.6566 | Primary baseline |
| **Cross-Attention** | **Headline model** | PAD-UFES-20 | **Test 0.6977, p≤0.006 vs. all** | **Official primary result** |
| 5-backbone comparison | Test if other backbones help | PAD-UFES-20 (expanded) | Val 0.5859–0.6224, all > non-expanded baseline | Ablation |
| Dataset-expansion only | Isolate data-volume effect | PAD-UFES-20 (expanded) | Val 0.6186 ≈ baseline 0.6209 | Ablation — negligible effect alone |
| Dual-backbone ensemble | Test if combining 2 backbones helps | PAD-UFES-20 (expanded) | Val 0.6856; Test 0.7321, p=0.062 | Ablation — not statistically confirmed |
| Joint 3-way fusion | Test single-model joint fusion | PAD-UFES-20 (expanded) | Val 0.6721 (missed 0.6710 bar by +0.0011) | Ablation — no test read (pre-registered rule) |
| Ensemble+TTA | Test augmentation-boosted ensemble | PAD-UFES-20 | **Validation-split** aggregate 0.6213→0.5886; Melanoma F1 0.3636→0.20 (never reached test) | **Rejected** on clinical-relevance grounds |
| Cross-dataset (PAD→HAM) | Test generalization | HAM10000 (3 shared classes) | Cross-Attn 0.4654 (not sig. vs. image-only) | Generalization test |
| External validation | Test on fully external data | ISIC Archive 1 & 2 (overlap-excluded) | Image-only 0.4912 (Archive 2), 0.2421 (Archive 1) | External validation |
| Fairness | Test skin-tone equity | PAD-UFES-20 test | Best/tied-best per group, but tiny darkest-tone samples | Fairness check, limited conclusiveness |
| Abandoned "improved" cross-attention | Test FocalLoss + WeightedRandomSampler + stronger aug + CosineAnnealingLR | PAD-UFES-20 | Mean macro-F1 ≈0.509 (well below 0.6209 baseline); no surviving checkpoint/summary JSON | **Rejected/abandoned** — user-reported numbers only, `train_cross_attention_improved.py` |

---

## PART 17 — PRIMARY RESULTS (Deep)

**Headline metric: Macro-F1.** Averages F1 across all 6 classes equally, regardless of class size — prevents majority classes (BCC, AK) from masking poor performance on rare classes (Melanoma, SCC).

**Why not only accuracy:** Same test predictions give 0.763 accuracy — legitimate but **not interchangeable** with macro-F1; accuracy is dominated by majority classes and would look better even if rare-class performance were poor.

**0.6977:** the official, headline, statistically-validated test macro-F1 for Cross-Attention Fusion — the strongest of the 4 primary evaluated variants, significant vs. every baseline (**CORRECTED: p≤0.006**, not 0.002 — image p<0.001, metadata p=0.006, late fusion p=0.002, Bonferroni-corrected across image/metadata/late-fusion comparisons; source: `docs/main.tex:185`).

**Confusion matrix / per-class findings:**
| Class | Support (×3 seeds) | F1 |
|---|---|---|
| AK | 327 | 0.8323 |
| BCC | 399 | 0.8079 |
| MEL | 24 | 0.7273 |
| NEV | 105 | 0.8056 |
| SEK | 111 | 0.7574 |
| SCC | 96 | 0.2543 |

**Dominant confusion cluster:** AK/BCC/SCC — 172/252 (68.3%) of all misclassifications, with SCC→BCC (51 cases) the single largest confusion. This matches literature (Pacheco & Krohling, 2020) noting these classes share near-identical clinical profiles — a genuine clinical difficulty, not necessarily a pure architectural weakness, though the model's inability to fully resolve it remains a real limitation.

**Without SCC, macro-F1 would be ≈0.786** (computed from Table V's own per-class F1 values) — illustrating that one genuinely hard, low-support class materially lowers the average; this was not addressed by excluding SCC, since doing so would misrepresent real-world class difficulty.

---

## PART 18 — CROSS-DATASET GENERALIZATION

**PAD-UFES-20 → HAM10000 (3 shared classes: BCC, Melanoma, Nevus):**
| Model | Macro-F1 |
|---|---|
| Image-only | 0.4658 |
| Metadata-only | 0.2920 |
| Late Fusion | 0.4597 |
| Cross-Attention | 0.4654 |

**Cross-attention is NOT statistically distinguishable from image-only (p=0.970) or late fusion (p=0.590), but significantly beats metadata-only (p<0.001).**

**Why performance drops (0.6977→0.4654):** distribution shift — different imaging equipment/conditions, different patient populations, and critically, HAM10000's much sparser metadata schema than PAD-UFES-20's.

**Why this should NOT be treated as "another version of the primary score":** the primary test score measures in-distribution performance under the training data's own conditions; the cross-dataset score measures out-of-distribution generalization — two genuinely different questions. Conflating them would misrepresent what each number means.

**What this tells us about generalization:** the architecture's advantage in-distribution does not automatically transfer to a new domain — a real, honestly-reported limitation, not a hidden flaw.

---

## PART 19 — EXTERNAL VALIDATION

**HAM10000-trained models → ISIC Archive 1 & 2** (never used in training). Because HAM10000 shares substantial image provenance with both archives (98.6% overlap with Archive 2, 66.5% with Archive 1), overlapping images were **identified and excluded** before evaluation — ensuring genuinely unseen data. **CORRECTED:** this used **two separate exclusion lists**, not one shared list: Archive 1's list has 1,362 IDs (`data/processed/ISIC_Archive_1/external_validation_exclusions.csv`); Archive 2's list has **9,873 IDs** (`data/processed/ISIC_Archive_2/external_validation_exclusions.csv`), consistent with its much higher 98.6% overlap figure.

Image-only macro-F1: 0.4912 (Archive 2), 0.2421 (Archive 1, driven by near-zero support for 2 of 5 shared classes — a support/coverage issue, not necessarily worse generalization). On Archive 2, image-only significantly outperformed metadata-only (diff +0.2502, p<0.001) — the single largest, most statistically decisive effect in the whole study.

**Why this strengthens the study:** demonstrates awareness of a subtle, easy-to-miss leakage risk (dataset provenance overlap) that many multi-dataset studies overlook entirely.

---

## PART 20 — ABLATION STUDIES (Question/Change/Constant/Result/Conclusion)

**Dataset expansion (isolated):** Q: "Does more real data alone help?" Changed: dataset size for Melanoma/SCC. Constant: architecture (EfficientNet-B0 cross-attention). Result: 0.6186 val ≈ baseline 0.6209. Conclusion: architecture change, not data volume, was the dominant driver of the larger Step-4 gains — a genuinely useful negative-ish finding.

**Backbone substitution:** Q: "Does a different image backbone help?" Changed: backbone (5 variants tested). Constant: dataset (expanded), fusion mechanism. Result: all 5 exceeded non-expanded baseline (0.5859–0.6224). Conclusion: architecture/backbone capacity matters more than dataset volume alone at this scale.

**Ensemble:** Q: "Does combining two backbones' predictions help?" Result: val 0.6856 (cleared pre-registered bar), test 0.7321, p=0.062. Conclusion: promising but not statistically confirmed — reported honestly, not adopted as headline.

**Joint 3-way fusion:** Q: "Does training one model on both backbones jointly (vs. late-ensembling two separately-trained models) help?" Result: val 0.6721, missed the 0.6710 bar by only +0.0011 — below the "clear and meaningful" pre-registered threshold. Conclusion: no test evaluation was authorized; independently-trained models' ensembling appears to capture more complementary diversity than joint training through one shared attention bottleneck.

**Why p=0.062 matters:** narrowly misses the conventional 0.05 significance threshold — the correct, careful phrasing is "not statistically confirmed as significant," not "basically significant" or "almost proven."

**Why a non-improving experiment is still useful:** it rules out a plausible hypothesis (more data alone would fix low scores) and demonstrates methodological discipline (not chasing every promising-looking number).

**ADDED — the val→test score-gap pattern (missing from earlier drafts):** Across every primary and ablation variant evaluated in this project, the test-split macro-F1 came out *higher* than the validation-split macro-F1 (headline: 0.6209 val → 0.6977 test; ensemble: 0.6856 val → 0.7321 test). This consistent gap was explicitly used as part of the reasoning for **not** authorizing a third test-split read for joint 3-way fusion even though its 0.6721 val score narrowly missed the 0.6710 bar by only +0.0011 — a val score this close to the bar, combined with the established pattern of test typically running ~0.05–0.08 higher than val, meant a third read could not be justified as "clearly and meaningfully" exceeding the threshold in the way the pre-registered rule required. Source: `THESIS_OWNERSHIP_MASTER.md` §6. This is a sophisticated, citable methodological-discipline point worth having ready in defense — it shows the decision not to reopen the test set was itself evidence-based, not merely rule-following.

---

## PART 21 — ENSEMBLE RESULT (Careful Framing)

- Tested because prior single-backbone results suggested architecture capacity mattered — a natural next question.
- **CORRECTED — gating sequence:** The ensemble (Step 4 "Option B") received the test split's **second, pre-approved read** on 2026-08-01, as the primary sanctioned Step 4 experiment — not because it cleared a validation bar. The **0.6710 "clearly and meaningfully exceed" bar** was set the following day (2026-08-02) specifically to gate whether joint 3-way fusion ("Option A") could earn a **third** read; it never applied to the ensemble's own (earlier, already-authorized) read. Source: `docs/Project_Tracking.md:3728-3734`, `THESIS_OWNERSHIP_MASTER.md` §2.1/§3.8.
- Ensemble validation mean: 0.6856 (reported alongside the test result, but this was not the gating condition for the ensemble's own test read — see above).
- Test result: 0.7321. p=0.062 against the headline 0.6977 — **narrowly, not clearly, misses significance.**
- **What CAN be claimed:** a promising, numerically higher, but not statistically confirmed finding, reported transparently.
- **What CANNOT be claimed:** that 0.7321 is "our best result" or the true model performance — 0.6977 remains the official, defensible headline because it IS statistically proven.
- **Why we must not selectively present only 0.7321:** doing so would be a form of results cherry-picking, directly contradicting the thesis's stated rigor.

---

## PART 22 — QUALITY OVER SCORE (Research Philosophy)

**Why not chase an artificially high score:**
- Leakage risk: an inflated number from a leaky feature is not real diagnostic performance.
- Test-set overfitting/repeated access: re-tuning based on test performance turns the test set into a second validation set, invalidating its purpose.
- Cherry-picking: selectively reporting only favorable ablation results misrepresents the actual evidence.
- Incomparable datasets/metrics: comparing our macro-F1 to another paper's binary accuracy is scientifically meaningless despite superficially resembling "comparison."

**How our practices strengthen credibility:** leakage audit, patient-level split, code-enforced guarded test set (unsanctioned re-access blocked, not literally single-use), multiple seeds (variance visibility), class-weighted loss (fair per-class learning), cross-dataset + external validation (generalization evidence), pre-registered ablation rules (anti-p-hacking), confusion/per-class analysis (honest error accounting), bootstrap significance testing (distinguishing real effects from noise), and transparent reporting of rejected/non-significant findings.

**Important caveat:** these practices strengthen **methodological credibility**, not **clinical safety** — the two are related but not identical; do not conflate rigor-in-evaluation with proof-of-clinical-readiness.

---

## PART 23 — MEDICAL AI SAFETY AND CLINICAL VALIDITY

| What our thesis actually did | What real clinical deployment would still require |
|---|---|
| Patient-level split | Prospective (not just retrospective) validation |
| Leakage audit | Regulatory review (e.g., medical device pathway) |
| Cross-dataset + external validation | Multi-site clinical trial validation |
| Fitzpatrick fairness check (limited) | Statistically powered fairness study across full skin-tone spectrum |
| Statistical significance testing | Calibration study (does confidence match real-world accuracy?) |
| Confusion/per-class analysis | Sensitivity/specificity tuned to clinical risk tolerance |
| — | Human-oversight/decision-support workflow design |
| — | Data privacy/regulatory compliance framework |
| Reproducible, documented pipeline | Independent third-party audit/reproduction |

**This distinction must be kept explicit in any defense or paper discussion — do not let "rigorous evaluation" be conflated with "clinically ready."**

---

## PART 24 — FAIRNESS

**What we did:** Evaluated cross-attention (and other primary variants) across Fitzpatrick skin-tone categories on PAD-UFES-20's test partition; found cross-attention best-or-tied-best in every group with *sufficient* sample size.

**What remains incomplete:** Types V and VI (darkest tones) had only 2 and 1 test images respectively; 38% of test records lacked a recorded Fitzpatrick value at all. **This makes the fairness finding suggestive, not statistically conclusive, especially for the darkest skin tones — this is explicitly a limitation, not a validated fairness guarantee.**

**Do not claim:** "our model is fair across skin tones" — the correct claim is "our model shows no evidence of disparity where sufficient data exists, but this cannot be confirmed for underrepresented groups."

---

## PART 25 — STRENGTHS OF THE RESEARCH (Ranked)

1. **Statistically-grounded leakage audit** (phi/chi-square; commonly cited as ~22 columns, exact figure NEEDS VERIFICATION — see Part 8) — rare in comparable work, directly protects result validity.
2. **Pre-registered ablation protocol with honest negative reporting** (ensemble p=0.062, joint fusion missed its bar, TTA rejected) — demonstrates genuine scientific discipline, not just claimed rigor.
3. **Cross-dataset + external validation with overlap exclusion** — catches a subtle leakage risk (dataset provenance overlap) most multi-dataset studies miss entirely.
4. **Patient-wise splitting with a code-enforced test-access guard** — operationalizes good practice as a hard constraint, not just a stated intention (note: the guard prevents *unsanctioned* re-access, not literally single use — see Part 9).
5. **Fitzpatrick fairness analysis with honest limitation disclosure** — attempted where most comparable work does not, and honestly caveated where data was insufficient.
6. **ADDED — found and fixed a gap in its own rigor:** `evaluate_fairness.py` was discovered to bypass the `--confirm-final` CLI guard entirely by calling helper functions directly rather than the guarded entry point, prompting the addition of a file-based `TEST_SPLIT_CONSUMED.json` guard (2026-07-25) that closes this loophole at a lower level. Catching and fixing a self-introduced gap in the project's own leakage/re-access discipline — rather than only claiming the discipline existed — is itself evidence of genuine rigor. Source: `THESIS_OWNERSHIP_MASTER.md` §5.

---

## PART 26 — WEAKNESSES AND LIMITATIONS (Brutally Honest)

| Limitation | Why it matters | Severity | Fixable before publication? |
|---|---|---|---|
| Fairness sample size (Types V/VI: 1-2 images) | Cannot support a real fairness claim for darkest skin tones | High for fairness claims specifically | No — needs new data collection |
| Ensemble (0.7321) unconfirmed (p=0.062) | Tempting but unsupported "better" number exists in the record | Medium — already handled by NOT adopting it as headline | Already appropriately handled |
| Cross-dataset limited to 3 shared classes | Generalization claim is partial, not full-taxonomy | Medium | No — inherent to dataset taxonomy mismatch |
| No inference latency/memory profiling | Deployment-readiness unknown | Low-medium | Yes, if desired as future work |
| **RESOLVED (was: LR/batch/epoch undocumented)** — the "22 columns" leakage-count figure does not reconcile against per-dataset feature-whitelist records | Reduces reproducibility precision for the leakage-audit claim specifically (LR/batch/epoch are now fully documented — see Part 15) | Medium (methodological transparency) | Yes — re-derive and correct the exact leakage-column count before final publication (also affects `main.tex`) |
| No independent *clinical/dermatologist* review of the excluded leakage columns (a documented researcher/supervisor review-and-approve step did occur, per `Project_Tracking.md`) | Small risk a legitimate feature was over-conservatively removed without domain-expert sign-off | Low-medium | Possible as a sensitivity-check future step |
| Architecture head-count/dimension choices not independently ablated | "Why 8 heads / why 256-d" answered by convention, not experiment | Low | Could be added as future ablation |
| Geographic/institutional concentration of all datasets | Limits real-world generalizability claims | Medium | No — needs new external data sources |

---

## PART 27 — CHALLENGES AND SOLUTIONS

- **Challenge:** Severe class imbalance (Melanoma/SCC). **Solution:** real, biopsy-confirmed data expansion — **trade-off:** still didn't meaningfully raise the isolated-architecture score, revealing the deeper constraint was model capacity, not data volume.
- **Challenge:** Avoiding p-hacking across many candidate ablations. **Solution:** pre-registered decision rules fixed *before* seeing results — **trade-off:** some promising-looking results (ensemble) had to be held back from "headline" status despite being numerically attractive.
- **Challenge:** Hidden dataset overlap risk in external validation. **Solution:** built an explicit image-ID exclusion list — **trade-off:** additional engineering effort, but essential for validity.
- **Challenge:** Balancing architecture explanation depth vs. presentation simplicity. **Solution:** layered explanation (intuitive first, technical detail only if asked).

---

## PART 28 — WHAT MAKES THIS RESEARCH DIFFERENT?

| Category | Claim status |
|---|---|
| Novel architecture | **Not strongly supported** — similar cross-attention direction (image K/V, metadata Q) already found superior in prior comparative work |
| Novel methodology | **Reasonably supported** — the specific combination of leakage audit + pre-registration + cross-dataset + fairness in one pipeline is uncommon in comparable literature |
| Novel evaluation protocol | **Reasonably supported** — pre-registered, access-guarded test discipline (sanctioned reads only, unsanctioned re-access blocked) is above typical practice at this research stage |
| Novel dataset handling | **Reasonably supported** — real-data (not synthetic) class-imbalance correction with verified zero patient overlap |
| Novel analysis | **Supported** — the per-class/confusion-based honest explanation of "why the score is lower" is a genuine, evidence-based interpretive contribution |
| Engineering contribution | Solid, reproducible pipeline |
| Scientific contribution | The methodology-over-raw-score framing itself |

**Evidence needed to justify a stronger novelty claim:** a systematic literature review (not an informal survey) confirming no prior work combines all these elements; formal ablation isolating the cross-attention direction's contribution specifically.

---

## PART 29 — PUBLISHABILITY

- **Is it publishable?** Yes, as a regional/mid-tier IEEE conference paper or a strong undergraduate thesis chapter, on the strength of its methodology.
- **Current level:** Not a top-tier, novel-architecture venue paper; framed correctly, it's a credible methodology/evaluation-rigor paper.
- **Strongest points:** leakage audit, pre-registration discipline, honest negative-result reporting, cross-dataset + external validation with overlap handling.
- **Biggest blockers for a higher-tier venue:** architecture isn't independently novel; SCC/Melanoma performance is weak; fairness evidence is thin; no clinical validation.
- **What to add:** fresh independent test set for the ensemble; targeted data for the AK/BCC/SCC cluster; broader fairness data.
- **What to weaken/qualify:** any language implying architectural novelty or SOTA status — both should remain explicitly unclaimed.

---

## PART 30 — PAPER WRITING (IEEE Structure)

**ADDED:** A real IEEE paper draft exists at `docs/main.tex` (326 lines: abstract, introduction, related work, methodology, results tables, discussion). Some earlier project records (`THESIS_OWNERSHIP_MASTER.md` §9.5, dated 2026-08-14) stated no paper source file existed locally — that is now stale; `main.tex` should be treated as the authoritative paper text and cross-checked directly. It is the actual source of the correct p-value breakdown used to fix this document (line 185: image p<0.001, metadata p=0.006, late fusion p=0.002), but it also independently repeats the unreconciled "22 columns" figure (line 46) — that correction (see Part 8) needs to be applied to `main.tex` as well, not just to this document.

| Section | Include | Exclude | Needs Citation? |
|---|---|---|---|
| Title/Abstract | Core contribution + headline result | Excess detail | No |
| Introduction | Motivation, gap, contributions list | Full methodology | Yes (background claims) |
| Related Work | TRACE/MetaBlock/Shrestha & Palit comparison, gap table | Unrelated general ML history | Yes |
| Methodology | Datasets, leakage audit, architecture, training config | Raw code | Dataset citations |
| Results | Primary, confusion, cross-dataset, external, ablations | Every intermediate log | No |
| Discussion | Interpretation, honest score explanation | New results | Cite corroborating clinical literature (e.g., Pacheco & Krohling 2020) |
| Limitations | All from Part 26 | Minimizing language | No |
| Conclusion | Summary, no new information | New claims | No |
| References | All cited works | — | — |

---

## PART 31 — FIGURES AND TABLES (What Should Exist)

| Figure/Table | Purpose | Belongs In |
|---|---|---|
| Dataset preparation workflow diagram | Show audit→leakage→split pipeline | Methodology |
| Architecture diagram | Show input→output flow | Methodology |
| Class-imbalance correction bar chart | Show real-data expansion | Methodology |
| Primary results table/bar chart | Show headline comparison | Results |
| Confusion matrix | Show error pattern | Results |
| Per-class precision/recall/F1 table | Explain "why lower score" | Results/Discussion |
| Cross-dataset results table | Show generalization | Results |
| Ablation results table | Show extended experiments | Results |
| Related-work comparison table | Show literature gap | Related Work |

---

## PART 32 — TECHNICAL GLOSSARY (Selected Key Terms)

| Term | Definition | In our thesis |
|---|---|---|
| Macro-F1 | Unweighted average of per-class F1 scores | Headline metric, 0.6977 |
| Data Leakage | Information not genuinely available at prediction time inflating performance | Columns removed via phi/chi-square audit (commonly cited as 22; exact count NEEDS VERIFICATION — see Part 8) |
| Patient-Level Split | Splitting by patient, not image | Prevents patient-overlap leakage |
| Cross-Attention | Attention where Q comes from one modality, K/V from another | Metadata=Q, Image=K/V |
| Multi-Head Attention | Parallel attention subspaces, concatenated | 8 heads used |
| Domain/Distribution Shift | Statistical differences between train and deployment/test data | Explains PAD→HAM drop |
| External Validation | Testing on data never used in any training | HAM→ISIC 1/2 |
| Ablation | Removing/changing one component to measure its effect | Backbone, dataset-expansion, ensemble, joint-fusion |
| Bootstrap | Resampling-based method to estimate a statistic's variability/significance | 1,000 resamples used |
| p-value | Probability of observing the effect (or more extreme) under the null hypothesis | 0.062 for ensemble = not significant at 0.05 |
| Class Imbalance | Unequal class representation | Melanoma/SCC most affected |
| Calibration | Whether predicted confidence matches real-world accuracy | Not established in this thesis |

*(Full glossary available on request — this covers the highest-yield terms for viva defense.)*

---

## PART 33-38 — Q&A BANKS (Condensed — see also `THESIS_FULL_A_TO_Z_DOCUMENTS.txt` for additional basic/intermediate Q&A already compiled earlier in this project)

### Basic
- **What is your thesis about?** → Part 1, answer 1.
- **What datasets?** → Part 6 table.
- **What is Macro-F1?** → Part 17/32.
- **What is cross-attention?** → Part 13.

### Intermediate
- **Why patient-level splitting?** → Part 9.
- **Why metadata as Query?** → Part 12-13.
- **Why remove [the leakage-flagged] columns?** → Part 8 (note: exact count needs re-verification before publication).
- **Why multiple seeds?** → Part 15.

### Advanced/Researcher-Level
- **What's the theoretical justification for metadata-conditioned cross-attention?** → It lets a lower-dimensional, clinically-structured signal (metadata) selectively query a higher-dimensional, spatially-structured signal (image), avoiding the dimensional-imbalance failure mode of naive concatenation, while remaining directionally consistent with clinical reasoning (context informs where to look).
- **Could simple concatenation outperform cross-attention under another representation size?** → **[UNCERTAIN — not tested with alternative late-fusion dimension-matching schemes in this thesis; a fair follow-up experiment, not something we can currently claim either way.]**
- **How do you separate architectural gain from dataset-size gain?** → Directly addressed: the dataset-expansion-only ablation (Part 20) isolates this, showing minimal gain from data alone.
- **Is the test set truly independent?** → Yes, by construction (patient-wise split + code-enforced access guard blocking unsanctioned re-reads — note the set was legitimately read twice under pre-registered rules, not literally once, see Part 9/21), assuming no undetected patient-ID errors in source metadata — **[this underlying assumption is not independently re-verified beyond the dataset's own patient-ID field, a reasonable caveat if pushed].**
- **Why is p=0.062 not significant at 0.05?** → By definition/convention; 0.05 is the standard significance threshold, and 0.062 exceeds it, meaning we cannot rule out the observed difference arose by chance at the conventional confidence level.

### Trick/Critical Questions (Safe answer / Stronger answer / What NOT to say)
- **"Your score isn't SOTA, why should we care?"**
  - Safe: "We don't claim SOTA — our contribution is methodological rigor under a harder, more honest evaluation setup."
  - Stronger: "Comparing raw scores across different metrics/tasks/splits is scientifically invalid; under a comparable rigorous setup, our result is competitive, and the evaluation protocol itself is a contribution."
  - Do NOT say: "Our score would be higher if we did it like they did" as an excuse — instead frame it as "we deliberately chose the harder setup."
- **"Why did your ensemble score higher — why not use that as your main result?"**
  - Safe: "It wasn't statistically confirmed (p=0.062), so we kept the proven result as official."
  - Stronger: "Adopting an unconfirmed higher number would be a form of cherry-picking, which we explicitly avoided per our pre-registered protocol."
  - Do NOT say: "It's basically significant" — it is not.
- **"Can this model diagnose cancer?"**
  - Safe: "No — it's a research-stage classification result, not a validated diagnostic tool."
  - Stronger: explain the full clinical-validation gap from Part 23.
  - Do NOT say: anything implying deployment-readiness.

### Unexpected Questions — Reasoning Framework
When you don't know: *"I don't have direct evidence for that in the current experiments, but based on [general ML/clinical knowledge], I would expect..."* or *"That specific test wasn't performed, so I'd treat it as an open limitation/future work item rather than guess."*

---

## PART 39 — "WHAT IF" SCENARIOS (Selected)

- **What if we used image-level splitting?** → Test score would likely be inflated (patient-overlap leakage), less trustworthy.
- **What if we kept the leaked metadata (e.g., "biopsed")?** → Score would look artificially higher but the model would be learning a diagnostic-proxy shortcut, not real signal.
- **What if metadata is missing at inference?** → **[NOT DOCUMENTED — no missing-metadata robustness test was performed; a reasonable limitation to acknowledge.]**
- **What if a new hospital provides very different images?** → Expect a generalization drop similar in kind to the PAD→HAM10000 result — this is exactly what that experiment was designed to anticipate.
- **What if ensemble significance became p<0.05 on a fresh test?** → It could then legitimately replace 0.6977 as the headline, per the pre-registered logic — but this must be tested on a genuinely fresh set, not the already-consumed one.

---

## PART 40 — RESEARCHER'S MEMORY SHEET

```
Title: Metadata-Guided Cross-Attention Fusion for Multimodal Skin Lesion 
       Classification — A Leakage-Audited, Cross-Dataset Validated Approach

Problem: Image-only skin lesion AI ignores clinical metadata; evaluation 
         rigor (leakage, generalization, fairness) is often weak

Datasets: PAD-UFES-20 (2,298/6cls, primary) · HAM10000 (10,015/7cls, 
          cross-dataset) · ISIC 1 (2,357) · ISIC 2 (25,076) — externals
          + DERM12345/MED-NODE (Melanoma expansion, real data)

Leakage audit: phi + chi-square → columns removed (e.g. biopsed, phi=0.80);
               commonly cited as "22 columns" but this NEEDS VERIFICATION —
               does not cleanly reconcile against per-dataset feature
               whitelists (strict leakage-tested ~5-12; all-exclusions 45)
Split: patient-wise, 70/15/15, seed=42, code-enforced test-access guard
       (NOT literally single-use — sanctioned second read occurred, see Pt 9/21)

Architecture: EfficientNet-B0 (49 tokens, 1280-d) → metadata-conditioned
              channel gate → + MLP (89-d→64-d) → shared 256-d space
              → 8-head cross-attention (Q=metadata, K/V=gated image)
              → 320-d joint → FC→BN→ReLU→Drop→FC(6)
LR/batch/epochs: 1e-4/1e-3/1e-5 (image/meta/fusion), batch 32, 30-epoch cap
Compute: no local GPU — trained on Kaggle

Primary results: Image 0.6175 | Meta 0.6077 | Late-Fusion 0.6566 | 
                 Cross-Attn 0.6977 (p≤0.006 vs all, headline)

Cross-dataset: 0.6977 → 0.4654 (PAD→HAM, expected drop)
External validation: overlap-excluded (1,362 IDs), image-only 0.4912 (Archive 2)
Ensemble: 0.7321 test, p=0.062 (NOT headline, reported honestly)
Joint fusion: 0.6721 val (missed 0.6710 bar, no test read)
TTA+Ensemble: REJECTED (Melanoma F1 0.36→0.20 despite OK aggregate)

Per-class weak point: SCC F1=0.2543 (support 96) — without it, macro-F1≈0.786
Confusion cluster: AK/BCC/SCC = 68.3% of all errors (clinically explainable)

Strongest contribution: methodology + honest reporting, not raw architecture novelty
Biggest limitation: fairness sample size (Types V/VI: 1-2 test images)
Future work: fresh test set for ensemble, AK/BCC/SCC targeted data, 
             multi-institution fairness data

Final conclusion: methodological rigor matters as much as raw accuracy 
                  for clinically-oriented AI
```

---

## PART 41 — LIVE DEFENSE SIMULATION

This part requires an interactive conversation, not a static document. **Reply "start the simulation" (or similar) in your next message**, and I will act as the 6-panel examiner (Supervisor, AI/ML Researcher, CV Researcher, Medical AI Researcher, Statistics Evaluator, Critical Examiner), asking one question at a time, starting easy and escalating — waiting for your answer before evaluating it and giving the ideal answer, exactly as you specified.

---
**END OF MASTERY DOCUMENT.** Every claim above is grounded in your thesis/paper/project record; anything not directly documented is explicitly flagged rather than invented.
