# Dataset Strategy

This research uses multiple publicly available dermatology datasets.

---

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Data Processing Philosophy](#data-processing-philosophy)
3. [Raw Data Structure](#raw-data-structure)
4. [Data Cleaning](#data-cleaning)
5. [Metadata Standardization](#metadata-standardization)
6. [Label Mapping](#label-mapping)
7. [Dataset Split](#dataset-split)
8. [Final Processed Dataset](#final-processed-dataset)

---

<a id="dataset-overview"></a>
## 1. Dataset Overview

### 1.1 PAD-UFES-20

- **Role:** Primary multimodal dataset
- **Contains:** Skin images, clinical metadata, patient information, symptoms
- **Usage:** Training and validation of the multimodal model.

### 1.2 HAM10000

- **Role:** Benchmark dataset
- **Contains:** Dermoscopic images, disease labels, limited metadata
- **Usage:** Comparison with existing image classification methods.

### 1.3 ISIC Archive 1

> *(Corrected 2026-07-08: this archive was later split into two separately
> audited/cleaned collections, ISIC Archive 1 and ISIC Archive 2 — see
> below. This section originally described them as one "ISIC Archive.")*

- **Role:** External validation dataset (with cross-dataset leakage
  exclusions applied against HAM10000 — see `Project_Tracking.md` and
  `Dataset_Preparation_Final_Report.md`)
- **Usage:** Evaluate model generalization ability. No `metadata.csv` ships
  with this archive — class labels come from folder structure.

### 1.4 ISIC Archive 2

- **Role:** Additional external evaluation dataset (with cross-dataset
  leakage exclusions applied against HAM10000 — same as ISIC Archive 1)
- **Usage:** Evaluate model generalization ability. Richer metadata than
  ISIC Archive 1 (27-column `metadata.csv`), though most clinical fields
  are excluded from model input as label-derived — see
  `data/processed/ISIC_Archive_2/feature_whitelist.md`.

---

<a id="data-processing-philosophy"></a>
## 2. Data Processing Philosophy

Raw data will never be modified.

**Pipeline:**

```
Raw Dataset → Audit → Cleaning → Standardization → Processed Dataset
```

---

<a id="raw-data-structure"></a>
## 3. Raw Data Structure

```
data/raw/
├── HAM10000
├── PAD_UFES20
├── ISIC_Archive_1
└── ISIC_Archive_2
```

---

<a id="data-cleaning"></a>
## 4. Data Cleaning

**Operations:**

- Corrupted image removal
- Duplicate detection
- Missing value analysis
- Label correction

---

<a id="metadata-standardization"></a>
## 5. Metadata Standardization

**Final format:**

- `image_path`
- `patient_id`
- `age`
- `sex`
- `location`
- `symptoms`
- `clinical_features`
- `disease_label`
- `dataset_source`

---

<a id="label-mapping"></a>
## 6. Label Mapping

Different datasets use different naming.

**Example:**

`MEL`, `mel`, `Melanoma` → converted to **`Melanoma`**

---

<a id="dataset-split"></a>
## 7. Dataset Split

For patient-based datasets:

| Split | Proportion |
|---|---|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

Patient-wise splitting will be applied to avoid data leakage.

---

<a id="final-processed-dataset"></a>
## 8. Final Processed Dataset

> *(Corrected 2026-07-08: file naming below was updated to match what the
> pipeline actually produces — see each dataset's `dataset_description.md`
> for full detail.)*

```
data/processed/
└── PAD_UFES20/  (and identically for HAM10000/ISIC_Archive_1/ISIC_Archive_2)
    ├── metadata_train.csv
    ├── metadata_val.csv
    ├── metadata_test.csv
    ├── label_mapping.csv
    ├── split_quality_report.csv
    ├── dataset_description.md
    └── feature_whitelist.md
```

Images are not copied here — `image_path` in each CSV points back to
`data/raw/.../`.
