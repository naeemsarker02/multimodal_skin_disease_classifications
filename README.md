# Multimodal Skin Disease Classification System

A research project on multimodal deep learning for skin disease diagnosis using images and clinical metadata.

## Overview

Skin disease diagnosis is a challenging healthcare problem that requires expert knowledge and clinical experience. Deep learning has shown promising results in automated skin disease classification from medical images, but most existing approaches rely on image information alone and ignore clinical factors such as patient demographics, symptoms, and lesion characteristics.

This research develops a **multimodal skin disease classification system** that combines skin lesion images, clinical metadata, patient information, and symptoms into a more reliable and generalizable AI-assisted diagnostic framework.

## Research Goal

Design and evaluate a multimodal deep learning framework that improves skin disease classification by integrating visual and clinical information — developing strong image-based baselines, using clinical metadata effectively, designing multimodal fusion approaches, and evaluating robustness on external datasets.

## Datasets

| Dataset | Role | Contains |
|---|---|---|
| PAD-UFES-20 | Primary multimodal dataset | Skin images, clinical metadata, patient info, symptoms |
| HAM10000 | Benchmark dataset | Dermoscopic images, disease labels, limited metadata |
| ISIC Archive 1 & 2 | External validation datasets | Images, with varying metadata richness |

Full dataset roles, pipeline rules, and current cross-dataset findings are documented in `docs/AI_Assistant_Instructions.md`, `docs/Dataset_Strategy.md`, and `docs/Dataset_Preparation_Final_Report.md`.

## Research Workflow

```
Literature Review → Dataset Collection → Dataset Preparation
  → Baseline Model Development → Multimodal Model Development
  → Experiments & Evaluation → Thesis & Paper Writing
```

## Project Structure

```
Multimodal_Skin_Disease_Research/
├── data/            raw / interim / processed datasets
├── docs/            research documentation (see below)
├── logs/            per-dataset audit & cleaning run logs
├── notebooks/        EDA notebooks (upcoming phase)
├── papers/           reviewed literature and reference material
├── reports/          per-dataset audit reports and figures
├── experiments/       model development tracking (upcoming phase)
├── results/           evaluation outputs (upcoming phase)
├── src/
│   ├── data_audit/    dataset audit pipeline
│   └── data_cleaning/ dataset cleaning pipeline
├── README.md
└── requirements.txt
```

## Documentation

- `docs/PROJECT_PLAN.md` — canonical project plan: confirmed design decisions, folder structure, current phase
- `docs/AI_Assistant_Instructions.md` — working directive: role, pipeline phases, hard rules
- `docs/Dataset_Strategy.md` — dataset roles and processing philosophy
- `docs/Project_Tracking.md` — current project status, decision log, and progress tracker (source of truth)
- `docs/Dataset_Preparation_Final_Report.md` — independent cross-dataset verification report (frozen historical report)

Superseded docs (e.g. the original `Research_Plan.md` and `Project_Cleanup_Review.md`) are kept, never deleted, in `_archive/`.

## Current Status

Dataset preparation is complete and leakage-verified for PAD-UFES-20, HAM10000, ISIC Archive 1, and ISIC Archive 2, individually. See `docs/Project_Tracking.md` for the live status of all phases and `docs/Dataset_Preparation_Final_Report.md` for a critical cross-dataset image-overlap finding that must be resolved before HAM10000 and the ISIC archives are combined in any train/external-validation protocol.

## Research Philosophy

"Build scientifically reliable AI systems, not only high accuracy models." Data quality, reproducibility, explainability, and clinical relevance are treated as equally important as accuracy.

## Author

Naeem Sarker — Medical AI, Computer Vision, Multimodal Learning, Deep Learning.
# multimodal_skin_disease_classifications
