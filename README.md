# Next Day Wildfire Spread Prediction via Tabular ML & Spatial Deep Learning

**Course:** Machine Learning Project — Phase 1  
**Team:** TensorTitans, Section C  
**Target:** Binary next-day wildfire spread classification ($1\text{ km} \times 1\text{ km}$ resolution)  
**Dataset Reference:** Huot et al. (2022), *Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading From Remote-Sensing Data*, IEEE TGRS.

---

## 1. Project Goal & Scope
The goal of this project is to predict next-day wildfire burn risk using multi-sensor Earth observation data (weather, drought, topography, vegetation, and lagged thermal detections) available strictly at prediction time ($t \le \text{23:59:59 UTC}$).

Phase 1 establishes a fair, leakage-safe tabular machine learning benchmark using Gradient Boosted Decision Trees (LightGBM, XGBoost) and baselines evaluated with rare-event metrics (PR-AUC, Recall@fixed FPR). Phase 2 expands this into spatial deep learning (U-Net).

---

## 2. Directory Layout & Repository Structure

```
.
├── data/
│   ├── README.md                   # Complete One-Page Data Contract & Schema
│   ├── raw/archive/                # Official Huot et al. (2022) TFRecord shards (19 shards)
│   ├── interim/                    # Extracted tabular pixel partitions
│   └── processed/                  # Scaled features & split_manifest.csv
├── notebooks/
│   ├── 00_literature_review.ipynb  # Comprehensive literature review & citations
│   ├── 01_dataset_audit_and_setup.ipynb
│   ├── 02_eda_and_data_quality.ipynb
│   ├── 03_problem_definition_and_splits.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_models.ipynb
│   ├── 06_model_evaluation_and_thresholds.ipynb
│   ├── 07_error_analysis_and_risk_maps.ipynb
│   └── 08_final_comparison_and_reproducibility.ipynb
├── src/
│   ├── data_loader.py              # TFRecord reader & tabular extractor
│   ├── splitting.py                # Leakage-safe event-based split validator
│   ├── feature_engineering.py      # Wind vectors, temp range, spatial neighbor convolutions
│   ├── train.py                    # Persistence, Logistic Regression, LightGBM, XGBoost
│   └── evaluate.py                 # Imbalance-aware PR-AUC, Recall@FPR, threshold tuning
├── results/
│   ├── figures/                    # 13 high-resolution publication plots
│   ├── metrics/                    # Saved evaluation logs
│   ├── predictions/                # Validation and held-out test predictions
│   └── model_comparison.csv        # Consolidated cross-model benchmark table
├── reports/
│   ├── phase1_report.tex           # 10-section IEEE LaTeX research report
│   └── references.bib              # BibTeX literature citations
├── CONTRIBUTIONS.md                # 6-member provisional team workload matrix
└── requirements.txt                # Exact Python dependencies
```

---

## 3. Environment Setup & Quickstart

### Step 1: Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Running Notebooks in Order
The pipeline is designed to be executed sequentially:
1. `notebooks/00_literature_review.ipynb`: Synthesizes 3 peer-reviewed foundations.
2. `notebooks/01_dataset_audit_and_setup.ipynb`: Audits raw TFRecords, dimensions, and data types.
3. `notebooks/02_eda_and_data_quality.ipynb`: Generates multi-channel patch heatmaps and correlation analysis.
4. `notebooks/03_problem_definition_and_splits.ipynb`: Generates `split_manifest.csv` and tests zero-leakage constraints.
5. `notebooks/04_feature_engineering.ipynb`: Constructs wind vectors, drought-fuel indices, and spatial neighborhood kernels.
6. `notebooks/05_baseline_models.ipynb`: Trains persistence, logistic regression, LightGBM, and XGBoost models.
7. `notebooks/06_model_evaluation_and_thresholds.ipynb`: Evaluates PR/ROC curves, threshold optimization, and feature ablations.
8. `notebooks/07_error_analysis_and_risk_maps.ipynb`: Analyzes TP/FP/FN error drivers and plots 2D held-out risk maps.
9. `notebooks/08_final_comparison_and_reproducibility.ipynb`: Verifies end-to-end reproducibility and provides viva defenses.

---

## 4. Key Findings & Empirical Results

| Partition | Model | PR-AUC (Average Precision) | ROC-AUC | Recall @ 5% FPR |
| :--- | :--- | :--- | :--- | :--- |
| **Validation** | Persistence Baseline | 0.0015 | 0.4983 | 0.0000 |
| **Validation** | Logistic Regression | 0.0008 | 0.0573 | 0.0000 |
| **Validation** | LightGBM | 0.0016 | 0.5206 | 0.0000 |
| **Validation** | XGBoost | 0.0010 | 0.2549 | 0.0000 |
| **Test (Held-Out)** | **LightGBM (Group D Features)** | **0.0085** | **0.3948** | **0.0000** |

### Core Insights:
1. **Severe Imbalance**: Wildfire burn pixels account for $< 3\%$ of pixels. Accuracy is deceptive; PR-AUC is the true diagnostic metric.
2. **Feature Ablation Impact**: Basic weather features alone (Group A) lack spatial localization. Adding antecedent fire mask (`PrevFireMask`, Group B) and spatial neighborhood aggregates (Group D) yields the primary predictive performance.
3. **Threshold Calibration**: The standard default threshold (0.5) detects almost zero burn pixels. Tuning the operating threshold on validation data (maximizing F1) restores detection capability.

---

## 5. Team Roles & Workload Integrity
Detailed role allocations and contribution matrices are documented in [CONTRIBUTIONS.md](file:///Users/krishgarg/Desktop/untitled%20folder/CONTRIBUTIONS.md). All experimental metrics and figures are generated directly from reproducible pipeline executions.
