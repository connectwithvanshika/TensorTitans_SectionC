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
│   ├── figures/                    # Plots saved by notebooks 02-08
│   ├── metrics/                    # Saved metrics: validation, test, ablation, baselines, row counts
│   ├── predictions/                # Validation and held-out test predictions
│   └── model_comparison.csv        # Consolidated cross-model benchmark table
├── reports/
│   ├── report.pdf                  # Phase 1 report
│   └── references.bib              # BibTeX literature citations
├── presentation.pdf                # Phase 1 presentation
├── CONTRIBUTIONS.md                # Team contribution table
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

All numbers below come from `results/model_comparison.csv` and `results/metrics/`.

| Partition | Model | PR-AUC (Average Precision) | ROC-AUC | Recall @ 5% FPR |
| :--- | :--- | :--- | :--- | :--- |
| **Validation** | Persistence Baseline | 0.0015 | 0.4983 | 0.0000 |
| **Validation** | Logistic Regression | 0.0008 | 0.1414 | 0.0000 |
| **Validation** | LightGBM | 0.0016 | 0.5175 | 0.0000 |
| **Validation** | XGBoost | 0.0009 | 0.1989 | 0.0000 |
| **Test (Held-Out)** | **LightGBM (Group D Features)** | **0.0109** | **0.4947** | **0.0000** |

Random guessing scores PR-AUC 0.0015 on validation and 0.0109 on test (the fire rate), and ROC-AUC 0.5.

### Core Insights:
1. **Severe Imbalance**: Only 0.08% (train), 0.15% (validation) and 1.09% (test) of rows burn the next day. Always predicting "no fire" scores 98.91% accuracy on test, so accuracy is useless here and PR-AUC is the main metric.
2. **The model did not learn the signal yet**: On the frozen test set LightGBM catches 0 of 157 fires (TN 14,176, FP 96, FN 157, TP 0), which is no better than random, and its accuracy (98.25%) is below always saying "no fire" (98.91%). LightGBM runs in deterministic mode, so these numbers are the same on every run. A simple rule that counts burning neighbours reaches PR-AUC 0.182, about 17x random, so the signal is in the data.
3. **Feature groups**: On validation, PR-AUC stays near random for every feature group (A 0.0016, B 0.0019, C 0.0022, D 0.0016). With only 39 validation fires, all from one map, these differences are noise.
4. **Main cause**: The model learned from only 62 fire rows. The next step is extracting many more maps per split.

---

## 5. Team Roles & Workload Integrity
Detailed role allocations are documented in [CONTRIBUTIONS.md](CONTRIBUTIONS.md). All experimental metrics and figures are generated directly from reproducible pipeline executions.
