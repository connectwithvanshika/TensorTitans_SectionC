# Team Roles & Contribution Matrix

**Project:** Next Day Wildfire Spread Prediction via Tabular ML and Spatial Deep Learning  
**Team:** TensorTitans, Section C  
**Course:** Machine Learning Project — Phase 1

---

## 1. Provisional Role Allocation & Responsibilities

| Team Member | Functional Area | Primary Responsibilities | Concrete Deliverables |
| :--- | :--- | :--- | :--- |
| **Member 1** | **Literature & Formulation** | Research prior work; formalize prediction task, time cutoffs, assumptions, and research gap. | `notebooks/00_literature_review.ipynb`, `literature_review/paper_tracking.csv`, `data/README.md` (Data Contract). |
| **Member 2** | **Dataset & Geospatial Audit** | Access raw TFRecords; verify channels, dimensions, CRS, temporal stamps, licensing, and schema. | `notebooks/01_dataset_audit_and_setup.ipynb`, `src/data_loader.py`, dataset schema documentation. |
| **Member 3** | **EDA & Data Quality** | Quantify class imbalance, missing values, coordinate distributions, correlation structure, and mask visualizations. | `notebooks/02_eda_and_data_quality.ipynb`, `results/figures/` (EDA plots), data quality notes. |
| **Member 4** | **Splits & Feature Engineering** | Formulate leakage-free spatial and temporal partitioning; extract tabular features and ablations. | `notebooks/03_problem_definition_and_splits.ipynb`, `notebooks/04_feature_engineering.ipynb`, `src/splitting.py`, `src/feature_engineering.py`. |
| **Member 5** | **ML Models & Tuning** | Build baseline models, train XGBoost and LightGBM models, optimize hyperparameters via validation data. | `notebooks/05_baseline_models.ipynb`, `src/train.py`, model artifacts. |
| **Member 6** | **Evaluation & Integration** | Compute PR-AUC, ROC-AUC, Recall@fixed FPR, select optimal decision threshold, generate risk maps and LaTeX report. | `notebooks/06_model_evaluation_and_thresholds.ipynb`, `notebooks/07_error_analysis_and_risk_maps.ipynb`, `notebooks/08_final_comparison_and_reproducibility.ipynb`, `src/evaluate.py`, `reports/phase1_report.tex`. |

---

## 2. Commit & Workload Integrity Standards

- **Shared Protocol**: All members review the target definition and split protocol before training starts.
- **Top-to-Bottom Execution**: Every notebook in `notebooks/` must execute end-to-end cleanly on the documented environment.
- **Experimental Integrity**: All reported metrics and figures are grounded in verified experimental runs; results are never fabricated or manually overwritten.
- **Leakage Prevention**: No feature utilizes information beyond day $t$ (end of observation cutoff).
