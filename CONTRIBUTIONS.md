# Team Roles & Contribution Matrix

**Project:** Next Day Wildfire Spread Prediction via Tabular ML and Spatial Deep Learning  
**Team:** TensorTitans, Section C  
**Course:** Machine Learning Project — Phase 1 and Phase 2

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


---

## 3. Phase 2 (Deep Learning) Role Allocation

Every member works on two things. Pairs share both of their pieces of work.

| Team Member | Work 1 | Work 2 |
| :--- | :--- | :--- |
| **Vanshika** | Literature review (with Nitanshu) | Model training (with Nitanshu) |
| **Nitanshu** | Literature review (with Vanshika) | Model training (with Vanshika) |
| **Riya** | Data pipeline (with Tejas) | Ablation study (with Tejas) |
| **Tejas** | Data pipeline (with Riya) | Ablation study (with Riya), report and README assembly |
| **Krish** | Evaluation and validation | Phase 2 slides |

### Deliverables by area

| Area | Owners | Files |
| :--- | :--- | :--- |
| Literature review | Vanshika, Nitanshu | `literature_review/phase2_dl_literature.md`, `literature_review/phase2_model_choices.md`, `reports/phase2/sections/introduction.tex`, `related_work.tex` |
| Data pipeline | Riya, Tejas | `src/dl_data.py`, `notebooks/09_dl_data_pipeline.ipynb`, `results/models/unet_norm.npz`, `results/metrics/dl_data_summary.csv`, `dl_normalization.csv`, `reports/phase2/sections/dataset.tex` |
| Model training | Vanshika, Nitanshu | `src/dl_model.py`, `notebooks/10_unet_training.ipynb`, `results/models/unet.keras`, `results/metrics/dl_training_history.csv`, `results/figures/dl_unet_architecture.png`, `dl_training_curves.png`, `reports/phase2/sections/methodology.tex` |
| Evaluation | Krish | `notebooks/11_unet_evaluation.ipynb`, `results/metrics/dl_test_results.csv`, `dl_test_metrics.json`, `results/predictions/dl_test_per_map.csv`, `reports/phase2/sections/results.tex` |
| Ablation study | Riya, Tejas | `notebooks/12_unet_ablation.ipynb`, `results/metrics/dl_ablation_results.csv`, `results/figures/dl_ablation.png`, `reports/phase2/sections/ablation.tex`, `conclusion.tex` |
| Slides | Krish | `phase2_presentation.pdf` |
| Report and README assembly | Tejas | `reports/phase2/main.tex`, `references.bib`, `README.md`, `CONTRIBUTIONS.md` |
