# Reproducible Statistical and Machine Learning Pipeline: Bangladeshi University Admission Examinees Study

## Overview

This repository contains the complete, reproducible statistical and machine-learning workflow for investigating the associations of adverse childhood experiences (childhood abuse and household dysfunction) with non-suicidal self-injury (self-harm) and past-year suicidal thoughts among Bangladeshi university admission examinees ($N = 1,139$).

## Key Reproducibility Features

1. **Explicit Whitespace & Missing-Data Cleaning (`src/data_preparation.py`)**:
   - Strips leading/trailing whitespace, converts space-only entries (`" "`) to `NaN`, validates permitted category codes across all 30 predictors and 2 outcomes, and exports `outputs/predictor_missingness_audit.csv` (identifying 75 missing values across 12 predictors in 63 participants; 1,076 complete cases).
   - Saves exact 80/20 joint-stratified train/holdout split identifiers (`outputs/split_indices.csv`, `SEED = 42`, development $n = 911$, untouched holdout $n = 228$).
2. **Conventional Bivariate & Multivariable Statistical Workflow (`src/statistical_tables_audit.py`)**:
   - Reproduces Table 1 (`outputs/Table1_bivariate_audit.csv`), Table 2 (`outputs/Table2_bivariate_audit.csv`), NSSI–suicidal thoughts cross-tabulation (`outputs/NSSI_vs_ST_crosstab_audit.csv`), Table 4 (`outputs/Table4_NSSI_multivariable_regression.csv`), Table 5 (`outputs/Table5_ST_multivariable_regression.csv`), and multicollinearity diagnostics (`outputs/Multicollinearity_VIF_audit.csv`).
3. **Cumulative ACE Dose–Response Audit (`src/ace_audit.py`)**:
   - Computes both the 7-domain ACE score (primary construct matching Felitti et al.) and the 17-item ACE count (sensitivity comparison), exporting crude ($N = 1,139$) and multivariable-adjusted ($N = 1,076$) odds ratios, 95% CIs, Wald tests, and Nagelkerke $R^2$ (`outputs/ACE_domain_score_audit.csv`, `outputs/ACE_dose_response_OR_AOR_audit.csv`, `outputs/Table3_ACE_dose_response_summary.csv`).
4. **Leakage-Free Machine Learning Pipeline (`src/modeling.py`, `src/evaluation.py`, `src/explainability.py`, `src/plots.py`)**:
   - Performs most-frequent imputation, one-hot encoding (71 valid category columns), and `RandomOverSampler` strictly inside training folds during 5-fold `StratifiedKFold` `RandomizedSearchCV` (25 configurations per model-outcome pair, refit on `average_precision`).
   - Uses a consistent probability threshold of 0.50 (`predict_proba(X)[:, 1] >= 0.50`) across both cross-validation and untouched holdout evaluation.
   - Computes additive one-hot-grouped CatBoost SHAP values and XGBoost gain importance without file overwrites.

## Installation

```bash
conda env create -f environment.yml
conda activate r2-ml-analysis
```
or via `pip`:
```bash
pip install -r requirements.txt
```

## Running the Complete Pipeline

To execute the entire statistical and machine-learning pipeline end-to-end:
```bash
python main.py
```
To run only the data cleaning, missingness audit, conventional statistical tables (Tables 1–5), and cumulative ACE dose–response audit:
```bash
python main.py --stats-only
```
