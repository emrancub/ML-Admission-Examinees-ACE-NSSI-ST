# Machine Learning Pipeline: Bangladeshi University Admission Examinees Study

## Overview

This repository contains the complete, reproducible machine-learning workflow for investigating the associations of adverse childhood experiences (childhood abuse and household dysfunction) with non-suicidal self-injury (self-harm) and past-year suicidal thoughts among Bangladeshi university admission examinees ($N = 1,139$).

## Key Reproducibility Features

1. **Explicit Whitespace & Missing-Data Cleaning (`src/data_preparation.py`)**:
   - Strips leading/trailing whitespace, converts space-only entries (`" "`) to `NaN`, validates permitted category codes across all 30 predictors and 2 outcomes.
2. **Leakage-Free Machine Learning Pipeline (`src/modeling.py`, `src/evaluation.py`, `src/explainability.py`, `src/plots.py`)**:
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

To execute the entire machine-learning pipeline end-to-end:
```bash
python main.py
```
