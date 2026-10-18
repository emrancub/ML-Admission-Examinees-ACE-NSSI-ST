import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score,
    average_precision_score, log_loss
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from .config import OUTDIR, SEED, USE_CUDA, CUDA_DEVICE, SEARCH_N_JOBS, display
from .data_preparation import TARGETS

THRESHOLD = 0.50


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


imputer = SimpleImputer(strategy='most_frequent')
ohe = make_ohe()


def _prob_pos(estimator, X):
    return np.asarray(estimator.predict_proba(X)[:, 1], dtype=float)


def _pred_at_threshold(estimator, X, threshold=THRESHOLD):
    return (_prob_pos(estimator, X) >= threshold).astype(int)


def cv_accuracy_proba(estimator, X, y_true):
    return accuracy_score(y_true, _pred_at_threshold(estimator, X))


def cv_balanced_accuracy_proba(estimator, X, y_true):
    return balanced_accuracy_score(y_true, _pred_at_threshold(estimator, X))


def cv_precision_pos_proba(estimator, X, y_true):
    return precision_score(y_true, _pred_at_threshold(estimator, X), pos_label=1, zero_division=0)


def cv_recall_pos_proba(estimator, X, y_true):
    return recall_score(y_true, _pred_at_threshold(estimator, X), pos_label=1, zero_division=0)


def cv_f1_pos_proba(estimator, X, y_true):
    return f1_score(y_true, _pred_at_threshold(estimator, X), pos_label=1, zero_division=0)


def cv_specificity_proba(estimator, X, y_true):
    y_pred = _pred_at_threshold(estimator, X)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else np.nan


def cv_gmean_proba(estimator, X, y_true):
    y_pred = _pred_at_threshold(estimator, X)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return math.sqrt(max(sens, 0.0) * max(spec, 0.0))


def cv_roc_auc_proba(estimator, X, y_true):
    return roc_auc_score(y_true, _prob_pos(estimator, X))


def cv_average_precision_proba(estimator, X, y_true):
    return average_precision_score(y_true, _prob_pos(estimator, X))


def cv_neg_log_loss_proba(estimator, X, y_true):
    p = _prob_pos(estimator, X)
    return -log_loss(y_true, np.column_stack([1.0 - p, p]), labels=[0, 1])


# Consistent probability-based scoring across CV and holdout (threshold = 0.50)
scoring = {
    'accuracy': cv_accuracy_proba,
    'balanced_accuracy': cv_balanced_accuracy_proba,
    'precision_pos': cv_precision_pos_proba,
    'recall_pos': cv_recall_pos_proba,
    'f1_pos': cv_f1_pos_proba,
    'specificity': cv_specificity_proba,
    'gmean': cv_gmean_proba,
    'roc_auc': cv_roc_auc_proba,
    'average_precision': cv_average_precision_proba,
    'neg_log_loss': cv_neg_log_loss_proba,
}

xgb_cuda_kwargs = {'device': 'cuda'} if USE_CUDA else {'device': 'cpu'}
catboost_cuda_kwargs = {'task_type': 'GPU', 'devices': CUDA_DEVICE} if USE_CUDA else {'task_type': 'CPU'}

MODELS = {
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(random_state=SEED, n_jobs=1),
    'SVM': SVC(probability=True, random_state=SEED),
    'GBM': GradientBoostingClassifier(random_state=SEED),
    'XGBoost': XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        random_state=SEED, n_jobs=1, tree_method='hist',
        **xgb_cuda_kwargs
    ),
    'CatBoost': CatBoostClassifier(
        loss_function='Logloss', eval_metric='Logloss',
        random_seed=SEED, verbose=False, allow_writing_files=False,
        thread_count=1, **catboost_cuda_kwargs
    ),
}

PARAM_DISTS = {
    'KNN': {
        'model__n_neighbors': [3, 5, 7, 9, 11, 15, 21, 31],
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2],
    },
    'Random Forest': {
        'model__n_estimators': [200, 400, 600, 800],
        'model__max_depth': [None, 5, 10, 15, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4, 8],
        'model__max_features': ['sqrt', 'log2', 0.5, None],
    },
    'SVM': {
        'model__kernel': ['rbf', 'linear'],
        'model__C': [0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100],
        'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
    },
    'GBM': {
        'model__n_estimators': [50, 100, 200, 300],
        'model__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'model__max_depth': [1, 2, 3, 4, 5],
        'model__min_samples_leaf': [1, 2, 4, 8, 12],
        'model__subsample': [0.6, 0.8, 1.0],
    },
    'XGBoost': {
        'model__n_estimators': [100, 200, 400, 600],
        'model__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'model__max_depth': [2, 3, 4, 5, 6],
        'model__min_child_weight': [1, 3, 5, 10],
        'model__subsample': [0.6, 0.8, 1.0],
        'model__colsample_bytree': [0.6, 0.8, 1.0],
        'model__reg_alpha': [0, 0.01, 0.1, 1.0],
        'model__reg_lambda': [0.5, 1, 2, 5, 10],
    },
    'CatBoost': {
        'model__iterations': [100, 200, 400, 600],
        'model__depth': [3, 4, 5, 6, 7, 8],
        'model__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'model__l2_leaf_reg': [1, 3, 5, 7, 10],
        'model__random_strength': [0, 0.5, 1, 2],
    },
}

N_ITER = int(os.environ.get('R2_N_ITER', '25'))
INNER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def tune_models(X_train, Y_train):
    print('RandomizedSearch iterations/model/outcome:', N_ITER)
    print('RandomizedSearch n_jobs:', SEARCH_N_JOBS)

    best_models = {t: {} for t in TARGETS}
    cv_rows = []
    hyperparam_rows = []

    for outcome in TARGETS:
        print('\n' + '=' * 90)
        print('OUTCOME:', outcome)
        print('=' * 90)
        ytr = Y_train[outcome]

        for model_name, base_model in MODELS.items():
            print(f'\nTuning {model_name} ...')
            pipe = ImbPipeline([
                ('imputer', clone(imputer)),
                ('ohe', clone(ohe)),
                ('oversample', RandomOverSampler(random_state=SEED)),
                ('model', clone(base_model)),
            ])

            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=PARAM_DISTS[model_name],
                n_iter=N_ITER,
                scoring=scoring,
                refit='average_precision',
                cv=INNER_CV,
                random_state=SEED,
                n_jobs=SEARCH_N_JOBS,
                return_train_score=False,
                verbose=0,
                error_score='raise',
            )
            search.fit(X_train, ytr)
            best_models[outcome][model_name] = search.best_estimator_

            idx = search.best_index_
            r = search.cv_results_
            row = {
                'Outcome': outcome,
                'Model': model_name,
                'CV_accuracy_mean': r['mean_test_accuracy'][idx],
                'CV_accuracy_sd': r['std_test_accuracy'][idx],
                'CV_balanced_accuracy_mean': r['mean_test_balanced_accuracy'][idx],
                'CV_balanced_accuracy_sd': r['std_test_balanced_accuracy'][idx],
                'CV_precision_pos_mean': r['mean_test_precision_pos'][idx],
                'CV_precision_pos_sd': r['std_test_precision_pos'][idx],
                'CV_recall_pos_mean': r['mean_test_recall_pos'][idx],
                'CV_recall_pos_sd': r['std_test_recall_pos'][idx],
                'CV_specificity_mean': r['mean_test_specificity'][idx],
                'CV_specificity_sd': r['std_test_specificity'][idx],
                'CV_gmean_mean': r['mean_test_gmean'][idx],
                'CV_gmean_sd': r['std_test_gmean'][idx],
                'CV_f1_pos_mean': r['mean_test_f1_pos'][idx],
                'CV_f1_pos_sd': r['std_test_f1_pos'][idx],
                'CV_AUROC_mean': r['mean_test_roc_auc'][idx],
                'CV_AUROC_sd': r['std_test_roc_auc'][idx],
                'CV_AUPRC_mean': r['mean_test_average_precision'][idx],
                'CV_AUPRC_sd': r['std_test_average_precision'][idx],
                'CV_log_loss_mean': -r['mean_test_neg_log_loss'][idx],
                'CV_log_loss_sd': r['std_test_neg_log_loss'][idx],
            }
            cv_rows.append(row)

            hyperparam_rows.append({
                'Outcome': outcome,
                'Model': model_name,
                'Best_parameters_JSON': json.dumps(search.best_params_, sort_keys=True, default=str),
                'Best_CV_AUPRC': search.best_score_,
            })
            print('Best CV AUPRC:', round(search.best_score_, 4))
            print('Best params:', search.best_params_)

    cv_summary = pd.DataFrame(cv_rows)
    best_hyperparameters = pd.DataFrame(hyperparam_rows)
    cv_summary.to_csv(OUTDIR / 'cv_metrics_summary.csv', index=False)
    best_hyperparameters.to_csv(OUTDIR / 'best_hyperparameters.csv', index=False)

    display(cv_summary)
    display(best_hyperparameters)
    return best_models, cv_summary, best_hyperparameters
