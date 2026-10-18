import math
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, f1_score,
    log_loss, roc_auc_score, average_precision_score, confusion_matrix
)

from .config import OUTDIR, display
from .data_preparation import TARGETS


def holdout_metrics(y_true, y_prob, threshold=0.50):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp/(tp+fn) if (tp+fn) else np.nan
    spec = tn/(tn+fp) if (tn+fp) else np.nan
    return {
        'N_test': len(y_true),
        'Positive_test_n': int(np.sum(y_true)),
        'Threshold': threshold,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision_positive': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Recall_sensitivity': sens,
        'Specificity': spec,
        'G_mean': math.sqrt(max(sens,0)*max(spec,0)) if not (np.isnan(sens) or np.isnan(spec)) else np.nan,
        'F1_positive': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'AUROC': roc_auc_score(y_true, y_prob),
        'AUPRC': average_precision_score(y_true, y_prob),
        'Log_loss': log_loss(y_true, np.column_stack([1-np.asarray(y_prob), np.asarray(y_prob)]), labels=[0,1]),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
    }


def evaluate_models(best_models, X_test, Y_test, cv_summary):
    holdout_rows = []
    probabilities = {t:{} for t in TARGETS}
    predictions = {t:{} for t in TARGETS}

    for outcome in TARGETS:
        yte = Y_test[outcome]
        for model_name, model in best_models[outcome].items():
            prob = model.predict_proba(X_test)[:,1]
            pred = (prob >= 0.50).astype(int)
            probabilities[outcome][model_name] = prob
            predictions[outcome][model_name] = pred
            m = holdout_metrics(yte.to_numpy(), prob, 0.50)
            holdout_rows.append({'Outcome':outcome,'Model':model_name,**m})

    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df.to_csv(OUTDIR/'test_metrics.csv', index=False)
    display(holdout_df.sort_values(['Outcome','AUPRC'], ascending=[True,False]))

    cols = ['Outcome','Model','Accuracy','Balanced_accuracy','Precision_positive','Recall_sensitivity',
            'Specificity','G_mean','F1_positive','AUROC','AUPRC','Log_loss']
    review_table = holdout_df[cols].copy()
    for c in cols[2:]:
        review_table[c] = review_table[c].astype(float).round(4)
    review_table.to_csv(OUTDIR/'reviewer_ready_test_table.csv', index=False)
    display(review_table)

    cv_display_cols = ['Outcome','Model','CV_balanced_accuracy_mean','CV_precision_pos_mean','CV_recall_pos_mean',
                       'CV_specificity_mean','CV_gmean_mean','CV_f1_pos_mean','CV_AUROC_mean','CV_AUPRC_mean','CV_log_loss_mean']
    cv_review_table = cv_summary[cv_display_cols].copy().round(4)
    cv_review_table.to_csv(OUTDIR/'reviewer_ready_cv_table.csv', index=False)
    display(cv_review_table)

    return holdout_df, probabilities, predictions, review_table, cv_review_table
