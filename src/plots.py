import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix
)

from .config import OUTDIR, FIG_DPI, CONFUSION_CMAP, safe_outcome_filename, maybe_show
from .data_preparation import TARGETS


def plot_roc_pr(probabilities, X_test, Y_test):
    for outcome in TARGETS:
        yte = Y_test[outcome].to_numpy()
        outcome_file = safe_outcome_filename(outcome)

        plt.figure(figsize=(8, 6))
        for model_name, prob in probabilities[outcome].items():
            fpr, tpr, _ = roc_curve(yte, prob)
            aucv = roc_auc_score(yte, prob)
            plt.plot(fpr, tpr, label=f'{model_name} (AUROC={aucv:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Chance')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate (sensitivity)')
        plt.title(f'ROC curves for {outcome}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(OUTDIR / f'ROC_{outcome_file}.png', dpi=FIG_DPI, bbox_inches='tight')
        maybe_show()
        plt.close()

        plt.figure(figsize=(8, 6))
        baseline = yte.mean()
        for model_name, prob in probabilities[outcome].items():
            precision, recall, _ = precision_recall_curve(yte, prob)
            ap = average_precision_score(yte, prob)
            plt.plot(recall, precision, label=f'{model_name} (AUPRC={ap:.3f})')
        plt.axhline(baseline, linestyle='--', label=f'Prevalence baseline={baseline:.3f}')
        plt.xlabel('Recall (sensitivity)')
        plt.ylabel('Precision')
        plt.title(f'Precision-recall curves for {outcome}')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(OUTDIR / f'PR_{outcome_file}.png', dpi=FIG_DPI, bbox_inches='tight')
        maybe_show()
        plt.close()

    # Combined 4-panel Figure 1 (A, B: ROC; C, D: PR)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    panel_labels_roc = ['(A)', '(B)']
    panel_labels_pr = ['(C)', '(D)']

    for idx, outcome in enumerate(TARGETS):
        yte = Y_test[outcome].to_numpy()
        ax_roc = axes[0, idx]
        for model_name, prob in probabilities[outcome].items():
            fpr, tpr, _ = roc_curve(yte, prob)
            aucv = roc_auc_score(yte, prob)
            ax_roc.plot(fpr, tpr, label=f'{model_name} (AUROC={aucv:.3f})')
        ax_roc.plot([0, 1], [0, 1], linestyle='--', color='orchid', label='Chance')
        ax_roc.set_xlabel('False positive rate')
        ax_roc.set_ylabel('True positive rate (sensitivity)')
        ax_roc.set_title(f'ROC curves for {outcome}', fontsize=13)
        ax_roc.legend(loc='lower right', fontsize=9.5)
        ax_roc.text(-0.12, 1.05, panel_labels_roc[idx], transform=ax_roc.transAxes, fontsize=16, fontweight='bold')

        ax_pr = axes[1, idx]
        baseline = yte.mean()
        for model_name, prob in probabilities[outcome].items():
            precision, recall, _ = precision_recall_curve(yte, prob)
            ap = average_precision_score(yte, prob)
            ax_pr.plot(recall, precision, label=f'{model_name} (AUPRC={ap:.3f})')
        ax_pr.axhline(baseline, linestyle='--', color='steelblue', label=f'Prevalence baseline={baseline:.3f}')
        ax_pr.set_xlabel('Recall (sensitivity)')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title(f'Precision-recall curves for {outcome}', fontsize=13)
        ax_pr.legend(loc='upper right', fontsize=9.5)
        ax_pr.text(-0.12, 1.05, panel_labels_pr[idx], transform=ax_pr.transAxes, fontsize=16, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUTDIR / 'Figure1_ROC_PR_Combined.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrices(predictions, Y_test):
    for outcome in TARGETS:
        yte = Y_test[outcome].to_numpy()
        outcome_file = safe_outcome_filename(outcome)

        for model_name, pred in predictions[outcome].items():
            cm = confusion_matrix(yte, pred, labels=[0, 1])

            fig, ax = plt.subplots(figsize=(5.4, 4.6))
            im = ax.imshow(cm, interpolation='nearest', cmap=CONFUSION_CMAP)

            ax.set_xticks([0, 1], labels=['No', 'Yes'])
            ax.set_yticks([0, 1], labels=['No', 'Yes'])
            ax.set_xlabel('Predicted class')
            ax.set_ylabel('Actual class')
            ax.set_title(f'{outcome} — {model_name}')

            threshold = cm.max() / 2.0 if cm.size else 0
            for i in range(2):
                for j in range(2):
                    text_color = 'white' if cm[i, j] > threshold else 'black'
                    ax.text(
                        j, i, f'{cm[i, j]:d}',
                        ha='center', va='center',
                        color=text_color, fontsize=13, fontweight='bold'
                    )

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Number of participants')
            fig.tight_layout()

            safe_model = safe_outcome_filename(model_name)
            fig.savefig(
                OUTDIR / f'CM_{outcome_file}_{safe_model}.png',
                dpi=FIG_DPI,
                bbox_inches='tight'
            )
            maybe_show()
            plt.close(fig)
