import argparse
from src.config import print_runtime_banner
from src.data_preparation import load_and_prepare_data
from src.ace_audit import run_ace_dose_response_audit
from src.statistical_tables_audit import run_statistical_tables_audit


def main():
    parser = argparse.ArgumentParser(
        description="Rmachine-learning pipeline for Bangladeshi university admission examinees study."
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Run data cleaning, missingness audit, conventional statistical tables (Tables 1-5), and ACE dose-response audit only.",
    )
    args = parser.parse_args()

    print_runtime_banner()

    (
        df_raw, df, X, Y,
        X_train, X_test, Y_train, Y_test,
        missing_audit, audit_df, joint_strata
    ) = load_and_prepare_data()

    if args.stats_only:
        print("\nCompleted statistical, missingness, and ACE audits (--stats-only mode).")
        return

    from src.modeling import tune_models
    from src.evaluation import evaluate_models
    from src.plots import plot_roc_pr, plot_confusion_matrices
    from src.explainability import run_shap_summary_plots, run_xgboost_gain_importance
    from src.archive import save_models_and_reproducibility, create_outputs_zip

    best_models, cv_summary, best_hyperparameters = tune_models(X_train, Y_train)

    holdout_df, probabilities, predictions, review_table, cv_review_table = evaluate_models(
        best_models, X_test, Y_test, cv_summary
    )

    plot_roc_pr(probabilities, X_test, Y_test)
    plot_confusion_matrices(predictions, Y_test)

    run_shap_summary_plots(best_models, X, X_test)
    run_xgboost_gain_importance(best_models, X)

    save_models_and_reproducibility(best_models)
    create_revision_outputs_zip()
    print("\nFull machine-learning and statistical pipeline completed successfully.")


if __name__ == "__main__":
    main()
