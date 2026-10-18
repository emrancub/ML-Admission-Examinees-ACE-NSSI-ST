import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from PIL import Image, ImageDraw, ImageFont

from .config import OUTDIR, FIG_DPI, safe_outcome_filename, maybe_show
from .data_preparation import TARGETS, feature_names

OUTCOME_DISPLAY = {
    'Self_Harm': 'Self-injury',
    'Self-injury': 'Self-injury',
    'Suicidal_Ideation': 'Suicidal thoughts',
    'Suicidal thoughts': 'Suicidal thoughts'
}


def get_display_outcome(outcome):
    return OUTCOME_DISPLAY.get(outcome, outcome)


def get_display_feature(col):
    return feature_names.get(col, col)


def _stitch_side_by_side(img_a_path, img_b_path, out_path, target_width=3600):
    if not (img_a_path.exists() and img_b_path.exists()):
        return
    im_a = Image.open(img_a_path).convert('RGB')
    im_b = Image.open(img_b_path).convert('RGB')
    h = max(im_a.height, im_b.height)
    pad_top = int(h * 0.02)
    pad_mid = int(im_a.width * 0.03)
    pad_side = int(im_a.width * 0.02)
    w = im_a.width + im_b.width + pad_mid + 2 * pad_side
    canvas = Image.new('RGB', (w, h + pad_top), (255, 255, 255))
    canvas.paste(im_a, (pad_side, pad_top))
    canvas.paste(im_b, (pad_side + im_a.width + pad_mid, pad_top))
    draw = ImageDraw.Draw(canvas)
    font_size = int(h * 0.038)
    font = None
    for font_candidate in ['arialbd.ttf', 'DejaVuSans-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']:
        try:
            font = ImageFont.truetype(font_candidate, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    draw.text((pad_side + 10, int(pad_top * 0.3)), '(A)', fill=(0, 0, 0), font=font)
    draw.text((pad_side + im_a.width + pad_mid + 10, int(pad_top * 0.3)), '(B)', fill=(0, 0, 0), font=font)
    if w > target_width:
        scale = target_width / float(w)
        canvas = canvas.resize((target_width, int(canvas.height * scale)), Image.Resampling.LANCZOS)
    canvas.save(out_path, dpi=(300, 300))


def encoded_feature_groups(fitted_pipeline, original_columns):
    """Map one-hot encoded model columns back to original predictors."""
    ohe = fitted_pipeline.named_steps['ohe']
    original_columns = list(original_columns)

    groups = []
    for col, cats in zip(original_columns, ohe.categories_):
        groups.extend([col] * len(cats))

    encoded_names = ohe.get_feature_names_out(original_columns)
    if len(groups) != len(encoded_names):
        raise RuntimeError(
            f'One-hot mapping error: {len(groups)} groups vs {len(encoded_names)} encoded features.'
        )
    return np.asarray(encoded_names), np.asarray(groups)


def catboost_shap_values(model, X_encoded):
    """Return a 2-D (samples x features) tree-path-dependent SHAP matrix."""
    if hasattr(X_encoded, 'toarray'):
        X_encoded = X_encoded.toarray()

    explainer = shap.TreeExplainer(model)
    try:
        explanation = explainer(X_encoded)
        sv = explanation.values if hasattr(explanation, 'values') else explanation
    except Exception:
        sv = explainer.shap_values(X_encoded)

    if isinstance(sv, list):
        sv = sv[-1]

    sv = np.asarray(sv)
    if sv.ndim == 3:
        if sv.shape[-1] in (1, 2):
            sv = sv[:, :, -1]
        elif sv.shape[0] in (1, 2):
            sv = sv[-1, :, :]

    if sv.ndim != 2:
        raise RuntimeError(f'Unexpected SHAP array shape: {sv.shape}')
    return sv


def make_numeric_feature_values(X_imputed, fitted_pipeline, original_columns):
    """Creates numeric category codes used ONLY for SHAP beeswarm dot colors."""
    X_imputed = np.asarray(X_imputed)
    ohe = fitted_pipeline.named_steps['ohe']

    n_samples = X_imputed.shape[0]
    n_features = len(original_columns)
    result = np.zeros((n_samples, n_features), dtype=float)

    for j, (col, cats) in enumerate(zip(original_columns, ohe.categories_)):
        values = pd.Series(X_imputed[:, j])
        numeric_values = pd.to_numeric(values, errors='coerce')

        if numeric_values.notna().all():
            result[:, j] = numeric_values.to_numpy(dtype=float)
        else:
            category_map = {cat: i + 1 for i, cat in enumerate(cats)}
            mapped = values.map(category_map)
            if mapped.isna().any():
                category_map_str = {str(cat): i + 1 for i, cat in enumerate(cats)}
                mapped = values.astype(str).map(category_map_str)
            result[:, j] = mapped.astype(float)

    return result


def aggregate_shap_to_original_features(shap_encoded, encoded_groups, original_columns):
    """
    By SHAP additivity, sum SHAP values across one-hot columns belonging to each
    original predictor for each participant before computing mean(|SHAP|).
    """
    n_samples = shap_encoded.shape[0]
    grouped_shap = np.zeros((n_samples, len(original_columns)), dtype=float)

    for j, predictor in enumerate(original_columns):
        idx = np.where(encoded_groups == predictor)[0]
        if len(idx) == 0:
            raise RuntimeError(f'No encoded columns found for predictor: {predictor}')
        grouped_shap[:, j] = shap_encoded[:, idx].sum(axis=1)

    return grouped_shap


def run_shap_summary_plots(best_models, X, X_test):
    """
    Compute CatBoost SHAP values using a single unified aggregation definition:
    participant-level sum across one-hot columns per predictor, followed by mean(|SHAP|).
    Saves both beeswarm summary plots (SHAP_Summary_*.png), bar plots (SHAP_*.png),
    grouped tables (SHAP_grouped_*.csv), and encoded-level tables (SHAP_encoded_*.csv).
    """
    shap_tables = []
    original_columns = list(X.columns)

    for outcome in TARGETS:
        pipe = best_models[outcome]['CatBoost']
        cb_model = pipe.named_steps['model']

        Xte_imp = pipe.named_steps['imputer'].transform(X_test)
        Xte_enc = pipe.named_steps['ohe'].transform(Xte_imp)
        if hasattr(Xte_enc, 'toarray'):
            Xte_enc = Xte_enc.toarray()

        encoded_names, encoded_groups = encoded_feature_groups(pipe, original_columns)
        shap_encoded = catboost_shap_values(cb_model, Xte_enc)

        # Save encoded-level SHAP table separately (never overwriting grouped table)
        outcome_file = safe_outcome_filename(get_display_outcome(outcome))
        encoded_df = pd.DataFrame({
            'Outcome': get_display_outcome(outcome),
            'Encoded_feature': encoded_names,
            'Predictor': [get_display_feature(g) for g in encoded_groups],
            'Mean_abs_SHAP_encoded': np.abs(shap_encoded).mean(axis=0),
        }).sort_values('Mean_abs_SHAP_encoded', ascending=False)
        encoded_df.to_csv(OUTDIR / f'SHAP_encoded_{outcome_file}.csv', index=False)

        # Unified participant-level additive grouping across one-hot columns
        grouped_shap = aggregate_shap_to_original_features(
            shap_encoded, encoded_groups, original_columns
        )
        grouped_feature_values = make_numeric_feature_values(
            Xte_imp, pipe, original_columns
        )
        display_feature_names = [get_display_feature(col) for col in original_columns]

        mean_abs_shap = np.abs(grouped_shap).mean(axis=0)
        grouped_table = pd.DataFrame({
            'Outcome': get_display_outcome(outcome),
            'Predictor': display_feature_names,
            'Mean_abs_SHAP': mean_abs_shap,
        }).sort_values('Mean_abs_SHAP', ascending=False)
        shap_tables.append(grouped_table)
        grouped_table.to_csv(OUTDIR / f'SHAP_grouped_{outcome_file}.csv', index=False)

        # 1. Bar plot of Mean |SHAP|
        top = grouped_table.head(15).sort_values('Mean_abs_SHAP')
        fig_bar, ax_bar = plt.subplots(figsize=(9, 7))
        ax_bar.barh(top['Predictor'], top['Mean_abs_SHAP'])
        ax_bar.set_xlabel('Mean absolute SHAP value')
        ax_bar.set_ylabel('Predictor')
        ax_bar.set_title(f'CatBoost SHAP importance for {get_display_outcome(outcome)}')
        fig_bar.tight_layout()
        fig_bar.savefig(OUTDIR / f'SHAP_{outcome_file}.png', dpi=FIG_DPI, bbox_inches='tight')
        plt.close(fig_bar)

        # 2. Beeswarm Summary Plot
        plt.close('all')
        shap.summary_plot(
            grouped_shap,
            features=grouped_feature_values,
            feature_names=display_feature_names,
            max_display=20,
            plot_type='dot',
            plot_size=(10.5, 11.5),
            show=False
        )
        fig = plt.gcf()
        ax = fig.axes[0]
        ax.set_title(f'SHAP Summary Plot for {get_display_outcome(outcome)}', fontsize=17, pad=14)
        ax.set_xlabel('SHAP value (impact on model output)', fontsize=15)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=13)
        ax.axvline(x=0, linewidth=0.9, alpha=0.65, zorder=0)
        ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.35)
        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.savefig(OUTDIR / f'SHAP_Summary_{outcome_file}.png', dpi=FIG_DPI, bbox_inches='tight', facecolor='white')
        maybe_show()
        plt.close(fig)

    _stitch_side_by_side(
        OUTDIR / 'SHAP_Summary_Self-injury.png',
        OUTDIR / 'SHAP_Summary_Suicidal_thoughts.png',
        OUTDIR / 'Figure2_SHAP_Summary_Combined.png'
    )

    all_shap = pd.concat(shap_tables, ignore_index=True)
    all_shap.to_csv(OUTDIR / 'SHAP_grouped_all_outcomes.csv', index=False)
    return shap_tables, all_shap


def run_grouped_shap_importance(best_models, X, X_test):
    """Backward-compatible wrapper returning unified grouped SHAP tables."""
    return run_shap_summary_plots(best_models, X, X_test)


def run_xgboost_gain_importance(best_models, X):
    xgb_tables = []

    for outcome in TARGETS:
        pipe = best_models[outcome]['XGBoost']
        model = pipe.named_steps['model']
        enc_names, groups = encoded_feature_groups(pipe, X.columns)
        raw = model.get_booster().get_score(importance_type='gain')

        gains = np.zeros(len(enc_names), dtype=float)
        for key, value in raw.items():
            if key.startswith('f') and key[1:].isdigit():
                idx = int(key[1:])
                if idx < len(gains):
                    gains[idx] = float(value)
            else:
                matches = np.where(enc_names == key)[0]
                if len(matches):
                    gains[matches[0]] = float(value)

        t = pd.DataFrame({
            'Encoded_feature': enc_names,
            'Predictor': [get_display_feature(g) for g in groups],
            'Gain': gains,
        })
        grouped = (
            t.groupby('Predictor', as_index=False)['Gain']
             .sum()
             .sort_values('Gain', ascending=False)
        )
        grouped.insert(0, 'Outcome', outcome)
        xgb_tables.append(grouped)
        outcome_file = safe_outcome_filename(outcome)
        grouped.to_csv(OUTDIR / f'XGBoost_gain_grouped_{outcome_file}.csv', index=False)

        top = grouped.head(15).sort_values('Gain')
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(top['Predictor'], top['Gain'])
        ax.set_xlabel('Gain')
        ax.set_ylabel('Predictor')
        ax.set_title(f'XGBoost gain-based importance for {outcome}')
        fig.tight_layout()
        fig.savefig(OUTDIR / f'XGBoost_gain_{outcome_file}.png', dpi=FIG_DPI, bbox_inches='tight')
        maybe_show()
        plt.close(fig)

    _stitch_side_by_side(
        OUTDIR / 'XGBoost_gain_Self-injury.png',
        OUTDIR / 'XGBoost_gain_Suicidal_thoughts.png',
        OUTDIR / 'Figure3_XGBoost_Gain_Combined.png'
    )

    all_xgb = pd.concat(xgb_tables, ignore_index=True)
    all_xgb.to_csv(OUTDIR / 'XGBoost_gain_grouped_all_outcomes.csv', index=False)
    return xgb_tables, all_xgb
