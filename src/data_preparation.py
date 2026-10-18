import math
import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None

from .config import DATA_PATH, OUTDIR, SEED, display

SOURCE_OUTCOME_RENAME = {
    'Self_Harm': 'Self-injury',
    'Suicidal_Ideation': 'Suicidal thoughts',
}

feature_names = {
    'Gender': 'Gender',
    'Residence': 'Permanent residence',
    'Family_type': 'Family type',
    'F_Education': 'Father’s education',
    'M_Education': 'Mother’s education',
    'MFI_Cat': 'Monthly family income (BDT)',
    'Health_Status': 'Self-reported health status',
    'Health_Problems': 'Self-reported health problems',
    'Test_Status': 'Appearance in the admission test',
    'SSC_Cat': 'GPA in SSC',
    'HSC_Cat': 'GPA in HSC',
    'Coaching_Type': 'Coached by professional coaching centers',
    'Prep_Satisfaction': 'Satisfied with previous mock test performance',
    'Desired_Institute': '-',
    'Add_Expenditure': 'Monthly expenditure during admission test (BDT)',
    'HSC_Background': 'Educational background',
    'Child_abuse1_1': 'Psychological abuse: Item 1',
    'Child_abuse1_2': 'Psychological abuse: Item 2',
    'Child_abuse2_1': 'Physical abuse: Item 1',
    'Child_abuse2_2': 'Physical abuse: Item 2',
    'Child_abuse3_1': 'Sexual abuse: Item 1',
    'Child_abuse3_2': 'Sexual abuse: Item 2',
    'Child_abuse4_1': 'Mental illness: Item 1',
    'Child_abuse4_2': 'Mental illness: Item 2',
    'Child_abuse5_1': 'Mother treated violently: Item 1',
    'Child_abuse5_2': 'Mother treated violently: Item 2',
    'Child_abuse5_3': 'Mother treated violently: Item 3',
    'Child_abuse5_4': 'Mother treated violently: Item 4',
    'Child_abuse6_1': 'Criminal behavior: Item 1',
    'Child_abuse7_1': 'Substance abuse: Item 1',
    'Child_abuse7_2': 'Substance abuse: Item 2',
}

PERMITTED_CODES = {
    'Gender': {'1', '2'},
    'Permanent residence': {'1', '2'},
    'Family type': {'1', '2'},
    'Father’s education': {'1', '2', '3'},
    'Mother’s education': {'1', '2', '3'},
    'Monthly family income (BDT)': {'1', '2', '3'},
    'Self-reported health status': {'1', '2'},
    'Self-reported health problems': {'0', '1'},
    'Appearance in the admission test': {'1', '2'},
    'GPA in SSC': {'1', '2', '3'},
    'GPA in HSC': {'1', '2', '3'},
    'Coached by professional coaching centers': {'1', '2'},
    'Satisfied with previous mock test performance': {'0', '1'},
    'Monthly expenditure during admission test (BDT)': {'1', '2', '3'},
    'Educational background': {'1', '2', '3'},
    **{f'Psychological abuse: Item {i}': {'0', '1'} for i in (1, 2)},
    **{f'Physical abuse: Item {i}': {'0', '1'} for i in (1, 2)},
    **{f'Sexual abuse: Item {i}': {'0', '1'} for i in (1, 2)},
    **{f'Mental illness: Item {i}': {'0', '1'} for i in (1, 2)},
    **{f'Mother treated violently: Item {i}': {'0', '1'} for i in (1, 2, 3, 4)},
    'Criminal behavior: Item 1': {'0', '1'},
    **{f'Substance abuse: Item {i}': {'0', '1'} for i in (1, 2)},
}

COLUMN_RENAME = {**SOURCE_OUTCOME_RENAME, **feature_names}
TARGETS = ['Self-injury', 'Suicidal thoughts']
DROP_COLUMNS = ['-']
POSITIVE_LABEL_OVERRIDES = {}


def clean_categorical_series(s: pd.Series, col_name: str = '') -> pd.Series:
    """
    Trim whitespace, convert empty or whitespace-only strings to np.nan,
    normalize integer-like strings (e.g. '1.0' -> '1'), and validate permitted codes.
    """
    out = pd.Series(np.nan, index=s.index, dtype='object')
    for idx, val in s.items():
        if pd.isna(val):
            continue
        txt = str(val).strip()
        if txt == '' or txt.lower() in {'nan', 'none', 'null', 'na'}:
            continue
        try:
            fval = float(txt)
            if fval.is_integer():
                txt = str(int(fval))
        except ValueError:
            pass
        out.at[idx] = txt

    if col_name in PERMITTED_CODES:
        observed = set(out.dropna().unique())
        allowed = PERMITTED_CODES[col_name]
        invalid = observed - allowed
        if invalid:
            raise ValueError(
                f'Column {col_name!r} contains unexpected code(s) {sorted(invalid)}; '
                f'permitted codes: {sorted(allowed)}'
            )
    return out


def coerce_binary_target(s: pd.Series, name: str, positive_override=None) -> pd.Series:
    """Convert common binary encodings to 0/1 without arbitrary label ordering."""
    x = clean_categorical_series(s, name)
    if x.isna().any():
        raise ValueError(f'{name} contains missing values; resolve them before modeling.')

    vals = set(pd.unique(x))
    if vals.issubset({'0', '1'}):
        return x.astype(int)

    if positive_override is not None:
        uniq = pd.unique(x)
        if str(positive_override) not in uniq:
            raise ValueError(f'Positive override {positive_override!r} not found in {name}: {uniq}')
        return (x == str(positive_override)).astype(int)

    low = x.astype(str).str.strip().str.lower()
    pos = {'1', 'yes', 'y', 'true', 'positive', 'present', 'case', 'self harm', 'self-harm', 'suicidal ideation', 'suicidal thoughts'}
    neg = {'0', 'no', 'n', 'false', 'negative', 'absent', 'control', 'none'}
    observed = set(low.unique())
    if observed.issubset(pos | neg):
        return low.map(lambda v: 1 if v in pos else 0).astype(int)

    raise ValueError(f'{name} is not binary. Unique values: {sorted(observed)}')


def wilson_interval(k, n, alpha=0.05):
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return np.nan, np.nan
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_and_prepare_data():
    print('Using dataset:', DATA_PATH)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'Dataset not found: {DATA_PATH}')

    df_raw = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=True)
    df_raw = df_raw.rename(columns=COLUMN_RENAME)

    duplicated_labels = df_raw.columns[df_raw.columns.duplicated()].tolist()
    if duplicated_labels:
        raise ValueError(f'Duplicate column labels after manuscript-facing renaming: {duplicated_labels}')

    for c in df_raw.columns:
        df_raw[c] = clean_categorical_series(df_raw[c], c)

    print('Raw shape:', df_raw.shape)

    missing_targets = [c for c in TARGETS if c not in df_raw.columns]
    if missing_targets:
        raise KeyError(f'Missing required target columns: {missing_targets}')

    df = df_raw.copy()
    for c in DROP_COLUMNS:
        if c in df.columns:
            df = df.drop(columns=c)

    Y = pd.DataFrame(index=df.index)
    for t in TARGETS:
        Y[t] = coerce_binary_target(df[t], t, POSITIVE_LABEL_OVERRIDES.get(t))

    # Convert cleaned categorical codes to float64 (with np.nan for missing cells)
    # so SimpleImputer and OneHotEncoder work seamlessly across pandas 2.x/3.x and numpy 2.x
    X = df.drop(columns=TARGETS).copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce').astype(float)

    n_total_missing_cells = int(X.isna().sum().sum())
    n_rows_with_missing = int(X.isna().any(axis=1).sum())
    n_complete_rows = int(X.notna().all(axis=1).sum())

    print('Final rows available for ML:', len(X))
    print('Number of predictors before one-hot encoding:', X.shape[1])
    print(f'Total missing cells across 30 predictors: {n_total_missing_cells} (in {n_rows_with_missing} participants; {n_complete_rows} complete cases)')

    missing_audit = pd.DataFrame({
        'Variable': X.columns,
        'Missing_n': [int(X[c].isna().sum()) for c in X.columns],
        'Missing_percent': [round(100.0 * float(X[c].isna().mean()), 4) for c in X.columns],
        'Observed_n': [int(X[c].notna().sum()) for c in X.columns],
    }).sort_values(['Missing_n', 'Variable'], ascending=[False, True])
    missing_audit.to_csv(OUTDIR / 'predictor_missingness_audit.csv', index=False)
    print('Predictors with missing values:', int((missing_audit['Missing_n'] > 0).sum()))
    display(missing_audit[missing_audit['Missing_n'] > 0])

    rows = []
    for t in TARGETS:
        n = len(Y)
        k = int(Y[t].sum())
        p = k / n
        lo, hi = wilson_interval(k, n, alpha=0.05)
        print(f'{t}: positives={k}, negatives={n - k}, prevalence={p:.4f} ({100 * p:.2f}%)')
        rows.append({
            'Outcome': t,
            'N': n,
            'Positive_n': k,
            'Prevalence': p,
            'Prevalence_percent': round(100.0 * p, 2),
            'Wilson_95CI_low': lo,
            'Wilson_95CI_high': hi,
            'Approx_margin_of_error': (hi - lo) / 2,
            'Majority_class_accuracy_baseline': max(p, 1 - p),
        })

    audit_df = pd.DataFrame(rows)
    display(audit_df)
    audit_df.to_csv(OUTDIR / 'data_audit.csv', index=False)

    joint_strata = Y[TARGETS].astype(str).agg('_'.join, axis=1)
    if joint_strata.value_counts().min() < 2:
        raise ValueError('At least one joint outcome stratum has <2 observations; joint stratification is not possible.')

    if train_test_split is not None:
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y,
            test_size=0.20,
            random_state=SEED,
            stratify=joint_strata
        )
    else:
        rng = np.random.RandomState(SEED)
        train_idx, test_idx = [], []
        for _, grp in X.groupby(joint_strata):
            idxs = grp.index.to_numpy().copy()
            rng.shuffle(idxs)
            n_te = int(round(len(idxs) * 0.20))
            test_idx.extend(idxs[:n_te])
            train_idx.extend(idxs[n_te:])
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        Y_train, Y_test = Y.loc[train_idx], Y.loc[test_idx]

    print('Train N:', len(X_train), 'Test N:', len(X_test))
    split_audit = []
    for split_name, yy in [('train', Y_train), ('test', Y_test)]:
        for t in TARGETS:
            split_audit.append({
                'split': split_name,
                'outcome': t,
                'N': len(yy),
                'positive_n': int(yy[t].sum()),
                'prevalence': yy[t].mean(),
            })
    pd.DataFrame(split_audit).to_csv(OUTDIR / 'split_audit.csv', index=False)

    split_indices = pd.DataFrame({
        'row_index': df.index,
        'split': np.where(df.index.isin(X_train.index), 'train', 'test'),
        'Self_injury': Y['Self-injury'].to_numpy(),
        'Suicidal_thoughts': Y['Suicidal thoughts'].to_numpy(),
        'joint_stratum': joint_strata.to_numpy(),
    })
    split_indices.to_csv(OUTDIR / 'split_indices.csv', index=False)

    return df_raw, df, X, Y, X_train, X_test, Y_train, Y_test, missing_audit, audit_df, joint_strata
