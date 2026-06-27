"""
brfss_utils.py
==============
Shared BRFSS data utilities for the myocardial-infarction modeling project.

This module owns everything about *reading and labeling* BRFSS data:
constants, the display-label lookups, the humanize() helper, data loading
into a feature/target frame, and the standalone EDA helpers (missingness
and outlier analysis).

Display-only rule: the readable labels here are used for plot labels and
tables ONLY. They are never applied to the model data itself — feature
selection, encoding, and the model all operate on the original BRFSS codes.

Maintained by: Maria
"""

import pickle

import numpy as np
import pandas as pd


# =====================================================================
# CONSTANTS
# =====================================================================

RANDOM_STATE = 42          # fixed seed for reproducible splits, sampling, and models
CV_FOLDS     = 5           # folds for cross-validation
TARGET       = "_MICHD"    # binary target: history of MI / coronary heart disease


# =====================================================================
# DISPLAY LABELS  (display-only — never applied to model data)
# =====================================================================

# Maps each of the 27 selected feature codes to a readable label.
# Definitions follow the BRFSS 2024 codebook (USCODE24_LLCP_082125.HTML).
LABELS = {
    "_AGEG5YR": "Age group",
    "_RFHLTH":  "General health",
    "_INCOMG1": "Income group",
    "MARITAL":  "Marital status",
    "PRIMINS2": "Primary insurance",        # note: Medicare acts as an age proxy
    "VETERAN3": "Veteran",
    "PREGNANT": "Pregnant",
    "_SMOKER3": "Smoking status",
    "_LCSYSMK": "Years smoked",
    "LCSNUMCG": "Cigarettes per day",
    "_LCSYQTS": "Years since quitting",
    "DIABETE4": "Diabetes",
    "CVDSTRK3": "Stroke history",           # most irreplaceable feature by ablation
    "CHCCOPD3": "COPD",
    "CHCKDNY2": "Kidney disease",
    "HAVARTH4": "Arthritis",
    "CHCOCNC1": "Cancer history",
    "DIFFWALK": "Difficulty walking",
    "DIFFDRES": "Difficulty dressing",
    "DIFFALON": "Difficulty doing errands alone",
    "DEAF":     "Hearing difficulty",
    "PHYSHLTH": "Physically unhealthy days",   # continuous
    "POORHLTH": "Poor-health days",            # continuous
    "CHECKUP1": "Last checkup",
    "PERSDOC3": "Has personal doctor",
    "PNEUVAC4": "Pneumonia vaccine",
    "_TOTINDA": "Physical activity",           # also the uplift treatment variable
}

# BRFSS _AGEG5YR code -> readable age range (display-only)
AGE_LABELS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
    6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
    11: "70-74", 12: "75-79", 13: "80+", 14: "Unknown",
}


def humanize(name):
    """Convert a BRFSS feature code into a human-readable label.

    Handles two cases:
      1. Exact match -> "_AGEG5YR"                  -> "Age group"
      2. One-hot column from the ColumnTransformer, where the encoder
         appends the category value after an underscore:
                      -> "_RFHLTH_Fair or Poor Health"
                      -> "General health = Fair or Poor Health"

    Codes are checked longest-first so a longer, more specific code is
    matched before a shorter code that is a prefix of it. Any name with
    no matching code is returned unchanged, so this is safe to map over
    a full list of (possibly already-clean) column names.

    Note: strip the ColumnTransformer lane prefix ("categorical__" /
    "continuous__") BEFORE calling, or the lookup will not match.
    """
    # Longest codes first -> avoids a short code matching a longer code's column
    for code in sorted(LABELS, key=len, reverse=True):
        if name == code:                       # exact feature match
            return LABELS[code]
        if name.startswith(code + "_"):        # one-hot column: "CODE_<value>"
            return f"{LABELS[code]} = {name[len(code) + 1:]}"
    return name                                # unknown -> leave unchanged


# =====================================================================
# DATA LOADING
# =====================================================================

def load_data(data_path, features_path):
    """Load the processed BRFSS dataset and the selected-feature list.

    Builds the categorical/continuous column lists used by the modeling
    pipeline (only _MICHD, the target, is binary), and
    drops rows with a missing target.

    Parameters
    ----------
    data_path : str
        Path to the processed CSV (e.g. processed_data_v2.csv on Drive).
    features_path : str
        Path to the selected-features pickle (e.g. selected_features_v2.pkl).

    Returns
    -------
    dict with keys:
        df               : the loaded DataFrame (rows with NaN target dropped)
        FEATURES         : the 27 selected features that survived selection
        categorical_cols : all categorical columns in df (full dataset)
        continuous_cols  : all continuous columns in df (full dataset)
        cat_features     : selected features that are categorical
        cont_features    : selected features that are continuous
    """
    # --- Load the processed dataset (cleaning/selection done upstream) ---
    df = pd.read_csv(data_path, low_memory=False)

    # --- Load the selected features (two-stage significance +
    #     multicollinearity selection from the EDA notebook) -----------
    with open(features_path, "rb") as f:
        selected_features = pickle.load(f)
    FEATURES = list(selected_features)   # the 27 features that survived selection

    # --- Build the two feature-type lanes for the ColumnTransformer ------
    # Logic mirrors the EDA notebook
    # Categorical: object/category dtype OR low-cardinality numeric (<10 unique).
    categorical_cols = [
        c for c in df.columns
        if c != TARGET
        and (df[c].dtype == "object" or df[c].dtype.name == "category" or df[c].nunique() < 10)
    ]
    # Continuous: remaining numeric columns that aren't categorical.
    continuous_cols = [
        c for c in df.select_dtypes("number").columns
        if c != TARGET
        and c not in categorical_cols
    ]

    # --- Drop rows with a missing target (unusable for supervised learning) ---
    df = df[df[TARGET].notna()]

    # --- Separate the SELECTED features by type (excludes target) --------
    cat_features  = [f for f in FEATURES if f in categorical_cols]
    cont_features = [f for f in FEATURES if f in continuous_cols]

    return {
        "df": df,
        "FEATURES": FEATURES,
        "categorical_cols": categorical_cols,
        "continuous_cols": continuous_cols,
        "cat_features": cat_features,
        "cont_features": cont_features,
    }


# =====================================================================
# EDA HELPERS
# =====================================================================

def find_missingness_drivers(df, missing_threshold=0.10, top_n=3, min_group=200):
    """For each high-missing feature, find which column best explains its missingness.

    Determines whether missingness is random or structural by checking, for
    each feature missing >= `missing_threshold`, which other column most
    sharply splits its missing/not-missing pattern.
    """
    # Fraction of rows missing in each column (True=1/False=0, so .mean() = % missing)
    miss_rate = df.isna().mean()

    # Keep only features missing at least `missing_threshold` — worth investigating
    high_missing = miss_rate[miss_rate >= missing_threshold].index.tolist()

    summary = []
    for feature in high_missing:

        is_missing = df[feature].isna()

        candidates = []
        for col in df.columns:
            if col == feature:          # don't test a feature against itself
                continue

            # Split the missingness flag into groups defined by the candidate driver
            grp = is_missing.groupby(df[col])

            rates  = grp.mean()         # within each group, what % is missing
            counts = grp.size()         # how many people are in each group

            # Drop tiny groups — a 100% rate from 3 people is noise, not signal
            rates = rates[counts >= min_group]
            if len(rates) < 2:          # need at least 2 groups to compare
                continue

            # The key metric: how far apart are the highest and lowest missing-rates?
            # Big spread = this column strongly controls whether the feature is missing
            spread = rates.max() - rates.min()
            candidates.append((col, round(spread, 3)))

        # Rank candidates so the strongest driver is first
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:top_n]

        summary.append({
            'feature': feature,
            'pct_missing': round(miss_rate[feature] * 100, 1),
            'top_driver': top[0][0] if top else None,      # best explainer
            'driver_spread': top[0][1] if top else None,   # how clean the split is
            # spread >= 0.7 means missingness flips ~0% to ~100% across groups = skip logic
            'verdict': 'STRUCTURAL' if top and top[0][1] >= 0.7 else 'check',
            'runners_up': top[1:],
        })

    # Show the most-missing features first
    return pd.DataFrame(summary).sort_values('pct_missing', ascending=False)


def outlier_summary(df, cont_features):
    """Flag IQR outliers in each continuous feature.

    A value is an outlier if it falls below Q1 - 1.5*IQR or above
    Q3 + 1.5*IQR. Returns count, percentage (of non-missing), and the
    cutoff bounds per feature, sorted most-outlier-heavy first.
    (Note: tree models like RF are robust to outliers.)
    """
    outlier_rows = []
    for col in cont_features:
        # Quartiles and interquartile range for this feature
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1

        # IQR fences: standard 1.5 * IQR rule
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Flag values outside the fences
        mask  = (df[col] < lower) | (df[col] > upper)
        n_out = mask.sum()

        outlier_rows.append({
            'feature':      col,
            'n_outliers':   n_out,
            # percentage is of NON-missing values, so NaNs don't deflate the rate
            'pct_outliers': round(100 * n_out / df[col].notna().sum(), 1),
            'lower_bound':  round(lower, 2),
            'upper_bound':  round(upper, 2),
        })

    return pd.DataFrame(outlier_rows).sort_values('pct_outliers', ascending=False)
