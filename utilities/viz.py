"""
viz.py
======
All figure-producing functions for the myocardial-infarction project.

Covers the EDA plots, the SHAP beeswarm, the min_samples_leaf
sensitivity plot, and the failure-analysis visuals (record selection +
error-rate-by-age chart). Anything that draws a figure lives here.

Display labels come from brfss_utils — humanize() is imported, never
redefined, so the data dictionary has a single source of truth.

Maintained by: Maria
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from brfss_utils import humanize, LABELS, AGE_LABELS, TARGET, RANDOM_STATE


# =====================================================================
# EDA PLOTS
# =====================================================================

def plot_outlier_boxplot(df, savepath, cols=("_LCSYSMK", "_LCSYQTS", "LCSNUMCG")):
    """Boxplot of the smoking-screen continuous features.

    Boxes show the IQR; points beyond the whiskers are outliers. Feature
    codes are humanized for readable axis labels (display-only).
    """
    cols = list(cols)
    labels = [humanize(c) for c in cols]   # display-only; data unchanged

    fig, ax = plt.subplots(figsize=(8, 5))
    df[cols].boxplot(ax=ax)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Distribution & Outliers — Smoking-Screen Features", fontsize=12)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def plot_age_health_interaction(df, savepath):
    """Heatmap of MI rate for each (age group x general health) combination.

    Mean of the 0/1 target within each cell = the MI rate for that
    combination. Darker = higher MI rate. Age codes are replaced with
    readable ranges (display-only; the pivot data is unchanged).
    """
    pivot = df.pivot_table(values="_MICHD", index="_AGEG5YR",
                           columns="_RFHLTH", aggfunc="mean")
    pivot.index = [AGE_LABELS.get(int(i), str(i)) for i in pivot.index]

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="Reds",
                cbar_kws={"label": "MI rate"}, linewidths=0.5)
    plt.title("MI Rate by Age Group × General Health", fontsize=12)
    plt.xlabel("General health"); plt.ylabel("Age range")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def plot_mi_by_comorbidity(df,
                           savepath,
                           comorbid_cols=("DIABETE4", "CVDSTRK3", "CHCCOPD3",
                                          "CHCKDNY2", "HAVARTH4")):
    """Horizontal bar of MI prevalence among people WITH each condition.

    Names the diseases rather than collapsing them into a count, with the
    overall MI rate drawn as a reference line.
    """
    comorbid_cols = list(comorbid_cols)

    def is_positive(series):
        s = series.astype(str).str.lower()
        negative = s.str.contains("no", na=True) | s.str.contains("unknown", na=True) \
                   | s.isin(["nan", "none", "0", "0.0"])
        return ~negative

    # For each disease, MI rate among people who HAVE that condition
    rates = {}
    for c in comorbid_cols:
        has_condition = is_positive(df[c])
        rates[humanize(c)] = df.loc[has_condition, "_MICHD"].mean()   # readable key

    rates_s = pd.Series(rates).sort_values()

    plt.figure(figsize=(9, 5))
    rates_s.plot(kind="barh", color="#C0392B")     # horizontal so names are readable
    plt.axvline(df["_MICHD"].mean(), color="black", linestyle="--", lw=1,
                label=f"Overall MI rate ({df['_MICHD'].mean():.1%})")
    plt.xlabel("MI rate among people with the condition")
    plt.ylabel("")
    plt.title("MI Rate by Individual Comorbidity", fontsize=12)
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


# =====================================================================
# SHAP — global feature importance (Random Forest)
# =====================================================================

def plot_shap_beeswarm(rf_result, X_test, savepath, n_sample=500, max_display=12):
    """SHAP beeswarm for the Random Forest: global feature importance.

    Shows how each feature contributes across many patients: importance
    (row rank), direction (left/right), and value (color). Explains a
    sampled subset of the test set for speed; feature codes are humanized
    for display only.

    Parameters
    ----------
    rf_result : dict
        The captured RF result (must contain "pipeline").
    X_test : DataFrame
        Test features (code-named columns).
    """
    # --- Pull the fitted pieces from the captured RF result -------------
    pipeline     = rf_result["pipeline"]
    model        = pipeline.named_steps["model"]          # the fitted RandomForest
    preprocessor = pipeline.named_steps["preprocessor"]   # the ColumnTransformer

    # --- Sample and transform the test set into model space -------------
    X_sample = X_test.sample(n=min(n_sample, len(X_test)), random_state=RANDOM_STATE)
    X_enc = preprocessor.transform(X_sample)
    if hasattr(X_enc, "toarray"):       # densify if the encoder returned sparse
        X_enc = X_enc.toarray()

    # --- Build readable feature names -----------------------------------
    feat_names = preprocessor.get_feature_names_out()
    clean_names = pd.Series(feat_names).str.replace(
        r"^(categorical__|continuous__)", "", regex=True).tolist()   # codes only
    readable_names = [humanize(n) for n in clean_names]              # display labels

    # DataFrame of encoded values, columns = clean codes (data stays code-named)
    X_enc_df = pd.DataFrame(X_enc, columns=clean_names)

    # --- Compute SHAP values across the sample --------------------------
    explainer = shap.TreeExplainer(model)   # exact, fast explainer for tree models
    sv = explainer(X_enc_df)

    # RandomForest returns SHAP values per class; keep the positive class (MI = 1).
    if len(sv.values.shape) == 3:
        vals = sv.values[:, :, 1]       # contributions toward MI
        base = sv.base_values[:, 1]     # baseline (average) prediction for MI
    else:
        vals = sv.values
        base = sv.base_values

    # Repackage with readable feature names so the plot shows labels, not codes
    expl = shap.Explanation(
        values=vals,
        base_values=base,
        data=X_enc_df.values,
        feature_names=readable_names,
    )

    # --- Plot the beeswarm: top features by importance ------------------
    shap.plots.beeswarm(expl, max_display=max_display, show=False)
    plt.title("SHAP Feature Importance — Random Forest (MI prediction)", fontsize=11)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")

    plt.figtext(0.5, -0.04,
        "Each dot = one patient.  Right of 0 → contributes toward MI;  left → contributes toward no-MI.\n"
        "Color = feature value (red = high, blue = low).  Features ranked by importance (top = most).",
        ha="center", fontsize=8, color="#444444")
    plt.show()


# =====================================================================
# SENSITIVITY PLOT — min_samples_leaf
# =====================================================================

def plot_sensitivity(res, savepath, selected=20, stable_start=5):
    """Plot CV AUC vs min_samples_leaf from the sensitivity sweep results.

    A flat curve = robust (not fragile to the setting). Marks the selected
    value and shades the stable region.

    Parameters
    ----------
    res : DataFrame
        Output of sensitivity_min_samples_leaf — must contain
        param_model__min_samples_leaf, mean_test_score, std_test_score.
    selected : int
        The min_samples_leaf chosen for the final model (circled).
    stable_start : int
        Left edge of the shaded "stable region".
    """
    x   = res["param_model__min_samples_leaf"].astype(int)   # values probed
    y   = res["mean_test_score"]                             # mean CV AUC per value
    err = res["std_test_score"]                              # std across folds

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # ± 1 std band around the mean (shows fold-to-fold variability)
    ax.fill_between(x, y - err, y + err, color="#C0392B", alpha=0.12, label="± 1 std")

    # Mean CV AUC line
    ax.plot(x, y, color="#C0392B", marker="o", markersize=7,
            linewidth=2, label="Mean CV AUC", zorder=3)

    # Circle + annotate the value selected for the final model
    sel_y = y[x == selected].iloc[0]
    ax.scatter([selected], [sel_y], s=200, facecolors="none",
               edgecolors="#1D3557", linewidths=2.5, zorder=4)
    ax.annotate(f"Selected\n(min_samples_leaf = {selected})",
                xy=(selected, sel_y), xytext=(selected, sel_y - 0.006),
                ha="center", fontsize=9, color="#1D3557", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1D3557", lw=1.2))

    # Shade the stable region (where AUC is essentially flat)
    ax.axvspan(stable_start, x.max(), color="#1D9E75", alpha=0.05)
    ax.text(x.max() * 0.55, y.max() + 0.0015, "stable region",
            fontsize=9, color="#1D9E75", style="italic")

    # Log x-axis (instructor convention — values span 1 to 100)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 20, 50, 100])
    ax.set_xticklabels([1, 5, 10, 20, 50, 100])
    ax.set_xlabel("min_samples_leaf (log scale)", fontsize=11)
    ax.set_ylabel("Cross-validated AUC", fontsize=11)
    ax.set_title("Random Forest Sensitivity to min_samples_leaf", fontsize=12, pad=12)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


# =====================================================================
# FAILURE ANALYSIS
# =====================================================================

def select_failure_records(rf_result, X_test, y_test):
    """Select three representative misclassified records, by rule not by hand.

    Returns (ex1, ex2, ex3) single-row frames:
      ex1 — most CONFIDENT false negative (lowest prob among missed MIs):
            expected profile young, healthy.
      ex2 — most CONFIDENT false positive (highest prob among false alarms):
            expected profile old, comorbid.
      ex3 — most BORDERLINE error (probability closest to 0.5): the model
            was maximally uncertain and tipped the wrong way.
    """
    y_prob = rf_result["y_prob"]     # predicted MI probability per test record
    y_pred = rf_result["y_pred"]     # predicted class (0/1) per test record

    # Build an error frame: test features + truth, prediction, probability
    err = X_test.copy()
    err["_MICHD_true"] = y_test.values
    err["pred"]        = y_pred
    err["prob_MI"]     = y_prob

    fn = err[(err["_MICHD_true"] == 1) & (err["pred"] == 0)].copy()   # false negatives
    fp = err[(err["_MICHD_true"] == 0) & (err["pred"] == 1)].copy()   # false positives

    ex1 = fn.sort_values("prob_MI").head(1)                 # most confident miss
    ex2 = fp.sort_values("prob_MI", ascending=False).head(1)  # most confident false alarm

    err_mis = err[err["_MICHD_true"] != err["pred"]].copy()
    err_mis["dist_from_5"] = (err_mis["prob_MI"] - 0.5).abs()
    ex3 = err_mis.sort_values("dist_from_5").head(1)        # most borderline

    return ex1, ex2, ex3


def print_failure_records(ex1, ex2, ex3):
    """Print the three selected records (full feature values, transposed)."""
    for name, rec in [("EX1_false_neg_young_lowrisk",   ex1),
                      ("EX2_false_pos_comorbid",        ex2),
                      ("EX3_borderline_conflicting",    ex3)]:
        print(f"\n========== {name} ==========")
        if rec.empty:
            print("(no matching record)")
        else:
            print("row index:", rec.index[0], "| prob_MI:", round(rec["prob_MI"].iloc[0], 3))
            print(rec.T.to_string())


def plot_failure_by_age(rf_result, X_test, y_test, savepath, example_records=None):
    """Plot false-negative and false-positive rates by age group.

    Optionally marks where the three Example records fall on the
    population error pattern.

    Parameters
    ----------
    rf_result : dict   -- captured RF result (uses y_pred)
    X_test, y_test : held-out test data
    example_records : dict
        {label: (age_code, label_height)} for vertical markers. Defaults to
        the three current failure records.
    """
    if example_records is None:
        example_records = {
            "Example 1 (44727)":  (2,  0.30),
            "Example 2 (324306)": (9,  0.30),
            "Example 3 (177052)": (11, 0.30),
        }

    err = X_test.copy()
    err["true"] = y_test.values
    err["pred"] = rf_result["y_pred"]
    err["false_neg"] = ((err["true"] == 1) & (err["pred"] == 0)).astype(int)
    err["false_pos"] = ((err["true"] == 0) & (err["pred"] == 1)).astype(int)

    by_age = err.groupby("_AGEG5YR").agg(
        fn_rate=("false_neg", "mean"),
        fp_rate=("false_pos", "mean"),
        n=("true", "size"),
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(by_age.index, by_age["fn_rate"], color="#C0392B", marker="o",
            lw=2, label="False negative rate (missed MIs)")
    ax.plot(by_age.index, by_age["fp_rate"], color="#F0AD4E", marker="s",
            lw=2, label="False positive rate (false alarms)")

    for label, (age_code, y_pos) in example_records.items():
        ax.axvline(age_code, color="#1D3557", linestyle=":", lw=1.2, alpha=0.7)
        ax.annotate(f"{label}", xy=(age_code, y_pos), rotation=90,
                    fontsize=7.5, color="#1D3557", va="top", ha="right")

    ax.set_xticks(by_age.index)
    ax.set_xticklabels([AGE_LABELS.get(int(c), str(int(c))) for c in by_age.index],
                       rotation=45, ha="right")
    ax.set_xlabel("Age range", fontsize=11)
    ax.set_ylabel("Error rate within age group", fontsize=11)
    ax.set_title("Where the Model Fails — Error Rates by Age", fontsize=12, pad=12)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()

    # Table with readable age ranges
    by_age_display = pd.DataFrame({
        "Age": [AGE_LABELS.get(int(c), str(int(c))) for c in by_age.index],
        "Misses (%)":       (by_age["fn_rate"] * 100).round(1).values,
        "False Alarms (%)": (by_age["fp_rate"] * 100).round(1).values,
        "n":                by_age["n"].values,
    })
    print(by_age_display.to_string(index=False))
    return by_age_display
