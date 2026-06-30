"""
model_pipeline.py
=================
Modeling machinery for the myocardial-infarction project.

Flow:  make_preprocessor -> build_pipeline -> train_pipeline
       -> evaluate_pipeline -> plot_dashboard,  orchestrated by run_model.

Also includes the single-feature ablation and the min_samples_leaf
sensitivity sweep, both of which reuse the same pipeline building blocks.

Constants (RANDOM_STATE, CV_FOLDS, TARGET) live in brfss_utils; import
them from there so every module shares one source of truth.

Maintained by: Maria
"""

import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute        import SimpleImputer
from sklearn.pipeline      import Pipeline
from sklearn.compose       import ColumnTransformer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
)
from imblearn.under_sampling import RandomUnderSampler

from brfss_utils import RANDOM_STATE, CV_FOLDS, TARGET


# =====================================================================
# MODEL PARAMETERS
# Per-model hyperparameters. Only RF was tuned (min_samples_leaf);
# LR and XGBoost use sensible defaults.
# =====================================================================

# --- Logistic Regression (baseline: linear, probabilistic) ----------
LR_PARAMS = {
    "max_iter": 1000,      # avoid non-convergence warnings on the scaled features
    "random_state": RANDOM_STATE,
}

# --- Random Forest (bagging ensemble — the selected/tuned model) -----
# min_samples_leaf = 20 chosen via the sensitivity sweep (stable region).
RF_PARAMS = {
    "n_estimators": 1000,      # number of trees
    "max_depth": None,         # let trees grow fully; leaf size controls overfitting
    "min_samples_leaf": 20,    # tuned: minimum samples per leaf (regularization)
    "random_state": RANDOM_STATE,
    "n_jobs": -1,              # use all cores when fitting the final model
}

# --- XGBoost (boosting ensemble) ------------------------------------
XGB_PARAMS = {
    "eval_metric": "logloss",  # evaluation metric during boosting
    "random_state": RANDOM_STATE,
}


# =====================================================================
# MODELING FUNCTIONS
# =====================================================================

def make_preprocessor(cat_features, cont_features):
    """Build a fresh ColumnTransformer with two feature lanes.

    A new preprocessor is returned on every call so that each model (and
    each ablation run) gets an unfitted transformer, preventing state from
    leaking between fits.

    Lanes:
      - categorical: most-frequent imputation, then one-hot encoding
      - continuous:  median imputation, then standard scaling

    Parameters
    ----------
    cat_features, cont_features : list of str
        Column names routed to each lane.

    Returns
    -------
    ColumnTransformer  (unfitted)
    """
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    cont_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    return ColumnTransformer([
        ("categorical", cat_transformer,  cat_features),
        ("continuous",  cont_transformer, cont_features),
    ])


def build_pipeline(model, preprocessor):
    """Chain a preprocessor and an estimator into one sklearn Pipeline.

    Keeping preprocessing inside the pipeline means imputation, encoding,
    and scaling are all learned from the training fold only during
    cross-validation, avoiding leakage from the validation/test data.

    Parameters
    ----------
    model : sklearn-compatible estimator
    preprocessor : ColumnTransformer

    Returns
    -------
    Pipeline  (unfitted)
    """
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def train_pipeline(model, preprocessor, X_train, y_train):
    """Undersample the majority class, then build and fit the pipeline.

    The MI outcome is imbalanced (~9% positive). RandomUnderSampler drops
    majority-class rows so the model trains on a balanced (~50/50) set,
    which raises recall on the minority (MI) class. Only the TRAINING data
    is resampled; the test set keeps its real-world prevalence.

    Parameters
    ----------
    model : sklearn-compatible estimator
    preprocessor : ColumnTransformer
    X_train, y_train : training features and labels

    Returns
    -------
    (fitted Pipeline, X_train_res, y_train_res)
        The fitted pipeline plus the resampled training data, which is
        reused for cross-validation in evaluate_pipeline.
    """
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    pipeline = build_pipeline(model, preprocessor)
    pipeline.fit(X_train_res, y_train_res)
    return pipeline, X_train_res, y_train_res


def evaluate_pipeline(pipeline, X_train_res, y_train_res, X_test, y_test,
                      model_name, verbose=True):
    """Cross-validate on the resampled training data, then score the test set.

    Reports CV AUC (mean +/- SD across folds) for stable cross-model
    comparison, plus test AUC, average precision, and a full
    classification report (precision/recall/F1 per class).

    Parameters
    ----------
    pipeline : fitted Pipeline
    X_train_res, y_train_res : resampled training data (from train_pipeline)
    X_test, y_test : held-out test data (real-world class balance)
    model_name : str  -- label for printed output
    verbose : bool    -- print metrics if True

    Returns
    -------
    dict with cv_auc_mean, cv_auc_std, test_auc, test_ap, y_pred, y_prob
    """
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(pipeline, X_train_res, y_train_res,
                             cv=cv, scoring="roc_auc", n_jobs=-1)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)
    test_ap  = average_precision_score(y_test, y_prob)
    if verbose:
        print(f"=== {model_name} ===")
        print(f"CV AUC ({CV_FOLDS}-fold): {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
        print(f"Test AUC: {test_auc:.4f} | Test AP: {test_ap:.4f}")
        print(f"\n{classification_report(y_test, y_pred)}")
    return {
        "cv_auc_mean": cv_auc.mean(), "cv_auc_std": cv_auc.std(),
        "test_auc": test_auc, "test_ap": test_ap,
        "y_pred": y_pred, "y_prob": y_prob,
    }


def plot_dashboard(pipeline, metrics, y_test, model_name):
    """Four-panel evaluation dashboard for one fitted model.

    Panels: ROC curve, precision-recall curve, confusion matrix, and the
    top-20 feature importances. Importance source adapts to the model:
    impurity decrease for tree models (feature_importances_), coefficient
    magnitude for linear models (coef_). Lane prefixes are stripped from
    feature names for readability and colored by feature type.

    Parameters
    ----------
    pipeline : fitted Pipeline
    metrics : dict from evaluate_pipeline (uses y_pred, y_prob, AUC, AP)
    y_test : true test labels
    model_name : str -- figure title
    """
    y_pred, y_prob = metrics["y_pred"], metrics["y_prob"]
    test_auc, test_ap = metrics["test_auc"], metrics["test_ap"]
    transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    def get_feature_type(name):
        # Lane is encoded in the ColumnTransformer prefix
        if name.startswith("categorical__"):
            return "categorical"
        return "continuous"

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1 -- ROC curve
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, color="#1D9E75", lw=2, label=f"AUC = {test_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve"); ax1.legend(fontsize=9)

    # Panel 2 -- Precision-Recall curve (baseline = positive class rate)
    ax2 = fig.add_subplot(gs[0, 1])
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ax2.plot(rec, prec, color="#378ADD", lw=2, label=f"AP = {test_ap:.3f}")
    ax2.axhline(y_test.mean(), color="grey", linestyle="--", lw=0.8,
                label=f"Baseline = {y_test.mean():.3f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve"); ax2.legend(fontsize=9)

    # Panel 3 -- Confusion matrix
    ax3 = fig.add_subplot(gs[0, 2])
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
                xticklabels=[f"Pred {c}" for c in [0, 1]],
                yticklabels=[f"Act {c}" for c in [0, 1]])
    ax3.set_title("Confusion Matrix")

    # Panel 4 -- Top-20 feature importance (model-dependent source)
    ax4 = fig.add_subplot(gs[1, :])
    fitted = pipeline.named_steps["model"]
    if hasattr(fitted, "feature_importances_"):
        importance = fitted.feature_importances_
        xlabel = "Mean decrease in impurity"
    elif hasattr(fitted, "coef_"):
        importance = np.abs(fitted.coef_[0])
        xlabel = "Coefficient magnitude (|weight|)"
    else:
        importance = None

    if importance is not None:
        imp_df = pd.DataFrame({"feature": transformed_names, "importance": importance})
        imp_df["type"] = imp_df["feature"].apply(get_feature_type)
        # Strip lane prefix for display only
        imp_df["feature"] = imp_df["feature"].str.replace(
            r"^(categorical__|continuous__)", "", regex=True)
        imp_df = imp_df.sort_values("importance", ascending=False).head(20)
        colors_map = {"categorical": "#8E44AD", "continuous": "#378ADD"}
        imp_colors = imp_df["type"].map(colors_map).tolist()
        ax4.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color=imp_colors[::-1])
        ax4.set_xlabel(xlabel)
        ax4.set_title("Feature Importance (top 20)")
        ax4.legend(handles=[Patch(color=v, label=k) for k, v in colors_map.items()], fontsize=9)

    plt.suptitle(f"{model_name} — {TARGET}", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def run_model(model, model_name,
              X_train, y_train, X_test, y_test,
              cat_features, cont_features,
              plot=True):
    """Orchestrate the full flow for one model: train -> evaluate -> plot.

    Unlike the notebook version (which fell back to module-level globals),
    this takes data and feature lists explicitly so the function is
    self-contained and importable. Keep notebook calls short by passing the
    same handful of variables every time, e.g.:

        run_model(RandomForestClassifier(**RF_PARAMS), "Random Forest",
                  X_train, y_train, X_test, y_test, cat_features, cont_features)

    Parameters
    ----------
    model : sklearn-compatible estimator (already constructed with params)
    model_name : str
    X_train, y_train, X_test, y_test : train/test split
    cat_features, cont_features : the two feature-type lanes
    plot : bool -- draw the dashboard if True

    Returns
    -------
    dict with name, fitted pipeline, cv_auc_mean, test_auc, test_ap,
    y_pred, y_prob
    """
    preprocessor = make_preprocessor(cat_features, cont_features)
    pipeline, X_train_res, y_train_res = train_pipeline(model, preprocessor, X_train, y_train)
    metrics = evaluate_pipeline(pipeline, X_train_res, y_train_res, X_test, y_test, model_name)
    if plot:
        plot_dashboard(pipeline, metrics, y_test, model_name)

    return {
        "name": model_name, "pipeline": pipeline,
        "cv_auc_mean": metrics["cv_auc_mean"],
        "test_auc": metrics["test_auc"], "test_ap": metrics["test_ap"],
        "y_pred": metrics["y_pred"], "y_prob": metrics["y_prob"],
    }


# =====================================================================
# SINGLE-FEATURE ABLATION
# Remove one feature at a time, retrain the RF, measure the change in
# test AUC. Measures IRREPLACEABILITY (does removing it hurt), which
# differs from SHAP's average-contribution importance.
# =====================================================================

def ablate_one(feature, X_train, y_train, X_test, y_test, cat_features, cont_features):
    """Retrain the RF with one feature removed; return (cv_auc, test_auc).

    The feature is dropped from whichever lane it belongs to (categorical
    or continuous); the other lane is unchanged. Passing feature=None
    removes nothing, which gives the full-model baseline.
    """
    # Rebuild the two lane lists without the feature being ablated
    cat_f  = [c for c in cat_features  if c != feature]
    cont_f = [c for c in cont_features if c != feature]

    # Fresh preprocessor on the reduced feature set, then train the RF
    pre = make_preprocessor(cat_f, cont_f)
    pipeline, X_res, y_res = train_pipeline(
        RandomForestClassifier(**RF_PARAMS), pre, X_train, y_train)

    # Cross-validated AUC on the resampled training data (n_jobs=1 = memory safe)
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(pipeline, X_res, y_res, cv=cv, scoring="roc_auc", n_jobs=1)

    # Test AUC on the held-out set (real-world class balance)
    test_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    return cv_auc.mean(), test_auc


def run_ablation(X_train, y_train, X_test, y_test, cat_features, cont_features,
                 verbose=True):
    """Ablate each feature in turn and rank by impact on test AUC.

    Returns a DataFrame sorted by delta_auc (most negative first); a large
    negative delta means removing that feature hurt the model most.
    """
    # Baseline: full model with all features (feature=None removes nothing)
    base_cv, base_test = ablate_one(None, X_train, y_train, X_test, y_test,
                                    cat_features, cont_features)
    if verbose:
        print(f"Baseline — full model: CV {base_cv:.4f} | Test {base_test:.4f}")

    all_features = cat_features + cont_features
    rows = []
    for feat in all_features:
        cv_auc, test_auc = ablate_one(feat, X_train, y_train, X_test, y_test,
                                      cat_features, cont_features)
        # delta_auc < 0 means removing the feature HURT (it was useful)
        rows.append({"feature": feat, "cv_auc": cv_auc, "test_auc": test_auc,
                     "delta_auc": test_auc - base_test})
        if verbose:
            print(f"  removed {feat:12s}  Test AUC {test_auc:.4f}  ΔAUC {test_auc - base_test:+.4f}")

    # Rank by impact: most negative delta (biggest AUC drop) = most important
    return pd.DataFrame(rows).sort_values("delta_auc").round(4).reset_index(drop=True)

def run_ablation_groups(X_train, y_train, X_test, y_test,
                        cat_features, cont_features, feature_groups=None, verbose=True):
    """Ablate each feature GROUP and rank by impact on test AUC."""
    base_cv, base_test = ablate_one(
        None, X_train, y_train, X_test, y_test, cat_features, cont_features)
    if verbose:
        print(f"Baseline — full model: CV {base_cv:.4f} | Test {base_test:.4f}\n")

    rows = []
    for group, members in feature_groups.items():
        present = [f for f in members if f in cat_features + cont_features]
        if not present:
            continue
        cat_f  = [c for c in cat_features  if c not in present]
        cont_f = [c for c in cont_features if c not in present]

        pre = make_preprocessor(cat_f, cont_f)
        pipeline, X_res, y_res = train_pipeline(
            RandomForestClassifier(**RF_PARAMS), pre, X_train, y_train)
        cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_auc = cross_val_score(
            pipeline, X_res, y_res, cv=cv, scoring="roc_auc", n_jobs=1).mean()
        test_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

        rows.append({"group": group, "n_removed": len(present),
                     "features_removed": ", ".join(present),
                     "cv_auc": cv_auc, "test_auc": test_auc,
                     "delta_auc": test_auc - base_test})
        if verbose:
            print(f"  removed {group:25s} ({len(present)} feats)  "
                  f"Test AUC {test_auc:.4f}  ΔAUC {test_auc - base_test:+.4f}")

    return pd.DataFrame(rows).sort_values("delta_auc").round(4).reset_index(drop=True)

# =====================================================================
# SENSITIVITY ANALYSIS — RandomForest min_samples_leaf
# Sweeps the key hyperparameter and returns the CV-AUC results so the
# notebook can plot stability. Also serves as the tuning step.
# =====================================================================

def sensitivity_min_samples_leaf(X_train, y_train, cat_features, cont_features,
                                 param_values=(1, 5, 10, 20, 50, 100)):
    """Sweep min_samples_leaf via GridSearchCV; return the cv_results frame.

    Undersamples the training data (same balanced setup as the main
    models), then grid-searches min_samples_leaf scoring CV AUC. Returns
    a DataFrame with one row per parameter value (mean/std test score).
    """
    # Undersample the training data (same balanced setup as the main models)
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

    # RF pipeline; only min_samples_leaf will vary in the grid search
    pipeline = Pipeline([
        ("preprocessor", make_preprocessor(cat_features, cont_features)),
        ("model", RandomForestClassifier(**RF_PARAMS)),
    ])

    param_grid = {"model__min_samples_leaf": list(param_values)}
    grid = GridSearchCV(
        pipeline, param_grid, scoring="roc_auc", cv=CV_FOLDS,
        n_jobs=1,            # serial — no data duplication across workers (memory safe)
        pre_dispatch=1,      # don't queue extra fits in memory
    )
    grid.fit(X_train_res, y_train_res)

    res = pd.DataFrame(grid.cv_results_)[
        ["param_model__min_samples_leaf", "mean_test_score", "std_test_score"]
    ]

    # Free the fitted grid (n_values x folds forests) to reclaim memory
    del grid
    gc.collect()

    return res
