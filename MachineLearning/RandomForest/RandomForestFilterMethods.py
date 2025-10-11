import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc, make_scorer, f1_score

# Filter methods
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE, SequentialFeatureSelector


# =======================
# ---- Filter Methods ---
# =======================
def filter_importance_table(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compute model-agnostic filter scores:
      - ANOVA F
      - Chi-square (MinMax scaled)
      - Mutual Information
      - |Correlation| with binary target
    Returns DataFrame sorted by CombinedScore.
    """
    Xn = X.select_dtypes(include=[np.number]).copy()
    Xn.replace([np.inf, -np.inf], np.nan, inplace=True)
    Xn.fillna(0.0, inplace=True)

    y_arr = y.values.astype(float)

    # 1) ANOVA F
    F_vals, F_p = f_classif(Xn, y_arr)

    # 2) Chi-square
    X_nonneg = pd.DataFrame(MinMaxScaler().fit_transform(Xn), columns=Xn.columns)
    chi2_vals, chi2_p = chi2(X_nonneg, y_arr)

    # 3) Mutual Information
    MI_vals = mutual_info_classif(Xn, y_arr, random_state=0)

    # 4) Point-biserial correlation
    R_vals = np.array([np.corrcoef(Xn[col].values, y_arr)[0, 1] for col in Xn.columns])
    R_vals = np.nan_to_num(R_vals, nan=0.0)
    AbsR_vals = np.abs(R_vals)

    df = pd.DataFrame({
        "Feature": Xn.columns,
        "F (ANOVA)": F_vals,
        "p_F": F_p,
        "chi2": chi2_vals,
        "p_chi2": chi2_p,
        "MI": MI_vals,
        "R_pointbiserial": R_vals,
        "|R|": AbsR_vals,
    })

    def _norm(col):
        v = df[col].values
        vmin, vmax = float(np.min(v)), float(np.max(v))
        rng = (vmax - vmin) if vmax > vmin else 1.0
        return (v - vmin) / rng

    for col in ["F (ANOVA)", "chi2", "MI", "|R|"]:
        df[col + " (norm)"] = _norm(col)

    df["CombinedScore"] = df[[c for c in df.columns if c.endswith("(norm)")]].mean(axis=1)
    return df.sort_values("CombinedScore", ascending=False).reset_index(drop=True)


def plot_filter_importance(scores_df: pd.DataFrame, top_n: int = 20, title="Filter Feature Importance (Combined)"):
    top = scores_df.head(top_n)
    plt.figure(figsize=(10, max(4, 0.40 * len(top))))
    plt.barh(top["Feature"], top["CombinedScore"])
    plt.gca().invert_yaxis()
    plt.xlabel("Combined Normalized Score")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =======================
# ---- Wrapper Methods ----
# =======================


# RFE (Recursive Feature Elimination)

# A combination of backward elimination with a recursive approach.
# Ranks all the features based on their importance.
# Recursively removes the least important ones and retrains the model.
# Continues until the desired number of features is reached.
# Example use case: Widely used with support vector machines and linear models for model interpretability.

def rfe_select(X, y, n_features=20, estimator=None, step=1):
    print(f"\n[RFE] Selecting top {n_features} features using RFE...")
    """Recursive Feature Elimination (RFE)."""
    base = estimator or RandomForestClassifier(random_state=100, n_jobs=-1)
    rfe = RFE(estimator=base, n_features_to_select=n_features, step=step)
    rfe.fit(X, y)
    selected = X.columns[rfe.support_].tolist()
    print(f"\n[RFE] Selected {len(selected)} features:\n{selected}")
    return selected


#Sequential Feature Selection (SFS)

# Forward Selection
# Starts with no features.
# Adds one feature at a time.
# At each step, it adds the feature that improves model performance the most.
# Stops when adding more features does not improve the model.
# Example use case: When you expect only a few features to be useful and want a quick way to build up a model.
# 
# Backward Elimination
# Starts with all features.
# Removes one feature at a time.
# At each step, it removes the feature that contributes the least to the model.
# Stops when removing more features degrades performance.

def sequential_select(X, y, n_features=20, direction="forward",
                      estimator=None, scoring="f1", cv=5):
    print(f"\n[SFS-{direction}] Selecting top {n_features} features using SFS...")
    """Sequential Feature Selection (forward/backward)."""
    base = estimator or RandomForestClassifier(random_state=100, n_jobs=-1)
    scorer = make_scorer(f1_score) if scoring == "f1" else scoring
    sfs = SequentialFeatureSelector(
        base, n_features_to_select=n_features,
        direction=direction, scoring=scorer, cv=cv, n_jobs=-1
    )
    sfs.fit(X, y)
    selected = X.columns[sfs.get_support()].tolist()
    print(f"\n[SFS-{direction}] Selected {len(selected)} features:\n{selected}")
    return selected


# =======================
# ---- Main Pipeline ----
# =======================

#Default parameters for do_model

def do_model(path: str,
             graph: bool = False,
             show_importance: bool = True,
             use_filters: bool = False,
             filter_k: int | None = None,
             filter_plot: bool = True,
             save_filter_csv: bool = False,
             filter_csv_path: str | None = None,
             use_permutation: bool = False,
             perm_metric: str = "f1",
             perm_repeats: int = 20,
             use_wrapper: bool = False,
             wrapper_method: str = "rfe",    # "rfe" or "sfs"
             wrapper_k: int = 20,
             sfs_direction: str = "forward"):
    """
    Train Random Forest on `path` with optional filter and/or wrapper feature selection.
    """

    # 1) Load
    df = pd.read_csv(path)

    # 2) Prepare X, y
    y = df['anomaly']
    X = df.drop(columns=['anomaly', 'timestamp', 'channel', 'label'], errors='ignore')
    X = X.select_dtypes(include=[np.number]).copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0.0, inplace=True)

    # ----- Filter stage -----
    scores_df = None
    if use_filters:
        scores_df = filter_importance_table(X, y)

        print("\nFilter-method feature scores (top 30 by CombinedScore):")
        print(scores_df.head(30).to_string(index=False))

        if save_filter_csv:
            out_path = filter_csv_path or (path.rsplit("/", 1)[0] + "/filter_scores.csv" if "/" in path else "filter_scores.csv")
            scores_df.to_csv(out_path, index=False)
            print(f"\nSaved filter scores to: {out_path}")

        if filter_plot:
            plot_filter_importance(scores_df, top_n=min(20, len(scores_df)))

        if filter_k is not None and filter_k > 0:
            top_features = scores_df["Feature"].head(filter_k).tolist()
            print(f"\nUsing top-{filter_k} features from filters:\n{top_features}")
            X = X[top_features]

    # ----- Wrapper stage -----
    if use_wrapper:
        if wrapper_method.lower() == "rfe":
            top_features = rfe_select(X, y, n_features=wrapper_k)
        elif wrapper_method.lower() == "sfs":
            top_features = sequential_select(
                X, y, n_features=wrapper_k, direction=sfs_direction, scoring="f1"
            )
        else:
            raise ValueError("wrapper_method must be 'rfe' or 'sfs'")
        print(f"\nUsing top-{wrapper_k} features from {wrapper_method.upper()}:\n{top_features}")
        X = X[top_features]

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100,
        stratify=y if len(y.unique()) == 2 else None
    )

    # 4) Train model
    rf = RandomForestClassifier(random_state=100)
    rf.fit(X_train, y_train)

    # 5) Evaluate
    print("\nModel Performance:")
    print("Training Set Performance:")
    print(classification_report(y_train, rf.predict(X_train)))
    print("Test Set Performance:")
    print(classification_report(y_test, rf.predict(X_test)))

    # 6) ROC curve
    if graph and hasattr(rf, "predict_proba") and (len(np.unique(y_test)) == 2):
        fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
        auc_val = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 7) RandomForest importance
    feat_imp = None
    if show_importance:
        importances = rf.feature_importances_
        feat_imp = pd.DataFrame({
            "Feature": X.columns,
            "Importance (RF, impurity)": importances
        }).sort_values("Importance (RF, impurity)", ascending=False).reset_index(drop=True)

        print("\nRandom Forest Feature Importance (impurity-based):")
        print(feat_imp.to_string(index=False))

        plt.figure(figsize=(10, max(4, 0.35 * len(feat_imp))))
        plt.barh(feat_imp["Feature"], feat_imp["Importance (RF, impurity)"])
        plt.gca().invert_yaxis()
        plt.xlabel("Importance (strength)")
        plt.title("Random Forest — Feature Importance (Impurity)")
        plt.tight_layout()
        plt.show()

    # 8) Permutation importance
    if use_permutation:
        perm = permutation_importance(
            rf, X_test, y_test, n_repeats=perm_repeats,
            random_state=100, n_jobs=-1, scoring=perm_metric
        )
        perm_df = pd.DataFrame({
            "Feature": X.columns,
            "Perm Importance (Mean Δscore)": perm.importances_mean,
            "Perm Importance (Std)": perm.importances_std
        }).sort_values("Perm Importance (Mean Δscore)", ascending=False).reset_index(drop=True)

        print(f"\nPermutation Importance on test set (scoring='{perm_metric}'):")
        print(perm_df.to_string(index=False))

        plt.figure(figsize=(10, max(4, 0.35 * len(perm_df))))
        plt.barh(perm_df["Feature"], np.abs(perm_df["Perm Importance (Mean Δscore)"]))
        plt.gca().invert_yaxis()
        plt.xlabel("Abs mean Δscore")
        plt.title("Permutation Importance (Absolute)")
        plt.tight_layout(); plt.show()

        plt.figure(figsize=(10, max(4, 0.35 * len(perm_df))))
        plt.barh(perm_df["Feature"], perm_df["Perm Importance (Mean Δscore)"])
        plt.gca().invert_yaxis()
        plt.xlabel("Mean Δscore (signed)")
        plt.title("Permutation Importance (Signed)")
        plt.tight_layout(); plt.show()

    return feat_imp


# === Example usage ===
if __name__ == "__main__":
    do_model(
        'CSVs\ToBeMerged\dataset.csv',
        use_filters=True, filter_k=20, filter_plot=True,
        use_wrapper=True, wrapper_method="sfs", wrapper_k=15, sfs_direction="forward",
        graph=False, show_importance=True
    )
    do_model(
        'CSVs\ToBeMerged\dataset.csv',
        use_filters=True, filter_k=20, filter_plot=True,
        use_wrapper=True, wrapper_method="sfs", wrapper_k=15, sfs_direction="backward",
        graph=False, show_importance=True
    )
    do_model(
        'CSVs\ToBeMerged\dataset.csv',
        use_filters=True, filter_k=20, filter_plot=True,
        use_wrapper=True, wrapper_method="rfe", wrapper_k=15, graph=False, show_importance=True
    )
