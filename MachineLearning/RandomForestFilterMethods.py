# RandomForest_with_filters.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc

# Filter-methods
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance  # optional, for signed PI

def filter_importance_table(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compute model-agnostic (filter) scores per feature:
      - ANOVA F
      - Chi-square (requires non-negative inputs; run on MinMax-scaled copy)
      - Mutual Information (nonlinear dependence)
      - |Correlation| with binary target (point-biserial ≡ Pearson with {0,1})
    Returns a DataFrame ranked by CombinedScore (mean of normalized F, chi2, MI, |R|).
    """
    # Keep numeric, clean NaNs/Infs
    Xn = X.select_dtypes(include=[np.number]).copy()
    Xn.replace([np.inf, -np.inf], np.nan, inplace=True)
    Xn.fillna(0.0, inplace=True)

    y_arr = y.values.astype(float)

    # 1) ANOVA F
    F_vals, F_p = f_classif(Xn, y_arr)

    # 2) Chi-square on non-negative copy
    X_nonneg = pd.DataFrame(MinMaxScaler().fit_transform(Xn), columns=Xn.columns)
    chi2_vals, chi2_p = chi2(X_nonneg, y_arr)

    # 3) Mutual Information
    MI_vals = mutual_info_classif(Xn, y_arr, random_state=0)

    # 4) Point-biserial correlation (signed). Use Pearson with binary y.
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

    # Normalize to [0,1] for a blended strength score
    def _norm(col):
        v = df[col].values
        vmin, vmax = float(np.min(v)), float(np.max(v))
        rng = (vmax - vmin) if vmax > vmin else 1.0
        return (v - vmin) / rng

    for col in ["F (ANOVA)", "chi2", "MI", "|R|"]:
        df[col + " (norm)"] = _norm(col)

    # Equal-weight blend (tweak if you want to emphasize MI, etc.)
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
             perm_repeats: int = 20):
    """
    Train Random Forest on `path`, print metrics, and (optionally) show feature importance.

    New params for filter methods:
      - use_filters: compute filter scores (ANOVA F, χ², MI, |corr|)
      - filter_k: keep only top-K by CombinedScore (if None, keep all)
      - filter_plot: show bar chart of top features by CombinedScore
      - save_filter_csv: save filter score table to CSV

    Extra:
      - use_permutation: also compute permutation importance (can be negative/signed)
      - perm_metric: scoring metric for permutation_importance (e.g., "f1", "accuracy")
      - perm_repeats: n_repeats for permutation importance

    Returns:
      feat_imp (pd.DataFrame) — RF impurity importances on the final feature set
    """
    # 1) Load
    df = pd.read_csv(path)

    # 2) Prepare X, y
    y = df['anomaly']
    X = df.drop(columns=['anomaly', 'timestamp', 'channel', 'label'], errors='ignore')
    # Ensure numeric for training
    X = X.select_dtypes(include=[np.number]).copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0.0, inplace=True)

    # ----- Optional: filter stage -----
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

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100, stratify=y if len(y.unique()) == 2 else None
    )

    # 4) Model
    rf = RandomForestClassifier(random_state=100)
    rf.fit(X_train, y_train)

    # 5) Evaluate
    y_pred_train = rf.predict(X_train)
    y_pred_test  = rf.predict(X_test)

    print("\nModel Performance:")
    print("Training Set Performance:")
    print(classification_report(y_train, y_pred_train))
    print("Test Set Performance:")
    print(classification_report(y_test, y_pred_test))

    # 6) ROC (binary)
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

    # 7) RF impurity importances (non-negative, sum to 1)
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

    # 8) Optional: Permutation Importance (can be negative; shows direction wrt chosen metric)
    if use_permutation:
        perm = permutation_importance(
            rf, X_test, y_test, n_repeats=perm_repeats, random_state=100, n_jobs=-1, scoring=perm_metric
        )
        perm_df = pd.DataFrame({
            "Feature": X.columns,
            "Perm Importance (Mean Δscore)": perm.importances_mean,
            "Perm Importance (Std)": perm.importances_std
        }).sort_values("Perm Importance (Mean Δscore)", ascending=False).reset_index(drop=True)

        print(f"\nPermutation Importance on test set (scoring='{perm_metric}'):")
        print(perm_df.to_string(index=False))

        # Plot absolute and signed views
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

# === Example usage (matches your original call) ===
# Plain RF:
# do_model("CSVs/Output/merged_OPSSAT_segments.csv", graph=False, show_importance=True)

# RF with filter stage (top-20):
do_model("CSVs\OPSAT-AD_modified.csv",
         use_filters=True, filter_k=20, filter_plot=True,
         graph=False, show_importance=True)

# RF + filters + permutation importance:
# do_model("CSVs/Output/merged_OPSSAT_segments.csv",
#          use_filters=True, filter_k=20, use_permutation=True, perm_metric="f1")
