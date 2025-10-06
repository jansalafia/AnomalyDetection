# SVM_FeatureImportance.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

def linear_coef_importance(model: SVC | Pipeline,
                           feature_names: Sequence[str],
                           plot: bool = True) -> pd.DataFrame:
    """
    If the SVM uses a *linear* kernel, show signed coefficients as feature importance.
    Works with a bare SVC or a Pipeline([StandardScaler, SVC]).

    Returns a DataFrame sorted by |coef|.
    """
    # Extract underlying SVC if it's a pipeline
    if isinstance(model, Pipeline):
        clf = model.named_steps.get('clf')
    else:
        clf = model

    if not isinstance(clf, SVC) or clf.kernel != 'linear' or not hasattr(clf, 'coef_'):
        raise ValueError("linear_coef_importance requires a linear-kernel SVC with learned coef_.")

    coefs = clf.coef_.ravel()  # binary case
    feat_imp = pd.DataFrame({
        "Feature": list(feature_names),
        "Coefficient (Raw)": coefs,
    })
    feat_imp["Importance (Abs)"] = np.abs(feat_imp["Coefficient (Raw)"])
    feat_imp = feat_imp.sort_values(by="Importance (Abs)", ascending=False).reset_index(drop=True)

    print("\nSVM (linear) — Coefficient-based Importance:")
    print(feat_imp.to_string(index=False))

    if plot:
        # Absolute importance
        plt.figure(figsize=(10, max(4, 0.35 * len(feat_imp))))
        plt.barh(feat_imp["Feature"], feat_imp["Importance (Abs)"])
        plt.gca().invert_yaxis()
        plt.xlabel("|Coefficient| (Strength)")
        plt.title("SVM (linear) — Feature Importance (Absolute)")
        plt.tight_layout(); plt.show()

        # Signed coefficients
        plt.figure(figsize=(10, max(4, 0.35 * len(feat_imp))))
        plt.barh(feat_imp["Feature"], feat_imp["Coefficient (Raw)"])
        plt.gca().invert_yaxis()
        plt.xlabel("Coefficient (Signed)")
        plt.title("SVM (linear) — Coefficients (Direction)")
        plt.tight_layout(); plt.show()

    return feat_imp

def permutation_importance_report(model: SVC | Pipeline,
                                  X_test: pd.DataFrame,
                                  y_test: pd.Series,
                                  scoring: str = "f1",
                                  n_repeats: int = 20,
                                  plot: bool = True) -> pd.DataFrame:
    """
    Model-agnostic permutation importance on the TEST set.
    Works for any kernel (rbf/poly/sigmoid/linear).
    """
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats,
        random_state=100, n_jobs=-1, scoring=scoring
    )

    perm_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Perm Importance (Mean Δscore)": perm.importances_mean,
        "Perm Importance (Std)": perm.importances_std
    }).sort_values("Perm Importance (Mean Δscore)", ascending=False).reset_index(drop=True)

    print(f"\nPermutation Importance on test set (scoring='{scoring}'):")
    print(perm_df.to_string(index=False))

    if plot:
        # Absolute bar
        plt.figure(figsize=(10, max(4, 0.35 * len(perm_df))))
        plt.barh(perm_df["Feature"], np.abs(perm_df["Perm Importance (Mean Δscore)"]))
        plt.gca().invert_yaxis()
        plt.xlabel("Abs mean Δscore")
        plt.title("Permutation Importance (Absolute)")
        plt.tight_layout(); plt.show()

        # Signed bar
        plt.figure(figsize=(10, max(4, 0.35 * len(perm_df))))
        plt.barh(perm_df["Feature"], perm_df["Perm Importance (Mean Δscore)"])
        plt.gca().invert_yaxis()
        plt.xlabel("Mean Δscore (signed)")
        plt.title("Permutation Importance (Signed)")
        plt.tight_layout(); plt.show()

    return perm_df

# === Example usage ===
if __name__ == "__main__":
    from SVM_ML import do_model
    # For linear coefficients, pass svm_params={'kernel':'linear'}
    model, X_train, X_test, y_train, y_test, feature_names = do_model('CSVs/OPSAT-AD_modified.csv', graph=False, svm_params={'kernel':'linear'})
    # Coefficient-based importance (linear only)
    try:
        linear_coef_importance(model, feature_names, plot=True)
    except ValueError as e:
        print("Linear coef importance not available:", e)
    # Permutation importance (any kernel)
    permutation_importance_report(model, X_test, y_test, scoring="f1", n_repeats=20, plot=True)
