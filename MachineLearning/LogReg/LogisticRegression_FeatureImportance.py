# LogisticRegression_FeatureImportance.py
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence
from sklearn.linear_model import LogisticRegression

def feature_importance(model: LogisticRegression,
                       feature_names: Sequence[str],
                       plot: bool = True) -> pd.DataFrame:
    """
    Compute and (optionally) plot Logistic Regression feature importance.

    Views:
      - Coefficient (Raw): keeps sign (direction of effect on anomaly odds)
      - Importance (Abs):  absolute magnitude (strength of effect, regardless of sign)

    Returns:
      feat_imp (pd.DataFrame)
    """
    # 8) Coefficients → Feature importance
    coefs = model.coef_[0]
    feat_imp = pd.DataFrame({
        "Feature": list(feature_names),
        "Coefficient (Raw)": coefs,
    })
    feat_imp["Importance (Abs)"] = feat_imp["Coefficient (Raw)"].abs()
    feat_imp = feat_imp.sort_values(by="Importance (Abs)", ascending=False).reset_index(drop=True)

    # Print compact table
    print("\nFeature Importance (Logistic Regression):")
    print(feat_imp.to_string(index=False))

    if plot:
        # Plot 1: Absolute importance (strength)
        plt.figure(figsize=(10, max(4, 0.35 * len(feat_imp))))
        plt.barh(feat_imp["Feature"], feat_imp["Importance (Abs)"])
        plt.gca().invert_yaxis()
        plt.xlabel("Absolute Coefficient (Strength)")
        plt.title("Logistic Regression — Feature Importance (Absolute)")
        plt.tight_layout()
        plt.show()

        # Plot 2: Signed coefficients (direction)
        plt.figure(figsize=(10, max(4, 0.35 * len(feat_imp))))
        plt.barh(feat_imp["Feature"], feat_imp["Coefficient (Raw)"])
        plt.gca().invert_yaxis()
        plt.xlabel("Coefficient (Signed)")
        plt.title("Logistic Regression — Feature Coefficients (Direction)")
        plt.tight_layout()
        plt.show()

    return feat_imp

# === Example usage with the ML module (mirrors original workflow without duplication) ===
if __name__ == "__main__":
    from LogisticRegression_ML import do_model
    model, X_train, X_test, y_train, y_test, feature_names = do_model('CSVs\dataset.csv', graph=False)
    feature_importance(model, feature_names, plot=True)
