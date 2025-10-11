# XGBoost_FeatureImportance.py
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence
import xgboost as xgb

def feature_importance(model: xgb.XGBClassifier,
                       feature_names: Sequence[str],
                       plot: bool = True) -> pd.DataFrame:
    """
    Compute and (optionally) plot XGBoost feature importance.

    View:
      - Importance (Gain/Weight exposed via scikit API as feature_importances_)

    Returns:
      feat_imp (pd.DataFrame)
    """
    importances = model.feature_importances_

    feat_imp = pd.DataFrame({
        "Feature": list(feature_names),
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    print("\nFeature Importance (XGBoost):")
    print(feat_imp.to_string(index=False))

    if plot:
        plt.figure(figsize=(10, max(4, 0.35 * len(feat_imp))))
        plt.barh(feat_imp["Feature"], feat_imp["Importance"])
        plt.gca().invert_yaxis()
        plt.xlabel("Importance")
        plt.title("XGBoost — Feature Importance")
        plt.tight_layout()
        plt.show()

    return feat_imp

# === Example usage with the ML module (mirrors your LogisticRegression_FeatureImportance style) ===
if __name__ == "__main__":
    from XGBoost_ML import do_model
    model, X_train, X_test, y_train, y_test, feature_names = do_model('CSVs\dataset.csv', graph=False)
    feature_importance(model, feature_names, plot=True)
