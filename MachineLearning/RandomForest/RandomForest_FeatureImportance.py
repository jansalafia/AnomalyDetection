# RandomForest_FeatureImportance.py
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence
from sklearn.ensemble import RandomForestClassifier

def feature_importance(model: RandomForestClassifier,
                       feature_names: Sequence[str],
                       plot: bool = True) -> pd.DataFrame:
    """
    Compute and (optionally) plot Random Forest feature importance.
    
    View:
      - Importance (Impurity): same as model.feature_importances_

    Returns:
      feat_imp (pd.DataFrame)
    """
    importances = model.feature_importances_

    feat_imp = pd.DataFrame({
        "Feature": list(feature_names),
        "Importance (Impurity)": importances,
    })
    # For consistent sorting/plotting
    feat_imp["Importance (Abs)"] = feat_imp["Importance (Impurity)"].abs()
    feat_imp = feat_imp.sort_values(by="Importance (Abs)", ascending=False).reset_index(drop=True)

    print("\nFeature Importance (Random Forest):")
    print(feat_imp[["Feature", "Importance (Impurity)"]].to_string(index=False))

    if plot:
        # Plot: Absolute impurity-based importance (strength)
        plt.figure(figsize=(10, max(4, 0.35 * len(feat_imp))))
        plt.barh(feat_imp["Feature"], feat_imp["Importance (Abs)"])
        plt.gca().invert_yaxis()
        plt.xlabel("Importance (Impurity)")
        plt.title("Random Forest — Feature Importance (Impurity)")
        plt.tight_layout()
        plt.show()

    return feat_imp

# === Example usage with the ML module ===
if __name__ == "__main__":
    from RandomForest_ML import do_model
    model, X_train, X_test, y_train, y_test, feature_names = do_model('CSVs/OPSAT-AD_modified.csv', graph=False)
    feature_importance(model, feature_names, plot=False)


# TODO: Fit n most important features and compare performance
