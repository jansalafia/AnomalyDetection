# NeuralNet_FeatureImportance.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

def input_weight_importance(model: MLPClassifier,
                            feature_names: Sequence[str],
                            plot: bool = True) -> pd.DataFrame:
    """
    Proxy importance for MLPs using the absolute weights from INPUT -> FIRST HIDDEN layer.
    Importance = sum(abs(W_input_to_hidden), axis=1) per input feature.

    Returns a DataFrame sorted by importance (descending).
    """
    if not hasattr(model, "coefs_") or len(model.coefs_) == 0:
        raise ValueError("Model has no learned weights. Fit the MLPClassifier first.")

    W0 = model.coefs_[0]  # shape: (n_features, n_hidden)
    scores = np.sum(np.abs(W0), axis=1)

    feat_imp = pd.DataFrame({
        "Feature": list(feature_names),
        "Importance (|W| sum)": scores
    }).sort_values("Importance (|W| sum)", ascending=False).reset_index(drop=True)

    print("\nNeural Net Input-Weight Importance:")
    print(feat_imp.to_string(index=False))

    if plot:
        plt.figure(figsize=(10, max(4, 0.35 * len(feat_imp))))
        plt.barh(feat_imp["Feature"], feat_imp["Importance (|W| sum)"])
        plt.gca().invert_yaxis()
        plt.xlabel("Sum of |weights| to first hidden layer")
        plt.title("MLP — Input Weight Importance")
        plt.tight_layout()
        plt.show()

    return feat_imp

def permutation_importance_report(model: MLPClassifier,
                                  X_test: pd.DataFrame,
                                  y_test: pd.Series,
                                  scoring: str = "accuracy",
                                  n_repeats: int = 20,
                                  plot: bool = True) -> pd.DataFrame:
    """
    Model-agnostic permutation importance on the TEST set.
    Returns a DataFrame sorted by mean delta score.
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

# === Example usage with the ML module ===
if __name__ == "__main__":
    from NeuralNet_ML import do_model
    model, X_train, X_test, y_train, y_test, feature_names = do_model('CSVs/OPSAT-AD_modified.csv', graph=False)
    # Weight-based proxy importance
    input_weight_importance(model, feature_names, plot=True)
    # Permutation-based importance on test set
    permutation_importance_report(model, X_test, y_test, scoring="f1", n_repeats=20, plot=True)
