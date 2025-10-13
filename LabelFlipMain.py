# RunAll_Poisoned.py — run all models on poisoned dataset and save classification reports

import os
import pandas as pd
from sklearn.metrics import classification_report

# --- Imports for model runners (each must return y_true, y_pred)
from MachineLearning.LogReg.LogisticRegression_ML import run_best_model as run_logreg
from MachineLearning.NeuralNetworks.NeuralNet_ML import run_best_model as run_neuralnet
from MachineLearning.RandomForest.RandomForest_ML import run_best_model as run_randomforest
from MachineLearning.SVM.SVM_ML import run_best_model as run_svm
from MachineLearning.XGBoost.XGBoost_ML import run_best_model as run_xgboost


# --- CONFIG ---
POISONED_CSV = "CSVs/newDataset_poisoned.csv"

RESULT_PATHS = {
    "LogReg":       "Results/LogRegResults/results_logreg_poisoned.csv",
    "NeuralNet":    "Results/NeuralNetworksResults/results_neuralnet_poisoned.csv",
    "RandomForest": "Results/RandomForestResults/results_randomforest_poisoned.csv",
    "SVM":          "Results/SVMResults/results_svm_poisoned.csv",
    "XGBoost":      "Results/XGBoostResults/results_xgboost_poisoned.csv",
}


def save_report(y_true, y_pred, path):
    """Save sklearn classification report to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(rep).transpose().to_csv(path, index=True)
    print(f"Saved report to: {path}")


def main():
    # Model registry
    models = {
        "LogReg": run_logreg,
        "NeuralNet": run_neuralnet,
        "RandomForest": run_randomforest,
        "SVM": run_svm,
        "XGBoost": run_xgboost,
    }

    for name, func in models.items():
        try:
            print(f"\n=== Running {name} on poisoned dataset ===")
            y_true, y_pred = func(POISONED_CSV)  # ensure each returns y_true, y_pred
            save_report(y_true, y_pred, RESULT_PATHS[name])
        except Exception as e:
            print(f"[WARN] {name} failed: {e}")

    print("\nAll models complete.")


if __name__ == "__main__":
    main()
