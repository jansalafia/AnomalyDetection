# LabelFlip.py — create a poisoned dataset and run all models on it
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


# --- CONFIG ---
ORIG_CSV        = "CSVs/newDataset.csv"
POISONED_CSV    = "CSVs/newDataset_poisoned.csv"
POISON_FRACTION = 0.05      # 5% of labels flipped
RANDOM_STATE    = 100


# --- Helper: Label flipping (binary) ---
def flip_labels_series(
    y: pd.Series,
    fraction: float = 0.25,
    random_state: int = 100,
    flip_from=None,
    flip_to=None
) -> tuple[pd.Series, int]:
    """
    Flip a fraction of labels in a binary or targeted way.
    Returns (flipped_series, number_of_flips)
    """
    rng = np.random.RandomState(random_state)
    y_new = y.copy().reset_index(drop=True)
    n = len(y_new)
    n_poison = int(np.ceil(fraction * n))
    if n_poison == 0:
        return y_new, 0

    idx = rng.choice(np.arange(n), size=n_poison, replace=False)
    if flip_from is None:
        # Binary flip 0 ↔ 1
        before = y_new.iloc[idx].values
        y_new.iloc[idx] = 1 - y_new.iloc[idx]
        flips = int((before != y_new.iloc[idx].values).sum())
    else:
        # Targeted flip
        mask = (y_new.iloc[idx] == flip_from)
        y_new.iloc[idx[mask]] = flip_to
        flips = int(mask.sum())

    return y_new, flips


# --- Helper: Save poisoning summary ---
def save_poisoning_report(y_before: pd.Series, y_after: pd.Series, out_csv: str):
    """Save a classification-style report comparing original vs poisoned labels."""
    report_dict = classification_report(y_before, y_after, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    report_df.to_csv(out_csv, index=True)
    print(f"Saved label-flip report to: {out_csv}")


# --- Run each model ---
def run_all_models(poisoned_csv_path: str):
    """
    Runs all ML models on the poisoned dataset.
    Each model saves its results to its own Results/{model}/results_*_poisoned.csv file.
    """
    try:
        from MachineLearning.LogReg.LogisticRegression_ML import run_best_model as run_logreg
        from MachineLearning.SVM.SVM_ML import run_best_model as run_svm
        from MachineLearning.RandomForest.RandomForest_ML import run_best_model as run_rf
        from MachineLearning.XGBoost.XGBoost_ML import run_best_model as run_xgb
        from MachineLearning.NeuralNetworks.NeuralNet_ML import run_best_model as run_nn
    except Exception as e:
        raise ImportError("Failed to import one or more ML model modules. Check import paths.") from e

    models = [
        ("LogReg",        run_logreg,      "results_logreg_poisoned.csv"),
        ("SVM",           run_svm,         "results_svm_poisoned.csv"),
        ("RandomForest",  run_rf,          "results_randomforest_poisoned.csv"),
        ("XGBoost",       run_xgb,         "results_xgboost_poisoned.csv"),
        ("NeuralNet",     run_nn,          "results_neuralnet_poisoned.csv"),
    ]

    print("\n=== Running all models on poisoned dataset ===")
    for name, runner, out_file in models:
        try:
            out_dir = f"Results/{name}"
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n→ Running {name}...")
            runner(poisoned_csv_path)

            # Move or overwrite model output if it saves somewhere else
            print(f"Expected output: {out_dir}/{out_file}")
        except Exception as e:
            print(f"[WARN] {name} failed: {e}")


# --- Main workflow ---
def main():
    # 1) Check dataset
    orig = Path(ORIG_CSV)
    if not orig.exists():
        raise FileNotFoundError(f"Original CSV not found: {ORIG_CSV}")

    # 2) Load and validate
    df = pd.read_csv(orig)
    if "anomaly" not in df.columns:
        raise KeyError("Expected column 'anomaly' not found in CSV. Adjust column name accordingly.")

    # 3) Poison labels
    y_original = df["anomaly"]
    y_poisoned, n_flips = flip_labels_series(y_original, fraction=POISON_FRACTION, random_state=RANDOM_STATE)

    df_poisoned = df.copy()
    df_poisoned["anomaly"] = y_poisoned.values

    # 4) Save poisoned dataset
    os.makedirs(os.path.dirname(POISONED_CSV), exist_ok=True)
    df_poisoned.to_csv(POISONED_CSV, index=False)
    print(f"\nSaved poisoned dataset to {POISONED_CSV} (flipped {n_flips} labels)")

    # 5) Save poisoning summary
    save_poisoning_report(y_original, y_poisoned, "Results/Poisoning/label_flip_report.csv")

    # 6) Run all models
    run_all_models(POISONED_CSV)


if __name__ == "__main__":
    main()
