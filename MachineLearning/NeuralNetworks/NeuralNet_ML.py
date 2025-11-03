import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    confusion_matrix,
)
import numpy as np


def run_best_model(path: str = "CSVs/newDataset.csv",
                   param_path: str = "MachineLearning/NeuralNetworks/best_params.csv",
                   output_dir: str = "Results/NeuralNetworksResults",
                   random_state: int = 100,
                   test_size: float = 0.80):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    last_result_csv = os.path.join(output_dir, "results_neuralnet.csv")
    best_result_csv = os.path.join(output_dir, "results_neuralnet_best.csv")
    roc_data_csv = os.path.join(output_dir, "roc_neuralnet_clean.csv")
    summary_csv = os.path.join(output_dir, "neuralnet_summary.csv")

    # 1) Load dataset
    df = pd.read_csv(path)
    y = df["anomaly"]
    X = df.drop(columns=["anomaly", "timestamp", "channel", "label"], errors="ignore")

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 3) Load best parameters
    p = pd.read_csv(param_path).iloc[0].to_dict()
    print("Using parameters:", p)

    # Parse hidden_layer_sizes safely
    hls = tuple(int(x) for x in str(p["hidden_layer_sizes"]).split(","))

    # 4) Build pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=hls,
            activation=str(p["activation"]),
            solver=str(p["solver"]),
            alpha=float(p["alpha"]),
            batch_size=(str(p["batch_size"]) if str(p["batch_size"]) == "auto" else int(p["batch_size"])),
            learning_rate=str(p["learning_rate"]),
            learning_rate_init=float(p["learning_rate_init"]),
            max_iter=int(p["max_iter"]),
            early_stopping=bool(p["early_stopping"]),
            n_iter_no_change=int(p["n_iter_no_change"]),
            random_state=int(p["random_state"]),
            verbose=False
        ))
    ])

    # 5) Fit & predict
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # 6) Classification report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(3)

    # 7) Save current run
    report_df.to_csv(last_result_csv, index=True)
    print(f"\nSaved last run results to {last_result_csv}")

    # 8) Update best result if improved
    def update_best_result(last_df, best_path):
        if os.path.exists(best_path):
            best_df = pd.read_csv(best_path)
            if "f1-score" in best_df.columns:
                last_mean = last_df["f1-score"].mean()
                best_mean = best_df["f1-score"].mean()
                if last_mean > best_mean:
                    print(f"New best model found! (F1 {last_mean:.3f} > {best_mean:.3f})")
                    last_df.to_csv(best_path, index=True)
            else:
                last_df.to_csv(best_path, index=True)
        else:
            last_df.to_csv(best_path, index=True)

    update_best_result(report_df, best_result_csv)

    # 9) ROC and AUC computation
    y_proba = None
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        roc_df.to_csv(roc_data_csv, index=False)
        print(f"Saved ROC data to {roc_data_csv} (AUC = {roc_auc:.3f})")
    except Exception as e:
        print(f"[Warning] ROC data could not be generated: {e}")
        roc_auc = np.nan

    # 10) Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cm_df = pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"])

    # 11) Combined summary
    summary_data = {
        "dataset": ["clean"],
        "auc": [roc_auc],
        "accuracy": [report_dict["accuracy"]],
        "precision_0": [report_dict["0"]["precision"]],
        "recall_0": [report_dict["0"]["recall"]],
        "f1_0": [report_dict["0"]["f1-score"]],
        "precision_1": [report_dict["1"]["precision"]],
        "recall_1": [report_dict["1"]["recall"]],
        "f1_1": [report_dict["1"]["f1-score"]],
        "tp": [tp],
        "fp": [fp],
        "tn": [tn],
        "fn": [fn],
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary (AUC + Confusion Matrix) to {summary_csv}")

    # 12) Print results
    print("\n=== Neural Network Test Set Report ===")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", cm_df)
    print(f"\nAUC: {roc_auc:.3f}")
    print("\n=== All results and summaries saved successfully ===")

    return y_test, y_pred, y_proba


if __name__ == "__main__":
    run_best_model()
