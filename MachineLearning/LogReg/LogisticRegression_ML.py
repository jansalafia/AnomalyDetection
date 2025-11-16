import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    confusion_matrix,
)
import numpy as np


def run_best_model(path: str = "CSVs/newDataset.csv",
                   param_path: str = "MachineLearning/LogReg/best_params.csv",
                   output_dir: str = "Results/LogRegResults",
                   random_state: int = 100,
                   test_size: float = 0.80):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    last_result_csv = os.path.join(output_dir, "results_logreg.csv")
    best_result_csv = os.path.join(output_dir, "results_logreg_best.csv")
    roc_data_csv = os.path.join(output_dir, "roc_logreg_clean.csv")
    summary_csv = os.path.join(output_dir, "logreg_summary.csv")

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

    # 4) Build pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver=p["clf__solver"],
            penalty=p["clf__penalty"],
            C=float(p["clf__C"]),
            class_weight=None if p["clf__class_weight"] == "None" else p["clf__class_weight"],
            max_iter=20000,
            random_state=random_state
        ))
    ])

    # 5) Fit & predict
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # 6) Generate classification report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(3)

    # 7) Save last run
    report_df.to_csv(last_result_csv, index=True)
    print(f"\nSaved last run results to {last_result_csv}")

    # 8) Update best run
    def update_best_result(last_df, best_path):
        def mean_f1(df):
            if "f1-score" not in df.columns:
                return None
            # Convert to numeric; any junk becomes NaN and is ignored in the mean
            return pd.to_numeric(df["f1-score"], errors="coerce").mean()
    
        if os.path.exists(best_path):
            best_df = pd.read_csv(best_path)
    
            last_mean = mean_f1(last_df)
            best_mean = mean_f1(best_df)
    
            # If we cannot compute a mean for the old file, just overwrite it
            if best_mean is None or pd.isna(best_mean):
                last_df.to_csv(best_path, index=True)
            else:
                if last_mean > best_mean:
                    print(f"New best model found! (F1 {last_mean:.3f} > {best_mean:.3f})")
                    last_df.to_csv(best_path, index=True)
        else:
            last_df.to_csv(best_path, index=True)


    update_best_result(report_df, best_result_csv)

    # 9) Compute ROC and AUC
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
    else:
        y_proba = pipe.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Save ROC curve data
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_df.to_csv(roc_data_csv, index=False)
    print(f"Saved ROC data to {roc_data_csv} (AUC = {roc_auc:.3f})")

    # 10) Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cm_df = pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"])

    # 11) Combined summary (AUC + confusion matrix + main metrics)
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

    # 12) Print overview
    print("\n=== Test Set Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", cm_df)
    print(f"\nAUC: {roc_auc:.3f}")
    print("\n=== All results and summaries saved successfully ===")

    return y_test, y_pred, y_proba


if __name__ == "__main__":
    run_best_model()
