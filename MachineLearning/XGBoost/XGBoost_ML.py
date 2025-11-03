import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    confusion_matrix,
)
import xgboost as xgb
import numpy as np


def run_best_model(path: str = "CSVs/newDataset.csv",
                   param_path: str = "MachineLearning/XGBoost/best_params.csv",
                   output_dir: str = "Results/XGBoostResults",
                   random_state: int = 100,
                   test_size: float = 0.80):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    last_result_csv = os.path.join(output_dir, "results_xgboost.csv")
    best_result_csv = os.path.join(output_dir, "results_xgboost_best.csv")
    roc_data_csv = os.path.join(output_dir, "roc_xgboost_clean.csv")
    summary_csv = os.path.join(output_dir, "xgboost_summary.csv")

    # 1) Load dataset
    df = pd.read_csv(path)
    y = df["anomaly"]
    X = df.drop(columns=["anomaly", "timestamp", "channel", "label"], errors="ignore")

    # 2) Load best hyperparameters
    p = pd.read_csv(param_path).iloc[0].to_dict()
    random_state = int(p.get("random_state", random_state))
    test_size = float(p.get("test_size", test_size))

    # 3) Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 4) Prepare XGBoost parameters
    xgb_params = {
        "n_estimators": int(p.get("n_estimators", 100)),
        "learning_rate": float(p.get("learning_rate", 0.1)),
        "max_depth": int(p.get("max_depth", 3)),
        "min_child_weight": int(p.get("min_child_weight", 1)),
        "tree_method": str(p.get("tree_method", "hist")),
        "eval_metric": str(p.get("eval_metric", "logloss")),
        "random_state": random_state,
        "n_jobs": -1,
        "use_label_encoder": False
    }

    # 5) Train model
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 6) Classification report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(3)
    report_df.to_csv(last_result_csv, index=True)
    print(f"\nSaved XGBoost last run results to {last_result_csv}")

    # 7) Update best result
    def update_best(last_df, best_path):
        if os.path.exists(best_path):
            best_df = pd.read_csv(best_path)
            if "f1-score" in best_df.columns:
                last_mean = last_df["f1-score"].mean()
                best_mean = best_df["f1-score"].mean()
                if last_mean > best_mean:
                    print(f"New best XGBoost model found! (F1 {last_mean:.3f} > {best_mean:.3f})")
                    last_df.to_csv(best_path, index=True)
            else:
                last_df.to_csv(best_path, index=True)
        else:
            last_df.to_csv(best_path, index=True)

    update_best(report_df, best_result_csv)

    # 8) Compute ROC & AUC
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        roc_df.to_csv(roc_data_csv, index=False)
        print(f"Saved ROC data to {roc_data_csv} (AUC = {roc_auc:.3f})")
    except Exception as e:
        print(f"[Warning] ROC data could not be generated: {e}")
        roc_auc = np.nan

    # 9) Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cm_df = pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"])

    # 10) Combined summary
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

    # 11) Print results
    print("\n=== XGBoost Test Set Report ===")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", cm_df)
    print(f"\nAUC: {roc_auc:.3f}")
    print("\n=== All results and summaries saved successfully ===")

    return y_test, y_pred, y_proba


if __name__ == "__main__":
    run_best_model()
