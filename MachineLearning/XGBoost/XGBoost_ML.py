# XGBoost_Run.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb


def run_best_model(path: str = "CSVs/newDataset.csv",
                   param_path: str = "MachineLearning/XGBoost/best_params.csv",
                   output_dir: str = "Results/XGBoostResults"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define output CSV paths
    last_result_csv = os.path.join(output_dir, "results_xgboost.csv")
    best_result_csv = os.path.join(output_dir, "results_xgboost_best.csv")
    poisoned_result_csv = os.path.join(output_dir, "results_xgboost_poisoned.csv")

    # 1) Load dataset
    df = pd.read_csv(path)

    # 2) Prepare X, y
    y = df["anomaly"]
    X = df.drop(columns=["anomaly", "timestamp", "channel", "label"], errors="ignore")

    # 3) Load best hyperparameters
    p = pd.read_csv(param_path).iloc[0].to_dict()
    random_state = int(p.get("random_state", 100))
    test_size = float(p.get("test_size", 0.80))

    # 4) Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 5) Prepare XGBoost parameters
    xgb_params = {
        "n_estimators": int(p["n_estimators"]),
        "learning_rate": float(p["learning_rate"]),
        "max_depth": int(p["max_depth"]),
        "min_child_weight": int(p["min_child_weight"]),
        "tree_method": str(p.get("tree_method", "hist")),
        "eval_metric": str(p.get("eval_metric", "logloss")),
        "random_state": random_state,
        "n_jobs": -1,
        "use_label_encoder": False
    }

    # 6) Train the optimized XGBoost model
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nXGBoost — Test Set Performance:")
    print(classification_report(y_test, y_pred))

    # 7) Save last run results
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(3)
    report_df.to_csv(last_result_csv, index=True)
    print(f"Saved XGBoost last run results to {last_result_csv}")

    # 8) Compare & update best results
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

    # 9) Placeholder poisoned data CSV
    if not os.path.exists(poisoned_result_csv):
        pd.DataFrame({"message": ["No poisoned data results yet"]}).to_csv(poisoned_result_csv, index=False)

    print("\n=== All XGBoost results updated successfully ===")
    
    return y_test, y_pred


if __name__ == "__main__":
    run_best_model()
