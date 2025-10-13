import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def run_best_model(path: str = "CSVs/newDataset.csv",
                   param_path: str = "MachineLearning/LogReg/best_params.csv",
                   output_dir: str = "Results/LogRegResults",
                   random_state: int = 100,
                   test_size: float = 0.80):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    last_result_csv = os.path.join(output_dir, "results_logreg.csv")
    best_result_csv = os.path.join(output_dir, "results_logreg_best.csv")
    poisoned_result_csv = os.path.join(output_dir, "results_logreg_poisoned.csv")

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

    # 9) Placeholder for poisoned results
    if not os.path.exists(poisoned_result_csv):
        pd.DataFrame({"message": ["No poisoned data results yet"]}).to_csv(poisoned_result_csv, index=False)

    # 10) Print test set performance
    print("\nLogisticRegression — Test Set Performance:")
    print(classification_report(y_test, y_pred))
    print("\n=== All results updated successfully ===")
    
    return y_test, y_pred

if __name__ == "__main__":
    run_best_model()
