# SVM_Run.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def run_best_model(path: str = "CSVs/newDataset.csv",
                   param_path: str = "MachineLearning/SVM/best_params.csv",
                   output_dir: str = "Results/SVMResults"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Output CSV paths
    last_result_csv = os.path.join(output_dir, "results_svm.csv")
    best_result_csv = os.path.join(output_dir, "results_svm_best.csv")
    poisoned_result_csv = os.path.join(output_dir, "results_svm_poisoned.csv")

    # 1) Load dataset
    df = pd.read_csv(path)

    # 2) Prepare features & labels
    y = df["anomaly"]
    X = df.drop(columns=["anomaly", "timestamp", "channel", "label"], errors="ignore")

    # 3) Read best params and split data consistently
    p = pd.read_csv(param_path).iloc[0].to_dict()
    random_state = int(p.get("random_state", 100))
    test_size = float(p.get("test_size", 0.30))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 4) Build optimized pipeline
    kernel = str(p["kernel"])
    C = float(p["C"])
    class_weight = None if str(p["class_weight"]) in ("None", "nan") else str(p["class_weight"])
    gamma = p.get("gamma", "scale")
    if kernel == "linear":
        gamma = "auto"

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel=kernel,
            C=C,
            class_weight=class_weight,
            gamma=gamma,
            probability=True,
            random_state=random_state
        ))
    ])

    # 5) Fit model & evaluate
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("\nSVM — Test Set Performance:")
    print(classification_report(y_test, y_pred))

    # 6) Save report to CSV
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(3)
    report_df.to_csv(last_result_csv, index=True)
    print(f"Saved last SVM results to {last_result_csv}")

    # 7) Compare and update best result
    def update_best(last_df, best_path):
        if os.path.exists(best_path):
            best_df = pd.read_csv(best_path)
            if "f1-score" in best_df.columns:
                last_mean = last_df["f1-score"].mean()
                best_mean = best_df["f1-score"].mean()
                if last_mean > best_mean:
                    print(f"New best SVM model found! (F1 {last_mean:.3f} > {best_mean:.3f})")
                    last_df.to_csv(best_path, index=True)
            else:
                last_df.to_csv(best_path, index=True)
        else:
            last_df.to_csv(best_path, index=True)

    update_best(report_df, best_result_csv)

    # 8) Placeholder poisoned data CSV
    if not os.path.exists(poisoned_result_csv):
        pd.DataFrame({"message": ["No poisoned data results yet"]}).to_csv(poisoned_result_csv, index=False)

    print("\n=== All SVM results updated successfully ===")


if __name__ == "__main__":
    run_best_model()
