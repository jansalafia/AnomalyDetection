# SVM_Run.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve, auc


def run_best_model(path: str = "CSVs/newDataset.csv",
                   param_path: str = "MachineLearning/SVM/best_params.csv",
                   output_dir: str = "Results/SVMResults"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Output CSV paths
    last_result_csv = os.path.join(output_dir, "results_svm.csv")
    best_result_csv = os.path.join(output_dir, "results_svm_best.csv")
    poisoned_result_csv = os.path.join(output_dir, "results_svm_poisoned.csv")

    # ROC data files (NEW)
    roc_data_csv = os.path.join(output_dir, "roc_svm_clean.csv")
    auc_summary_csv = os.path.join(output_dir, "roc_auc_summary.csv")

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
            probability=True,  # Needed for ROC curves
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

    # 8) Compute and save ROC data
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        # Save ROC data
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        roc_df.to_csv(roc_data_csv, index=False)
        print(f"Saved ROC data to {roc_data_csv} (AUC = {roc_auc:.3f})")

        # Save AUC summary
        pd.DataFrame([{"dataset": "clean", "auc": roc_auc}]).to_csv(auc_summary_csv, index=False)
    except Exception as e:
        print(f"[Warning] ROC data could not be generated: {e}")

    print("\n=== All SVM results and ROC data updated successfully ===")

    return y_test, y_pred, y_proba if "y_proba" in locals() else None


if __name__ == "__main__":
    run_best_model()
