# RandomForest_Run.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def run_best_model(path: str = "CSVs/newDataset.csv",
                   param_path: str = "MachineLearning/RandomForest/best_params.csv",
                   output_dir: str = "Results/RandomForestResults/",
                   test_size: float = 0.80):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Output paths
    last_result_csv = os.path.join(output_dir, "results_randomforest.csv")
    best_result_csv = os.path.join(output_dir, "results_randomforest_best.csv")
    poisoned_result_csv = os.path.join(output_dir, "results_randomforest_poisoned.csv")

    # 1) Load dataset
    df = pd.read_csv(path)

    # 2) Prepare X, y
    y = df["anomaly"]
    X = df.drop(columns=["anomaly", "timestamp", "channel", "label"], errors="ignore")

    # 3) Load tuned parameters
    p = pd.read_csv(param_path).iloc[0].to_dict()
    random_state = int(p.get("random_state", 100))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 4) Handle class weight
    cw = p.get("class_weight", None)
    if isinstance(cw, str) and cw.lower() == "none":
        cw = None

    # 5) Build RandomForest with tuned params
    rf = RandomForestClassifier(
        n_estimators=int(p["n_estimators"]),
        max_depth=int(p["max_depth"]) if str(p["max_depth"]).lower() != "none" else None,
        min_samples_split=int(p["min_samples_split"]),
        min_samples_leaf=int(p["min_samples_leaf"]),
        class_weight=cw,
        random_state=random_state,
        n_jobs=-1
    )

    # 6) Fit & evaluate
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print("\nRandomForest — Test Set Performance:")
    print(classification_report(y_test, y_pred))

    # 7) Save classification report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(3)

    # Save "Last result"
    report_df.to_csv(last_result_csv, index=True)
    print(f"Saved last run results to {last_result_csv}")

    # 8) Update "Best result" if improved
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

    # 9) Create placeholder for poisoned results if not present
    if not os.path.exists(poisoned_result_csv):
        pd.DataFrame({"message": ["No poisoned data results yet"]}).to_csv(poisoned_result_csv, index=False)

    print("\n=== All RandomForest results updated successfully ===")


if __name__ == "__main__":
    run_best_model()
