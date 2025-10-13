# NeuralNet_Run.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


def run_best_model(path: str = "CSVs/newDataset.csv",
                   param_path: str = "MachineLearning/NeuralNetworks/best_params.csv",
                   output_dir: str = "Results/NeuralNetworksResults",
                   random_state: int = 100,
                   test_size: float = 0.80):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    last_result_csv = os.path.join(output_dir, "results_neuralnet.csv")
    best_result_csv = os.path.join(output_dir, "results_neuralnet_best.csv")
    poisoned_result_csv = os.path.join(output_dir, "results_neuralnet_poisoned.csv")

    # 1) Load
    df = pd.read_csv(path)

    # 2) Prepare X, y
    y = df["anomaly"]
    X = df.drop(columns=["anomaly", "timestamp", "channel", "label"], errors="ignore")

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 4) Load best params
    p = pd.read_csv(param_path).iloc[0].to_dict()

    # Parse hidden_layer_sizes string -> tuple[int,...]
    hls = tuple(int(x) for x in str(p["hidden_layer_sizes"]).split(","))

    # 5) Build the final (optimized) pipeline
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

    # 6) Fit & evaluate ONLY the optimized model
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # 7) Generate classification report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Round to 3 decimals for consistency with Dash display
    report_df = report_df.round(3)

    # 8) Save current run
    report_df.to_csv(last_result_csv, index=True)
    print(f"\n Saved last run results to {last_result_csv}")

    # 9) Update best result if this run improved
    def update_best_result(last_df, best_path):
        if os.path.exists(best_path):
            best_df = pd.read_csv(best_path)
            if "f1-score" in best_df.columns:
                # Compare mean F1-score (ignoring support/avg types)
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

    # 10) Placeholder poisoned result file if not existing
    if not os.path.exists(poisoned_result_csv):
        pd.DataFrame({"message": ["No poisoned data results yet"]}).to_csv(poisoned_result_csv, index=False)

    print("\nNeuralNet — Test Set Performance:")
    print(classification_report(y_test, y_pred))
    print("\n=== All results updated successfully ===")
    
    return y_test, y_pred


if __name__ == "__main__":
    run_best_model()