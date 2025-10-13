# LogisticRegression_Run.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def run_best_model(
    path: str, 
    param_path: str = "MachineLearning/LogReg/best_params.csv", 
    output_csv: str = "Results/Logistic Regression/results_logreg.csv"
):
    # Load dataset
    df = pd.read_csv(path)
    y = df["anomaly"]
    X = df.drop(columns=["anomaly", "timestamp", "channel", "label"], errors="ignore")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, stratify=y, random_state=100
    )

    # Read best parameters
    best_params = pd.read_csv(param_path).iloc[0].to_dict()
    print("Using parameters:", best_params)

    # Build pipeline using the best parameters
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver=best_params["clf__solver"],
            penalty=best_params["clf__penalty"],
            C=float(best_params["clf__C"]),
            class_weight=None if best_params["clf__class_weight"] == "None" else best_params["clf__class_weight"],
            max_iter=20000,
            random_state=100
        ))
    ])

    # Fit and evaluate
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Print performance
    print("\n=== Test Set Performance ===")
    report = classification_report(y_test, y_pred)
    print(report)

    # Save report to CSV
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(output_csv, index=True)
    print(f"\nSaved classification report to {output_csv}")

if __name__ == "__main__":
    run_best_model("CSVs/newDataset.csv")
