# SVM_Run.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def run_best_model(path: str = "CSVs/newDataset.csv",
                   param_path: str = "MachineLearning/SVM/best_params.csv"):
    # 1) Load
    df = pd.read_csv(path)

    # 2) Prepare X, y
    y = df["anomaly"]
    X = df.drop(columns=["anomaly", "timestamp", "channel", "label"], errors="ignore")

    # 3) Read best params and keep split consistent
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
    gamma = gamma if kernel != "linear" else "auto"  # gamma not used for linear

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel=kernel,
            C=C,
            class_weight=class_weight,
            gamma=gamma,
            probability=True
        ))
    ])

    # 5) Fit & evaluate ONLY the optimized model
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("\nSVM — Test Set Performance:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_best_model()
