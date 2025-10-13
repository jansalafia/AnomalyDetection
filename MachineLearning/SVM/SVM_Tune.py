# SVM_Tune.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def tune_model(path: str = "CSVs/newDataset.csv",
               save_path: str = "MachineLearning/SVM/best_params.csv",
               random_state: int = 100,
               test_size: float = 0.90):
    # 1) Load
    df = pd.read_csv(path)

    # 2) Prepare X, y
    y = df["anomaly"]
    X = df.drop(columns=["anomaly", "timestamp", "channel", "label"], errors="ignore")

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 4) Pipeline + compact search space (as in your current workflow)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True))
    ])

    param_grid = [
        {
            "clf__kernel": ["rbf", "linear"],
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", "auto", 0.01, 0.001],
            "clf__class_weight": [None, "balanced"],
        },
    ]

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_params_
    print("Best params:", best)

    # Normalize for CSV (ensure all keys exist)
    kernel = best["clf__kernel"]
    out = {
        "kernel": kernel,
        "C": best["clf__C"],
        "class_weight": best.get("clf__class_weight", None),
        "gamma": best.get("clf__gamma", "scale" if kernel == "rbf" else "auto"),
        "random_state": random_state,
        "test_size": test_size
    }
    pd.DataFrame([out]).to_csv(save_path, index=False)
    print(f"Saved best parameters to {save_path}")

if __name__ == "__main__":
    tune_model()
