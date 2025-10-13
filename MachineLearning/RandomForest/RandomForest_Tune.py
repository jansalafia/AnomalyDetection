# RandomForest_Tune.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def tune_model(path: str = "CSVs/newDataset.csv",
               save_path: str = "MachineLearning/RandomForest/best_params.csv",
               random_state: int = 100,
               test_size: float = 0.80):
    # 1) Load
    df = pd.read_csv(path)

    # 2) Prepare X, y
    y = df["anomaly"]
    X = df.drop(columns=["anomaly", "timestamp", "channel", "label"], errors="ignore")

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 4) Grid
    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [10, 12, 16],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_grid=param_grid,
        cv=5, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best = grid.best_params_
    print("Best params:", best)

    # Save to CSV (ensure class_weight serializes cleanly)
    out = best.copy()
    out["random_state"] = random_state
    pd.DataFrame([out]).to_csv(save_path, index=False)
    print(f"Saved best parameters to {save_path}")

if __name__ == "__main__":
    tune_model()
