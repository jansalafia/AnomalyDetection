# XGBoost_Tune.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import xgboost as xgb

def tune_model(path: str = "CSVs/newDataset.csv",
               save_path: str = "MachineLearning/XGBoost/best_params.csv",
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

    # 4) Compact search space (aligned with your file)
    param_grid = {
        "n_estimators":     [200, 400, 600],
        "learning_rate":    [0.05, 0.1, 0.2],
        "max_depth":        [3, 4, 6],
        "min_child_weight": [1, 3, 5],
        "tree_method":      ["hist"],
        "eval_metric":      ["logloss"],
    }

    grid = GridSearchCV(
        estimator=xgb.XGBClassifier(
            n_jobs=-1,
            random_state=random_state,
        ),
        param_grid=param_grid,
        cv=5,
        scoring=make_scorer(accuracy_score),
        n_jobs=-1,
        refit=True
    )

    grid.fit(X_train, y_train)
    best = grid.best_params_
    print("Best params:", best)

    # Save plus random_state for reproducible split later
    out = best.copy()
    out["random_state"] = random_state
    out["test_size"] = test_size
    pd.DataFrame([out]).to_csv(save_path, index=False)
    print(f"Saved best parameters to {save_path}")

if __name__ == "__main__":
    tune_model()
