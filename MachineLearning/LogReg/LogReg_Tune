# LogisticRegression_Tune.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def tune_model(path: str, save_path: str = "MachineLearning/LogReg/best_params.csv"):
    df = pd.read_csv(path)
    y = df["anomaly"]
    X = df.drop(columns=["anomaly", "timestamp", "channel", "label"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, stratify=y, random_state=100
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=20000, random_state=100))
    ])

    param_grid = {
        "clf__solver": ["liblinear"],
        "clf__penalty": ["l2"],
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__class_weight": [None, "balanced"]
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    print("Best parameters found:", best_params)

    # Save to CSV
    pd.DataFrame([best_params]).to_csv(save_path, index=False)
    print(f"Saved best parameters to {save_path}")

if __name__ == "__main__":
    tune_model("CSVs/newDataset.csv")
