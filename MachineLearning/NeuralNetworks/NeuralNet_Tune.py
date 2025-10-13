# NeuralNet_Tune.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def tune_model(path: str = "CSVs/newDataset.csv",
               save_path: str = "MachineLearning/NeuralNetworks/best_params.csv",
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

    # 4) Pipeline + grid
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=random_state,
            verbose=False
        ))
    ])

    param_grid = {
        "clf__hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
        "clf__alpha": [1e-4, 5e-4, 1e-3, 1e-2],
        "clf__learning_rate_init": [1e-3, 5e-3, 1e-2],
        "clf__batch_size": ["auto", 64, 128],
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_params_
    print("Best params:", best)

    # Save only the classifier params in a simple CSV-friendly way
    hls = best["clf__hidden_layer_sizes"]
    hls_str = ",".join(str(n) for n in hls)  # e.g., "64,32" or "64"

    out = {
        "hidden_layer_sizes": hls_str,
        "alpha": best["clf__alpha"],
        "learning_rate_init": best["clf__learning_rate_init"],
        "batch_size": best["clf__batch_size"],
        # fixed parts of your structure (kept explicit for clarity/consistency)
        "activation": "relu",
        "solver": "adam",
        "learning_rate": "adaptive",
        "max_iter": 1000,
        "early_stopping": True,
        "n_iter_no_change": 20,
        "random_state": random_state
    }
    pd.DataFrame([out]).to_csv(save_path, index=False)
    print(f"Saved best parameters to {save_path}")

if __name__ == "__main__":
    tune_model()
