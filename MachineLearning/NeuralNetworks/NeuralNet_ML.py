# NeuralNet_Run.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def run_best_model(path: str = "CSVs/newDataset.csv",
                   param_path: str = "MachineLearning/NeuralNetworks/best_params.csv",
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

    # 4) Load best params
    p = pd.read_csv(param_path).iloc[0].to_dict()

    # parse hidden_layer_sizes string -> tuple[int,...]
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

    print("\nNeuralNet — Test Set Performance:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_best_model()
