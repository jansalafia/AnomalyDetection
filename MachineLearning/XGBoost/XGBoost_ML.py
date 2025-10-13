# XGBoost_Run.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb

def run_best_model(path: str = "CSVs/newDataset.csv",
                   param_path: str = "MachineLearning/XGBoost/best_params.csv"):
    # 1) Load
    df = pd.read_csv(path)

    # 2) Prepare X, y
    y = df["anomaly"]
    X = df.drop(columns=["anomaly", "timestamp", "channel", "label"], errors="ignore")

    # 3) Read best params and keep split consistent
    p = pd.read_csv(param_path).iloc[0].to_dict()
    random_state = int(p.get("random_state", 100))
    test_size = float(p.get("test_size", 0.80))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 4) Build final XGB with tuned params
    xgb_params = {
        "n_estimators":     int(p["n_estimators"]),
        "learning_rate":    float(p["learning_rate"]),
        "max_depth":        int(p["max_depth"]),
        "min_child_weight": int(p["min_child_weight"]),
        "tree_method":      str(p.get("tree_method", "hist")),
        "eval_metric":      str(p.get("eval_metric", "logloss")),
        "random_state":     random_state,
        "n_jobs":           -1,
    }

    model = xgb.XGBClassifier(**xgb_params)

    # 5) Fit & evaluate ONLY the optimized model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nXGBoost — Test Set Performance:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_best_model()
