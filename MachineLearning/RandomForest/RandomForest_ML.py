# RandomForest_Run.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def run_best_model(path: str = "CSVs/newDataset.csv",
                   param_path: str = "MachineLearning/RandomForest/best_params.csv",
                   test_size: float = 0.80):
    # 1) Load
    df = pd.read_csv(path)

    # 2) Prepare X, y
    y = df["anomaly"]
    X = df.drop(columns=["anomaly", "timestamp", "channel", "label"], errors="ignore")

    # 3) Split
    # Use the same random_state saved from tuning to keep behavior consistent
    p = pd.read_csv(param_path).iloc[0].to_dict()
    random_state = int(p.get("random_state", 100))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 4) Convert class_weight text -> Python value
    cw = p.get("class_weight", None)
    if isinstance(cw, str) and cw.lower() == "none":
        cw = None

    # 5) Build final RF with tuned params
    rf = RandomForestClassifier(
        n_estimators=int(p["n_estimators"]),
        max_depth=int(p["max_depth"]) if str(p["max_depth"]).lower() != "none" else None,
        min_samples_split=int(p["min_samples_split"]),
        min_samples_leaf=int(p["min_samples_leaf"]),
        class_weight=cw,
        random_state=random_state,
        n_jobs=-1
    )

    # 6) Fit & evaluate ONLY the optimized model
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print("\nRandomForest — Test Set Performance:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_best_model()
