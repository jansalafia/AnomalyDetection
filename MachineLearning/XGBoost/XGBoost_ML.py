# XGBoost_ML.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, ConfusionMatrixDisplay
import xgboost as xgb

def do_model(path: str,
             graph: bool = False,
             random_state: int = 100,
             test_size: float = 0.30,
             xgb_params: dict | None = None):
    """
    Train XGBoost on the CSV at `path`, print metrics, and (optionally) show graphs.

    Simplified version — fixed overfitting by reducing n_estimators and max_depth.

    Returns:
      (clf, X_train, X_test, y_train, y_test, feature_names)
    """
    # 1) Load
    df = pd.read_csv(path)

    # 2) Prepare X, y
    y = df['anomaly']
    X = df.drop(columns=['anomaly', 'timestamp', 'channel', 'label'], errors='ignore')

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 4) Model (simplified — reduced tree depth & estimators)
    params = dict(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='logloss',
    )
    if xgb_params:
        params.update(xgb_params)

    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train)

    # 5) Evaluate
    y_pred_train = clf.predict(X_train)
    y_pred_test  = clf.predict(X_test)

    print("Model Performance:")
    print("Training Set Performance:")
    print(classification_report(y_train, y_pred_train))
    print("Test Set Performance:")
    print(classification_report(y_test, y_pred_test))

    # Optional: Confusion Matrix + ROC
    if graph:
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

        if hasattr(clf, "predict_proba"):
            fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
            auc_val = auc(fpr, tpr)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.tight_layout()
            plt.show()

    feature_names = list(X.columns)
    return clf, X_train, X_test, y_train, y_test, feature_names

# === Example ===
clf, X_train, X_test, y_train, y_test, feature_names = do_model('CSVs/OPSAT-AD_modified.csv', graph=False)
