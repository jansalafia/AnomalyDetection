# XGBoost_feature_importance.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
# from sklearn.preprocessing import StandardScaler  # Not needed for tree models
from sklearn.metrics import ConfusionMatrixDisplay
import xgboost as xgb


def do_model(path: str,
             graph: bool = False,
             show_importance: bool = True,
             xgb_params: dict | None = None):
    """
    Train XGBoost on `OPSAT-AD_modified`, print metrics, and (optionally) show feature importance.

    Mirrors the structure of your LogisticRegression script:
      1) load -> 2) prepare X,y -> 3) split -> 4) model -> 5) evaluate
      Optional: confusion matrix, ROC, and feature-importance plots.

    Returns:
      feat_imp (pd.DataFrame) if show_importance else None
    """

    # 1) Load
    df = pd.read_csv(path)

    # 2) Prepare X, y (align with your LR file: 'anomaly' is target; drop misc. columns if present)
    y = df['anomaly']
    X = df.drop(columns=['anomaly', 'timestamp', 'channel', 'label'], errors='ignore')

    # NOTE: Scaling is generally unnecessary for tree models like XGBoost
    # If you truly want it, uncomment below:
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # X = pd.DataFrame(X_scaled, columns=X.columns)

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=100, stratify=y
    )

    # 4) Model
    default_params = dict(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=4,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        random_state=100,
        n_jobs=-1,
        eval_metric='logloss',

    )
    if xgb_params:
        default_params.update(xgb_params)

    clf = xgb.XGBClassifier(**default_params)
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
        # Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

        # ROC Curve (binary)
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

    feat_imp = None

    # Feature Importance
    if show_importance:
        # XGBoost's native importance from the trained booster
        # (This is the default "gain/weight"-based overall importance exposed by scikit API)
        importances = clf.feature_importances_
        feat_imp = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        print("\nFeature Importance (XGBoost):")
        print(feat_imp.to_string(index=False))

        # Plot: Absolute importance (already non-negative for tree importances)
        plt.figure(figsize=(10, max(4, 0.35 * len(feat_imp))))
        plt.barh(feat_imp["Feature"], feat_imp["Importance"])
        plt.gca().invert_yaxis()
        plt.xlabel("Importance")
        plt.title("XGBoost — Feature Importance")
        plt.tight_layout()
        plt.show()

    return feat_imp


# === Example usage ===
do_model('CSVs/OPSAT-AD_modified.csv', graph=False, show_importance=True)
