# XGBoost_ML.py
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc, ConfusionMatrixDisplay
import xgboost as xgb
import numpy as np


def do_model(path: str,
             graph: bool = False,
             random_state: int = 100,
             test_size: float = 0.80,
             xgb_params: dict | None = None):
    """
    Train XGBoost on the CSV at `path`, print metrics, and (optionally) show graphs.

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

    # 4) Model (good default, light regularization)
    params = dict(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        gamma=0.0,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='logloss',
        tree_method="hist"  # fast & robust default
    )

    # If class imbalance is present, set scale_pos_weight automatically
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    if pos > 0:
        spw = neg / pos
        # Only apply if meaningfully imbalanced
        if spw >= 1.5:
            params["scale_pos_weight"] = spw

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


# === Grid Search (mirrors your LogisticRegression/SVM/NN pattern) ===
if __name__ == "__main__":
    # 1) Build base model & get the split
    clf, X_train, X_test, y_train, y_test, feature_names = do_model('CSVs/newDataset.csv', graph=False)

    # 2) Create a small validation slice from the TRAINING data (for early stopping during CV fits)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=100
    )

    # 3) Detect imbalance for scoring & scale_pos_weight search
    pos = (y_tr == 1).sum()
    neg = (y_tr == 0).sum()
    spw = (neg / pos) if pos > 0 else 1.0
    imbalanced = spw >= 1.5

    # 4) Define search space (kept compact but effective)
    #    If imbalanced, include scale_pos_weight around the heuristic value
    spw_candidates = [1.0] if not imbalanced else [max(1.0, spw*0.5), spw, spw*1.5]

    param_grid = {
        "n_estimators":       [200, 400, 600],
        "learning_rate":      [0.05, 0.1, 0.2],
        "max_depth":          [3, 4, 6],
        "min_child_weight":   [1, 3, 5],
        "tree_method":        ["hist"],
        "eval_metric":        ["logloss"],
    }

    # 5) Choose scoring: accuracy for balanced; F1 for imbalanced
    scoring = "accuracy"

    grid = GridSearchCV(
        estimator=xgb.XGBClassifier(n_jobs=-1),
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
        refit=True
    )

    # 6) Fit with early stopping on the inner validation slice (no leakage from X_test)
    # NOTE: XGBoost's sklearn wrapper accepts these as fit() kwargs.
    grid.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_model = grid.best_estimator_
    print("Best params:", grid.best_params_)

    # 7) Refit on the full training split using early stopping vs. the same inner val slice
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # 8) Evaluate on the held-out test split
    y_pred = best_model.predict(X_test)
    print("\nGridSearchCV Best Model — Test Performance:")
    print(classification_report(y_test, y_pred))

    # Optional: Confusion Matrix + ROC for the tuned model
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    # plt.title("Confusion Matrix (Tuned XGBoost)")
    # plt.tight_layout()
    # plt.show()

    # if hasattr(best_model, "predict_proba"):
    #     fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
    #     auc_val = auc(fpr, tpr)
    #     plt.figure(figsize=(6, 5))
    #     plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.3f})")
    #     plt.plot([0, 1], [0, 1], linestyle="--")
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")
    #     plt.title("ROC Curve (Tuned XGBoost)")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
