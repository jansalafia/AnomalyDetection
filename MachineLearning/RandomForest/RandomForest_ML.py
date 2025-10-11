# RandomForest_ML.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc

def do_model(path: str,
             graph: bool = False,
             random_state: int = 100,
             test_size: float = 0.80):
    # Train Random Forest on `path` and print metrics.

    # This module handles:
    #   - Loading data
    #   - Preparing X and y (drops 'anomaly','timestamp','channel','label' if present)
    #   - Train/test split (stratified if binary)
    #   - Training RandomForestClassifier
    #   - Printing train & test classification reports
    #   - Optional evaluation plots (Confusion Matrix, ROC)

    # Returns:
    #   (rf, X_train, X_test, y_train, y_test, feature_names)

    # 1) Load
    df = pd.read_csv(path)

    # 2) Prepare X, y
    y = df['anomaly']
    X = df.drop(columns=['anomaly','timestamp','channel','label'], errors='ignore')

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 4) Model (kept similar to your prior settings)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=random_state,
        oob_score=True,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 5) Evaluate
    y_pred_train = rf.predict(X_train)
    y_pred_test  = rf.predict(X_test)

    print("Model Performance:")
    print("Training Set Performance:")
    print(classification_report(y_train, y_pred_train))
    print("Test Set Performance:")
    print(classification_report(y_test, y_pred_test))

    # 6) Optional graphs
    if graph:
        from sklearn.metrics import ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
        disp.ax_.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

        # ROC Curve (binary)
        if hasattr(rf, "predict_proba"):
            fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
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
    return rf, X_train, X_test, y_train, y_test, feature_names

# === Example (kept aligned with the LogisticRegression_ML style) ===
rf, X_train, X_test, y_train, y_test, feature_names = do_model('CSVs/newDataset.csv', graph=False)

# Simple GridSearchCV example for RandomForest (mirrors the template pattern)
param_grid = {
    "n_estimators": [200, 300, 400],
    "max_depth": [10, 12, 16],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "class_weight": [None, "balanced", "balanced_subsample"]
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=100, n_jobs=-1),
    param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

print("Best params:", grid.best_params_)

# Refit on the full training split (recommended)
best_model.fit(X_train, y_train)

# Evaluate on the test split
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
