# SVM_ML.py
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve, auc, ConfusionMatrixDisplay

def do_model(path: str,
             graph: bool = False,
             random_state: int = 100,
             test_size: float = 0.30,
             svm_params: dict | None = None):
    """
    Train an SVM on the CSV at `path`, print metrics, and (optionally) show graphs.

    Steps (matches your other ML files):
      1) Load -> 2) Prepare X,y -> 3) Split -> 4) Model -> 5) Evaluate
      Optional: Confusion Matrix + ROC

    Returns:
      (model, X_train, X_test, y_train, y_test, feature_names)
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

    # 4) Model (SVM benefits from scaling → use a Pipeline)
    params = dict(
        C=1.0,
        kernel='rbf',        # use 'linear' if you want coefficient-based importance
        gamma='scale',
        probability=True,    # enables predict_proba for ROC
    )
    if svm_params:
        params.update(svm_params)

    model = Pipeline(steps=[
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('clf', SVC(**params))
    ])

    model.fit(X_train, y_train)

    # 5) Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

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

        # ROC Curve (binary)
        clf = model.named_steps['clf']
        if hasattr(clf, "predict_proba") and getattr(clf, "probability", False):
            scores = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
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
    return model, X_train, X_test, y_train, y_test, feature_names


# === Example training (same usage style) ===
# model, X_train, X_test, y_train, y_test, feature_names = do_model('CSVs\ToBeMerged\dataset.csv', graph=False)


# === Grid Search (similar style to LogisticRegression_ML.py) ===
if __name__ == "__main__":
    # Run once to get the split
    model, X_train, X_test, y_train, y_test, feature_names = do_model('CSVs\ToBeMerged\dataset.csv', graph=False)

    # Pipeline for tuning (always scale for SVM)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True))
    ])

    # Reasonable search space (kept compact to avoid super long runs)
    param_grid = [
        {
            "clf__kernel": ["rbf"],
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", "auto", 0.01, 0.001],
            "clf__class_weight": [None, "balanced"],
        },
        {
            "clf__kernel": ["linear"],
            "clf__C": [0.1, 1, 10, 100],
            "clf__class_weight": [None, "balanced"],
        }
    ]

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("Best params:", grid.best_params_)
    # Refit on the full training split (recommended)
    best_model.fit(X_train, y_train)

    # Evaluate on the test split
    y_pred = best_model.predict(X_test)
    print("\nGridSearchCV Best Model — Test Performance:")
    print(classification_report(y_test, y_pred))
