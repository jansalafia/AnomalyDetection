# NeuralNet_ML.py
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc, ConfusionMatrixDisplay

def do_model(path: str,
             graph: bool = False,
             random_state: int = 100,
             test_size: float = 0.80,
             nn_params: dict | None = None):
    """
    Train a simple Neural Network (MLPClassifier) on the CSV at `path` and print metrics.

    Steps (mirrors your other ML files):
      1) Load -> 2) Prepare X,y -> 3) Split -> 4) Model -> 5) Evaluate
      Optional: Confusion Matrix + ROC

    Returns:
      (mlp, X_train, X_test, y_train, y_test, feature_names)
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

    # 4) Model
    params = dict(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0005,          # L2 regularization
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=random_state,
        early_stopping=True,   # prevent overfitting & auto-stop
        n_iter_no_change=20,
        verbose=False
    )
    if nn_params:
        params.update(nn_params)

    mlp = MLPClassifier(**params)
    mlp.fit(X_train, y_train)

    # 5) Evaluate
    y_pred_train = mlp.predict(X_train)
    y_pred_test  = mlp.predict(X_test)

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
        if hasattr(mlp, "predict_proba"):
            fpr, tpr, _ = roc_curve(y_test, mlp.predict_proba(X_test)[:, 1])
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
    return mlp, X_train, X_test, y_train, y_test, feature_names


# === Example (same usage style) ===
# mlp, X_train, X_test, y_train, y_test, feature_names = do_model('CSVs\dataset.csv', graph=False)


# === Grid Search (similar style to LogisticRegression_ML.py) ===
if __name__ == "__main__":
    # Run once to get the split
    mlp, X_train, X_test, y_train, y_test, feature_names = do_model('CSVs\dataset.csv', graph=False)

    # Scale features before MLP (important)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=100,
            verbose=False
        ))
    ])

    # Compact but effective search space
    param_grid = {
        "clf__hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
        "clf__alpha": [1e-4, 5e-4, 1e-3, 1e-2],         # L2
        "clf__learning_rate_init": [1e-3, 5e-3, 1e-2],
        "clf__batch_size": ["auto", 64, 128]
    }

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
