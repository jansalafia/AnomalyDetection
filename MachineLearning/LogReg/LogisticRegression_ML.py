# LogisticRegression_ML.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler  # STANDARDIZING ALTERS FEATURE IMPORTANCE
from sklearn.metrics import classification_report, roc_curve, auc

def do_model(path: str,
             graph: bool = False,
             random_state: int = 100,
             test_size: float = 0.80):

    # Train Logistic Regression on `path` and print metrics.

    # This module handles:
    #   - Loading data
    #   - Preparing X and y (drops 'anomaly', 'timestamp', 'channel', 'label' if present)
    #   - Train/test split (stratified)
    #   - Training LogisticRegression
    #   - Printing train & test classification reports
    #   - Optional evaluation plots (Confusion Matrix, ROC)

    # Returns:
    #   (logreg, X_train, X_test, y_train, y_test, feature_names)


    # 1) Load
    df = pd.read_csv(path)

    # 2) Prepare X, y
    y = df['anomaly']
    X = df.drop(columns=['anomaly', 'timestamp', 'channel', 'label'], errors='ignore')

    # 3) Standardize (disabled to keep raw coefficient interpretability consistent)
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # X = pd.DataFrame(X_scaled, columns=X.columns)

    # 4) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 5) Model
    logreg = LogisticRegression(max_iter=100_000, random_state=random_state)
    logreg.fit(X_train, y_train)

    # 6) Evaluate
    y_pred_train = logreg.predict(X_train)
    y_pred_test  = logreg.predict(X_test)

    print("Model Performance:")
    print("Training Set Performance:")
    print(classification_report(y_train, y_pred_train))
    print("Test Set Performance:")
    print(classification_report(y_test, y_pred_test))

    # 7) Optional graphs
    if graph:
        from sklearn.metrics import ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
        disp.ax_.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

        # ROC Curve (binary)
        if hasattr(logreg, "predict_proba"):
            fpr, tpr, _ = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
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
    return logreg, X_train, X_test, y_train, y_test, feature_names

# === Example ===
logreg, X_train, X_test, y_train, y_test, feature_names = do_model('CSVs/OPSAT-AD_modified.csv', graph=False)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=20000, random_state=100))
])

param_grid = {
    "clf__solver": ["liblinear"],   
    "clf__penalty": ["l2", "l1"],                
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__class_weight": [None, "balanced"]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
# after: grid.fit(X_train, y_train)
best_model = grid.best_estimator_         # already fitted on the CV folds

print("Best params:", grid.best_params_)

# Refit on the full training split (recommended)
best_model.fit(X_train, y_train)

# Evaluate on the test split
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))