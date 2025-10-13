# poisoning_utils.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Simple label-flip poisoner for binary labels (0/1)
def label_flip_poison(X: pd.DataFrame, y: pd.Series, flip_fraction: float = 0.1,
                      flip_from: int | None = None, flip_to: int | None = None,
                      random_state: int = 0):
    """
    Flip labels for a fraction of training data.

    Args:
      X: features (DataFrame)
      y: labels (Series) - binary expected (0/1)
      flip_fraction: fraction of total train samples to flip (0..1)
      flip_from: label value to flip FROM (if None, choose majority class)
      flip_to: label value to flip TO (if None, choose the other class)
      random_state: RNG seed

    Returns:
      X_poison, y_poison (copies; original inputs are not modified)
    """
    rng = np.random.RandomState(random_state)
    Xp = X.copy().reset_index(drop=True)
    yp = y.copy().reset_index(drop=True)

    n = len(yp)
    num_to_flip = int(np.round(flip_fraction * n))
    if num_to_flip == 0:
        return Xp, yp

    unique_labels = np.unique(yp)
    if flip_from is None:
        # choose the majority label to flip by default
        counts = yp.value_counts()
        flip_from = int(counts.idxmax())
    if flip_to is None:
        flip_to = int([l for l in unique_labels if l != flip_from][0])

    # indices of candidates with label == flip_from
    candidates = yp[yp == flip_from].index.values
    if len(candidates) == 0:
        raise ValueError(f"No examples with label {flip_from} to flip.")
    chosen = rng.choice(candidates, size=min(num_to_flip, len(candidates)), replace=False)
    yp.loc[chosen] = flip_to

    return Xp, yp

# Integration example: split → poison training set → train and evaluate
def run_with_poison(path: str,
                    poison_fraction: float = 0.1,
                    flip_from: int | None = None,
                    flip_to: int | None = None,
                    test_size: float = 0.2,
                    random_state: int = 100):
    """
    Load CSV (expects an 'anomaly' column), poison a fraction of training labels,
    fit a logistic regression pipeline, and print results.
    """
    df = pd.read_csv(path)
    if "anomaly" not in df.columns:
        raise ValueError("'anomaly' column not found in CSV")

    # Prepare X/y same as your ML script
    y = df["anomaly"]
    X = df.drop(columns=["anomaly", "timestamp", "channel", "label"], errors="ignore")

    # Train/test split (poison only the training set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Poison the training labels
    X_train_p, y_train_p = label_flip_poison(
        X_train, y_train,
        flip_fraction=poison_fraction,
        flip_from=flip_from,
        flip_to=flip_to,
        random_state=random_state
    )

    print(f"Poisoned {poison_fraction*100:.1f}% of training examples (labels flipped).")

    # Build pipeline consistent with your project
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=20000, random_state=random_state))
    ])

    pipe.fit(X_train_p, y_train_p)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    print("\n=== Test Set Performance (after poisoning) ===")
    print(classification_report(y_test, y_pred))
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_test, y_proba)
            print(f"ROC AUC: {auc:.4f}")
        except Exception:
            pass
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # return objects if you want to inspect further
    return dict(pipe=pipe,
                X_train_p=X_train_p, y_train_p=y_train_p,
                X_test=X_test, y_test=y_test, y_pred=y_pred)

# Quick CLI-style run
if __name__ == "__main__":
    # Example: change the path to your CSV
    res = run_with_poison("CSVs/dataset.csv", poison_fraction=0.10, test_size=0.2)
