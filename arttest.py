# apply_poison_then_run.py
import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path

# --- CONFIG ---
ORIG_CSV = "CSVs/newDataset.csv"             # original dataset (keep a backup!)
POISONED_CSV = "CSVs/newDataset_poisoned.csv" # generated poisoned copy
POISON_FRACTION = 0.05                        # 5% of labels flipped
RANDOM_STATE = 100

# --- Helper: label flipping (binary) ---
def flip_labels_series(y: pd.Series, fraction: float = 0.05, random_state: int = 100,
                       flip_from=None, flip_to=None) -> tuple[pd.Series, int]:
    """
    Returns a new Series with up to `fraction` of labels flipped.
    If flip_from is None: performs binary flip 0->1 and 1->0 on chosen indices.
    Otherwise flips only labels equal to flip_from to flip_to among chosen indices.
    Also returns number of flips performed.
    """
    rng = np.random.RandomState(random_state)
    y_new = y.copy().reset_index(drop=True)
    n = len(y_new)
    n_poison = int(np.ceil(fraction * n))
    if n_poison == 0:
        return y_new, 0
    idx = rng.choice(np.arange(n), size=n_poison, replace=False)

    if flip_from is None:
        # binary flip
        before = y_new.iloc[idx].values
        y_new.iloc[idx] = 1 - y_new.iloc[idx]
        flips = (before != y_new.iloc[idx].values).sum()
    else:
        # targeted flip
        before = y_new.iloc[idx].values
        mask = (y_new.iloc[idx] == flip_from)
        y_new.iloc[idx[mask]] = flip_to
        flips = mask.sum()
    return y_new, int(flips)

# --- Main: create poisoned CSV then call run_logreg on it ---
def main():
    # 1) confirm original exists
    orig = Path(ORIG_CSV)
    if not orig.exists():
        raise FileNotFoundError(f"Original CSV not found: {ORIG_CSV}")

    # 2) load CSV
    df = pd.read_csv(orig)
    if "anomaly" not in df.columns:
        raise KeyError("Expected column 'anomaly' not found in CSV. Adjust column name accordingly.")

    # 3) flip labels (you can change flip_from/flip_to if you want targeted flips)
    y_original = df["anomaly"]
    y_poisoned, n_flips = flip_labels_series(y_original, fraction=POISON_FRACTION, random_state=RANDOM_STATE)

    # replace labels in dataframe
    df_poisoned = df.copy()
    df_poisoned["anomaly"] = y_poisoned.values

    # 4) save poisoned CSV (safe: don't overwrite original)
    poisoned_path = Path(POISONED_CSV)
    poisoned_dir = poisoned_path.parent
    poisoned_dir.mkdir(parents=True, exist_ok=True)
    df_poisoned.to_csv(poisoned_path, index=False)
    print(f"Saved poisoned CSV to: {poisoned_path}  (flipped {n_flips} labels)")

    # 5) call your existing run_logreg function using the poisoned CSV.
    #    Import it from your LogisticRegression_ML.py (adjust import if function in different module)
    try:
        # If your file lives at project root and contains run_logreg, import it:
        from MachineLearning.LogReg.LogisticRegression_ML import run_best_model as run_logreg
    except Exception as e:
        # fallback: try alternative name or show helpful error
        raise ImportError("Could not import run_logreg from LogisticRegression_ML.py. "
                          "Ensure the file is in PYTHONPATH and defines run_logreg(path).") from e

    print("Calling run_logreg on poisoned CSV — output below:")
    # This will run the same way you already call it:
    run_logreg(str(poisoned_path))

    # Optionally: keep or remove poisoned file. Comment out the next lines to keep it.
    # poisoned_path.unlink()    # uncomment to delete the poisoned file after run
    # print("Removed poisoned CSV.")

if __name__ == "__main__":
    main()
