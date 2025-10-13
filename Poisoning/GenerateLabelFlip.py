# LabelFlip_Simple.py — minimal label-poisoning utility
import os
from pathlib import Path
import numpy as np
import pandas as pd

# --- DEFAULTS (edit or override via function args) ---
ORIG_CSV        = "CSVs/newDataset.csv"
POISONED_CSV    = "CSVs/newDataset_poisoned.csv"
LABEL_COL       = "anomaly"
POISON_FRACTION = 0.05      # 5% of labels flipped
RANDOM_STATE    = 100


def poison_labels(
    input_csv: str = ORIG_CSV,
    output_csv: str = POISONED_CSV,
    label_col: str = LABEL_COL,
    fraction: float = POISON_FRACTION,
    random_state: int = RANDOM_STATE,
    flip_from=None,
    flip_to=None,
) -> dict:
    """
    Load CSV -> flip a fraction of labels -> save the poisoned CSV.

    Args:
        input_csv:   Path to original dataset.
        output_csv:  Path to write poisoned dataset.
        label_col:   Name of the label column to flip.
        fraction:    Fraction of labels to flip (0.0–1.0).
        random_state:RNG seed for reproducibility.
        flip_from:   If set, perform targeted flip (only labels == flip_from) ...
        flip_to:     ... set those to this value. If None, do untargeted binary flip 0<->1.

    Returns:
        Basic stats dict about the operation.
    """
    # 1) Read & validate
    src = Path(input_csv)
    if not src.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(src)
    if label_col not in df.columns:
        raise KeyError(f"Expected label column '{label_col}' not found. Columns: {list(df.columns)}")

    y = df[label_col].reset_index(drop=True)
    n = len(y)
    if n == 0:
        raise ValueError("Empty dataset — nothing to flip.")
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"'fraction' must be in [0.0, 1.0], got {fraction}.")

    rng = np.random.RandomState(random_state)

    # 2) Choose indices to flip and perform flip
    if flip_from is None:
        # Untargeted: binary flip 0<->1 on a random subset
        unique_vals = set(pd.unique(y))
        if not unique_vals.issubset({0, 1}):
            raise ValueError("Untargeted flip expects binary labels {0,1}. Use targeted flip_from/flip_to.")
        n_poison = int(np.ceil(fraction * n))
        idx = rng.choice(np.arange(n), size=n_poison, replace=False) if n_poison > 0 else np.array([], dtype=int)

        y_new = y.copy()
        if len(idx) > 0:
            y_new.iloc[idx] = 1 - y_new.iloc[idx]
        n_flipped = len(idx)
        targeted = False
    else:
        # Targeted: flip only labels equal to flip_from --> flip_to
        if flip_to is None:
            raise ValueError("For targeted flipping, 'flip_to' must be provided.")
        candidates = np.where(y.values == flip_from)[0]
        n_candidates = len(candidates)
        n_poison = int(np.ceil(fraction * n_candidates))
        chosen = rng.choice(candidates, size=n_poison, replace=False) if n_poison > 0 else np.array([], dtype=int)

        y_new = y.copy()
        if len(chosen) > 0:
            y_new.iloc[chosen] = flip_to
        n_flipped = len(chosen)
        targeted = True

    # 3) Save poisoned dataset
    df_out = df.copy()
    df_out[label_col] = y_new.values
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(output_csv, index=False)

    print(f"Saved poisoned dataset to {output_csv} (flipped {n_flipped} labels)")

    return {
        "n_rows": n,
        "n_flipped": n_flipped,
        "fraction": float(fraction),
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "label_col": str(label_col),
        "targeted": targeted,
    }


if __name__ == "__main__":
    # Minimal CLI-style usage with defaults
    poison_labels()
