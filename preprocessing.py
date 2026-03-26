"""
preprocessing.py
----------------
Data collection, preprocessing pipeline for Credit Card Fraud Detection.
Responsible: Yousen Xie (yx697)

Deliverables to other modules:
  - X_train, y_train : balanced training set (after undersampling)
  - X_test,  y_test  : original-ratio test set (reflects real-world distribution)
  - scaler_params    : dict of {'mean': ..., 'std': ...} for Time & Amount,
                       saved to saved_models/scaler_params.npy for Streamlit use
"""

import numpy as np
import pandas as pd
import os


# ─────────────────────────────────────────────
# 1. DATA LOADING & VALIDATION
# ─────────────────────────────────────────────

def load_and_validate(filepath: str) -> pd.DataFrame:
    """
    Load the CSV and run basic sanity checks.
    Prints a summary so teammates can verify the data is correct.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            "Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place it at data/creditcard.csv"
        )

    df = pd.read_csv(filepath)

    # ── Sanity checks ──────────────────────────────────────
    assert list(df.columns[1:29]) == [f"V{i}" for i in range(1, 29)], \
        "Unexpected column names — check the CSV file."
    assert "Time"   in df.columns, "Missing 'Time' column."
    assert "Amount" in df.columns, "Missing 'Amount' column."
    assert "Class"  in df.columns, "Missing 'Class' column."

    missing = df.isnull().sum().sum()
    n_total  = len(df)
    n_fraud  = df["Class"].sum()
    n_legit  = n_total - n_fraud
    fraud_pct = n_fraud / n_total * 100

    print("=" * 50)
    print("  Dataset Validation Summary")
    print("=" * 50)
    print(f"  Total samples   : {n_total:,}")
    print(f"  Fraud  (Class=1): {n_fraud:,}  ({fraud_pct:.4f}%)")
    print(f"  Legit  (Class=0): {n_legit:,}")
    print(f"  Features        : {df.shape[1] - 1}  (30 features + 1 label)")
    print(f"  Missing values  : {missing}")
    print(f"  Dtypes          : all numeric = {all(df.dtypes != object)}")
    print("=" * 50)

    assert missing == 0, f"Found {missing} missing values — handle them before proceeding."

    return df


# ─────────────────────────────────────────────
# 2. STRATIFIED TRAIN / TEST SPLIT
#    Must come BEFORE normalization & undersampling
#    to prevent data leakage.
# ─────────────────────────────────────────────

def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Stratified 80/20 split — preserves the 0.17% fraud ratio in both partitions.

    Returns
    -------
    train_df, test_df : pd.DataFrame
    """
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    rng = np.random.default_rng(random_state)

    # Sample test indices from each class separately
    fraud_test_idx = rng.choice(len(fraud), size=int(len(fraud) * test_size), replace=False)
    legit_test_idx = rng.choice(len(legit), size=int(len(legit) * test_size), replace=False)

    fraud_test  = fraud.iloc[fraud_test_idx]
    fraud_train = fraud.iloc[np.setdiff1d(np.arange(len(fraud)), fraud_test_idx)]

    legit_test  = legit.iloc[legit_test_idx]
    legit_train = legit.iloc[np.setdiff1d(np.arange(len(legit)), legit_test_idx)]

    train_df = pd.concat([fraud_train, legit_train]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df  = pd.concat([fraud_test,  legit_test ]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"\n  Train set: {len(train_df):,} samples  "
          f"| fraud={fraud_train.shape[0]}  legit={legit_train.shape[0]}")
    print(f"  Test  set: {len(test_df):,}  samples  "
          f"| fraud={fraud_test.shape[0]}  legit={legit_test.shape[0]}")

    return train_df, test_df


# ─────────────────────────────────────────────
# 3. Z-SCORE NORMALIZATION
#    Only applied to Time and Amount.
#    V1–V28 are already PCA-transformed (zero-mean, unit-variance).
#    μ and σ are computed from TRAIN only; same params applied to TEST.
# ─────────────────────────────────────────────

def fit_scaler(train_df: pd.DataFrame) -> dict:
    """
    Compute mean and std for Time and Amount from the training set.

    Returns
    -------
    scaler_params : {'mean': pd.Series, 'std': pd.Series}
    """
    cols_to_scale = ["Time", "Amount"]
    mean = train_df[cols_to_scale].mean()
    std  = train_df[cols_to_scale].std()

    # Guard against zero std (shouldn't happen, but be safe)
    std = std.replace(0, 1)

    scaler_params = {"mean": mean, "std": std}

    print(f"\n  Scaler fitted on training set:")
    for col in cols_to_scale:
        print(f"    {col:8s}  mean={mean[col]:.4f}  std={std[col]:.4f}")

    return scaler_params


def apply_scaler(df: pd.DataFrame, scaler_params: dict) -> pd.DataFrame:
    """
    Apply z-score normalization using pre-fitted scaler_params.
    Returns a new DataFrame (does not modify in place).
    """
    df = df.copy()
    cols_to_scale = ["Time", "Amount"]
    df[cols_to_scale] = (df[cols_to_scale] - scaler_params["mean"]) / scaler_params["std"]
    return df


def save_scaler(scaler_params: dict, save_dir: str = "saved_models") -> None:
    """
    Persist scaler params as .npy files so Streamlit can normalize new inputs.
    """
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "scaler_mean.npy"), scaler_params["mean"].values)
    np.save(os.path.join(save_dir, "scaler_std.npy"),  scaler_params["std"].values)
    print(f"\n  Scaler params saved to {save_dir}/")


def load_scaler(save_dir: str = "saved_models") -> dict:
    """
    Load scaler params back from disk (used by Streamlit prediction page).
    """
    cols = ["Time", "Amount"]
    mean = pd.Series(np.load(os.path.join(save_dir, "scaler_mean.npy")), index=cols)
    std  = pd.Series(np.load(os.path.join(save_dir, "scaler_std.npy")),  index=cols)
    return {"mean": mean, "std": std}


# ─────────────────────────────────────────────
# 4. RANDOM UNDERSAMPLING
#    Applied to TRAINING SET ONLY.
#    Test set stays untouched (real-world distribution).
# ─────────────────────────────────────────────

def undersample(
    train_df: pd.DataFrame,
    ratio: float = 1.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Randomly undersample the majority (legitimate) class.

    Parameters
    ----------
    ratio : float
        Desired ratio of legit:fraud in the balanced set.
        ratio=1.0  →  equal counts (1:1)
        ratio=2.0  →  twice as many legit as fraud (2:1)

    Returns
    -------
    balanced_df : pd.DataFrame, shuffled
    """
    fraud = train_df[train_df["Class"] == 1]
    legit = train_df[train_df["Class"] == 0]

    n_sample = int(len(fraud) * ratio)
    if n_sample > len(legit):
        n_sample = len(legit)

    rng = np.random.default_rng(random_state)
    legit_sampled = legit.iloc[rng.choice(len(legit), size=n_sample, replace=False)]

    balanced_df = pd.concat([fraud, legit_sampled]) \
                    .sample(frac=1, random_state=random_state) \
                    .reset_index(drop=True)

    print(f"\n  After undersampling (ratio={ratio}):")
    print(f"    Fraud : {len(fraud)}")
    print(f"    Legit : {n_sample}")
    print(f"    Total : {len(balanced_df)}")

    return balanced_df


# ─────────────────────────────────────────────
# 5. IQR OUTLIER CAPPING  (applied to Amount)
# ─────────────────────────────────────────────

def cap_outliers_iqr(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    col: str = "Amount"
):
    """
    Compute IQR bounds from training set and cap extreme values in both sets.
    Winsorizes values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    Returns
    -------
    train_df, test_df : with capped values (copies, not in-place)
    bounds : (lower, upper) for reference
    """
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[col] = train_df[col].clip(lower, upper)
    test_df[col]  = test_df[col].clip(lower, upper)

    print(f"\n  IQR outlier capping on '{col}':")
    print(f"    Q1={Q1:.2f}  Q3={Q3:.2f}  IQR={IQR:.2f}")
    print(f"    Capping range: [{lower:.2f}, {upper:.2f}]")

    return train_df, test_df, (lower, upper)


# ─────────────────────────────────────────────
# 6. FEATURE / LABEL EXTRACTION
# ─────────────────────────────────────────────

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
LABEL_COL    = "Class"


def to_arrays(df: pd.DataFrame):
    """
    Split a DataFrame into NumPy feature matrix X and label vector y.

    Returns
    -------
    X : np.ndarray, shape (N, 30), dtype float64
    y : np.ndarray, shape (N,),   dtype int64
    """
    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df[LABEL_COL].values.astype(np.int64)
    return X, y


# ─────────────────────────────────────────────
# 7. MASTER PIPELINE  (call this from notebooks / app)
# ─────────────────────────────────────────────

def build_pipeline(
    filepath: str = "data/creditcard.csv",
    test_size: float = 0.2,
    undersample_ratio: float = 5.0,
    cap_amount_outliers: bool = True,
    random_state: int = 42,
    save_dir: str = "saved_models"
):
    """
    End-to-end preprocessing pipeline.

    Steps (in correct order to avoid data leakage):
      1. Load & validate
      2. Stratified train/test split
      3. IQR outlier capping on Amount (bounds from train)
      4. Fit scaler on train, apply to both sets
      5. Save scaler params for Streamlit
      6. Undersample training set
      7. Convert to NumPy arrays

    Returns
    -------
    X_train, y_train : balanced training arrays
    X_test,  y_test  : original-ratio test arrays
    raw_df           : original DataFrame for EDA visualizations
    scaler_params    : dict for normalizing new inputs in Streamlit
    """
    print("\n>>> Step 1: Loading data")
    df = load_and_validate(filepath)
    raw_df = df.copy()

    print("\n>>> Step 2: Stratified train/test split")
    train_df, test_df = stratified_split(df, test_size=test_size, random_state=random_state)

    if cap_amount_outliers:
        print("\n>>> Step 3: IQR outlier capping on Amount")
        train_df, test_df, _ = cap_outliers_iqr(train_df, test_df, col="Amount")

    print("\n>>> Step 4: Fit & apply Z-score normalization")
    scaler_params = fit_scaler(train_df)
    train_df = apply_scaler(train_df, scaler_params)
    test_df  = apply_scaler(test_df,  scaler_params)

    print("\n>>> Step 5: Saving scaler params")
    save_scaler(scaler_params, save_dir=save_dir)

    print("\n>>> Step 6: Undersampling training set")
    train_balanced = undersample(train_df, ratio=undersample_ratio, random_state=random_state)

    print("\n>>> Step 7: Converting to NumPy arrays")
    X_train, y_train = to_arrays(train_balanced)
    X_test,  y_test  = to_arrays(test_df)

    print(f"\n  X_train shape: {X_train.shape}  y_train: {np.bincount(y_train)}")
    print(f"  X_test  shape: {X_test.shape}   y_test:  {np.bincount(y_test)}")
    print("\n>>> Pipeline complete.\n")

    return X_train, y_train, X_test, y_test, raw_df, scaler_params


# ─────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")

    SAVE_DIR = os.path.join(BASE_DIR, "saved_models")

    X_train, y_train, X_test, y_test, raw_df, scaler_params = build_pipeline(
        filepath=DATA_PATH,
        test_size=0.2,
        undersample_ratio=5.0,
        cap_amount_outliers=True,
        random_state=42,
        save_dir=SAVE_DIR
    )

    # Export arrays for teammates
    np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(SAVE_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(SAVE_DIR, "y_test.npy"),  y_test)
    print("Arrays saved to saved_models/")
    print("  X_train.npy  y_train.npy")
    print("  X_test.npy   y_test.npy")
