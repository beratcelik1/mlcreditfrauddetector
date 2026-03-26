# Data Preprocessing — Usage Guide
**Prepared by Yousen Xie (yx697)**

---

## What's in `saved_models/`

| File | Shape | Description |
|------|-------|-------------|
| `X_train.npy` | (2364, 30) | Training features — **balanced** (1970 legit + 394 fraud) |
| `y_train.npy` | (2364,) | Training labels |
| `X_test.npy` | (56961, 30) | Test features — **original ratio** (reflects real world) |
| `y_test.npy` | (56961,) | Test labels |
| `scaler_mean.npy` | (2,) | Mean of [Time, Amount] — for normalizing new inputs |
| `scaler_std.npy` | (2,) | Std  of [Time, Amount] — for normalizing new inputs |

Labels: `0` = legitimate, `1` = fraud

---

## Feature Columns (order matters)

```python
FEATURE_COLS = ['V1','V2',...,'V28', 'Time', 'Amount']  # 30 features total
```
- `V1–V28` : PCA-transformed (already normalized by dataset provider)
- `Time`, `Amount` : Z-score normalized by our pipeline

---

## How to Load the Data

```python
import numpy as np

X_train = np.load('saved_models/X_train.npy')
y_train = np.load('saved_models/y_train.npy')
X_test  = np.load('saved_models/X_test.npy')
y_test  = np.load('saved_models/y_test.npy')
```

---

## How to Normalize a New Input (for Streamlit prediction)

```python
import numpy as np

scaler_mean = np.load('saved_models/scaler_mean.npy')  # [mean_Time, mean_Amount]
scaler_std  = np.load('saved_models/scaler_std.npy')   # [std_Time,  std_Amount]

# x is a raw input vector of shape (30,) in order [V1..V28, Time, Amount]
x = x.copy()
x[28] = (x[28] - scaler_mean[0]) / scaler_std[0]  # normalize Time
x[29] = (x[29] - scaler_mean[1]) / scaler_std[1]  # normalize Amount
```

---

## How to Regenerate the Data

If you need to change `undersample_ratio` or other parameters:

```bash
python preprocessing.py
```

Key parameters in `build_pipeline()`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `undersample_ratio` | 5.0 | legit:fraud ratio in training set |
| `test_size` | 0.2 | 80/20 split |
| `random_state` | 42 | fixed seed for reproducibility |
