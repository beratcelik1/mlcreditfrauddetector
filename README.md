# Credit Card Fraud Detector

Machine learning pipeline for detecting fraudulent credit card transactions. Logistic Regression and K-Nearest Neighbors implemented from scratch using NumPy (no scikit-learn, no TensorFlow, no PyTorch).

Built for INFO 5368: Practical Applications in Machine Learning.

## Team

Berat Celik, Zijing Wu, Yuxiang Jiang, Yousen Xie, Gabrielle Xiao, Binyao Zhao

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 fraud (0.17%), 28 PCA features + Time + Amount.

Download and place at `data/creditcard.csv`.

## Setup

```bash
pip install -r requirements.txt
```

## Run the Pipeline

```bash
# 1. Preprocess data (generates train/test splits in saved_models/)
python preprocessing.py

# 2. Train Logistic Regression
python train_logistic.py

# 3. Train KNN
python train_knn.py

# 4. Launch Streamlit app
streamlit run app.py
```

## Project Structure

| File | Description |
|------|-------------|
| `preprocessing.py` | Data loading, validation, normalization, undersampling, train/test split |
| `train_logistic.py` | Logistic Regression from scratch with gradient descent, hyperparameter tuning |
| `knn.py` | KNN classifier from scratch with vectorized distance computation |
| `train_knn.py` | KNN training with 5-fold stratified cross-validation |
| `evaluation.py` | Shared metrics: precision, recall, F1, AUC-ROC, confusion matrix |
| `app.py` | Streamlit web application (4 pages) |

## Results

| Metric | Logistic Regression | KNN (k=5, distance) |
|--------|-------------------|-------------------|
| Precision | 0.378 | 0.339 |
| Recall | 0.806 | 0.837 |
| F1 | 0.515 | 0.482 |
| AUC-ROC | 0.959 | 0.939 |

Both models achieve recall above 0.80 on the imbalanced test set (56,961 samples, 98 fraud).
