[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/dJBAHCPL)

# Credit Card Fraud Detection with Machine Learning

INFO 5368: Practical Applications in Machine Learning — Final Project (Spring 2026)

**Team:** Berat Celik (bc729), Zijing Wu (zw795), Yuxiang Jiang (yj548), Yousen Xie (yx697), Gabrielle Xiao (mx262), Binyao Zhao (bz383)

This project detects fraudulent credit card transactions using two machine learning models implemented entirely from scratch with NumPy: Logistic Regression and K-Nearest Neighbors. We evaluate both on the Kaggle Credit Card Fraud Detection dataset (284,807 transactions, 492 fraud) and deploy the best model in a Streamlit web application.

## Results Summary

| Model | Precision | Recall | F1 | AUC-ROC |
|-------|-----------|--------|----|---------|
| Logistic Regression | 0.378 | 0.806 | **0.515** | **0.959** |
| KNN (k=5, distance) | 0.339 | **0.837** | 0.482 | 0.939 |

Both models exceed our 0.80 recall target on the imbalanced test set.

## Repository Contents

| File / Folder | Description |
|---|---|
| `app.py` | Streamlit application home page |
| `pages/` | Four Streamlit pages: Explore, Train, Test, Deploy |
| `helper_functions.py` | Model loading, prediction, normalization for the app |
| `preprocessing.py` | Data loading, validation, normalization, undersampling, train/test split |
| `evaluation.py` | Shared metrics (precision, recall, F1, AUC-ROC, confusion matrix, ROC curve) |
| `knn.py` | KNN classifier from scratch + 5-fold stratified CV tuning |
| `train_knn.py` | KNN training pipeline |
| `train_logistic.py` | Logistic Regression class + grid search training pipeline |
| `notebooks/` | Two Jupyter notebooks documenting offline experiments |
| `data/creditcard.csv` | Kaggle Credit Card Fraud dataset |
| `saved_models/` | Pre-trained model artifacts (`.npy` files) |
| `requirements.txt` | Python dependencies |

## Setup

```bash
git clone https://github.com/Cornell-Tech-PAML-Course-2026/final-project-submission-aag.git
cd final-project-submission-aag
pip install -r requirements.txt
```

## Run the Pipeline From Scratch

The `saved_models/` folder already contains pre-trained artifacts. To regenerate everything:

```bash
# 1. Preprocess data (creates X_train, y_train, X_test, y_test, scaler params)
python preprocessing.py

# 2. Train Logistic Regression (saves weights, bias, threshold, ROC data)
python train_logistic.py

# 3. Train KNN (saves training data, params, ROC data)
python train_knn.py
```

## Run the Streamlit App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` with four pages:

1. **Explore Data** — interactive Plotly charts for class distribution, feature histograms, box plots, scatter plots, and correlation heatmaps.
2. **Train Model** — hyperparameter sliders (learning rate, iterations, threshold for LR; k and weighting for KNN) and train buttons.
3. **Test Model** — side-by-side metrics, confusion matrix heatmaps, overlaid ROC curves, deployment recommendation.
4. **Deploy App** — manual feature input or CSV upload, returns color-coded fraud/legit predictions with confidence scores.

## Deployed Application

[Live demo on Streamlit Community Cloud](#) — link added after deployment.

## Implementation Notes

- All algorithms implemented from scratch using only NumPy and Pandas. No scikit-learn, no TensorFlow, no PyTorch.
- Logistic Regression trained via batch gradient descent on binary cross-entropy loss.
- KNN uses vectorized Euclidean distance computation with optional inverse-distance weighting.
- Hyperparameter tuning: grid search for LR (27 configs), 5-fold stratified cross-validation for KNN (10 configs).
- Class imbalance handled via random undersampling at 5:1 (legitimate:fraud) ratio in the training set; test set retains the original 0.17% fraud ratio for realistic evaluation.

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions from European cardholders in September 2013. Features V1–V28 are PCA-transformed (original features confidential). Time, Amount, and Class are raw.
