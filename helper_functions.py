import numpy as np
import pandas as pd
import streamlit as st
import os


def fetch_dataset():
    """
    Load the credit card fraud dataset. Checks session state first,
    then tries local file, then offers file upload.
    """
    df = None

    if "data" in st.session_state:
        df = st.session_state["data"]
    else:
        data_path = os.path.join(os.path.dirname(__file__), "data", "creditcard.csv")
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            st.session_state["data"] = df
        else:
            data = st.file_uploader("Upload creditcard.csv", type=["csv"])
            if data:
                df = pd.read_csv(data)
                st.session_state["data"] = df

    return df


def load_saved_models():
    """Load pre-trained model artifacts from saved_models/."""
    save_dir = os.path.join(os.path.dirname(__file__), "saved_models")

    artifacts = {}

    # Check if artifacts exist
    required = [
        "logreg_weights.npy",
        "logreg_bias.npy",
        "logreg_threshold.npy",
        "knn_X_train.npy",
        "knn_y_train.npy",
        "knn_params.npy",
        "scaler_mean.npy",
        "scaler_std.npy",
    ]

    for f in required:
        if not os.path.exists(os.path.join(save_dir, f)):
            return None

    # Logistic Regression
    artifacts["lr_weights"] = np.load(os.path.join(save_dir, "logreg_weights.npy"))
    artifacts["lr_bias"] = float(np.load(os.path.join(save_dir, "logreg_bias.npy"))[0])
    artifacts["lr_threshold"] = float(
        np.load(os.path.join(save_dir, "logreg_threshold.npy"))[0]
    )

    # LR metrics
    if os.path.exists(os.path.join(save_dir, "logreg_tuning_summary.npy")):
        artifacts["lr_summary"] = np.load(
            os.path.join(save_dir, "logreg_tuning_summary.npy"), allow_pickle=True
        ).item()

    # LR ROC
    if os.path.exists(os.path.join(save_dir, "logreg_roc_fpr.npy")):
        artifacts["lr_roc_fpr"] = np.load(os.path.join(save_dir, "logreg_roc_fpr.npy"))
        artifacts["lr_roc_tpr"] = np.load(os.path.join(save_dir, "logreg_roc_tpr.npy"))

    # KNN
    artifacts["knn_X_train"] = np.load(os.path.join(save_dir, "knn_X_train.npy"))
    artifacts["knn_y_train"] = np.load(os.path.join(save_dir, "knn_y_train.npy"))
    knn_params = np.load(os.path.join(save_dir, "knn_params.npy"), allow_pickle=True)
    artifacts["knn_k"] = int(knn_params[0])
    artifacts["knn_weights"] = str(knn_params[1])

    # KNN metrics
    if os.path.exists(os.path.join(save_dir, "knn_tuning_summary.npy")):
        artifacts["knn_summary"] = np.load(
            os.path.join(save_dir, "knn_tuning_summary.npy"), allow_pickle=True
        ).item()

    # KNN ROC
    if os.path.exists(os.path.join(save_dir, "knn_roc_fpr.npy")):
        artifacts["knn_roc_fpr"] = np.load(os.path.join(save_dir, "knn_roc_fpr.npy"))
        artifacts["knn_roc_tpr"] = np.load(os.path.join(save_dir, "knn_roc_tpr.npy"))

    # Scaler
    artifacts["scaler_mean"] = np.load(os.path.join(save_dir, "scaler_mean.npy"))
    artifacts["scaler_std"] = np.load(os.path.join(save_dir, "scaler_std.npy"))

    return artifacts


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def predict_lr(x, artifacts):
    """Predict with Logistic Regression. x is shape (n_features,) or (n_samples, n_features)."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    prob = sigmoid(np.dot(x, artifacts["lr_weights"]) + artifacts["lr_bias"])
    pred = (prob >= artifacts["lr_threshold"]).astype(int)
    return pred, prob


def predict_knn(x, artifacts, k=None):
    """Predict with KNN. x is shape (n_features,) or (n_samples, n_features)."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if k is None:
        k = artifacts["knn_k"]

    X_train = artifacts["knn_X_train"]
    y_train = artifacts["knn_y_train"]
    weights = artifacts["knn_weights"]

    preds = []
    probs = []

    for query in x:
        dists = np.sqrt(np.sum((X_train - query) ** 2, axis=1))
        knn_idx = np.argpartition(dists, k)[:k]
        knn_labels = y_train[knn_idx]
        knn_dists = dists[knn_idx]

        if weights == "distance":
            w = 1.0 / (knn_dists + 1e-8)
        else:
            w = np.ones(k)

        vote_1 = np.sum(w[knn_labels == 1])
        vote_0 = np.sum(w[knn_labels == 0])
        total = vote_0 + vote_1

        fraud_prob = vote_1 / total if total > 0 else 0.0
        preds.append(1 if vote_1 > vote_0 else 0)
        probs.append(fraud_prob)

    return np.array(preds), np.array(probs)


def normalize_input(x, artifacts):
    """Normalize Time and Amount features in a 30-dim input vector."""
    x = x.copy()
    # Time is at index 28, Amount at index 29
    x[28] = (x[28] - artifacts["scaler_mean"][0]) / artifacts["scaler_std"][0]
    x[29] = (x[29] - artifacts["scaler_mean"][1]) / artifacts["scaler_std"][1]
    return x
