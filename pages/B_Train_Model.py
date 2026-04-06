import streamlit as st
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from helper_functions import fetch_dataset

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")
st.markdown("### Final Project - Credit Card Fraud Detection")

#############################################

st.title("Train Model")

#############################################

df = fetch_dataset()

if df is not None:
    st.markdown("### Training Configuration")
    st.markdown(
        "Both models are implemented **from scratch using NumPy only**. No scikit-learn, TensorFlow, or PyTorch."
    )

    # Import our from-scratch implementations
    from train_logistic import LogisticRegression
    from knn import KNNClassifier
    from evaluation import compute_all_metrics, confusion_matrix

    # Load preprocessed data
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
    X_train = np.load(os.path.join(save_dir, "X_train.npy"))
    y_train = np.load(os.path.join(save_dir, "y_train.npy"))
    X_test = np.load(os.path.join(save_dir, "X_test.npy"))
    y_test = np.load(os.path.join(save_dir, "y_test.npy"))

    st.write(
        f"**Training set:** {X_train.shape[0]:,} samples (fraud={np.sum(y_train==1)}, legit={np.sum(y_train==0)})"
    )
    st.write(
        f"**Test set:** {X_test.shape[0]:,} samples (fraud={np.sum(y_test==1)}, legit={np.sum(y_test==0)})"
    )

    # ── Model Selection ─────────────────────────
    st.markdown("### Select Models to Train")
    model_options = ["Logistic Regression", "K-Nearest Neighbors"]
    model_select = st.multiselect("Select models", model_options, default=model_options)

    # ── Logistic Regression ─────────────────────
    if "Logistic Regression" in model_select:
        st.markdown("---")
        st.markdown("#### Logistic Regression Hyperparameters")

        lr_col1, lr_col2, lr_col3 = st.columns(3)
        with lr_col1:
            lr_rate = st.select_slider(
                "Learning rate", options=[0.001, 0.01, 0.1], value=0.001
            )
        with lr_col2:
            lr_iters = st.select_slider(
                "Iterations", options=[1000, 5000, 10000], value=10000
            )
        with lr_col3:
            lr_thresh = st.select_slider(
                "Decision threshold", options=[0.3, 0.4, 0.5], value=0.5
            )

        if st.button("Train Logistic Regression"):
            with st.spinner("Training Logistic Regression..."):
                t0 = time.time()
                model_lr = LogisticRegression(lr=lr_rate, n_iters=lr_iters)
                model_lr.fit(X_train, y_train)
                train_time = time.time() - t0

                probs = model_lr.predict_proba(X_test)
                preds = (probs >= lr_thresh).astype(int)
                metrics = compute_all_metrics(y_test, preds, probs)
                cm = confusion_matrix(y_test, preds)

                # Store in session state
                st.session_state["lr_model"] = model_lr
                st.session_state["lr_metrics"] = metrics
                st.session_state["lr_cm"] = cm
                st.session_state["lr_probs"] = probs
                st.session_state["lr_preds"] = preds
                st.session_state["lr_threshold"] = lr_thresh
                st.session_state["lr_params"] = {
                    "lr": lr_rate,
                    "iters": lr_iters,
                    "threshold": lr_thresh,
                }

            st.success(f"Logistic Regression trained in {train_time:.2f}s")

        if "lr_metrics" in st.session_state:
            st.markdown("**Logistic Regression Results:**")
            m = st.session_state["lr_metrics"]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Precision", f"{m['precision']:.4f}")
            mc2.metric("Recall", f"{m['recall']:.4f}")
            mc3.metric("F1", f"{m['f1']:.4f}")
            mc4.metric("AUC-ROC", f"{m['auc_roc']:.4f}")

            cm = st.session_state["lr_cm"]
            st.write(
                f"Confusion Matrix: TN={cm[0,0]:,} FP={cm[0,1]:,} FN={cm[1,0]:,} TP={cm[1,1]:,}"
            )

    # ── KNN ─────────────────────────────────────
    if "K-Nearest Neighbors" in model_select:
        st.markdown("---")
        st.markdown("#### KNN Hyperparameters")

        knn_col1, knn_col2 = st.columns(2)
        with knn_col1:
            knn_k = st.select_slider(
                "Number of neighbors (k)", options=[3, 5, 7, 11, 15], value=5
            )
        with knn_col2:
            knn_weights = st.selectbox("Weighting scheme", ["distance", "uniform"])

        if st.button("Train KNN"):
            with st.spinner("Training KNN (this stores training data)..."):
                t0 = time.time()
                model_knn = KNNClassifier(k=knn_k, weights=knn_weights)
                model_knn.fit(X_train, y_train)

                progress = st.progress(0)
                st.write("Running predictions on test set...")
                preds = model_knn.predict(X_test)
                probs = model_knn.predict_proba(X_test)[:, 1]
                progress.progress(100)
                train_time = time.time() - t0

                metrics = compute_all_metrics(y_test, preds, probs)
                cm = confusion_matrix(y_test, preds)

                st.session_state["knn_model"] = model_knn
                st.session_state["knn_metrics"] = metrics
                st.session_state["knn_cm"] = cm
                st.session_state["knn_probs"] = probs
                st.session_state["knn_preds"] = preds
                st.session_state["knn_params"] = {"k": knn_k, "weights": knn_weights}

            st.success(f"KNN trained and evaluated in {train_time:.2f}s")

        if "knn_metrics" in st.session_state:
            st.markdown("**KNN Results:**")
            m = st.session_state["knn_metrics"]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Precision", f"{m['precision']:.4f}")
            mc2.metric("Recall", f"{m['recall']:.4f}")
            mc3.metric("F1", f"{m['f1']:.4f}")
            mc4.metric("AUC-ROC", f"{m['auc_roc']:.4f}")

            cm = st.session_state["knn_cm"]
            st.write(
                f"Confusion Matrix: TN={cm[0,0]:,} FP={cm[0,1]:,} FN={cm[1,0]:,} TP={cm[1,1]:,}"
            )

    st.write("Continue to **Test Model** page.")
