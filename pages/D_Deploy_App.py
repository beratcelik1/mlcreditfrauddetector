import streamlit as st
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from helper_functions import load_saved_models, predict_lr, predict_knn, normalize_input

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")
st.markdown("### Final Project - Credit Card Fraud Detection")

#############################################

st.title("Deploy Application")

#############################################

artifacts = load_saved_models()

if artifacts is None:
    st.error("Model artifacts not found. Run the training scripts first:")
    st.code(
        "python preprocessing.py\npython train_logistic.py\npython train_knn.py",
        language="bash",
    )
    st.stop()

st.markdown("### Fraud Prediction Tool")
st.markdown(
    "Enter transaction features below or upload a CSV file to get fraud predictions."
)

# ── Model Selection ─────────────────────────
deploy_model = st.session_state.get("deploy_model", "Logistic Regression")
model_choice = st.selectbox(
    "Select model",
    ["Logistic Regression", "K-Nearest Neighbors"],
    index=0 if "Logistic" in deploy_model else 1,
)

# ── Input Method ────────────────────────────
input_method = st.radio("Input method", ["Manual Entry", "Upload CSV"])

if input_method == "Manual Entry":
    st.markdown("#### Enter Transaction Features")
    st.markdown(
        "V1-V28 are PCA-transformed features. Time and Amount are raw values (will be normalized automatically)."
    )

    col1, col2, col3 = st.columns(3)

    feature_values = {}
    for i in range(1, 29):
        col = [col1, col2, col3][(i - 1) % 3]
        with col:
            feature_values[f"V{i}"] = st.number_input(
                f"V{i}", value=0.0, format="%.4f", key=f"v{i}"
            )

    t_col, a_col = st.columns(2)
    with t_col:
        time_val = st.number_input(
            "Time (seconds since first transaction)",
            value=50000.0,
            min_value=0.0,
            format="%.1f",
        )
    with a_col:
        amount_val = st.number_input(
            "Amount (EUR)", value=100.0, min_value=0.0, format="%.2f"
        )

    if st.button("Predict", type="primary"):
        # Build the 30-dim feature vector
        x = np.array(
            [feature_values[f"V{i}"] for i in range(1, 29)] + [time_val, amount_val]
        )
        x = normalize_input(x, artifacts)

        if model_choice == "Logistic Regression":
            pred, prob = predict_lr(x, artifacts)
            pred = pred[0]
            prob = prob[0]
            model_name = f"Logistic Regression (threshold={artifacts['lr_threshold']})"
        else:
            pred, prob = predict_knn(x, artifacts)
            pred = pred[0]
            prob = prob[0]
            model_name = (
                f"KNN (k={artifacts['knn_k']}, weights={artifacts['knn_weights']})"
            )

        st.markdown("---")
        st.markdown("### Prediction Result")

        if pred == 1:
            st.error("**FRAUD DETECTED**")
            st.write(f"Confidence: {prob*100:.1f}%")
        else:
            st.success("**LEGITIMATE TRANSACTION**")
            st.write(f"Fraud probability: {prob*100:.1f}%")

        st.caption(f"Model: {model_name}")

elif input_method == "Upload CSV":
    st.markdown("#### Upload Transaction Data")
    st.markdown("CSV must have columns: V1-V28, Time, Amount (30 columns total)")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        input_df = pd.read_csv(uploaded)
        st.write(f"Uploaded {len(input_df)} transactions")
        st.dataframe(input_df.head())

        feature_cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
        missing_cols = [c for c in feature_cols if c not in input_df.columns]

        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            st.stop()

        if st.button("Predict All", type="primary"):
            X_input = input_df[feature_cols].values.astype(np.float64)

            # Normalize Time and Amount
            for i in range(len(X_input)):
                X_input[i] = normalize_input(X_input[i], artifacts)

            if model_choice == "Logistic Regression":
                preds, probs = predict_lr(X_input, artifacts)
            else:
                preds, probs = predict_knn(X_input, artifacts)

            results = input_df.copy()
            results["Prediction"] = ["FRAUD" if p == 1 else "LEGIT" for p in preds]
            results["Fraud_Probability"] = [f"{p*100:.1f}%" for p in probs]

            st.markdown("### Results")

            n_fraud = int(np.sum(preds == 1))
            n_legit = len(preds) - n_fraud

            r1, r2 = st.columns(2)
            r1.metric("Flagged as Fraud", n_fraud)
            r2.metric("Flagged as Legitimate", n_legit)

            # Color code the results
            def highlight_fraud(row):
                if row["Prediction"] == "FRAUD":
                    return ["background-color: #ffcdd2"] * len(row)
                return ["background-color: #c8e6c9"] * len(row)

            st.dataframe(
                results[
                    ["Prediction", "Fraud_Probability"]
                    + [
                        c
                        for c in results.columns
                        if c not in ["Prediction", "Fraud_Probability"]
                    ]
                ].style.apply(highlight_fraud, axis=1)
            )

            # Download results
            csv = results.to_csv(index=False)
            st.download_button(
                "Download Results CSV", csv, "fraud_predictions.csv", "text/csv"
            )
