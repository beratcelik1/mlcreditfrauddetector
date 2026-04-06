import streamlit as st
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from helper_functions import fetch_dataset, load_saved_models

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")
st.markdown("### Final Project - Credit Card Fraud Detection")

#############################################

st.title("Test Model")

#############################################

df = fetch_dataset()

if df is not None:
    # Try loading pre-trained results, or use session state from Train page
    artifacts = load_saved_models()

    lr_metrics = None
    knn_metrics = None
    lr_roc = None
    knn_roc = None

    # Get LR metrics
    if "lr_metrics" in st.session_state:
        lr_metrics = st.session_state["lr_metrics"]
        lr_params = st.session_state.get("lr_params", {})
    elif artifacts and "lr_summary" in artifacts:
        lr_metrics = artifacts["lr_summary"]["test_metrics"]
        lr_params = {
            "lr": artifacts["lr_summary"].get("best_lr", "?"),
            "iters": artifacts["lr_summary"].get("best_iters", "?"),
            "threshold": artifacts["lr_summary"].get("best_threshold", "?"),
        }

    # Get KNN metrics
    if "knn_metrics" in st.session_state:
        knn_metrics = st.session_state["knn_metrics"]
        knn_params = st.session_state.get("knn_params", {})
    elif artifacts and "knn_summary" in artifacts:
        knn_metrics = artifacts["knn_summary"]["test_metrics"]
        knn_params = {
            "k": artifacts.get("knn_k", "?"),
            "weights": artifacts.get("knn_weights", "?"),
        }

    # Get ROC data
    if artifacts and "lr_roc_fpr" in artifacts:
        lr_roc = (artifacts["lr_roc_fpr"], artifacts["lr_roc_tpr"])
    if artifacts and "knn_roc_fpr" in artifacts:
        knn_roc = (artifacts["knn_roc_fpr"], artifacts["knn_roc_tpr"])

    if lr_metrics is None and knn_metrics is None:
        st.warning(
            "No trained models found. Go to the **Train Model** page first, or run the training scripts."
        )
        st.stop()

    # ── Metrics Comparison ──────────────────────
    st.markdown("### Performance Comparison")

    metric_names = ["precision", "recall", "f1", "auc_roc"]
    display_names = ["Precision", "Recall", "F1-Score", "AUC-ROC"]

    col_header, col_lr, col_knn = st.columns([2, 2, 2])
    col_header.markdown("**Metric**")
    col_lr.markdown("**Logistic Regression**")
    col_knn.markdown("**KNN**")

    for metric, display in zip(metric_names, display_names):
        c1, c2, c3 = st.columns([2, 2, 2])
        c1.write(display)

        lr_val = lr_metrics[metric] if lr_metrics else None
        knn_val = knn_metrics[metric] if knn_metrics else None

        if lr_val is not None and knn_val is not None:
            better_lr = lr_val >= knn_val
            c2.write(
                f"{'**' if better_lr else ''}{lr_val:.4f}{'**' if better_lr else ''}"
            )
            c3.write(
                f"{'**' if not better_lr else ''}{knn_val:.4f}{'**' if not better_lr else ''}"
            )
        elif lr_val is not None:
            c2.write(f"{lr_val:.4f}")
            c3.write("---")
        elif knn_val is not None:
            c2.write("---")
            c3.write(f"{knn_val:.4f}")

    # ── Model Parameters ────────────────────────
    st.markdown("### Model Parameters")
    p1, p2 = st.columns(2)
    if lr_metrics:
        with p1:
            st.markdown("**Logistic Regression**")
            for k, v in lr_params.items():
                st.write(f"- {k}: {v}")
    if knn_metrics:
        with p2:
            st.markdown("**KNN**")
            for k, v in knn_params.items():
                st.write(f"- {k}: {v}")

    # ── Confusion Matrices ──────────────────────
    st.markdown("### Confusion Matrices")

    cm_col1, cm_col2 = st.columns(2)

    if lr_metrics:
        with cm_col1:
            cm = None
            if "lr_cm" in st.session_state:
                cm = st.session_state["lr_cm"]
            elif artifacts and "lr_summary" in artifacts:
                cm = artifacts["lr_summary"].get("confusion_matrix_test")

            if cm is not None:
                fig = go.Figure(
                    data=go.Heatmap(
                        z=cm,
                        text=[[f"{v:,}" for v in row] for row in cm],
                        texttemplate="%{text}",
                        textfont={"size": 16},
                        colorscale="Blues",
                        showscale=False,
                        x=["Pred Legit", "Pred Fraud"],
                        y=["True Legit", "True Fraud"],
                    )
                )
                fig.update_layout(
                    title="Logistic Regression",
                    height=350,
                    xaxis_title="Predicted",
                    yaxis_title="Actual",
                )
                st.plotly_chart(fig, use_container_width=True)

    if knn_metrics:
        with cm_col2:
            cm = None
            if "knn_cm" in st.session_state:
                cm = st.session_state["knn_cm"]
            elif artifacts and "knn_summary" in artifacts:
                cm = artifacts["knn_summary"].get("confusion_matrix_test")

            if cm is not None:
                fig = go.Figure(
                    data=go.Heatmap(
                        z=cm,
                        text=[[f"{v:,}" for v in row] for row in cm],
                        texttemplate="%{text}",
                        textfont={"size": 16},
                        colorscale="Oranges",
                        showscale=False,
                        x=["Pred Legit", "Pred Fraud"],
                        y=["True Legit", "True Fraud"],
                    )
                )
                fig.update_layout(
                    title=f'KNN (k={knn_params.get("k", "?")})',
                    height=350,
                    xaxis_title="Predicted",
                    yaxis_title="Actual",
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── ROC Curves ──────────────────────────────
    st.markdown("### ROC Curves")

    if lr_roc or knn_roc:
        fig = go.Figure()

        if lr_roc:
            auc_lr = lr_metrics["auc_roc"] if lr_metrics else 0
            fig.add_trace(
                go.Scatter(
                    x=lr_roc[0],
                    y=lr_roc[1],
                    mode="lines",
                    name=f"LR (AUC={auc_lr:.4f})",
                    line=dict(color="#1976D2", width=2),
                )
            )
        if knn_roc:
            auc_knn = knn_metrics["auc_roc"] if knn_metrics else 0
            fig.add_trace(
                go.Scatter(
                    x=knn_roc[0],
                    y=knn_roc[1],
                    mode="lines",
                    name=f"KNN (AUC={auc_knn:.4f})",
                    line=dict(color="#F44336", width=2),
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(color="gray", dash="dash"),
            )
        )

        fig.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ROC curve data not available. Run training scripts to generate.")

    # ── Deployment Recommendation ───────────────
    st.markdown("### Deployment Recommendation")

    if lr_metrics and knn_metrics:
        lr_f1 = lr_metrics["f1"]
        knn_f1 = knn_metrics["f1"]
        lr_recall = lr_metrics["recall"]
        knn_recall = knn_metrics["recall"]

        if lr_f1 >= knn_f1:
            st.success(
                f"**Logistic Regression** is recommended for deployment (F1={lr_f1:.4f} vs {knn_f1:.4f}). "
                f"It also has faster inference time O(d) vs O(Nd)."
            )
        else:
            st.success(
                f"**KNN** is recommended for deployment (F1={knn_f1:.4f} vs {lr_f1:.4f})."
            )

        if knn_recall > lr_recall:
            st.info(
                f"Note: KNN has higher recall ({knn_recall:.4f} vs {lr_recall:.4f}), "
                f"catching more fraud cases at the cost of more false alarms."
            )

    st.markdown("### Choose Deployment Model")
    deploy_options = []
    if lr_metrics:
        deploy_options.append("Logistic Regression")
    if knn_metrics:
        deploy_options.append("K-Nearest Neighbors")

    deploy_select = st.selectbox("Select model to deploy", deploy_options)
    st.session_state["deploy_model"] = deploy_select

    st.write("Continue to **Deploy App** page.")
