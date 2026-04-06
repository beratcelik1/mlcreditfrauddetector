import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from helper_functions import fetch_dataset

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")
st.markdown("### Final Project - Credit Card Fraud Detection")

#############################################

st.title("Explore & Preprocess Data")

#############################################

df = fetch_dataset()

if df is not None:
    st.markdown("### Dataset Overview")
    st.write(
        f"**Rows:** {len(df):,} | **Columns:** {df.shape[1]} | **Missing values:** {df.isnull().sum().sum()}"
    )

    st.dataframe(df.head(20))

    # ── Class Distribution ──────────────────────
    st.markdown("### Class Distribution")

    fraud_count = int(df["Class"].sum())
    legit_count = len(df) - fraud_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Legitimate", f"{legit_count:,}")
    col3.metric("Fraudulent", f"{fraud_count:,} ({fraud_count/len(df)*100:.3f}%)")

    fig = px.bar(
        x=["Legitimate", "Fraudulent"],
        y=[legit_count, fraud_count],
        color=["Legitimate", "Fraudulent"],
        color_discrete_map={"Legitimate": "#4CAF50", "Fraudulent": "#F44336"},
        log_y=True,
        labels={"x": "Class", "y": "Count (log scale)"},
        title="Class Distribution (Log Scale)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Feature Exploration ─────────────────────
    st.markdown("### Explore Features")

    feature_cols = [c for c in df.columns if c != "Class"]
    selected_feature = st.selectbox(
        "Select a feature to explore", feature_cols, index=feature_cols.index("Amount")
    )

    plot_type = st.selectbox(
        "Select plot type", ["Histogram", "Box Plot", "Scatter Plot"]
    )

    fraud_df = df[df["Class"] == 1]
    legit_df = df[df["Class"] == 0]

    if plot_type == "Histogram":
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=legit_df[selected_feature],
                name="Legitimate",
                marker_color="#4CAF50",
                opacity=0.7,
                nbinsx=50,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=fraud_df[selected_feature],
                name="Fraud",
                marker_color="#F44336",
                opacity=0.7,
                nbinsx=50,
            )
        )
        fig.update_layout(
            barmode="overlay",
            title=f"{selected_feature} Distribution by Class",
            xaxis_title=selected_feature,
            yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Box Plot":
        fig = px.box(
            df,
            x="Class",
            y=selected_feature,
            color="Class",
            color_discrete_map={0: "#4CAF50", 1: "#F44336"},
            labels={"Class": "Class (0=Legit, 1=Fraud)"},
            title=f"{selected_feature} Box Plot by Class",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Scatter Plot":
        feature2 = st.selectbox(
            "Select second feature for scatter",
            feature_cols,
            index=feature_cols.index("Time") if "Time" in feature_cols else 0,
        )
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig = px.scatter(
            sample,
            x=selected_feature,
            y=feature2,
            color="Class",
            color_discrete_map={0: "#4CAF50", 1: "#F44336"},
            opacity=0.5,
            title=f"{selected_feature} vs {feature2}",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Correlation Heatmap ─────────────────────
    st.markdown("### Correlation Heatmap")

    top_features = st.multiselect(
        "Select features for correlation heatmap",
        feature_cols,
        default=["V14", "V17", "V12", "V10", "V1", "V3", "Amount"],
    )

    if len(top_features) >= 2:
        corr_cols = top_features + ["Class"]
        corr = df[corr_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Feature Correlation Heatmap",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Preprocessing Summary ───────────────────
    st.markdown("### Preprocessing Pipeline")
    st.markdown("""
    Our preprocessing steps (implemented in `preprocessing.py`):
    1. **Missing value check** - verified: 0 missing values
    2. **Stratified 80/20 train/test split** - preserves the 0.17% fraud ratio in both sets
    3. **IQR outlier capping** on Amount - prevents extreme values from skewing training
    4. **Z-score normalization** on Time and Amount - matches PCA feature scale
    5. **Random undersampling** (5:1 ratio) - balances training set so model learns fraud patterns

    Test set remains at original imbalanced ratio for realistic evaluation.
    """)

    st.write("Continue to **Train Model** page.")
