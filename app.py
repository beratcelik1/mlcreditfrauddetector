import streamlit as st

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - Credit Card Fraud Detection")

#############################################

st.markdown("""
This application detects fraudulent credit card transactions using two machine learning
models implemented from scratch with NumPy:

- **Logistic Regression** - linear classifier with gradient descent
- **K-Nearest Neighbors (KNN)** - instance-based classifier with distance-weighted voting

**Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions, 492 fraud)

**Team:** Berat Celik, Zijing Wu, Yuxiang Jiang, Yousen Xie, Gabrielle Xiao, Binyao Zhao
""")

st.markdown("Click **Explore & Preprocess Data** in the sidebar to get started.")
