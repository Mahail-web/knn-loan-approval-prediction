"""
Streamlit Web Application for Loan Approval Prediction Using KNN.
Enables 1-click all-in-one Python deployment on Streamlit Community Cloud (https://share.streamlit.io).
"""

import json
import os
import joblib
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="LoanWise AI – Loan Approval Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# File paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "backend", "best_knn_model.joblib")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "backend", "model_config.json")
VIS_DIR = os.path.join(os.path.dirname(__file__), "ml", "visualizations")


@st.cache_resource
def load_model_and_config():
    """Loads serialized pipeline and config with Streamlit caching."""
    pipeline = joblib.load(MODEL_PATH)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    return pipeline, config


pipeline, config = load_model_and_config()

# Header
st.title("💳 LoanWise AI: Loan Approval Prediction")
st.caption("Machine Learning Capstone Project | Supervised K-Nearest Neighbors (KNN)")

# Tabs
tab_pred, tab_model, tab_vis = st.tabs(["🎯 Loan Application Prediction", "🧠 About the Model", "📊 Pipeline Visualizations"])

with tab_pred:
    st.subheader("Applicant Details")
    st.markdown("Fill out the applicant's financial and demographic profile to evaluate credit approval probability.")

    # Quick test presets
    preset = st.radio(
        "Load Quick Test Scenario:",
        ["Custom Input", "Prime Applicant (High Approval Probability)", "High Risk Applicant (Delinquency Risk)"],
        horizontal=True,
    )

    if preset == "Prime Applicant (High Approval Probability)":
        d_gender, d_married, d_dep = "Male", "Yes", "1"
        d_edu, d_self, d_income = "Graduate", "No", 6500.0
        d_coincome, d_loan, d_term = 2200.0, 135.0, 360.0
        d_credit, d_area = 1.0, "Semiurban"
    elif preset == "High Risk Applicant (Delinquency Risk)":
        d_gender, d_married, d_dep = "Female", "No", "0"
        d_edu, d_self, d_income = "Not Graduate", "Yes", 1400.0
        d_coincome, d_loan, d_term = 0.0, 380.0, 180.0
        d_credit, d_area = 0.0, "Rural"
    else:
        d_gender, d_married, d_dep = "Male", "Yes", "0"
        d_edu, d_self, d_income = "Graduate", "No", 5000.0
        d_coincome, d_loan, d_term = 1500.0, 120.0, 360.0
        d_credit, d_area = 1.0, "Semiurban"

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if d_gender == "Male" else 1)
        married = st.selectbox("Marital Status", ["Yes", "No"], index=0 if d_married == "Yes" else 1)
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=["0", "1", "2", "3+"].index(d_dep))
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0 if d_edu == "Graduate" else 1)

    with col2:
        self_employed = st.selectbox("Self Employed", ["No", "Yes"], index=0 if d_self == "No" else 1)
        property_area = st.selectbox("Property Area", ["Semiurban", "Urban", "Rural"], index=["Semiurban", "Urban", "Rural"].index(d_area))
        applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=0.0, value=float(d_income), step=100.0)
        coapplicant_income = st.number_input("Co-Applicant Monthly Income ($)", min_value=0.0, value=float(d_coincome), step=100.0)

    with col3:
        loan_amount = st.number_input("Loan Amount (Thousands $)", min_value=1.0, value=float(d_loan), step=5.0)
        loan_term = st.selectbox("Loan Term (Months)", [360.0, 240.0, 180.0, 120.0, 84.0, 60.0], index=[360.0, 240.0, 180.0, 120.0, 84.0, 60.0].index(d_term))
        credit_history = st.selectbox(
            "Credit History Guidelines",
            [1.0, 0.0],
            format_func=lambda x: "1.0 - Meets Credit Guidelines" if x == 1.0 else "0.0 - Adverse/Delinquent History",
            index=0 if d_credit == 1.0 else 1,
        )

    submit = st.button("🚀 Predict Loan Approval", type="primary", use_container_width=True)

    if submit:
        # Build DataFrame in training feature order
        feature_order = config["training_feature_names"]
        input_data = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "Property_Area": property_area,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
        }
        df_in = pd.DataFrame([input_data])[feature_order]

        # Pipeline prediction
        raw_pred = pipeline.predict(df_in)[0]
        probs = pipeline.predict_proba(df_in)[0]
        prob_approved = float(probs[1])
        prob_rejected = float(probs[0])

        st.divider()

        if raw_pred == 1:
            st.success(f"### 🎉 Loan Application: Approved\n**Model Confidence:** {prob_approved * 100:.1f}%")
        else:
            st.error(f"### ❌ Loan Application: Rejected\n**Model Confidence:** {prob_rejected * 100:.1f}%")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.write(f"**Approved Class Probability:** {prob_approved * 100:.1f}%")
            st.progress(prob_approved)
        with col_p2:
            st.write(f"**Rejected Class Probability:** {prob_rejected * 100:.1f}%")
            st.progress(prob_rejected)

        st.info(
            f"**Neighbor Breakdown (K={config['selected_k']})**: "
            f"Based on Euclidean proximity to the {config['selected_k']} nearest historical borrowers in scaled feature space, "
            f"{int(round(prob_approved * config['selected_k']))} neighbors voted Approved and "
            f"{int(round(prob_rejected * config['selected_k']))} neighbors voted Rejected."
        )

        st.caption("⚠️ **Educational Disclaimer**: This prediction is generated by a machine learning model for academic demonstration and is not an official banking credit approval.")

with tab_model:
    st.subheader("Model Specifications & Evaluation Metrics")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Selected Algorithm", "K-Nearest Neighbors")
    col_m2.metric("Optimal K", f"K = {config['selected_k']}")
    col_m3.metric("Test Accuracy", f"{config['test_accuracy'] * 100:.2f}%")
    col_m4.metric("5-Fold CV Score", f"{config['best_cv_accuracy'] * 100:.2f}%")

    st.markdown("""
    ### Key Architectural Highlights:
    1. **Zero Data Leakage**: Standardized using `StandardScaler` fitted strictly within cross-validation folds.
    2. **Hyperparameter Tuning**: Tuned over $K \in [1, 20]$, distance metrics (Euclidean, Manhattan), and weights.
    3. **Credit Risk Evaluation**:
       - **False Approvals (FP = 17)**: Credit default risk.
       - **False Rejections (FN = 1)**: Outstanding **98.8% Recall** on creditworthy applicants.
    """)

with tab_vis:
    st.subheader("Generated EDA & Model Performance Visualizations")
    vis_images = [
        ("Loan Approval Distribution", "loan_approval_distribution.png"),
        ("Missing Values Analysis", "missing_values_visualization.png"),
        ("Numerical Feature Distributions", "numerical_feature_distributions.png"),
        ("Correlation Heatmap", "correlation_heatmap.png"),
        ("K-Value Comparison & CV Curve", "k_value_accuracy_comparison.png"),
        ("Confusion Matrix (Default KNN)", "confusion_matrix_default_knn.png"),
        ("Confusion Matrix (Optimized KNN)", "confusion_matrix_optimized_knn.png"),
    ]

    for title, fname in vis_images:
        fpath = os.path.join(VIS_DIR, fname)
        if os.path.exists(fpath):
            st.write(f"#### {title}")
            st.image(fpath, use_column_width=True)
            st.divider()

