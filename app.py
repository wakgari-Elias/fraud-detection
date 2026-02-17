# app.py - Fraud Risk Explainer Dashboard
# Capstone improvement - Interactive Streamlit dashboard with SHAP explainability

import streamlit as st
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from src.explainability import load_model
import streamlit_shap          # ← This line was missing or incorrect

# ────────────────────────────────────────────────
# Page config & styling
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Risk Explainer | Adey Innovations",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional finance styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .sidebar .sidebar-content { background-color: #ffffff; }
    h1, h2, h3 { color: #0d47a1; }
    .stButton>button { background-color: #0d47a1; color: white; border-radius: 6px; }
    .stButton>button:hover { background-color: #002984; }
    .stAlert.success { background-color: #e8f5e9; border-left-color: #4caf50; }
    .stAlert.warning { background-color: #fff3e0; border-left-color: #ff9800; }
    .stAlert.error   { background-color: #ffebee; border-left-color: #f44336; }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# Load model (cached)
# ────────────────────────────────────────────────
@st.cache_resource
def get_model():
    model_path = Path("models/best_model_xgboost.joblib")
    if not model_path.exists():
        st.error("Model file not found! Check path: models/best_model_xgboost.joblib")
        st.stop()
    return load_model(str(model_path))

model = get_model()

# Your model's exact features
FEATURES = [
    'age', 'day_of_week', 'hour_of_day', 'purchase_value',
    'time_since_signup_hours', 'tx_per_device', 'tx_per_user'
]

# ────────────────────────────────────────────────
# Sidebar Inputs
# ────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ Fraud Risk Explainer")
    st.markdown("**Adey Innovations Inc.** – E-commerce & Banking Fraud Detection")
    st.markdown("Adjust transaction details and click Predict.")

    age = st.slider("Customer Age", 18, 90, value=32, step=1)
    day_of_week = st.selectbox(
        "Day of Week",
        options=[0,1,2,3,4,5,6],
        index=3,
        format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
    )
    hour_of_day = st.slider("Hour of Day (0-23)", 0, 23, value=14)
    purchase_value = st.number_input("Purchase Value (USD)", 0.0, 10000.0, value=120.0, step=5.0)
    time_since_signup_hours = st.number_input("Hours since Signup", 0.0, 8760.0, value=72.0, step=1.0)
    tx_per_device = st.number_input("Transactions per Device", 1.0, 200.0, value=4.0, step=1.0)
    tx_per_user = st.number_input("Transactions per User", 1.0, 200.0, value=7.0, step=1.0)

    st.markdown("---")
    predict_button = st.button("Calculate Fraud Risk", type="primary", use_container_width=True)

# ────────────────────────────────────────────────
# Main content
# ────────────────────────────────────────────────
st.title("Fraud Detection – Explainable Risk Assessment")
st.markdown("""
This dashboard shows how an XGBoost model detects fraud in e-commerce transactions.  
It provides **real-time prediction** + **SHAP explainability** so you can understand why a transaction is risky.
""")

if predict_button:
    with st.spinner("Analyzing transaction..."):
        # Prepare input
        input_dict = {
            'age': age,
            'day_of_week': day_of_week,
            'hour_of_day': hour_of_day,
            'purchase_value': purchase_value,
            'time_since_signup_hours': time_since_signup_hours,
            'tx_per_device': tx_per_device,
            'tx_per_user': tx_per_user
        }
        df_input = pd.DataFrame([input_dict])

        # Predict probability
        prob = model.predict_proba(df_input)[:, 1][0]

        # Display risk level
        if prob > 0.70:
            st.error(f"**HIGH FRAUD RISK** – Probability: **{prob:.1%}**")
        elif prob > 0.30:
            st.warning(f"**MEDIUM FRAUD RISK** – Probability: **{prob:.1%}**")
        else:
            st.success(f"**LOW FRAUD RISK** – Probability: **{prob:.1%}**")

        # SHAP explanation
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_input)

            st.subheader("Why this prediction? (SHAP Force Plot)")
            st.caption("Red pushes toward fraud · Blue pushes toward legitimate")

            # Interactive SHAP plot using streamlit-shap
            streamlit_shap.st_shap(
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[0],
                    df_input.iloc[0],
                    matplotlib=False
                ),
                height=300
            )

            # Feature contribution table
            contrib = pd.DataFrame({
                "Feature": FEATURES,
                "Value": df_input.iloc[0].values,
                "SHAP Impact": shap_values[0]
            }).round(4)
            st.subheader("Feature Contributions to Risk")
            st.dataframe(
                contrib.style.bar(
                    subset=["SHAP Impact"],
                    color=["#d62728", "#1f77b4"],
                    align="zero"
                )
            )

        except Exception as e:
            st.warning(f"SHAP visualization failed: {str(e)}")
            st.info("The fraud probability is still shown above.")

else:
    st.info("Enter transaction details in the sidebar and click **Calculate Fraud Risk**.")

# ────────────────────────────────────────────────
# Business Recommendations
# ────────────────────────────────────────────────
with st.expander("🔍 Business Recommendations (from SHAP insights)"):
    st.markdown("""
    1. **Require extra verification for recent signups**  
       Short `time_since_signup_hours` is a top fraud driver → OTP/SMS for < 24 hours.

    2. **Implement velocity limits**  
       High `tx_per_device` / `tx_per_user` strongly increase risk → limit to 3–5 tx per window.

    3. **Review unusual purchase amounts**  
       Extreme `purchase_value` often contributes to fraud risk → flag high or suspicious values.
    """)

# Footer
st.markdown("---")
st.caption("© 2026 Adey Innovations Inc. – Capstone Fraud Detection Project")