"""Customer Churn Predictor & Retention Assistant - Streamlit prototype.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict import load_model, predict_churn
from src.retention_assistant import draft_retention_email
from src.train import load_data

st.set_page_config(page_title="Churn Predictor & Retention Assistant",
                   page_icon="◈", layout="wide")

st.markdown("""
<style>
  .stApp { background: #0a1628; }
  section[data-testid="stSidebar"] { background: #0f2038; }
  div[data-testid="stMetric"] {
      background: #0f2038; border: 1px solid #1e3a5f; border-radius: 12px;
      padding: 12px 16px;
  }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def bootstrap():
    """Load the model once per session (trains on first run if needed)."""
    try:
        load_model()
    except FileNotFoundError:
        from src.train import train
        train()
    return load_data()


df = bootstrap()

st.title("Customer Churn Predictor & Retention Assistant")
st.caption("Predictive ML (Random Forest) + Generative AI retention drafting - fintech prototype")

with st.sidebar:
    st.header("Customer profile")
    age = st.slider("Age", 18, 85, 35)
    tenure = st.slider("Tenure (months)", 1, 120, 24)
    balance = st.number_input("Balance (ZAR)", 0.0, 2_500_000.0, 15_000.0, 1000.0)
    num_products = st.slider("Number of products", 1, 4, 2)
    has_cc = st.selectbox("Has credit card?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    active = st.selectbox("Active member?", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    salary = st.number_input("Estimated salary (ZAR/month)", 5_000.0, 250_000.0, 28_000.0, 1000.0)
    tx_30d = st.slider("Transactions (last 30 days)", 0, 60, 4)
    tickets_30d = st.slider("Support tickets (last 30 days)", 0, 10, 0)
    run = st.button("Predict churn risk", type="primary", use_container_width=True)

col1, col2, col3 = st.columns(3)
c1, c2, c3 = st.columns(3)
c1.metric("Dataset", f"{len(df):,} customers")
c2.metric("Observed churn rate", f"{df['Churn'].mean():.1%}")
c3.metric("Model features", "10")

profile = {
    "Age": age, "Tenure_Months": tenure, "Balance": balance,
    "Num_Products": num_products, "Has_Credit_Card": has_cc,
    "Is_Active_Member": active, "Estimated_Salary": salary,
    "Transaction_Count_30D": tx_30d, "Support_Tickets_30D": tickets_30d,
}

if run:
    with st.spinner("Scoring customer..."):
        result = predict_churn(profile)
    prob = result["churn_probability"]
    level = result["risk_level"]

    gauge_color = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}[level]

    col1, col2 = st.columns([1, 2])
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100,
            title={"text": "Churn risk (%)"},
            number={"suffix": "%", "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100], "bgcolor": "#0f2038"},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 30], "color": "rgba(34,197,94,.15)"},
                    {"range": [30, 70], "color": "rgba(245,158,11,.15)"},
                    {"range": [70, 100], "color": "rgba(239,68,68,.25)"},
                ],
                "threshold": {"line": {"color": "white", "width": 3},
                              "thickness": 0.9, "value": prob * 100},
            },
        ))
        fig.update_layout(height=320, paper_bgcolor="#0f2038",
                          font={"color": "#e2e8f0", "size": 14})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"### Risk level: <span style='color:{gauge_color}'>{level}</span>",
                    unsafe_allow_html=True)
        st.markdown(
            f"**Churn probability: {result['churn_probability_pct']}** "
            f"(High band starts at 70%)"
        )
        if level == "High":
            st.error("This customer is in the high-risk band. Consider triggering "
                     "the AI Retention Assistant below.")
        elif level == "Medium":
            st.warning("Medium risk. A proactive check-in may prevent escalation.")
        else:
            st.success("Low risk. No action needed today.")

        if st.button("✉ Draft retention email with AI assistant",
                     type="primary", use_container_width=True):
            with st.spinner("Drafting personalised retention email..."):
                draft = draft_retention_email(profile, prob)
            st.markdown("#### Generated retention draft")
            st.markdown(f"*Engine: {draft['source']}* | "
                        f"*Compliance check: {'PASS' if draft['compliant'] else 'FLAGGED - ' + ', '.join(draft['violations'])}*")
            st.markdown(draft["email"])
            st.caption("Inferred drivers: " + "; ".join(draft["reasons"]))
            if not draft["compliant"]:
                st.error("Draft blocked by compliance guardrail and replaced by "
                         "template fallback. Review the prompt constraints.")
    st.divider()
    st.caption("Synthetic demo data. Offers are illustrative and drawn only from "
               "the approved-offer whitelist.")
else:
    st.info("Set the customer profile in the sidebar and click "
            "**Predict churn risk** to score them.")
