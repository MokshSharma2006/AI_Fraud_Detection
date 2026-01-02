import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Fraud Detection Dashboard",
    layout="wide"
)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("output/fraud_results.csv")

df = load_data()

# -------------------------------
# TITLE
# -------------------------------
st.title("🧠 AI-Based Public Fraud & Anomaly Detection Dashboard")
st.markdown(
    "This dashboard presents AI-driven fraud risk analysis on public financial transactions."
)

# -------------------------------
# KPI METRICS
# -------------------------------
total_txns = len(df)
high_risk = len(df[df["fraud_risk"] == "High"])
medium_risk = len(df[df["fraud_risk"] == "Medium"])
low_risk = len(df[df["fraud_risk"] == "Low"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", total_txns)
col2.metric("🔴 High Risk", high_risk)
col3.metric("🟠 Medium Risk", medium_risk)
col4.metric("🟢 Low Risk", low_risk)

st.divider()

# -------------------------------
# FRAUD RISK DISTRIBUTION
# -------------------------------
st.subheader("📊 Fraud Risk Distribution")

risk_counts = df["fraud_risk"].value_counts().reset_index()
risk_counts.columns = ["Risk Level", "Count"]

fig_risk = px.pie(
    risk_counts,
    names="Risk Level",
    values="Count",
    color="Risk Level",
    color_discrete_map={
        "High": "red",
        "Medium": "orange",
        "Low": "green"
    }
)

st.plotly_chart(fig_risk, use_container_width=True)

# -------------------------------
# DEPARTMENT-WISE RISK
# -------------------------------
st.subheader("🏛️ Department-Wise Risk Overview")

dept_risk = (
    df.groupby(["department", "fraud_risk"])
    .size()
    .reset_index(name="count")
)

fig_dept = px.bar(
    dept_risk,
    x="department",
    y="count",
    color="fraud_risk",
    barmode="stack",
    color_discrete_map={
        "High": "red",
        "Medium": "orange",
        "Low": "green"
    }
)

st.plotly_chart(fig_dept, use_container_width=True)

# -------------------------------
# TOP RISK VENDORS
# -------------------------------
st.subheader("🏢 Top Vendors by Transaction Frequency")

top_vendors = (
    df.groupby("vendor")
    .size()
    .reset_index(name="transaction_count")
    .sort_values(by="transaction_count", ascending=False)
    .head(10)
)

fig_vendor = px.bar(
    top_vendors,
    x="vendor",
    y="transaction_count"
)

st.plotly_chart(fig_vendor, use_container_width=True)

# -------------------------------
# FILTERABLE TABLE
# -------------------------------
st.subheader("📄 Transaction Details")

risk_filter = st.multiselect(
    "Filter by Fraud Risk:",
    options=df["fraud_risk"].unique(),
    default=df["fraud_risk"].unique()
)

filtered_df = df[df["fraud_risk"].isin(risk_filter)]

st.dataframe(
    filtered_df.sort_values(by="anomaly_score"),
    use_container_width=True
)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption(
    "⚠️ AI flags risk for audit review. Final decisions require human verification."
)
