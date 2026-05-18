"""
PAGE 1 — DATA OVERVIEW
Trust Layer: Understand the dataset at a glance.
"""
import streamlit as st
import pandas as pd
from utils.data_loader import load_data, HARDCODED_KPIS

st.markdown(
    '<div class="section-header">📊 Dataset Overview — FDA FAERS 2015–2026</div>',
    unsafe_allow_html=True,
)

# Load a sample for UI performance
@st.cache_data(show_spinner="Loading dataset preview...")
def get_preview():
    return load_data(sample_n=50_000)

df = get_preview()
kpis = HARDCODED_KPIS

# ── KPI METRICS ─────────────────────────────────────────────────────────────
st.markdown("#### 📌 At a Glance")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Reports", "528,000", help="All FAERS reports 2015–2025")
c2.metric("Fatal Reports", "54,301", delta="-10.3% of total", delta_color="inverse")
c3.metric("Hospitalized", "188,042", help="35.6% of all reports")
c4.metric("Unique Drugs", "9,828", help="Distinct suspect drugs")
c5.metric("Countries", "162", help="Global coverage")

st.divider()

# ── DATASET SUMMARY ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.markdown("#### 🗂️ Schema Overview")
    schema_data = {
        "Column": [
            "report_id", "receive_date", "year/month/quarter", "serious",
            "is_fatal ⭐", "is_hospitalized", "is_life_threat", "primary_reaction",
            "suspect_drug", "manufacturer", "drug_route", "patient_age_years",
            "age_group", "patient_sex", "patient_weight_kg", "country",
            "num_drugs", "num_reactions", "drug_count_category",
        ],
        "Type": [
            "int64", "datetime", "int/object", "object",
            "bool", "bool", "bool", "object",
            "object", "object", "object", "float64",
            "object", "object", "float64", "object",
            "int64", "int64", "object",
        ],
        "Description": [
            "Unique report identifier",
            "Date FDA received the report",
            "Temporal breakdown columns",
            "Is the adverse event serious?",
            "🎯 MODEL TARGET — Was outcome fatal?",
            "Required hospitalization?",
            "Was the event life-threatening?",
            "Primary adverse reaction",
            "Drug suspected to cause AE",
            "Drug manufacturer",
            "Route of administration",
            "Patient age in years",
            "Binned age category",
            "Patient biological sex",
            "Patient weight (kg)",
            "Country of report",
            "Number of drugs taken",
            "Number of adverse reactions",
            "Polypharmacy category",
        ],
    }
    st.dataframe(pd.DataFrame(schema_data), use_container_width=True, hide_index=True)

with col_right:
    st.markdown("#### 📉 Missing Values")
    missing_data = pd.DataFrame({
        "Column": [
            "patient_weight_kg", "patient_age_years", "drug_indication",
            "brand_name", "drug_route", "pharm_class", "manufacturer",
            "patient_sex", "country",
        ],
        "Missing %": [72.0, 28.7, 45.2, 12.1, 8.4, 31.2, 5.1, 3.8, 2.1],
    }).sort_values("Missing %", ascending=False)

    import plotly.express as px

    fig = px.bar(
        missing_data, x="Missing %", y="Column", orientation="h",
        color="Missing %",
        color_continuous_scale=["#22c55e", "#f97316", "#ef4444"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
        height=320, font=dict(color="#e2e8f0"),
        coloraxis_showscale=False, margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Quick Stats")
    stats = {
        "Date Range": "Jan 2015 — Dec 2025",
        "Mean Age": "55.9 years",
        "% Female": "~52%",
        "% Male": "~48%",
        "Top Country": "United States",
        "Top Drug": "WARFARIN",
        "Top Reaction": "Drug ineffective",
    }
    for k, v in stats.items():
        st.markdown(f"**{k}**: {v}")

st.divider()

# ── DATA PREVIEW ─────────────────────────────────────────────────────────────
st.markdown("#### 🔍 Raw Data Preview")
n_rows = st.slider("Rows to display", 5, 100, 20)
st.dataframe(df.head(n_rows), use_container_width=True)

# ── COLUMN DISTRIBUTION ───────────────────────────────────────────────────────
st.markdown("#### 📊 Column Distribution Explorer")
col_to_explore = st.selectbox(
    "Select a column",
    ["patient_age_years", "num_drugs", "num_reactions", "patient_weight_kg",
     "patient_sex", "age_group", "drug_route", "drug_count_category", "year"],
)

import plotly.express as px  # noqa: E811

if df[col_to_explore].dtype in ["float64", "int64"]:
    fig = px.histogram(
        df[col_to_explore].dropna(), nbins=50,
        color_discrete_sequence=["#3b82f6"],
    )
else:
    vc = df[col_to_explore].value_counts().head(15).reset_index()
    vc.columns = [col_to_explore, "count"]
    fig = px.bar(vc, x=col_to_explore, y="count", color_discrete_sequence=["#00d9b8"])

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
    font=dict(color="#e2e8f0"), height=300, margin=dict(t=20, b=30),
)
st.plotly_chart(fig, use_container_width=True)
