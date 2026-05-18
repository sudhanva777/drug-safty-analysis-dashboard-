"""
PAGE 2 — EDA DASHBOARD
Exploration Layer: Interactive, alive, dynamic.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_data, render_common_sidebar
from utils.charts import (
    plot_annual_trend, plot_quarterly_trend, plot_top_drugs,
    plot_top_reactions, plot_age_fatality, plot_sex_distribution,
    plot_outcome_rates, plot_polypharmacy_fatal, plot_country_top,
    plot_route_distribution,
)

# ── Render unified sidebar and check dataset status ──────────────────────────
status = render_common_sidebar()
if not status["dataset_active"]:
    st.markdown(
        '<div class="section-header">🔍 Exploratory Data Analysis Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.warning("⚠️ **Dataset Offline**\nPlease upload the adverse events dataset on the Home tab to activate this page's analysis.")
    st.stop()

st.markdown(
    '<div class="section-header">🔍 Exploratory Data Analysis Dashboard</div>',
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading EDA data...")
def get_eda_data():
    return load_data(sample_n=100_000)


df = get_eda_data()

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Temporal", "💊 Drugs & Reactions", "👥 Demographics",
    "🌍 Geography", "🔬 Custom Explorer",
])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_annual_trend(df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_quarterly_trend(df), use_container_width=True)
    st.plotly_chart(plot_outcome_rates(df), use_container_width=True)

with tab2:
    n_top = st.slider("Top N drugs/reactions", 5, 20, 10)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_top_drugs(df, n_top), use_container_width=True)
    with col2:
        st.plotly_chart(plot_top_reactions(df, n_top), use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_route_distribution(df), use_container_width=True)
    with col4:
        st.plotly_chart(plot_polypharmacy_fatal(df), use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_age_fatality(df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_sex_distribution(df), use_container_width=True)

    # Age distribution
    age_data = df["patient_age_years"].dropna()
    fig_age = px.histogram(
        age_data, nbins=60, color_discrete_sequence=["#3b82f6"],
        title="Patient Age Distribution",
    )
    fig_age.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
        font=dict(color="#e2e8f0"), height=300,
    )
    st.plotly_chart(fig_age, use_container_width=True)

with tab4:
    n_countries = st.slider("Top N countries", 5, 30, 10)
    st.plotly_chart(plot_country_top(df, n_countries), use_container_width=True)

with tab5:
    # ── CUSTOM EXPLORER ─────────────────────────────────────────────
    st.markdown("#### 🧪 Custom Column Explorer")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    col_x = st.selectbox("X-axis", num_cols)
    col_y = st.selectbox("Y-axis (optional)", ["None"] + num_cols)
    color_by = st.selectbox(
        "Color by",
        ["None", "is_fatal", "patient_sex", "age_group",
         "drug_count_category", "serious"],
    )
    plot_type = st.radio("Plot type", ["Histogram", "Box", "Scatter"], horizontal=True)

    color_col = None if color_by == "None" else color_by

    if plot_type == "Histogram":
        fig = px.histogram(
            df, x=col_x, color=color_col,
            color_discrete_sequence=["#3b82f6", "#ef4444", "#22c55e", "#f97316"],
        )
    elif plot_type == "Box":
        fig = px.box(
            df, y=col_x, color=color_col,
            color_discrete_sequence=["#3b82f6", "#ef4444", "#22c55e", "#f97316"],
        )
    else:
        y_col = col_y if col_y != "None" else "patient_age_years"
        fig = px.scatter(
            df.sample(min(5000, len(df))), x=col_x, y=y_col, color=color_col,
            opacity=0.5,
            color_discrete_sequence=["#3b82f6", "#ef4444", "#22c55e", "#f97316"],
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
        font=dict(color="#e2e8f0"), height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
