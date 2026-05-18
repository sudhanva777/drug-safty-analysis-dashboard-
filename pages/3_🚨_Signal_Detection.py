"""
PAGE 3 — DRUG SAFETY SIGNAL DETECTION
Core Brain: Drug × Event risk scoring.
"""
import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data

st.markdown(
    '<div class="section-header">🚨 Drug Safety Signal Detection</div>',
    unsafe_allow_html=True,
)

st.markdown("""
> **Signal Detection** identifies drug-event combinations with statistically elevated
> reporting rates. Based on FDA FAERS 528K reports, 2015–2026.
""")


@st.cache_data(show_spinner="Computing signal table...")
def compute_signal_table(min_reports=50):
    df = load_data(sample_n=200_000)
    stats = df.groupby("suspect_drug").agg(
        total_reports=("report_id", "count"),
        fatal_count=("is_fatal", "sum"),
        hospitalized_count=("is_hospitalized", "sum"),
        life_threat_count=("is_life_threat", "sum"),
        mean_age=("patient_age_years", "mean"),
    ).query(f"total_reports >= {min_reports}").reset_index()

    stats["fatality_rate"] = stats["fatal_count"] / stats["total_reports"] * 100
    stats["hospitalization_rate"] = stats["hospitalized_count"] / stats["total_reports"] * 100
    stats["life_threat_rate"] = stats["life_threat_count"] / stats["total_reports"] * 100

    # Composite risk score (weighted)
    stats["risk_score"] = (
        stats["fatality_rate"] * 0.5
        + stats["life_threat_rate"] * 0.3
        + stats["hospitalization_rate"] * 0.2
    )

    # Signal level
    def assign_signal(score):
        if score >= 15:
            return "🔴 HIGH"
        elif score >= 6:
            return "🟡 MEDIUM"
        else:
            return "🟢 LOW"

    stats["signal_level"] = stats["risk_score"].apply(assign_signal)
    return stats.sort_values("risk_score", ascending=False).round(2)


# ── CONTROLS ─────────────────────────────────────────────────────────────────
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
with col_ctrl1:
    min_reps = st.slider("Min reports threshold", 10, 500, 100)
with col_ctrl2:
    signal_filter = st.multiselect(
        "Signal level",
        ["🔴 HIGH", "🟡 MEDIUM", "🟢 LOW"],
        default=["🔴 HIGH", "🟡 MEDIUM"],
    )
with col_ctrl3:
    top_n = st.slider("Top N drugs to show", 10, 100, 25)

signal_df = compute_signal_table(min_reps)
if signal_filter:
    signal_df = signal_df[signal_df["signal_level"].isin(signal_filter)]

# ── SIGNAL SUMMARY METRICS ────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("🔴 HIGH Signal Drugs", len(signal_df[signal_df["signal_level"] == "🔴 HIGH"]))
c2.metric("🟡 MEDIUM Signal Drugs", len(signal_df[signal_df["signal_level"] == "🟡 MEDIUM"]))
c3.metric("🟢 LOW Signal Drugs", len(signal_df[signal_df["signal_level"] == "🟢 LOW"]))

st.divider()

# ── DRUG LOOKUP ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown("#### 🔎 Drug Signal Lookup")
    drug_query = st.text_input("Search drug name", placeholder="e.g. WARFARIN, METFORMIN...")

    if drug_query:
        matches = signal_df[
            signal_df["suspect_drug"].str.upper().str.contains(drug_query.upper(), na=False)
        ].head(5)
        if len(matches) > 0:
            for _, row in matches.iterrows():
                color = (
                    "#ef4444" if "HIGH" in row["signal_level"]
                    else "#f97316" if "MEDIUM" in row["signal_level"]
                    else "#22c55e"
                )
                st.markdown(f"""
                <div style="background:#111827; border:1px solid {color}; border-radius:8px;
                             padding:1rem; margin:0.5rem 0;">
                    <b style="color:{color}; font-size:1.1rem;">{row['suspect_drug']}</b>
                    <span style="float:right; background:rgba(0,0,0,0.3); padding:2px 8px;
                                 border-radius:4px; color:{color};">{row['signal_level']}</span>
                    <br><br>
                    <span style="color:#94a3b8;">Reports: </span>
                    <b>{int(row['total_reports']):,}</b> &nbsp;&nbsp;
                    <span style="color:#94a3b8;">Fatality: </span>
                    <b style="color:#ef4444;">{row['fatality_rate']:.1f}%</b> &nbsp;&nbsp;
                    <span style="color:#94a3b8;">Risk Score: </span>
                    <b style="color:#f0b429;">{row['risk_score']:.1f}</b>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No matching drugs found with sufficient reports.")

with col_right:
    st.markdown("#### 📊 Top Risk Drugs")
    import plotly.graph_objects as go

    top_risk = signal_df.head(top_n).head(15)
    colors = [
        "#ef4444" if "HIGH" in s else "#f97316" if "MEDIUM" in s else "#22c55e"
        for s in top_risk["signal_level"]
    ]
    fig = go.Figure(go.Bar(
        x=top_risk["risk_score"][::-1],
        y=top_risk["suspect_drug"][::-1],
        orientation="h",
        marker=dict(color=colors[::-1]),
        hovertemplate="%{y}<br>Risk Score: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
        font=dict(color="#e2e8f0"), height=420,
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── FULL SIGNAL TABLE ─────────────────────────────────────────────────────────
st.markdown("#### 📋 Full Signal Detection Table")
display_cols = [
    "suspect_drug", "total_reports", "fatality_rate",
    "hospitalization_rate", "life_threat_rate", "risk_score", "signal_level",
]
st.dataframe(
    signal_df[display_cols].head(top_n).reset_index(drop=True),
    use_container_width=True,
    column_config={
        "fatality_rate": st.column_config.ProgressColumn("Fatality %", min_value=0, max_value=100),
        "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=50),
        "total_reports": st.column_config.NumberColumn("Reports", format="%d"),
    },
)
