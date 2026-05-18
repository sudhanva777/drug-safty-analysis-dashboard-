"""
PAGE 6 — INSIGHTS & STORYTELLING
Decision Layer: Where data becomes intelligence.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data_loader import load_data, HARDCODED_KPIS

st.markdown(
    '<div class="section-header">💡 Key Findings & Intelligence Summary</div>',
    unsafe_allow_html=True,
)

st.markdown("""
> This page distills 528,000 FDA FAERS reports (2015–2026) into actionable intelligence.
> These are statistically grounded findings from the full analysis pipeline.
""")

kpis = HARDCODED_KPIS


@st.cache_data(show_spinner="Computing insights...")
def get_insights_data():
    df = load_data(sample_n=150_000)

    # Top risky drugs (min 300 reports in sample)
    drug_stats = df.groupby("suspect_drug").agg(
        total=("report_id", "count"), fatal=("is_fatal", "sum"),
    ).query("total >= 300")
    drug_stats["fatal_pct"] = drug_stats["fatal"] / drug_stats["total"] * 100
    top_risky = drug_stats.nlargest(10, "fatal_pct")

    # Year-wise fatality trend
    year_trend = df.groupby("year")["is_fatal"].mean() * 100

    # Age group fatality
    age_fatal = df[df["age_group"].notna() & (df["age_group"] != "Unknown")].groupby("age_group")["is_fatal"].mean() * 100

    # Polypharmacy effect
    poly_fatal = df[
        df["drug_count_category"].notna() & (df["drug_count_category"] != "Unknown")
    ].groupby("drug_count_category")["is_fatal"].mean() * 100

    return top_risky, year_trend, age_fatal, poly_fatal


top_risky, year_trend, age_fatal, poly_fatal = get_insights_data()

# ── STORY 1 — FATALITY LANDSCAPE ──────────────────────────────────────────────
st.markdown("### 📖 Story 1 — The Fatality Landscape")
col1, col2 = st.columns([1, 1.5])
with col1:
    st.markdown(f"""
    <div style="background:#111827; border:1px solid #1e2d3d; border-radius:12px; padding:1.5rem;">
        <h3 style="color:#f0b429;">10.3% of all reported adverse events result in death</h3>
        <p style="color:#94a3b8;">That's <b style="color:#ef4444;">54,301 fatal reports</b> across 528,000 cases
        spanning 2015–2025. Among those, nearly one in three required hospitalization,
        and 4.3% were immediately life-threatening.</p>
        <hr style="border-color:#1e2d3d;">
        <p style="color:#94a3b8;">The year-on-year trend shows a <b style="color:#f0b429;">consistent rise
        in reporting volume</b> post-2020, coinciding with increased pharmacovigilance
        and COVID-era drug interactions.</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=year_trend.index.tolist(), y=year_trend.values.tolist(),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        line=dict(color="#ef4444", width=3),
        mode="lines+markers+text",
        text=[f"{v:.1f}%" for v in year_trend.values],
        textposition="top center",
        textfont=dict(color="#f0b429", size=10),
    ))
    fig_trend.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
        font=dict(color="#e2e8f0"), height=280,
        xaxis_title="Year", yaxis_title="Fatality Rate (%)",
        title=dict(text="Annual Fatality Rate Trend", font=dict(color="#f0b429", size=13), x=0),
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# ── STORY 2 — ELDERLY RISK ────────────────────────────────────────────────────
st.markdown("### 📖 Story 2 — The Age Factor")
col1, col2 = st.columns([1.5, 1])
with col1:
    # Detect actual age-group labels
    all_groups = age_fatal.index.tolist()
    age_order_candidates = [
        ["Pediatric(0-18)", "Adult(19-40)", "Middle-Aged(41-60)", "Senior(61-80)", "Elderly(81+)"],
        ["Pediatric(0-18)", "Adult(19-40)", "Middle-Aged(41-65)", "Senior(66-80)", "Elderly(81+)"],
    ]
    age_order = None
    for cand in age_order_candidates:
        if set(cand).issubset(set(all_groups)):
            age_order = cand
            break
    if age_order is None:
        age_order = sorted([g for g in all_groups if g])

    age_data = age_fatal.reindex(age_order).dropna().reset_index()
    age_data.columns = ["age_group", "fatal_pct"]
    colors_age = ["#22c55e", "#22c55e", "#f97316", "#f97316", "#ef4444"][:len(age_data)]
    fig_age = go.Figure(go.Bar(
        x=age_data["age_group"], y=age_data["fatal_pct"],
        marker=dict(color=colors_age),
        text=[f"{v:.1f}%" for v in age_data["fatal_pct"]],
        textposition="outside", textfont=dict(color="#ffffff"),
    ))
    fig_age.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
        font=dict(color="#e2e8f0"), height=300,
        title=dict(text="Fatality Rate by Age Group", font=dict(color="#f0b429", size=13)),
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig_age, use_container_width=True)
with col2:
    st.markdown(f"""
    <div style="background:#111827; border:1px solid #ef4444; border-radius:12px; padding:1.5rem;">
        <h3 style="color:#ef4444;">3.8× Higher Fatality Risk in Elderly Patients</h3>
        <p style="color:#94a3b8;">Patients aged <b style="color:#f0b429;">81+</b> face fatality rates
        nearly <b style="color:#ef4444;">4× higher</b> than young adults (19–40).
        This is the single strongest demographic signal in the dataset.</p>
        <hr style="border-color:#1e2d3d;">
        <p style="color:#94a3b8;">Senior (61–80) patients also show elevated risk,
        suggesting that geriatric pharmacovigilance is a critical clinical gap.
        Age ranks as a <b style="color:#f0b429;">top-3 feature</b> in the LightGBM model.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── STORY 3 — POLYPHARMACY ────────────────────────────────────────────────────
st.markdown("### 📖 Story 3 — The Polypharmacy Effect")
col1, col2 = st.columns([1, 1.5])
with col1:
    st.markdown(f"""
    <div style="background:#111827; border:1px solid #f97316; border-radius:12px; padding:1.5rem;">
        <h3 style="color:#f97316;">Patients on 6+ drugs face markedly higher fatality</h3>
        <p style="color:#94a3b8;">The data shows a near-linear relationship between
        concurrent drug count and fatality rate. Polypharmacy (6+ drugs) patients
        face significantly elevated risk compared to single-drug patients.</p>
        <hr style="border-color:#1e2d3d;">
        <p style="color:#94a3b8;">This is especially critical in elderly populations,
        where polypharmacy is common. <b style="color:#f0b429;">num_drugs</b> is a
        top-5 feature in the LightGBM model.</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    poly_order = ["Single", "2-3 drugs", "4-5 drugs", "Polypharmacy(6+)"]
    poly_data = poly_fatal.reindex(poly_order).dropna().reset_index()
    poly_data.columns = ["category", "fatal_pct"]
    colors_poly = ["#22c55e", "#00d9b8", "#f97316", "#ef4444"][:len(poly_data)]
    fig_poly = go.Figure(go.Bar(
        x=poly_data["category"], y=poly_data["fatal_pct"],
        marker=dict(color=colors_poly),
        text=[f"{v:.1f}%" for v in poly_data["fatal_pct"]],
        textposition="outside", textfont=dict(color="#ffffff"),
    ))
    fig_poly.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
        font=dict(color="#e2e8f0"), height=300,
        title=dict(text="Fatality Rate by Drug Count Category", font=dict(color="#f0b429", size=13)),
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig_poly, use_container_width=True)

st.divider()

# ── STORY 4 — HIGH RISK DRUGS ─────────────────────────────────────────────────
st.markdown("### 📖 Story 4 — Highest-Risk Drug Signals")
fig_drugs = go.Figure(go.Bar(
    x=top_risky["fatal_pct"].values[::-1],
    y=top_risky.index.tolist()[::-1],
    orientation="h",
    marker=dict(
        color=[
            f"rgba(239,68,68,{0.4 + 0.6 * v / top_risky['fatal_pct'].max()})"
            for v in top_risky["fatal_pct"].values[::-1]
        ],
        line=dict(color="#ef4444", width=1),
    ),
    text=[
        f"{v:.1f}%  ({int(n):,} reports)"
        for v, n in zip(top_risky["fatal_pct"].values[::-1], top_risky["total"].values[::-1])
    ],
    textposition="inside", textfont=dict(color="#ffffff", size=11),
    hovertemplate="%{y}<br>Fatality: %{x:.1f}%<extra></extra>",
))
fig_drugs.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
    font=dict(color="#e2e8f0"), height=380,
    title=dict(
        text="Top 10 Drugs by Fatality Rate (min 300 reports in sample)",
        font=dict(color="#f0b429", size=14),
    ),
    margin=dict(t=50, b=30, l=220, r=20),
)
st.plotly_chart(fig_drugs, use_container_width=True)

# ── KEY TAKEAWAYS ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 🎯 Executive Takeaways")
takeaways = [
    ("🔴", "10.3% fatality rate",
     "1 in 10 adverse event reports results in death — significantly higher than typical drug safety benchmarks."),
    ("🟡", "3.8× elderly risk multiplier",
     "Patients 81+ have nearly 4× the fatality rate of adults 19–40. Age is the strongest demographic predictor."),
    ("🟠", "Polypharmacy amplifies risk",
     "6+ concurrent drugs correlates with markedly higher fatality. Medication reconciliation is critical."),
    ("🔵", "is_hospitalized is top predictor",
     "Clinical severity flags (hospitalization, life-threat) are the model's strongest features — not demographics alone."),
    ("🟢", "LightGBM AUC 0.7829",
     "The model successfully identifies high-risk patients. Threshold tuning improves recall for the fatal class."),
]
for emoji, title, desc in takeaways:
    st.markdown(f"""
    <div style="background:#111827; border-left:4px solid #f0b429; border-radius:0 8px 8px 0;
                 padding:1rem 1.5rem; margin:0.5rem 0;">
        <b style="color:#f0b429;">{emoji} {title}</b><br>
        <span style="color:#94a3b8;">{desc}</span>
    </div>
    """, unsafe_allow_html=True)
