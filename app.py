"""
Drug Safety Signal Intelligence Dashboard
FDA FAERS 2015–2026 · LightGBM AUC 0.7829 · 528K Reports

Streamlit entry point + sidebar nav + landing page.
"""
import streamlit as st
import os

st.set_page_config(
    page_title="Drug Safety Intelligence | FDA FAERS",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "FDA FAERS 2015–2026 Drug Safety Signal Dashboard"},
)

# ── Inject custom CSS ──────────────────────────────────────────────────────
_css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
if os.path.isfile(_css_path):
    with open(_css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar brand ──────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; padding: 1.5rem 0 1rem 0;">
    <div style="font-size:2rem;">🧬</div>
    <div style="font-size:1.2rem; font-weight:700; color:#f0b429; letter-spacing:1px;">
        DRUG SAFETY
    </div>
    <div style="font-size:0.7rem; color:#94a3b8; letter-spacing:2px; text-transform:uppercase;">
        Intelligence Platform
    </div>
    <hr style="border-color:#1e2d3d; margin:1rem 0;">
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="color:#94a3b8; font-size:0.75rem; padding: 0.5rem;">
    <b style="color:#e2e8f0;">Dataset</b><br>
    FDA FAERS 2015–2026<br>
    528,000 Reports · 162 Countries<br><br>
    <b style="color:#e2e8f0;">Model</b><br>
    LightGBM · AUC 0.7829<br>
    Target: Fatal Outcome Prediction
</div>
""", unsafe_allow_html=True)

# ── Landing page content ──────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 3rem 0 1rem 0;">
    <div style="font-size:3.5rem; margin-bottom:1rem;">🧬</div>
    <h1 style="font-size:2.5rem; font-weight:800; color:#f0b429; margin:0; letter-spacing:2px;">
        DRUG SAFETY INTELLIGENCE
    </h1>
    <p style="color:#94a3b8; font-size:1.1rem; margin-top:0.5rem;">
        FDA FAERS 2015–2026 · 528,000 Reports · LightGBM AUC 0.7829
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ────────────────────────────────────────────────────────────────
from utils.data_loader import HARDCODED_KPIS  # noqa: E402

cols = st.columns(9)
kpi_items = [
    ("528K", "Total Reports"),
    ("10.3%", "Fatality Rate"),
    ("35.6%", "Hospitalized"),
    ("74.8%", "Serious Cases"),
    ("9,828", "Unique Drugs"),
    ("10,446", "Reactions"),
    ("162", "Countries"),
    ("AUC 0.78", "Model Score"),
    ("3.8×", "Elderly Risk"),
]
for col, (val, label) in zip(cols, kpi_items):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <span class="kpi-value">{val}</span>
            <span class="kpi-label">{label}</span>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.9rem; padding:1rem;">
    Navigate using the sidebar pages to explore the full intelligence dashboard.
</div>
""", unsafe_allow_html=True)
