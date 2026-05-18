"""
Data loading and KPI computation utilities for the Drug Safety Dashboard.
All heavy loads wrapped in @st.cache_data for performance.

Robust path resolution:
  1. data/fda_adverse_events_2015_2026_CLEAN.csv  (inside project)
  2. ../fda_adverse_events_2015_2026_CLEAN.csv     (parent dir)
  3. The project root itself
"""
import pandas as pd
import streamlit as st
import os
from pathlib import Path

# ── DATA PATH RESOLUTION ───────────────────────────────────────────────────
_CSV_NAME = "fda_adverse_events_2015_2026_CLEAN.csv"
_PROJECT_DIR = Path(__file__).resolve().parent.parent

_CANDIDATE_PATHS = [
    _PROJECT_DIR / "data" / _CSV_NAME,           # <project>/data/
    _PROJECT_DIR / _CSV_NAME,                     # <project>/
    _PROJECT_DIR.parent / _CSV_NAME,              # one level up from project
]

def get_resolved_data_path() -> str:
    """Dynamically resolve dataset file path on disk."""
    for _p in _CANDIDATE_PATHS:
        if _p.is_file():
            return str(_p)
    return str(_CANDIDATE_PATHS[0])

# ── HARDCODED KPIs ─────────────────────────────────────────────────────────
# From notebook output (ground truth); avoids recomputing on every page load
HARDCODED_KPIS = {
    "total_reports": 528_000,
    "fatal_count": 54_301,
    "fatal_pct": 10.3,
    "hospitalized_pct": 35.6,
    "serious_pct": 74.8,
    "life_threat_pct": 4.3,
    "unique_drugs": 9_828,
    "unique_reactions": 10_446,
    "countries": 162,
    "manufacturers": 1_570,
    "mean_age": 55.9,
    "elderly_risk_multiplier": 3.8,
    "model_auc": 0.7829,
    "date_range": "2015-01-01 → 2025-12-31",
}


@st.cache_data(show_spinner="Loading FDA FAERS dataset...")
def load_local_csv(path: str) -> pd.DataFrame:
    """Cached low-level reader for the FAERS CSV."""
    df = pd.read_csv(path, parse_dates=["receive_date"], low_memory=False)

    bool_cols = [
        "is_fatal", "is_hospitalized", "is_life_threat",
        "is_disabling", "patient_recovered",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


def load_data(sample_n: int | None = None) -> pd.DataFrame:
    """
    Load the FDA FAERS dataset. Dynamically checks resolved paths.
    Optionally sample for UI performance.
    """
    path = get_resolved_data_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found. Searched:\n"
            + "\n".join(f"  • {p}" for p in _CANDIDATE_PATHS)
        )
    df = load_local_csv(path)
    if sample_n:
        df = df.sample(n=min(sample_n, len(df)), random_state=42)
    return df


@st.cache_data(show_spinner=False)
def get_kpis(df: pd.DataFrame | None = None) -> dict:
    """Return KPIs. Uses hardcoded values for speed; compute live if df passed."""
    if df is None:
        return HARDCODED_KPIS
    return {
        "total_reports": len(df),
        "fatal_count": int(df["is_fatal"].sum()),
        "fatal_pct": round(df["is_fatal"].mean() * 100, 1),
        "hospitalized_pct": round(df["is_hospitalized"].mean() * 100, 1),
        "serious_pct": round((df["serious"] == "Yes").mean() * 100, 1),
        "life_threat_pct": round(df["is_life_threat"].mean() * 100, 1),
        "unique_drugs": df["suspect_drug"].nunique(),
        "unique_reactions": df["primary_reaction"].nunique(),
        "countries": df["country"].nunique(),
        "manufacturers": df["manufacturer"].nunique(),
        "mean_age": round(df["patient_age_years"].mean(), 1),
        "model_auc": HARDCODED_KPIS["model_auc"],
        "date_range": HARDCODED_KPIS["date_range"],
    }


def init_app_state():
    """Initialize application states on first load."""
    if "app_state" not in st.session_state:
        # Check if dataset and models exist on disk
        path = get_resolved_data_path()
        dataset_exists = os.path.isfile(path)
        
        model_dir = _PROJECT_DIR / "models"
        model_files = [
            "lgbm_model.pkl", "rf_model.pkl", "lr_model.pkl", 
            "label_encoders.pkl", "optimal_threshold.pkl", "optimal_thresholds.pkl",
            "feature_list.pkl", "test_results.parquet", "feature_importance.parquet"
        ]
        model_exists = all((model_dir / f).is_file() for f in model_files)
        
        if dataset_exists and model_exists:
            st.session_state["app_state"] = "READY"
        elif dataset_exists:
            st.session_state["app_state"] = "DATASET_UPLOADED"
        else:
            st.session_state["app_state"] = "APP_LOCKED"


def render_common_sidebar() -> dict:
    """
    Renders the platform's unified sidebar, including:
      - Custom CSS injection
      - Sidebar branding
      - Platform state indicators
    Returns a dictionary of platform status flags.
    """
    # Initialize state
    init_app_state()
    state = st.session_state["app_state"]

    # 1. Inject custom CSS
    css_path = _PROJECT_DIR / "assets" / "styles.css"
    if css_path.is_file():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # 2. Sidebar brand
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

    # 3. Status Section
    st.sidebar.markdown("<h3 style='font-size:1.1rem; color:#e2e8f0; margin-bottom:0.5rem;'>⚡ Platform Status</h3>", unsafe_allow_html=True)

    if state == "APP_LOCKED":
        st.sidebar.error("🔒 APP LOCKED")
        st.sidebar.info("💡 Upload the adverse events dataset on the Home tab to unlock the dashboard.")
    elif state == "PROCESSING":
        st.sidebar.warning("⏳ PROCESSING DATA")
    elif state == "DATASET_UPLOADED":
        st.sidebar.warning("⏳ COMPILING AI MODELS")
        st.sidebar.info("💡 Model training is starting automatically on the Home tab...")
    elif state == "READY":
        st.sidebar.success("✅ PLATFORM ACTIVE")
        st.sidebar.success("✅ Predictive AI: Active")
    elif state == "ERROR":
        st.sidebar.error("❌ INGESTION ERROR")
        st.sidebar.info("💡 Review the error logs on the Home tab and re-upload.")

    st.sidebar.markdown("""
    <div style="color:#94a3b8; font-size:0.75rem; padding: 0.5rem; margin-top: 2rem;">
        <b style="color:#e2e8f0;">Metadata</b><br>
        FDA FAERS 2015–2026<br>
        528,000 Reports · 162 Countries<br>
        Models: LightGBM · RF · LR
    </div>
    """, unsafe_allow_html=True)

    return {
        "dataset_active": state in ["READY", "DATASET_UPLOADED"],
        "model_active": state == "READY"
    }
