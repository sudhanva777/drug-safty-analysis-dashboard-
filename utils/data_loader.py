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

# Anchor to the project root (one level above utils/)
_PROJECT_DIR = Path(__file__).resolve().parent.parent

_CANDIDATE_PATHS = [
    _PROJECT_DIR / "data" / _CSV_NAME,           # <project>/data/
    _PROJECT_DIR / _CSV_NAME,                     # <project>/
    _PROJECT_DIR.parent / _CSV_NAME,              # one level up from project
]

DATA_PATH: str | None = None
for _p in _CANDIDATE_PATHS:
    if _p.is_file():
        DATA_PATH = str(_p)
        break

if DATA_PATH is None:
    # Last resort — give an actionable error later on load
    DATA_PATH = str(_CANDIDATE_PATHS[0])

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
def load_data(sample_n: int | None = None) -> pd.DataFrame:
    """
    Load the FDA FAERS dataset. Optionally sample for UI performance.
    Always parses dates and bool columns correctly.
    """
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found.  Searched:\n"
            + "\n".join(f"  • {p}" for p in _CANDIDATE_PATHS)
            + f"\nPlease place '{_CSV_NAME}' in one of these locations."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["receive_date"], low_memory=False)

    bool_cols = [
        "is_fatal", "is_hospitalized", "is_life_threat",
        "is_disabling", "patient_recovered",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

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
