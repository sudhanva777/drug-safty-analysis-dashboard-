"""
Model loading, encoding, and prediction utilities.
Wraps the LightGBM V2 model exported from the notebook.
"""
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

from utils.preprocessing import encode_input as _encode_input

# ── PATHS & CONSTANTS ──────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

FEATURES = [
    "num_drugs", "num_reactions", "patient_age_years", "patient_weight_kg",
    "patient_sex", "country", "drug_route", "age_group",
    "drug_count_category", "is_hospitalized", "is_life_threat", "is_disabling",
]

CAT_COLS = [
    "patient_sex", "country", "drug_route",
    "age_group", "drug_count_category",
]

NUM_MEDIANS = {
    "num_drugs": 2,
    "num_reactions": 2,
    "patient_age_years": 56.0,
    "patient_weight_kg": 75.0,
}


# ── MODEL LOADING ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading LightGBM model...")
def load_model():
    model = joblib.load(MODEL_DIR / "lgbm_model.pkl")
    le_dict = joblib.load(MODEL_DIR / "label_encoders.pkl")
    threshold = joblib.load(MODEL_DIR / "optimal_threshold.pkl")
    return model, le_dict, threshold


def encode_input(input_dict: dict, le_dict: dict) -> pd.DataFrame:
    """
    Encode a single prediction input dict using the notebook's LabelEncoders.
    Delegates to preprocessing.encode_input.
    """
    return _encode_input(input_dict, le_dict, FEATURES)


def predict_risk(input_dict: dict) -> dict:
    """
    Full prediction pipeline: encode → predict → return structured result.
    Returns: probability (0-1), label, risk_level, confidence_pct
    """
    model, le_dict, threshold = load_model()
    X = encode_input(input_dict, le_dict)

    proba = float(model.predict(X)[0])
    label = "Fatal" if proba >= threshold else "Not Fatal"

    if proba >= 0.7:
        risk_level = "HIGH"
        risk_color = "#ef4444"
        risk_emoji = "🔴"
    elif proba >= 0.4:
        risk_level = "MEDIUM"
        risk_color = "#f97316"
        risk_emoji = "🟡"
    else:
        risk_level = "LOW"
        risk_color = "#22c55e"
        risk_emoji = "🟢"

    return {
        "probability": proba,
        "probability_pct": round(proba * 100, 2),
        "label": label,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_emoji": risk_emoji,
        "threshold_used": threshold,
        "confidence": round(max(proba, 1 - proba) * 100, 2),
    }


@st.cache_data(show_spinner=False)
def load_test_results():
    """Load pre-computed test set results for performance page."""
    return pd.read_parquet(MODEL_DIR / "test_results.parquet")


@st.cache_data(show_spinner=False)
def load_feature_importance():
    return pd.read_parquet(MODEL_DIR / "feature_importance.parquet")
