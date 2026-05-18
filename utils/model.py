"""
Model loading, encoding, and prediction utilities.
Wraps LightGBM, Random Forest, and Logistic Regression models.
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
@st.cache_resource(show_spinner="Loading model encoder dict...")
def load_model():
    """Legacy helper for backward compatibility."""
    model = joblib.load(MODEL_DIR / "lgbm_model.pkl")
    le_dict = joblib.load(MODEL_DIR / "label_encoders.pkl")
    threshold = joblib.load(MODEL_DIR / "optimal_threshold.pkl")
    return model, le_dict, threshold


@st.cache_resource(show_spinner="Loading predictive AI model...")
def load_selected_model(model_name: str = "LightGBM"):
    """
    Dynamically loads the selected model along with shared encoders and tuned threshold.
    Supports LightGBM, Random Forest, and Logistic Regression.
    """
    le_dict = joblib.load(MODEL_DIR / "label_encoders.pkl")
    
    # Resolve threshold
    threshold_path = MODEL_DIR / "optimal_thresholds.pkl"
    if threshold_path.is_file():
        thresholds = joblib.load(threshold_path)
        threshold = thresholds.get(model_name, 0.5)
    else:
        threshold = joblib.load(MODEL_DIR / "optimal_threshold.pkl")

    # Load specific classifier
    if model_name == "Random Forest":
        model = joblib.load(MODEL_DIR / "rf_model.pkl")
    elif model_name == "Logistic Regression":
        model = joblib.load(MODEL_DIR / "lr_model.pkl")
    else:
        model = joblib.load(MODEL_DIR / "lgbm_model.pkl")

    return model, le_dict, threshold


def encode_input(input_dict: dict, le_dict: dict) -> pd.DataFrame:
    """
    Encode a single prediction input dict using the LabelEncoders.
    """
    return _encode_input(input_dict, le_dict, FEATURES)


def predict_risk(input_dict: dict, model_name: str = "LightGBM") -> dict:
    """
    Full prediction pipeline: encode → predict using selected model → return structured result.
    Returns: probability (0-1), label, risk_level, confidence_pct
    """
    model, le_dict, threshold = load_selected_model(model_name)
    X = encode_input(input_dict, le_dict)

    if model_name == "LightGBM":
        proba = float(model.predict(X)[0])
    else:
        # Scikit-learn models use predict_proba
        proba = float(model.predict_proba(X)[0, 1])

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
