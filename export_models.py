"""
export_models.py — Standalone model training & artifact export
==============================================================
Trains LightGBM, Random Forest, and Logistic Regression classifiers to predict `is_fatal` 
from the FDA FAERS dataset, then exports all artifacts required by the Streamlit dashboard.

Run from the drug-safety-dashboard/ directory:
    python export_models.py

Produces (in models/):
    lgbm_model.pkl          – trained LightGBM booster
    rf_model.pkl            – trained Random Forest classifier
    lr_model.pkl            – trained Logistic Regression classifier
    label_encoders.pkl      – dict of sklearn LabelEncoders
    optimal_threshold.pkl   – F1-tuned probability threshold (LightGBM)
    optimal_thresholds.pkl  – dict of F1-tuned thresholds for all models
    feature_list.pkl        – ordered feature name list
    test_results.parquet    – test-set predictions with probabilities for all models
    feature_importance.parquet – feature importance table containing LGBM & RF metrics
"""

import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR  = SCRIPT_DIR / "models"
CSV_NAME   = "fda_adverse_events_2015_2026_CLEAN.csv"

FEATURES = [
    "num_drugs",
    "num_reactions",
    "patient_age_years",
    "patient_weight_kg",
    "patient_sex",
    "country",
    "drug_route",
    "age_group",
    "drug_count_category",
    "is_hospitalized",
    "is_life_threat",
    "is_disabling",
]

CAT_COLS = [
    "patient_sex",
    "country",
    "drug_route",
    "age_group",
    "drug_count_category",
]

BOOL_COLS = ["is_hospitalized", "is_life_threat", "is_disabling"]
TARGET    = "is_fatal"

# Sampling budget — keeps memory under ~1 GB and trains all 3 models in < 45 s
SAMPLE_N  = 150_000


# ── Helpers ────────────────────────────────────────────────────────────────
def _find_csv() -> Path:
    """Locate the dataset CSV in common locations."""
    candidates = [
        SCRIPT_DIR / "data" / CSV_NAME,
        SCRIPT_DIR / CSV_NAME,
        SCRIPT_DIR.parent / CSV_NAME,
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Dataset CSV not found.  Looked in:\n"
        + "\n".join(f"  • {c}" for c in candidates)
    )


def _load_and_sample(path: Path, n: int) -> pd.DataFrame:
    """Read CSV, coerce types, sample to *n* rows (stratified on target)."""
    print(f"  Reading {path.name} …")
    df = pd.read_csv(path, low_memory=False)
    for c in BOOL_COLS + [TARGET, "patient_recovered"]:
        if c in df.columns:
            df[c] = df[c].astype(bool)

    # Stratified sample so the target ratio is preserved
    if n and len(df) > n:
        df, _ = train_test_split(
            df, train_size=n, stratify=df[TARGET], random_state=42
        )
        print(f"  Sampled to {len(df):,} rows (stratified).")
    return df


def _prepare_features(df: pd.DataFrame):
    """
    Encode categoricals with LabelEncoder, fill numeric nulls, cast bools.
    Returns (X DataFrame, y Series, le_dict).
    """
    le_dict: dict[str, LabelEncoder] = {}

    for col in CAT_COLS:
        df[col] = df[col].fillna("Unknown").astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Numeric nulls → median
    for col in ["patient_age_years", "patient_weight_kg",
                "num_drugs", "num_reactions"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Bool → int
    for col in BOOL_COLS:
        df[col] = df[col].astype(int)

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    return X, y, le_dict


def _train_lgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM with balanced sampling + early stopping."""
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "max_depth": 8,
        "learning_rate": 0.03,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "lambda_l1": 0.05,
        "lambda_l2": 0.05,
        "is_unbalance": True,       # handles class imbalance
        "seed": 42,
    }

    train_set = lgb.Dataset(X_train, y_train)
    valid_set = lgb.Dataset(X_test, y_test, reference=train_set)

    callbacks = [
        lgb.early_stopping(stopping_rounds=80),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[valid_set],
        callbacks=callbacks,
    )
    return model


def _find_optimal_threshold(y_true, y_proba):
    """Pick the threshold that maximises F1 on the precision-recall curve."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    # F1 = 2 * P * R / (P + R)
    f1 = np.where((prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))
    best_idx = np.argmax(f1)
    # Safe boundary check
    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    return 0.5


# ── Main pipeline ──────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 60)
    print("  Drug Safety Multi-Model — Training & Export Pipeline")
    print("=" * 60)

    # 1. Locate data
    csv_path = _find_csv()
    print(f"\n[1/6] Dataset found: {csv_path}")

    # 2. Load & sample
    print("[2/6] Loading data …")
    df = _load_and_sample(csv_path, SAMPLE_N)

    # 3. Prepare features
    print("[3/6] Preparing features …")
    X, y, le_dict = _prepare_features(df)

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"  Fatal rate – train: {y_train.mean()*100:.1f}%  test: {y_test.mean()*100:.1f}%")

    # 5. Train LightGBM Model
    print("[4/6] Training LightGBM …")
    lgb_model = _train_lgbm(X_train, y_train, X_test, y_test)
    lgb_proba = lgb_model.predict(X_test)
    lgb_auc = roc_auc_score(y_test, lgb_proba)
    lgb_thresh = _find_optimal_threshold(y_test.values, lgb_proba)

    # 6. Train Random Forest Model
    print("      Training Random Forest Classifier …")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_proba)
    rf_thresh = _find_optimal_threshold(y_test.values, rf_proba)

    # 7. Train Logistic Regression Model
    print("      Training Logistic Regression Classifier …")
    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_model.fit(X_train, y_train)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_proba)
    lr_thresh = _find_optimal_threshold(y_test.values, lr_proba)

    print("\n[5/6] Model Evaluations on Test Set:")
    print(f"  • LightGBM:             AUC-ROC = {lgb_auc:.4f} | Optimal Threshold = {lgb_thresh:.4f}")
    print(f"  • Random Forest:        AUC-ROC = {rf_auc:.4f} | Optimal Threshold = {rf_thresh:.4f}")
    print(f"  • Logistic Regression:  AUC-ROC = {lr_auc:.4f} | Optimal Threshold = {lr_thresh:.4f}")

    # 8. Export artifacts
    print("\n[6/6] Exporting artifacts …")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Serializations
    joblib.dump(lgb_model,   MODEL_DIR / "lgbm_model.pkl")
    joblib.dump(rf_model,    MODEL_DIR / "rf_model.pkl")
    joblib.dump(lr_model,    MODEL_DIR / "lr_model.pkl")
    joblib.dump(le_dict,     MODEL_DIR / "label_encoders.pkl")
    joblib.dump(lgb_thresh,  MODEL_DIR / "optimal_threshold.pkl")
    
    # Combined threshold directory
    thresholds = {
        "LightGBM": lgb_thresh,
        "Random Forest": rf_thresh,
        "Logistic Regression": lr_thresh
    }
    joblib.dump(thresholds,  MODEL_DIR / "optimal_thresholds.pkl")
    joblib.dump(FEATURES,    MODEL_DIR / "feature_list.pkl")

    # Save test results containing all predictions
    test_results = X_test.copy()
    test_results["y_true"] = y_test.values
    test_results["y_pred_proba"] = lgb_proba # default for legacy compat
    test_results["y_pred_proba_lgbm"] = lgb_proba
    test_results["y_pred_proba_rf"] = rf_proba
    test_results["y_pred_proba_lr"] = lr_proba
    test_results.to_parquet(MODEL_DIR / "test_results.parquet", index=False)

    # Save feature importances
    feat_imp = pd.DataFrame({
        "feature": FEATURES,
        "importance_lgbm": lgb_model.feature_importance(importance_type="gain"),
        "importance_rf": rf_model.feature_importances_,
    }).sort_values("importance_lgbm", ascending=False)
    feat_imp.to_parquet(MODEL_DIR / "feature_importance.parquet", index=False)

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"  [OK] All model artifacts saved to {MODEL_DIR}")
    print(f"     Elapsed:    {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
