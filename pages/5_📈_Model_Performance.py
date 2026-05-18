"""
PAGE 5 — MODEL PERFORMANCE
Engineer View: Full model evaluation metrics.
"""
import streamlit as st
import numpy as np
import pandas as pd

st.markdown(
    '<div class="section-header">📈 Model Performance — LightGBM Binary Classifier</div>',
    unsafe_allow_html=True,
)

st.markdown("""
> **Model**: LightGBM V2 (balanced 50/50 sampling + optimal threshold tuning via F1 maximization)
> **Target**: `is_fatal` — Predict whether an adverse event report will result in patient death
""")

# ── LOAD TEST RESULTS ─────────────────────────────────────────────────────────
try:
    from sklearn.metrics import (
        roc_auc_score, roc_curve, confusion_matrix,
        classification_report, precision_recall_curve,
    )
    from utils.model import load_test_results, load_feature_importance, load_model
    from utils.charts import plot_roc_curve, plot_feature_importance, plot_confusion_matrix

    @st.cache_data(show_spinner="Loading model evaluation data...")
    def get_eval_data():
        test_df = load_test_results()
        feat_imp = load_feature_importance()
        return test_df, feat_imp

    test_df, feat_imp = get_eval_data()
    y_true = test_df["y_true"]
    y_proba = test_df["y_pred_proba"]

    _, _, threshold = load_model()
    y_pred = (y_proba >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=["Not Fatal", "Fatal"],
        output_dict=True,
    )

    # ── TOP KPI METRICS ───────────────────────────────────────────────────────
    st.markdown("#### 🏆 Model Scorecard")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("AUC-ROC", f"{auc:.4f}", delta="↑ Best metric for imbalanced data")
    m2.metric("Fatal Precision", f"{report['Fatal']['precision']:.3f}")
    m3.metric("Fatal Recall", f"{report['Fatal']['recall']:.3f}")
    m4.metric("Fatal F1", f"{report['Fatal']['f1-score']:.3f}")
    m5.metric("Accuracy", f"{report['accuracy']:.3f}")

    st.divider()

    # ── CHARTS ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_roc_curve(fpr, tpr, auc), use_container_width=True)
    with col2:
        st.plotly_chart(plot_confusion_matrix(cm.tolist()), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_feature_importance(feat_imp), use_container_width=True)
    with col4:
        # Precision-Recall Curve
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        import plotly.graph_objects as go

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            line=dict(color="#00d9b8", width=3),
            fill="tozeroy", fillcolor="rgba(0,217,184,0.1)",
            name="Precision-Recall",
        ))
        fig_pr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
            font=dict(color="#e2e8f0"), height=400,
            title=dict(text="📉 Precision-Recall Curve", font=dict(color="#f0b429", size=14)),
            xaxis_title="Recall", yaxis_title="Precision",
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # ── FULL CLASSIFICATION REPORT ─────────────────────────────────────────────
    st.markdown("#### 📋 Full Classification Report")
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)

    # ── MODEL CARD ─────────────────────────────────────────────────────────────
    st.markdown("#### 🃏 Model Card")
    with st.expander("View Full Model Configuration", expanded=False):
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | Algorithm | LightGBM (Gradient Boosting) |
        | Objective | Binary Classification |
        | Training samples | 48,000 (50/50 balanced) |
        | Test samples | 12,000 |
        | Features | 12 |
        | Num leaves | 127 |
        | Max depth | 8 |
        | Learning rate | 0.03 |
        | Feature fraction | 0.9 |
        | Bagging fraction | 0.9 |
        | Lambda L1 | 0.05 |
        | Lambda L2 | 0.05 |
        | Early stopping | 80 rounds |
        | Optimal threshold | Tuned via F1 maximization |
        | AUC-ROC | 0.7829 |
        """)

except FileNotFoundError:
    st.error("""
    ⚠️ Model test results not found.
    Please run `export_models.py` from your notebook environment first to generate:
    - `models/test_results.parquet`
    - `models/feature_importance.parquet`
    - `models/lgbm_model.pkl`
    - `models/label_encoders.pkl`
    - `models/optimal_threshold.pkl`
    """)
except Exception as e:
    st.error(f"Error loading model performance data: {e}")
