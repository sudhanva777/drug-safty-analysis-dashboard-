"""
PAGE 5 — MODEL PERFORMANCE
Diagnostics Layer: Inspect classifier curves and evaluation matrices.
"""
import streamlit as st
import numpy as np
import pandas as pd
from utils.data_loader import render_common_sidebar

# ── Render unified sidebar and check model status ──────────────────────────
status = render_common_sidebar()

st.markdown(
    '<div class="section-header">📈 Model Performance & Diagnostics</div>',
    unsafe_allow_html=True,
)

st.markdown("""
> Compare metrics, curves, and diagnostic profiles across our trained predictive AI models.
""")

# ── Check model availability ──────────────────────────────────────────────
if not status["model_active"]:
    st.warning("""
    🔌 **Model Performance Metrics Offline**
    
    The predictive AI model performance statistics are currently offline.
    
    **How to activate:**
    1. Navigate to the **Home tab**.
    2. Upload the adverse events dataset.
    3. The platform will automatically train all three classifiers (LightGBM, Random Forest, Logistic Regression), perform evaluation, and unlock this performance diagnostics page.
    """)
    st.stop()

# ── Model Switcher ────────────────────────────────────────────────────────
model_name = st.selectbox(
    "🔮 Selected Predictive AI Model",
    ["LightGBM", "Random Forest", "Logistic Regression"],
    help="Switch between the different trained AI classifiers to view their performance metrics and curves."
)

try:
    from sklearn.metrics import (
        roc_auc_score, roc_curve, confusion_matrix,
        classification_report, precision_recall_curve,
    )
    from utils.model import load_test_results, load_feature_importance, load_selected_model
    from utils.charts import plot_roc_curve, plot_feature_importance, plot_confusion_matrix

    @st.cache_data(show_spinner="Loading model evaluation data...")
    def get_eval_data():
        test_df = load_test_results()
        feat_imp = load_feature_importance()
        return test_df, feat_imp

    test_df, feat_imp = get_eval_data()
    y_true = test_df["y_true"]
    
    # Select appropriate probability column
    if model_name == "Random Forest" and "y_pred_proba_rf" in test_df.columns:
        y_proba = test_df["y_pred_proba_rf"]
    elif model_name == "Logistic Regression" and "y_pred_proba_lr" in test_df.columns:
        y_proba = test_df["y_pred_proba_lr"]
    else:
        # Default/LightGBM
        y_proba = test_df["y_pred_proba_lgbm"] if "y_pred_proba_lgbm" in test_df.columns else test_df["y_pred_proba"]

    # Load optimal threshold
    _, _, threshold = load_selected_model(model_name)
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate metrics
    auc = roc_auc_score(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=["Not Fatal", "Fatal"],
        output_dict=True,
    )

    # Load thresholds dict if available
    import joblib
    from pathlib import Path
    model_dir = Path(__file__).resolve().parent.parent / "models"
    thresholds = {}
    if (model_dir / "optimal_thresholds.pkl").is_file():
        thresholds = joblib.load(model_dir / "optimal_thresholds.pkl")

    # ── TOP KPI METRICS ───────────────────────────────────────────────────────
    st.markdown("#### 🏆 Model Scorecard")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("AUC-ROC", f"{auc:.4f}", delta="Primary performance metric")
    m2.metric("Fatal Precision", f"{report['Fatal']['precision']:.3f}", help="Positive Predictive Value")
    m3.metric("Fatal Recall", f"{report['Fatal']['recall']:.3f}", help="True Positive Rate / Sensitivity")
    m4.metric("Fatal F1", f"{report['Fatal']['f1-score']:.3f}", help="Harmonic mean of Precision and Recall")
    m5.metric("Accuracy", f"{report['accuracy']:.3f}")

    st.divider()

    # ── CHARTS ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        # Update trace name dynamically in ROC figure
        roc_fig = plot_roc_curve(fpr, tpr, auc)
        roc_fig.data[0].name = f"{model_name} (AUC = {auc:.4f})"
        roc_fig.update_layout(title=dict(text=f"📊 ROC Curve — {model_name}"))
        st.plotly_chart(roc_fig, use_container_width=True)
    with col2:
        st.plotly_chart(plot_confusion_matrix(cm.tolist()), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if model_name in ["LightGBM", "Random Forest"]:
            imp_col = "importance_lgbm" if model_name == "LightGBM" else "importance_rf"
            feat_imp_display = feat_imp[["feature", imp_col]].rename(columns={imp_col: "importance"})
            fi_fig = plot_feature_importance(feat_imp_display)
            fi_fig.update_layout(title=dict(text=f"🤖 Feature Importance — {model_name}"))
            st.plotly_chart(fi_fig, use_container_width=True)
        else:
            st.info("""
            ℹ️ **Feature Importance Offline**
            
            Feature importance metrics (gain/impurity) are not natively available for Logistic Regression. 
            Use LightGBM or Random Forest models to analyze non-linear feature importances.
            """)
    with col4:
        # Precision-Recall Curve
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        import plotly.graph_objects as go

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            line=dict(color="#00d9b8", width=3),
            fill="tozeroy", fillcolor="rgba(0,217,184,0.1)",
            name=f"{model_name} PR",
        ))
        fig_pr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
            font=dict(color="#e2e8f0"), height=400,
            title=dict(text=f"📉 Precision-Recall Curve — {model_name}", font=dict(color="#f0b429", size=14)),
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
    with st.expander("View Full Model Configurations", expanded=False):
        lgb_t = thresholds.get("LightGBM", 0.5) if thresholds else threshold
        rf_t = thresholds.get("Random Forest", 0.5) if thresholds else threshold
        lr_t = thresholds.get("Logistic Regression", 0.5) if thresholds else threshold
        
        st.markdown(f"""
        | Attribute | LightGBM | Random Forest | Logistic Regression |
        |-----------|----------|---------------|----------------------|
        | **Algorithm** | Gradient Boosting Decision Trees | Bagged Decision Trees | L2 Regularized Linear Classifier |
        | **Objective** | Binary Classification | Binary Classification | Binary Classification |
        | **Imbalance Mitigation** | Class weights scaled | Balanced sample class weights | Balanced sample class weights |
        | **Features Used** | 12 | 12 | 12 |
        | **Tree/Solver Config** | 127 leaves, max depth 8 | 100 trees, max depth 8 | LBFGS solver, max_iter 1000 |
        | **Optimal Threshold** | Tuned via F1 Maximization ({lgb_t:.4f}) | Tuned via F1 Maximization ({rf_t:.4f}) | Tuned via F1 Maximization ({lr_t:.4f}) |
        | **Target Outcome** | `is_fatal` (Patient Death) | `is_fatal` (Patient Death) | `is_fatal` (Patient Death) |
        """)

except FileNotFoundError:
    st.warning(f"""
    🔌 **Model Diagnostics Offline**
    
    The performance metrics and files for **{model_name}** are currently missing.
    
    **How to activate:**
    1. Click the **⚡ Train Predictive AI Model** button in the sidebar.
    2. The platform will automatically compile all three classifiers (LightGBM, Random Forest, and Logistic Regression) and evaluate their performance.
    """)
    st.stop()
except Exception as e:
    st.error(f"Error loading model performance data: {e}")
