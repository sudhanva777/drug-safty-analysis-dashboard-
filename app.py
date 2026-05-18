"""
Drug Safety Signal Intelligence Dashboard
FDA FAERS 2015–2026 · 528K Reports · LightGBM, RF, LR Multi-Classifier

Streamlit entry point + sidebar nav + SaaS onboarding wizard.
"""
import streamlit as st
import os
import sys
import io
import importlib
import pandas as pd

st.set_page_config(
    page_title="Drug Safety Intelligence | FDA FAERS",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "FDA FAERS 2015–2026 Drug Safety Signal Dashboard"},
)

# ── Render unified sidebar status panel ─────────────────────────────────────
import utils.data_loader
import importlib
importlib.reload(utils.data_loader)
from utils.data_loader import render_common_sidebar, init_app_state, get_resolved_data_path, _CSV_NAME, _PROJECT_DIR
status = render_common_sidebar()
state = st.session_state.get("app_state", "APP_LOCKED")

# ── 1. LOCKED / UPLOADER STATE ONBOARDING ───────────────────────────────────
if state in ["APP_LOCKED", "ERROR", "PROCESSING"]:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0 1rem 0;">
        <div style="font-size:5rem; margin-bottom:1.5rem;">🔒</div>
        <h1 style="font-size:2.8rem; font-weight:900; color:#f0b429; margin:0; letter-spacing:2px;">
            DRUG SAFETY INTELLIGENCE
        </h1>
        <p style="color:#94a3b8; font-size:1.2rem; margin-top:0.5rem; font-weight: 500;">
            FDA FAERS Pharmacovigilance & Predictive Analytics Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #111827 0%, #1c2536 100%); border: 1px solid #30363d; border-radius: 12px; padding: 2rem; text-align: center; max-width: 800px; margin: 0 auto 2.5rem auto;">
        <h3 style="color:#f0b429; margin-top:0; font-size:1.35rem; font-weight:700; letter-spacing:1px;">
            🔌 PLATFORM LOCKED — DATASET INITIALIZATION REQUIRED
        </h3>
        <p style="color:#94a3b8; font-size:0.95rem; line-height:1.6; margin-bottom: 1.5rem;">
            Welcome to the Drug Safety platform. To activate the analytical dashboards, safety signals, 
            and AI risk prediction models, you must first initialize the platform by uploading a clinical reports dataset.
        </p>
        <div style="text-align:left; background:rgba(0,0,0,0.3); padding:1rem; border-radius:8px; border:1px solid #1e2d3d; font-size:0.85rem; color:#e2e8f0;">
            <b style="color:#f0b429;">📋 Required Dataset Rules:</b>
            <ul style="margin:0.5rem 0 0 1rem; padding:0; line-height:1.6;">
                <li>Supports standard clinical formats: <b>CSV</b> (.csv) or <b>Excel</b> (.xlsx, .xls)</li>
                <li>Validates and standardizes headers automatically using synonym dictionaries.</li>
                <li>Handles missing variables gracefully (auto-imputes age, weight, and clinical flags).</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if state == "ERROR":
        st.error("❌ **Dataset Validation Failed**: The uploaded file contains empty rows, corrupted formats, or incompatible columns. Please review your file and try uploading again below.")
    
    # Centered drag & drop file uploader
    col_l, col_m, col_r = st.columns([1, 4, 1])
    with col_m:
        uploaded_file = st.file_uploader(
            "Drag and drop clinical adverse events dataset here (CSV or XLSX)",
            type=["csv", "xlsx"],
            help="Upload the FDA FAERS adverse events spreadsheet to compile the platform."
        )
        
        if uploaded_file is not None:
            st.session_state["app_state"] = "PROCESSING"
            with st.status("🔍 Ingesting and validating dataset...", expanded=True) as loader:
                try:
                    loader.write("Reading file structure...")
                    if uploaded_file.name.endswith((".xlsx", ".xls")):
                        try:
                            df = pd.read_excel(uploaded_file)
                        except ImportError:
                            st.session_state["app_state"] = "ERROR"
                            loader.update(label="Ingestion failed: Excel engine dependencies missing.", state="error")
                            st.error("Excel loader is missing system dependencies. Please convert your file to a standard CSV format and upload again.")
                            st.stop()
                    else:
                        df = pd.read_csv(uploaded_file, low_memory=False)
                    
                    if df.empty:
                        raise ValueError("The uploaded spreadsheet contains no records (empty file).")
                    
                    loader.write("Applying clinical standardizations and mapping synonyms...")
                    from utils.preprocessing import standardize_dataframe
                    df_clean = standardize_dataframe(df)
                    
                    loader.write("Saving standardized dataset to platform disk container...")
                    dest_path = _PROJECT_DIR / _CSV_NAME
                    df_clean.to_csv(dest_path, index=False)
                    
                    loader.update(label="Dataset standardized successfully! Transitioning pipeline...", state="complete")
                    st.session_state["app_state"] = "DATASET_UPLOADED"
                    st.rerun()
                except Exception as e:
                    st.session_state["app_state"] = "ERROR"
                    loader.update(label=f"Ingestion failed: {e}", state="error")
            
    st.stop()

# ── 2. AUTO AI COMPILATION PIPELINE STATE ────────────────────────────────────
elif state == "DATASET_UPLOADED":
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0 1rem 0;">
        <div style="font-size:5rem; margin-bottom:1.5rem;">⚡</div>
        <h1 style="font-size:2.8rem; font-weight:900; color:#f0b429; margin:0; letter-spacing:2px;">
            COMPILING AI MODELS
        </h1>
        <p style="color:#94a3b8; font-size:1.2rem; margin-top:0.5rem; font-weight: 500;">
            Initializing Multi-Classifier Risk Predictor Pipeline
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #111827 0%, #1c2536 100%); border: 1px solid #f97316; border-radius: 12px; padding: 2rem; text-align: center; max-width: 800px; margin: 0 auto 2.5rem auto;">
        <h3 style="color:#f97316; margin-top:0; font-size:1.35rem; font-weight:700; letter-spacing:1px;">
            🚀 DATASET INSTALLED — TRAINING PREDICTIVE SUITE
        </h3>
        <p style="color:#94a3b8; font-size:0.95rem; line-height:1.6; margin-bottom: 0;">
            The platform is automatically compiling the AI model suite (<b>LightGBM</b>, <b>Random Forest</b>, and <b>Logistic Regression</b>) 
            directly from the uploaded dataset. This will take ~30 seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 4, 1])
    with col_m:
        with st.status("Training Predictive AI Models...", expanded=True) as status_box:
            class StreamToStreamlit:
                def __init__(self, container):
                    self.container = container
                    self.old_stdout = sys.stdout
                def __enter__(self):
                    sys.stdout = self
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    sys.stdout = self.old_stdout
                def write(self, string):
                    self.old_stdout.write(string)
                    stripped = string.strip()
                    if stripped:
                        self.container.write(stripped)
                def flush(self):
                    self.old_stdout.flush()
            
            try:
                with StreamToStreamlit(status_box):
                    import export_models
                    importlib.reload(export_models)
                    export_models.main()
                status_box.update(label="⚡ All models compiled & validated successfully!", state="complete")
                st.session_state["app_state"] = "READY"
                st.rerun()
            except Exception as e:
                st.session_state["app_state"] = "ERROR"
                status_box.update(label=f"❌ Training failed: {e}", state="error")
                st.stop()

# ── 3. READY STATE: MAIN SAAS PORTAL ──────────────────────────────────────────
elif state == "READY":
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0 1rem 0;">
        <div style="font-size:3.5rem; margin-bottom:1rem;">🧬</div>
        <h1 style="font-size:2.5rem; font-weight:800; color:#f0b429; margin:0; letter-spacing:2px;">
            DRUG SAFETY INTELLIGENCE PORTAL
        </h1>
        <p style="color:#94a3b8; font-size:1.1rem; margin-top:0.5rem;">
            FDA FAERS Pharmacovigilance Suite · Predictive AI Active
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ────────────────────────────────────────────────────────────────
    from utils.data_loader import HARDCODED_KPIS

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

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:#111827; border:1px solid #1e2d3d; border-radius:12px; padding:2rem; max-width:900px; margin:0 auto;">
        <h3 style="color:#f0b429; margin-top:0; font-size:1.25rem;">🚀 Welcome to Your Active Dashboard</h3>
        <p style="color:#94a3b8; font-size:0.95rem; line-height:1.6;">
            The uploaded dataset and clinical reporting features are fully processed. Navigate through the sidebar pages to inspect real-time metrics:
        </p>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top:1.5rem;">
            <div style="background:rgba(255,255,255,0.02); padding:1rem; border-radius:8px; border:1px solid #1e2d3d;">
                <b style="color:#e2e8f0;">📊 Data Overview</b><br>
                <span style="color:#94a3b8; font-size:0.85rem;">Inspect active reporting volume, row data schema, and variables.</span>
            </div>
            <div style="background:rgba(255,255,255,0.02); padding:1rem; border-radius:8px; border:1px solid #1e2d3d;">
                <b style="color:#e2e8f0;">🔍 EDA Dashboard</b><br>
                <span style="color:#94a3b8; font-size:0.85rem;">View suspect drugs, quarterly trend lines, and outcome ratios.</span>
            </div>
            <div style="background:rgba(255,255,255,0.02); padding:1rem; border-radius:8px; border:1px solid #1e2d3d;">
                <b style="color:#e2e8f0;">🤖 AI Risk Calculator</b><br>
                <span style="color:#94a3b8; font-size:0.85rem;">Score patient profiles and risk triggers using the 3 trained models.</span>
            </div>
            <div style="background:rgba(255,255,255,0.02); padding:1rem; border-radius:8px; border:1px solid #1e2d3d;">
                <b style="color:#e2e8f0;">📈 Performance scorecards</b><br>
                <span style="color:#94a3b8; font-size:0.85rem;">Examine diagnostic ROC-AUC curves, confusion matrices, and features.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
