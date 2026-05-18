"""
PAGE 4 — MODEL PREDICTION
User Tool: Real-time fatality risk prediction.
"""
import streamlit as st
from utils.model import predict_risk, load_model
from utils.charts import plot_risk_gauge

st.markdown(
    '<div class="section-header">🤖 Fatality Risk Prediction — LightGBM</div>',
    unsafe_allow_html=True,
)

st.markdown("""
> Enter patient and drug details below. The LightGBM model (AUC 0.7829) will predict
> the **probability of a fatal adverse event outcome**.
""")

# ── Check model availability ──────────────────────────────────────────────
try:
    _, le_dict, threshold = load_model()
    model_available = True
except FileNotFoundError:
    model_available = False
    st.error("""
    ⚠️ Model artifacts not found. Please run `export_models.py` from your
    notebook environment first to generate the required files in `models/`.
    """)

if model_available:
    # Build dropdown choices from the trained encoders so they always match
    def _enc_classes(col, defaults):
        """Return encoder classes for *col*, or *defaults* if unavailable."""
        if col in le_dict:
            return sorted(le_dict[col].classes_.tolist())
        return defaults

    _age_groups  = _enc_classes("age_group", ["Adult(19-40)", "Middle-Aged(41-65)", "Senior(66-80)", "Elderly(81+)", "Unknown"])
    _sex_opts    = _enc_classes("patient_sex", ["Female", "Male", "Unknown"])
    _drug_cats   = _enc_classes("drug_count_category", ["Single", "2-3 Drugs", "4-5 Drugs", "Polypharmacy(6+)", "Unknown"])
    _routes      = _enc_classes("drug_route", ["Unknown"])
    _countries   = _enc_classes("country", ["US", "JP", "GB", "DE", "FR", "CA", "Unknown"])

    # ── INPUT FORM ────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        st.markdown("#### 👤 Patient Profile")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Patient Age (years)", 1, 100, 55)
            age_group = st.selectbox("Age Group", _age_groups,
                                     index=_age_groups.index("Middle-Aged(41-65)") if "Middle-Aged(41-65)" in _age_groups else 0)
            sex = st.selectbox("Patient Sex", _sex_opts)

        with col2:
            weight = st.slider("Weight (kg)", 20.0, 200.0, 75.0, step=0.5)
            num_drugs = st.slider("Number of Drugs", 1, 15, 2)
            drug_count_cat = st.selectbox("Drug Count Category", _drug_cats,
                                          index=_drug_cats.index("Single") if "Single" in _drug_cats else 0)

        with col3:
            num_reactions = st.slider("Number of Adverse Reactions", 1, 20, 2)
            drug_route = st.selectbox("Drug Route", _routes,
                                      index=_routes.index("Unknown") if "Unknown" in _routes else 0)
            country = st.selectbox("Country", _countries,
                                   index=_countries.index("US") if "US" in _countries else 0)

        st.markdown("#### 🏥 Clinical Flags")
        cl1, cl2, cl3 = st.columns(3)
        with cl1:
            is_hospitalized = st.checkbox("Hospitalized", value=False)
        with cl2:
            is_life_threat = st.checkbox("Life-Threatening", value=False)
        with cl3:
            is_disabling = st.checkbox("Disabling", value=False)

        submitted = st.form_submit_button("🔮 Predict Risk", use_container_width=True)

    # ── PREDICTION OUTPUT ──────────────────────────────────────────────────
    if submitted:
        input_dict = {
            "num_drugs": num_drugs,
            "num_reactions": num_reactions,
            "patient_age_years": float(age),
            "patient_weight_kg": float(weight),
            "patient_sex": sex,
            "country": country,
            "drug_route": drug_route,
            "age_group": age_group,
            "drug_count_category": drug_count_cat,
            "is_hospitalized": int(is_hospitalized),
            "is_life_threat": int(is_life_threat),
            "is_disabling": int(is_disabling),
        }

        with st.spinner("Running prediction..."):
            result = predict_risk(input_dict)

        st.divider()
        st.markdown("#### 📊 Prediction Result")

        col_gauge, col_details = st.columns([1, 1.3])

        with col_gauge:
            st.plotly_chart(plot_risk_gauge(result["probability"]), use_container_width=True)

        with col_details:
            # Risk badge
            risk_class = f"risk-{result['risk_level'].lower()}"
            st.markdown(f"""
            <div style="text-align:center; padding:1rem 0;">
                <div class="risk-badge {risk_class}">
                    {result['risk_emoji']} {result['risk_level']} RISK
                </div>
                <div style="color:#e2e8f0; font-size:1.8rem; font-weight:700; margin:0.5rem 0;">
                    {result['probability_pct']:.1f}% Fatality Probability
                </div>
                <div style="color:#94a3b8; font-size:0.85rem;">
                    Model confidence: {result['confidence']:.1f}% &nbsp;|&nbsp;
                    Threshold used: {result['threshold_used']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(min(result["probability_pct"], 100)))

            # Contextual alerts
            if result["risk_level"] == "HIGH":
                st.markdown("""
                <div class="alert-box alert-fatal">
                    ⚠️ <b>HIGH RISK DETECTED</b> — This patient profile shows elevated fatality
                    probability. Clinical review strongly recommended.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-box alert-safe">
                    ✅ <b>LOW-MEDIUM RISK</b> — Fatality probability within expected range.
                    Standard monitoring protocols apply.
                </div>
                """, unsafe_allow_html=True)

            # Key risk factors
            st.markdown("**📌 Input Summary**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"Age: {age} yrs ({age_group})")
                st.write(f"Sex: {sex}")
                st.write(f"Drugs: {num_drugs} ({drug_count_cat})")
            with col_b:
                st.write(f"Reactions: {num_reactions}")
                st.write(f"Hospitalized: {'Yes' if is_hospitalized else 'No'}")
                st.write(f"Life-Threat: {'Yes' if is_life_threat else 'No'}")

        # ── BENCHMARK ──────────────────────────────────────────────────────
        st.divider()
        st.markdown("#### 📏 Benchmark Context")
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric(
            "Population Fatality Rate", "10.3%",
            delta=f"{result['probability_pct'] - 10.3:+.1f}% vs. this patient",
            delta_color="inverse",
        )
        bc2.metric("Elderly(81+) Rate", "~22%", help="Highest-risk age group in FAERS data")
        bc3.metric("Polypharmacy(6+) Rate", "~16%", help="Higher risk with 6+ concurrent drugs")
