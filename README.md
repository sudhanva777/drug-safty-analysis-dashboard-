# 🧬 Drug Safety Intelligence Platform
### FDA FAERS (2015–2026) · Multi-Model AI (LightGBM, RF, LR) · 528,000+ Adverse Event Reports

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit v1.35+](https://img.shields.io/badge/Streamlit-v1.35+-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LightGBM v4.3+](https://img.shields.io/badge/Model-LightGBM--v4.3-green.svg?style=for-the-badge&logo=analytics&logoColor=white)](https://lightgbm.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.4+-orange.svg?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A high-performance, clinical-grade pharmacological intelligence platform designed to ingest, analyze, and model drug-related adverse events using the **FDA FAERS (Adverse Event Reporting System)** database. Built on a custom glassmorphic dark design system, the dashboard transforms raw multi-million-row pharmacovigilance reports into actionable statistical and predictive insights.

---

## ⚡ SaaS Onboarding & Application State Manager

The platform behaves like a premium, production-grade SaaS product governed by a robust **Application State Manager** (`st.session_state["app_state"]`). The entire platform is dynamically controlled by dataset upload initialization:

### 🔒 1. Platform Lock State (`APP_LOCKED`)
- On startup, the dashboard freezes and locks all subpages, visualizers, predictions, and analytical summaries to protect against empty-dataset crashes and raw Python errors.
- Displays a prominent, centered drag-and-drop file uploader on the home tab, welcoming researchers.

### 🧼 2. Clinical Data Sanitizer & Synonyms Ingestion (`PROCESSING`)
- **Multi-Format Support**: Supports standard **CSV** (`.csv`) and **Excel** (`.xlsx`, `.xls`) uploads.
- **Synonyms Mapping**: Automatically parses headers to canonical form (e.g. `died`, `fatal_flag`, `death` ➔ `is_fatal`; `gender`, `patient sex` ➔ `patient_sex`).
- **Clinical Imputation**: Safely handles null values (median clinical imputations for ages, weights) and performs cohort binning (e.g. `Elderly(81+)`, `Middle-Aged(41-65)`).
- **Graceful Error Handling**: Catches corrupted spreadsheet formats or empty files cleanly, rendering beautiful warning cards in the UI rather than traceback screens.

### 🚀 3. Auto AI Model Compilation (`DATASET_UPLOADED`)
- Once the dataset is sanitized, the dashboard **automatically initiates** the training of the three AI models (**LightGBM**, **Random Forest**, and **Logistic Regression**) in the foreground.
- **Live Console Streaming**: Uses a `StreamToStreamlit` stdout interceptor to stream live Python compilation and estimator training logs directly in the Streamlit Welcome card in real-time!

### ✅ 4. Dynamic Dashboard Unlocking (`READY`)
- Once training completes, the state transitions to `READY`. All subpages are immediately unlocked and populated dynamically using the newly uploaded, standardized dataset.

---

## 🖥️ Platform Showcase

The application is structured as a premium **6-page research dashboard** tailored for clinical toxicologists and pharmacovigilance experts:

### 1. 📊 Data Overview (Trust Layer)
*   **High-Level KPIs**: Instant reporting counts, fatality rates (10.3%), hospitalization rate (35.6%), and unique active entities.
*   **Interactive Schema**: Clear mapping of all variables, data types, and their exact clinical significance.
*   **Missing Value Profiler**: Dynamic charts pinpointing data sparsity across key demographic and clinical dimensions.
*   **Raw Data Explorer**: Interactive preview table with customizable displays.

### 2. 🔍 EDA Dashboard (Exploration Layer)
*   **📅 Temporal Trends**: Annual and quarterly reporting volumes alongside dynamic yearly fatality rates.
*   **💊 Drugs & Reactions**: Bar graphs mapping the top suspect drugs and reactions, polypharmacy risks, and administration routes.
*   **👥 Patient Demographics**: Joint distributions of age vs. fatality rate, biological sex splits, and custom age-binned histograms.
*   **🌍 Geographic Mapping**: Distribution of adverse events across 162 countries.
*   **🔬 Custom Explorer**: Interactive analytical tool enabling custom X-Y scatter plots, box plots, and histograms colored by clinical outcomes.

### 3. 🚨 Safety Signal Detection (Statistical Screening)
*   **Composite Risk Scoring**: Custom statistical formula measuring the risk of a suspect drug:
    $$\text{Risk Score} = 0.5 \times \text{Fatality Rate} + 0.3 \times \text{Life-Threatening Rate} + 0.2 \times \text{Hospitalization Rate}$$
*   **Dynamic Flagging**: Automated categorization into **🔴 HIGH**, **🟡 MEDIUM**, and **🟢 LOW** risk alerts based on reporting thresholds.
*   **Granular Signal Lookup**: Instant search box allowing clinical researchers to look up any drug and view its reporting stats in real-time.

### 4. 🤖 AI Risk Calculator (Model Switcher & Prediction)
*   **Interactive Selector**: Switch between LightGBM, Random Forest, or Logistic Regression on the fly to compare risk predictions.
*   **AI Patient Profiler**: Input panel for age, weight, sex, concurrent drugs, primary route, country, and clinical flags.
*   **Real-time Risk Gauge**: Plotly gauge visualizes the predicted probability of a fatal outcome using the chosen model's optimal threshold.
*   **Clinical Alerts**: Interactive warning cards and population benchmark context cards comparing the patient's individual risk to baseline FAERS averages.

### 5. 📈 Model Performance (Classifier Diagnostics)
*   **Interactive Switcher**: Choose the active model to inspect metrics scorecards, ROC/PR curves, and confusion matrices dynamically.
*   **Performance Scorecard**: Live display of AUC-ROC, Precision, Recall, F1-Score, and overall accuracy.
*   **Feature Importance**: Analyze non-linear gains/impurities for LightGBM and Random Forest dynamically.
*   **Machine Learning Model Card**: Complete list of training parameters, objective functions, imbalance mitigation strategies, and early stopping thresholds for all models.

### 6. 💡 Analytical Insights (Decision Layer)
*   **Interactive Storytelling**: Grounded takeaways summarizing critical clinical trends:
    *   **The Age Factor**: 3.8× higher fatality risk in elderly patients (81+) compared to young adults (19–40).
    *   **The Polypharmacy Effect**: Linear rise in fatal outcomes as concurrent drug intake scales past 6+ drugs.
    *   **Annual Trends**: Post-2020 changes in adverse reporting volumes.

---

## 🏗️ Project Architecture

```text
├── app.py                   # Central Router, State Manager & Onboarding Welcome card
├── export_models.py         # Multi-model AI training pipeline (LightGBM, RF, LR)
├── pages/                   # Multi-page dashboard modules
│   ├── 1_📊_Data_Overview.py    # Schema & dataset summary (Locked until READY)
│   ├── 2_🔍_EDA_Dashboard.py    # Exploratory data analysis (Locked until READY)
│   ├── 3_🚨_Signal_Detection.py  # Statistical signal risk scoring (Locked until READY)
│   ├── 4_🤖_Model_Prediction.py  # Patient risk calculator (Locked until READY)
│   ├── 5_📈_Model_Performance.py # ROC/PR switcher diagnostics (Locked until READY)
│   └── 6_💡_Insights.py         # Executive storytelling summaries (Locked until READY)
├── utils/                   # Backend logic & helper utilities
│   ├── __init__.py          # Package initialization
│   ├── charts.py            # Reusable Plotly chart templates & layout engines
│   ├── data_loader.py       # Optimized data ingestion, common sidebar & state indicators
│   ├── model.py             # Multi-model dynamic loading & inference pipeline
│   └── preprocessing.py     # Synonym mapping, clinical type coercion & label encodings
├── assets/                  # Styling and static assets
│   └── styles.css           # Premium glassmorphic dark theme stylesheet
├── data/                    # Ingested datasets (Git-ignored)
├── models/                  # Trained classifier pickles & thresholds (Git-ignored)
└── requirements.txt         # Project dependencies
```

---

## ⚙️ Installation & Setup

Ensure you have **Python 3.9+** installed. Follow these steps to set up and run the dashboard locally:

### 1. Clone the Repository
```bash
git clone https://github.com/sudhanva777/drug-safety-dashboard.git
cd drug-safety-dashboard
```

### 2. Set Up a Virtual Environment
```powershell
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```
*The entry point script automatically checks the local disk container. If a dataset is missing, it will initialize the onboarding welcome uploader screen. Simply drag and drop your clinical file (CSV/XLSX) to compile and activate the entire platform automatically!*

---

## 🛠️ Technology Stack & Libraries

- **Frontend & App Frame**: [Streamlit v1.35.0](https://streamlit.io/) (High-performance multi-page Python framework)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
- **Machine Learning**: [LightGBM](https://lightgbm.readthedocs.io/) & [Scikit-Learn](https://scikit-learn.org/)
- **Data Serialization**: [Joblib](https://joblib.readthedocs.io/) & [PyArrow](https://arrow.apache.org/) (Parquet format support)
- **Data Visualizations**: [Plotly Express & Graph Objects](https://plotly.github.io/plotly.py-docs/)
- **UI Styling**: Custom HSL dark style with linear gradients & glassmorphism for premium typography and layout.

---

**Developed with ❤️ for Pharmacological Safety & Public Health Research**