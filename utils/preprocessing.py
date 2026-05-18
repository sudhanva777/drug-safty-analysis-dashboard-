"""
Preprocessing helpers for the Drug Safety Dashboard.
Encoding and null-filling logic factored out from the notebook.
"""
import pandas as pd
import numpy as np

# Approximate medians from FAERS data — used for null filling
NUM_MEDIANS = {
    "num_drugs": 2,
    "num_reactions": 2,
    "patient_age_years": 56.0,
    "patient_weight_kg": 75.0,
}

CAT_COLS = [
    "patient_sex", "country", "drug_route",
    "age_group", "drug_count_category",
]

# Column synonyms mapping for highly robust CSV ingestion
SYNONYMS = {
    "is_fatal": ["is_fatal", "fatal", "death", "died", "is fatal", "outcome_fatal", "patient_died"],
    "is_hospitalized": ["is_hospitalized", "hospitalized", "hospitalization", "is hospitalized", "hosp"],
    "is_life_threat": ["is_life_threat", "life_threat", "life-threatening", "is life threat", "life threat"],
    "is_disabling": ["is_disabling", "disabling", "disabled", "is disabling", "disability"],
    "patient_age_years": ["patient_age_years", "age", "patient_age", "age_years", "patient age"],
    "patient_weight_kg": ["patient_weight_kg", "weight", "patient_weight", "weight_kg", "patient weight"],
    "patient_sex": ["patient_sex", "sex", "gender", "patient_gender", "patient sex"],
    "country": ["country", "patient_country", "reporter_country"],
    "suspect_drug": ["suspect_drug", "drug", "drug_name", "drugname", "active_substance", "suspect drug"],
    "primary_reaction": ["primary_reaction", "reaction", "adverse_event", "adverse_reaction", "symptom", "primary reaction"],
    "num_drugs": ["num_drugs", "drug_count", "number_of_drugs", "num drugs", "drug count"],
    "num_reactions": ["num_reactions", "reaction_count", "number_of_reactions", "num reactions", "reaction count"],
    "receive_date": ["receive_date", "date", "report_date", "receive date"],
    "manufacturer": ["manufacturer", "mfg", "drug_manufacturer"],
    "serious": ["serious", "is_serious", "severity"],
    "report_id": ["report_id", "id", "report_number", "case_id", "report id"],
}


def get_age_group(age: float) -> str:
    """Binned age category definition."""
    if pd.isna(age):
        return "Unknown"
    if age <= 18:
        return "Pediatric(0-18)"
    elif age <= 40:
        return "Adult(19-40)"
    elif age <= 65:
        return "Middle-Aged(41-65)"
    elif age <= 80:
        return "Senior(66-80)"
    else:
        return "Elderly(81+)"


def get_drug_category(count: int) -> str:
    """Binned polypharmacy category definition."""
    if pd.isna(count):
        return "Unknown"
    if count == 1:
        return "Single"
    elif count <= 3:
        return "2-3 drugs"
    elif count <= 5:
        return "4-5 drugs"
    else:
        return "Polypharmacy(6+)"


def fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric nulls with median approximations."""
    for col, median in NUM_MEDIANS.items():
        if col in df.columns:
            df[col] = df[col].fillna(median)
    return df


def encode_input(input_dict: dict, le_dict: dict, features: list) -> pd.DataFrame:
    """
    Encode a single prediction input dict using the trained LabelEncoders.
    Handles unseen labels gracefully by mapping to 0 (Unknown class).
    """
    df = pd.DataFrame([input_dict])

    # Fill numeric nulls
    df = fill_nulls(df)

    # Encode categoricals
    for col in CAT_COLS:
        if col in df.columns and col in le_dict:
            le = le_dict[col]
            val = str(df[col].iloc[0])
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                # Unseen value — try "Unknown", else fallback to 0
                if "Unknown" in le.classes_:
                    df[col] = le.transform(["Unknown"])[0]
                else:
                    df[col] = 0

    # Bool → int
    for col in ["is_hospitalized", "is_life_threat", "is_disabling"]:
        if col in df.columns:
            df[col] = int(bool(df[col].iloc[0]))

    return df[features]


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the uploaded DataFrame:
      1. Map column synonyms to canonical names.
      2. Synthesize/impute missing required columns with smart clinical defaults.
      3. Coerce column types to prevent downstream crashes.
      4. Handle empty/corrupt records gracefully.
    """
    df = df.copy()

    # 1. Map synonyms
    col_mapping = {}
    for canonical, synonyms in SYNONYMS.items():
        for syn in synonyms:
            matches = [c for c in df.columns if c.strip().lower() == syn.lower()]
            if matches:
                col_mapping[matches[0]] = canonical
                break
    if col_mapping:
        df = df.rename(columns=col_mapping)

    # 2. Check target column availability
    if "is_fatal" not in df.columns:
        raise ValueError(
            "Fatal outcome column ('is_fatal') is missing in the uploaded dataset.\n"
            "Please make sure your CSV contains an outcome column representing patient death (e.g., 'is_fatal', 'died', 'death')."
        )

    # 3. Handle empty datasets
    if len(df) == 0:
        raise ValueError("The uploaded CSV file is empty. Please upload a valid dataset.")

    # 4. Fill required columns & synthesize missing columns
    if "receive_date" not in df.columns:
        df["receive_date"] = pd.to_datetime("2020-01-01")
    else:
        df["receive_date"] = pd.to_datetime(df["receive_date"], errors="coerce").fillna(pd.to_datetime("2020-01-01"))

    df["year"] = df["receive_date"].dt.year.fillna(2020).astype(int)

    if "report_id" not in df.columns:
        df["report_id"] = range(1000000, 1000000 + len(df))

    if "suspect_drug" not in df.columns:
        df["suspect_drug"] = "UNKNOWN DRUG"
    else:
        df["suspect_drug"] = df["suspect_drug"].fillna("UNKNOWN DRUG").astype(str).str.upper()

    if "primary_reaction" not in df.columns:
        df["primary_reaction"] = "Unknown adverse event"
    else:
        df["primary_reaction"] = df["primary_reaction"].fillna("Unknown adverse event").astype(str)

    if "manufacturer" not in df.columns:
        df["manufacturer"] = "Unknown Manufacturer"
    else:
        df["manufacturer"] = df["manufacturer"].fillna("Unknown Manufacturer").astype(str)

    if "serious" not in df.columns:
        df["serious"] = df["is_fatal"].apply(lambda x: "Yes" if x else "No")
    else:
        df["serious"] = df["serious"].fillna("No").astype(str)

    if "patient_age_years" not in df.columns:
        df["patient_age_years"] = 56.0
    else:
        df["patient_age_years"] = pd.to_numeric(df["patient_age_years"], errors="coerce").fillna(56.0)

    if "patient_weight_kg" not in df.columns:
        df["patient_weight_kg"] = 75.0
    else:
        df["patient_weight_kg"] = pd.to_numeric(df["patient_weight_kg"], errors="coerce").fillna(75.0)

    if "num_drugs" not in df.columns:
        df["num_drugs"] = 2
    else:
        df["num_drugs"] = pd.to_numeric(df["num_drugs"], errors="coerce").fillna(2).astype(int)

    if "num_reactions" not in df.columns:
        df["num_reactions"] = 2
    else:
        df["num_reactions"] = pd.to_numeric(df["num_reactions"], errors="coerce").fillna(2).astype(int)

    # Bools
    for col in ["is_fatal", "is_hospitalized", "is_life_threat", "is_disabling"]:
        if col not in df.columns:
            df[col] = False
        else:
            def map_bool(val):
                if pd.isna(val):
                    return False
                s = str(val).strip().lower()
                return s in ["true", "1", "yes", "y", "fatal", "hospitalized", "disabling", "life threat"]
            df[col] = df[col].apply(map_bool)

    # Categoricals
    if "patient_sex" not in df.columns:
        df["patient_sex"] = "Unknown"
    else:
        df["patient_sex"] = df["patient_sex"].fillna("Unknown").astype(str)

    if "country" not in df.columns:
        df["country"] = "Unknown"
    else:
        df["country"] = df["country"].fillna("Unknown").astype(str).str.upper()

    if "drug_route" not in df.columns:
        df["drug_route"] = "Unknown"
    else:
        df["drug_route"] = df["drug_route"].fillna("Unknown").astype(str)

    # Binned columns
    if "age_group" not in df.columns:
        df["age_group"] = df["patient_age_years"].apply(get_age_group)
    else:
        df["age_group"] = df["age_group"].fillna("Unknown").astype(str)

    if "drug_count_category" not in df.columns:
        df["drug_count_category"] = df["num_drugs"].apply(get_drug_category)
    else:
        df["drug_count_category"] = df["drug_count_category"].fillna("Unknown").astype(str)

    return df
