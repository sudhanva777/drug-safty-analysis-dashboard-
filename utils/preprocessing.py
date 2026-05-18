"""
Preprocessing helpers for the Drug Safety Dashboard.
Encoding and null-filling logic factored out from the notebook.
"""
import pandas as pd

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
