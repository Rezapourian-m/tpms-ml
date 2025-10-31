from __future__ import annotations
import pandas as pd

def categorize_size(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def _cat(row):
        h, r = row.get("Height", None), row.get("Radius", None)
        if h is None or r is None:
            return "Unknown"
        if h <= 9 and r <= 4:
            return "Small-sized"
        elif h <= 15 and r <= 7:
            return "Medium-sized"
        else:
            return "Large-sized"
    df["Implant Size"] = df.apply(_cat, axis=1)
    return df

def categorize_application(df: pd.DataFrame, ea_threshold=30.0):
    """Categorize samples for applications based on your notebook logic.

    Defaults:
      - Load-Bearing if:
        Elastic Modulus >= 7 GPa,
        Yield Stress >= 70 MPa,
        Ultimate Stress >= 150 MPa
      - Else Energy-Absorbing if Energy Absorption >= ea_threshold (default 30 MJ/m^3)
      - Else Other
    """
    df = df.copy()
    def _app(row):
        if (
            row.get("Elastic Modulus", 0) >= 7 and
            row.get("Yield Stress", 0) >= 70 and
            row.get("Ultimate Stress", 0) >= 150
        ):
            return "Load-Bearing"
        elif row.get("Energy Absorption", 0) >= ea_threshold:
            return "Energy-Absorbing"
        return "Other"
    df["Category"] = df.apply(_app, axis=1)
    return df

def finalize_dataset(df: pd.DataFrame, add_size=True, add_app=True, ea_threshold=30.0) -> pd.DataFrame:
    if add_size:
        df = categorize_size(df)
    if add_app:
        df = categorize_application(df, ea_threshold=ea_threshold)
    return df
