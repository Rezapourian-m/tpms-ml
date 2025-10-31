from __future__ import annotations
import pandas as pd
from typing import Dict

# Map variant column names to canonical ones
CANONICAL = {
    # design
    "Thickness": ["Thickness (mm)", "Thickness_mm", "thickness", "Thickness"],
    "Xcell": ["Xcell", "xCell", "xcell"],
    "Ycell": ["Ycell", "yCell", "ycell"],
    "Zcell": ["Zcell", "zCell", "zcell"],
    "Angle": ["Angle (°)", "Angle_deg", "Angle", "angle"],
    "Height": ["Height (mm)", "Height_mm", "Height", "height"],
    "Radius": ["Radius (mm)", "Radius_mm", "Radius", "radius"],
    "Diameter": ["Diameter", "diameter", "Dia"],
    "type": ["type", "Type", "CODE", "Code"],
    # outputs
    "Elastic Modulus": ["Elastic Modulus (GPa)", "ElasticModulus", "E_modulus", "Elastic Modulus"],
    "Yield Stress": ["Yield Stress (MPa)", "YieldStress", "Yield Stress"],
    "Ultimate Stress": ["Ultimate Stress (MPa)", "UltimateStress", "Ultimate Stress"],
    "Energy Absorption": ["Energy Absorption (MJ/m^3)", "EnergyAbsorption", "Energy Absorption"],
    "Plateau Stress": ["Plateau Stress (MPa)", "PlateauStress", "Plateau Stress"],
    "SA": ["SA (cm^2)", "SA_cm2", "SA"],
    "SA/VR": ["SA/VR (1/cm)", "SA_VR", "SA/VR"],
    "Porosity": ["Porosity", "porosity"],
    "RD": ["RD", "Relative Density (%)", "RelativeDensity"],
}

def _reverse_map() -> Dict[str, str]:
    rev = {}
    for canonical, variants in CANONICAL.items():
        for v in variants:
            rev[v.lower()] = canonical
    return rev

REV = _reverse_map()

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        key = REV.get(c.strip().lower(), c)
        new_cols.append(key)
    df = df.copy()
    df.columns = new_cols

    # type normalization
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip().str.upper()
    elif "Code" in df.columns:
        df["type"] = df["Code"].astype(str).str.strip().str.upper()

    # Diameter derivation if missing
    if "Diameter" not in df.columns and "Radius" in df.columns:
        df["Diameter"] = df["Radius"] * 2.0

    # H/D ratios
    if "Height" in df.columns and "Radius" in df.columns:
        df["H_over_R"] = df["Height"] / df["Radius"].replace(0, pd.NA)
    if "Height" in df.columns and "Diameter" in df.columns:
        df["H_over_D"] = df["Height"] / df["Diameter"].replace(0, pd.NA)

    return df

def load_csv(path: str, encoding: str | None = None) -> pd.DataFrame:
    encodings = [encoding] if encoding else ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            return standardize_columns(df)
        except Exception as e:
            last_err = e
    raise last_err
