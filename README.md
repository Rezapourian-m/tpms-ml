# TPMS-ML: Forward & Inverse Modeling Toolkit

This repository organizes your original Colab script into a clean, modular Python project for
**TPMS lattice design ↔ mechanical properties** modeling. It includes:
- **Data loading & cleaning** (column unification across multiple naming styles)
- **Feature engineering** (H/D ratio, implant size categories, application categories)
- **Baselines** (Linear Regression, Random Forest, XGBoost via MultiOutput wrapper)
- **ANN forward models** (design ➜ mechanical properties)
- **ANN inverse models** (mechanical targets ➜ design parameters) including per-lattice models (D/G/S)
- **Evaluation plots** and **CLI utilities**
- Per-section **README** files and a simple **Makefile** for common tasks

> ✅ Drop your CSV (e.g., `results_table.csv`) into `data/` and follow the quick start below.

---

## Repo Structure

```
tpms-ml/
├─ data/
│  └─ README.md
├─ models/                  # Saved models & scalers will go here
│  └─ .gitkeep
├─ src/
│  ├─ data/                 # Loading, cleaning, splits
│  ├─ features/             # Feature engineering & categorization
│  ├─ models/               # Baselines, forward & inverse ANNs, forward validation
│  ├─ eval/                 # Plotting utilities
│  ├─ utils/                # I/O, paths, metrics, seeding helpers
│  └─ cli/                  # Command-line entry points
├─ docs/                    # Section readmes (duplicated inside each package too)
├─ .gitignore
├─ requirements.txt
├─ LICENSE
├─ Makefile
└─ README.md
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **TensorFlow note:** If you prefer CPU-only, keep as-is. For GPU, install the matching TF build for your CUDA setup.

## Data: expected columns

Your original code used different names. The loader **standardizes** them automatically. Any of these will be unified:

- **Design inputs**  
  - `Thickness (mm)` / `Thickness_mm` / `Thickness` → `Thickness`  
  - `xCell`/`Xcell` → `Xcell`, `yCell`/`Ycell` → `Ycell`, `zCell`/`Zcell` → `Zcell`  
  - `Angle (°)` / `Angle_deg` / `Angle` → `Angle`  
  - `Height (mm)` / `Height_mm` / `Height` → `Height`  
  - `Radius (mm)` / `Radius_mm` / `Radius` → `Radius` (and `Diameter` is derived if missing)  
  - `Code` / `type` (values like D/G/S) → `type` (uppercase)

- **Mechanical/geom outputs**  
  - `Elastic Modulus (GPa)` → `Elastic Modulus`  
  - `Yield Stress (MPa)` → `Yield Stress`  
  - `Ultimate Stress (MPa)` → `Ultimate Stress`  
  - `Energy Absorption (MJ/m^3)` → `Energy Absorption`  
  - `Plateau Stress (MPa)` → `Plateau Stress`  
  - `SA (cm^2)` → `SA`, `SA/VR (1/cm)` → `SA/VR`, `Porosity` stays the same, `RD` stays the same

## Quick Start

1) **Clean & standardize your dataset** (also creates helper features like H/D):
```bash
python -m src.cli.categorize --data data/results_table.csv --out data/dataset_clean.csv
```

2) **(Optional) Label for applications** (Load-Bearing vs Energy-Absorbing), and save as `categorized_designs.csv`:
```bash
python -m src.cli.categorize --data data/dataset_clean.csv --categorize-app --out data/categorized_designs.csv
```

3) **Train forward model (design ➜ mech)** for 3-output (E/Y/UTS) or 2-output (EA/Plateau):
```bash
# A: Elastic Modulus, Yield Stress, Ultimate Stress
python -m src.cli.train_forward --data data/dataset_clean.csv --task A --model-out models/forward_A.keras --scaler-out models/forward_A_scaler.joblib

# B: Energy Absorption, Plateau Stress
python -m src.cli.train_forward --data data/dataset_clean.csv --task B --model-out models/forward_B.keras --scaler-out models/forward_B_scaler.joblib
```

4) **Train inverse models** (two flavors):
- Global by **application** (Load-Bearing / Energy-Absorbing) using `categorized_designs.csv`
```bash
python -m src.cli.train_inverse --data data/categorized_designs.csv --mode app --app Load-Bearing  --model-out models/inv_load.keras  --scaler-x-out models/inv_load_x.joblib  --scaler-y-out models/inv_load_y.joblib
python -m src.cli.train_inverse --data data/categorized_designs.csv --mode app --app Energy-Absorbing --model-out models/inv_energy.keras --scaler-x-out models/inv_energy_x.joblib --scaler-y-out models/inv_energy_y.joblib
```
- Per **lattice type** (D / G / S):
```bash
python -m src.cli.train_inverse --data data/categorized_designs.csv --mode type --lattice D --model-out models/inv_D.keras --scaler-x-out models/inv_D_x.joblib --scaler-y-out models/inv_D_y.joblib
```

5) **Predict** from trained models:
```bash
# Forward: from design(+type) to properties
python -m src.cli.predict_forward --model models/forward_A.keras --scaler models/forward_A_scaler.joblib --json '[{"Xcell":4,"Ycell":4,"Zcell":4,"Angle":0,"Thickness":0.3,"Height":15,"Radius":6,"type":"G"}]'

# Inverse (e.g., Load-Bearing): from targets to design
python -m src.cli.predict_inverse --model models/inv_load.keras --scaler-x models/inv_load_x.joblib --scaler-y models/inv_load_y.joblib --json '{"Elastic Modulus":20,"Yield Stress":100,"Ultimate Stress":200,"Height":15,"Diameter":12}'
```

> Each CLI prints a compact table and also writes a CSV alongside your model by default.

## Notes & Differences vs Original Notebook

- Removed Colab-only bits (`!pip install ...`, inline figures scattered around, broken label encoders).
- Fixed **multi-output XGBoost** via `MultiOutputRegressor` wrapper.
- Centralized **column standardization** to support both `results_table.csv` and `results_table_.csv` and mixed headers.
- Provided **deterministic seeds**, **model persistence**, clean **plots**, and modular code.
- Kept your **categorization thresholds** as defaults (tweak on CLI).

## License

MIT — see `LICENSE`.

---

**Have fun modeling!**
