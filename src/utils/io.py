from pathlib import Path
import joblib

def save_scaler(scaler, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(path: str | Path):
    return joblib.load(path)
