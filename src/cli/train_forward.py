import argparse, json
import pandas as pd
from pathlib import Path
from src.data.load import load_csv
from src.models.ann_forward import train_forward
from src.utils.io import save_scaler
from joblib import dump

def main():
    ap = argparse.ArgumentParser(description="Train forward ANN (design->mechanics).")
    ap.add_argument("--data", required=True, help="Cleaned CSV path")
    ap.add_argument("--task", choices=["A","B"], default="A", help="A: [E,Y,UTS], B: [EA,Plateau]")
    ap.add_argument("--model-out", required=True, help="Path to save Keras model (.keras)")
    ap.add_argument("--scaler-out", required=True, help="Path to save input scaler (.joblib)")
    ap.add_argument("--metrics-out", default=None, help="Optional path to write metrics JSON")
    args = ap.parse_args()

    df = load_csv(args.data)
    model, scaler, ohe, metrics, targets = train_forward(df, task=args.task)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    save_scaler(scaler, args.scaler_out)

    # Save OHE as joblib
    dump(ohe, str(Path(args.scaler_out).with_name(Path(args.scaler_out).stem + "_ohe.joblib")))

    if args.metrics_out:
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump({"targets": targets, **metrics}, f, indent=2)
    print("✅ Forward model saved. Metrics:", metrics)

if __name__ == "__main__":
    main()
