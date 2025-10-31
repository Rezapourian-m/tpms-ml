import argparse, json
import pandas as pd
from pathlib import Path
from src.data.load import load_csv
from src.models.ann_inverse import train_inverse
from src.utils.io import save_scaler

def main():
    ap = argparse.ArgumentParser(description="Train inverse ANN (targets->design).")
    ap.add_argument("--data", required=True, help="Categorized CSV path")
    ap.add_argument("--mode", choices=["app","type"], default="app", help="Train by application or lattice type")
    ap.add_argument("--app", choices=["Load-Bearing","Energy-Absorbing"], help="Required if mode=app")
    ap.add_argument("--lattice", choices=["D","G","S"], help="Required if mode=type")
    ap.add_argument("--model-out", required=True, help="Path to save Keras model (.keras)")
    ap.add_argument("--scaler-x-out", required=True, help="Input scaler path (.joblib)")
    ap.add_argument("--scaler-y-out", required=True, help="Output scaler path (.joblib)")
    ap.add_argument("--metrics-out", default=None, help="Optional metrics JSON path")
    args = ap.parse_args()

    df = load_csv(args.data)
    model, sx, sy, scores = train_inverse(df, mode=args.mode, app=args.app, lattice=args.lattice)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    save_scaler(sx, args.scaler_x_out)
    save_scaler(sy, args.scaler_y_out)

    if args.metrics_out:
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)
    print("✅ Inverse model saved. Scores:", scores)

if __name__ == "__main__":
    main()
