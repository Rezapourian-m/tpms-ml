import argparse, pandas as pd
from src.data.load import load_csv, standardize_columns
from src.features.preprocess import finalize_dataset

def main():
    ap = argparse.ArgumentParser(description="Standardize columns and (optionally) add categories.")
    ap.add_argument("--data", required=True, help="Path to input CSV")
    ap.add_argument("--out", required=True, help="Path to write cleaned CSV")
    ap.add_argument("--categorize-app", action="store_true", help="Add application category labels")
    ap.add_argument("--ea-threshold", type=float, default=30.0, help="EA threshold for Energy-Absorbing category")
    args = ap.parse_args()

    df = load_csv(args.data)
    df = finalize_dataset(df, add_size=True, add_app=args.categorize_app, ea_threshold=args.ea_threshold)
    df.to_csv(args.out, index=False)
    print(f"✅ Saved cleaned dataset to: {args.out}  (rows={len(df)})")

if __name__ == "__main__":
    main()
